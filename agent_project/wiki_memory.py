"""
LLM Wiki Memory — Karpathy-style graph memory

设计哲学 (Page-First)
======================
1. 每个 entity / concept / episode 都是一个 **wiki page**：
   - title:   主题名（实体名）
   - content:  LLM 生成的"文章正文"（自然语言，而非 dict of properties）
   - sections: {section: paragraph}，按维度组织内容，支持增量更新
   - links:   指向其他页面的 ID 列表（hyperlinks → graph edges）

2. 图谱 = 页面之间的双向引用
   - edge = '{source_id} → {target_id}'
   - 每个页面维护自己的 links；WikiGraph 维护 backlinks 全局索引
   - 不存在没有标题/正文的"孤立节点"

3. 存储
   - ChromaDB（可选）存 page 的 *content embedding*
   - JSON 存 pages 全文（可 diff、可读、可检查）
   - graph json 存 edges（link pairs）
   - 降级路径：无 chromadb 时退回 keyword exact match + in-memory

4. LLM 集成
   - 页面创建/更新 由 LLM 生成自然语言摘要
   - 关系推断 由 LLM 推断 link type
   - 调用方式：接受 OpenAI-compatible client，统一接口
"""

from __future__ import annotations

from __future__ import annotations

import uuid, json, re, time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# ──────────────────── optional deps ────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _HAS_VECTOR = True
except ImportError:
    _HAS_VECTOR = False

from pydantic import BaseModel


# ===================================================================
# 0. 最小 LLM Client 接口（可注入）
# ===================================================================

class LLMClient:
    """
    最简 OpenAI-compatible 调用接口。
    传入一个实现 `chat(messages) -> str` 方法的具体 client 即可。
    也可以传入 NIMBackend / OpenAIBackend 实例。
    """

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.3,
             max_tokens: int = 512, **kwargs) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """用 openai SDK 直接调任意的 OpenAI-compatible endpoint。"""

    def __init__(self, api_key: str, base_url: str, model: str = "",
                 temperature: float = 0.3, max_tokens: int = 512,
                 timeout: int = 60):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                  timeout=self._timeout)
        return self._client

    def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs):
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature or self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()


class PassthroughBackendClient(LLMClient):
    """
    直接包裹已有 backend 对象（NIMBackend / OpenAIBackend）。
    这些 backend 的接口是 generate(prompt, n_loops, temperature, max_tokens, ...)
    我们用 system prompt 把 conversation 压进单条 user message。
    """

    def __init__(self, backend):
        self._backend = backend

    def chat(self, messages, *, temperature=0.3, max_tokens=512, **kwargs):
        system = ""
        user_parts: List[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system = content
            else:
                user_parts.append(f"[{role.upper()}]\n{content}")
        prompt = "\n".join(user_parts)
        return self._backend.generate(
            prompt=prompt,
            n_loops=1,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ).strip()


# ===================================================================
# 1. 核心数据类
# ===================================================================

@dataclass
class WikiPage:
    """User-facing data object: a single wiki page in the memory graph."""

    title: str
    content: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)        # outgoing page ids
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    page_type: str = "entity"   # entity | episode | concept | tool | person
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = _make_id(self.title)

    def access(self):
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1

    # ---- section helpers ----
    def set_section(self, key: str, value: str):
        self.sections[key] = value
        self._rebuild_content()

    def get_section(self, key: str) -> str:
        return self.sections.get(key, "")

    def append_section(self, key: str, value: str):
        old = self.sections.get(key, "")
        self.sections[key] = f"{old}\n\n{value}" if old else value
        self._rebuild_content()

    def _rebuild_content(self):
        """把 sections 拼成可读的 text content。"""
        parts = [f"# {self.title}\n"]
        for k, v in self.sections.items():
            if v.strip():
                parts.append(f"## {k}\n{v.strip()}")
        self.content = "\n\n".join(parts)


# ===================================================================
# Helpers
# ===================================================================

_SLUG_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]+")


def _make_id(title: str) -> str:
    slug = _SLUG_RE.sub("_", title.strip().lower()).strip("_")[:60]
    ts = int(time.time() * 1000) % 10000
    return f"wiki_{slug}_{ts}_{uuid.uuid4().hex[:6]}"


def _title_hash(title: str) -> str:
    return hashlib_simple(title)


def hashlib_simple(s: str) -> str:
    """纯 Python 的简单 hash，避免引入 hashlib。"""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return hex(h)[2:]


# ===================================================================
# 2. WikiGraph
# ===================================================================

class WikiGraph:
    """
    以页面为节点的有向知识图谱。

    存储
    -----
    - self._pages : Dict[id -> WikiPage]   内存主存
    - chromadb collection "wiki_pages"     向量（存 section 拼接文本）
    - links 存在每个页面自身，同时维护 _backlinks 全局索引

    检索
    -----
    - semantic_search(query, k=...) : 全页面语义检索（返回 pages）
    - find_pages(title=None, tag=None) : 精确/标签过滤
    - get_related(page_id, depth=2) : 沿 links 邻域遍历
    - shortest_path(a_id, b_id) : BFS
    """

    def __init__(self, storage_path: str = "./data/wiki_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._pages: Dict[str, WikiPage] = {}
        self._title_index: Dict[str, str] = {}    # lowercased title -> id
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._backlinks: Dict[str, Set[str]] = defaultdict(set)  # id -> incoming id set

        # vector (lazy init: do not load SentenceTransformer at startup)
        self._embedding_model_name = embedding_model
        self._embedding_device = device
        self._embedding_model = None
        self._client = None
        self._collection = None
        self._mode: str = "vector" if _HAS_VECTOR else "memory"

        if _HAS_VECTOR:
            try:
                self._client = chromadb.PersistentClient(path=str(self.storage_path))
                self._collection = self._client.get_or_create_collection(
                    name="wiki_pages",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                print(f"  \033[2mwiki memory: vector client failed ({exc}), using memory mode\033[0m")
                self._mode = "memory"
        self._load_from_disk()

    # ----------------------------------------------------------------
    # init
    # ----------------------------------------------------------------
    def _ensure_embedding_model(self):
        """Lazy-load SentenceTransformer when first needed."""
        if self._embedding_model is not None or not _HAS_VECTOR:
            return
        try:
            self._embedding_model = SentenceTransformer(
                self._embedding_model_name, device=self._embedding_device
            )
        except Exception as exc:
            print(f"  \033[2mwiki memory: embedding model load failed ({exc}), using memory mode\033[0m")
            self._mode = "memory"
            self._embedding_model = None

    # ----------------------------------------------------------------
    # CRUD
    # ----------------------------------------------------------------
    def upsert_page(self, page: WikiPage) -> str:
        """Create or update a page. Returns the page id."""
        # update indexes
        if page.title.lower() not in self._title_index:
            self._title_index[page.title.lower()] = page.id
        else:
            # title collision → keep existing id (merge semantics)
            existing_id = self._title_index[page.title.lower()]
            if existing_id != page.id:
                page.id = existing_id

        old = self._pages.get(page.id)
        if old:
            # migrate backlinks from old id
            if old.id != page.id and old.id in self._backlinks:
                self._backlinks[page.id].update(self._backlinks.pop(old.id, set()))
            # adjust outgoing links that point to old id
            page.links = [page.id if l == old.id else l for l in page.links]

        self._pages[page.id] = page
        for tag in page.tags:
            self._tag_index[tag].add(page.id)

        # rebuild backlinks for all pages (cheap for small graphs)
        self._rebuild_backlinks()

        # vector upsert
        if self._mode == "vector" and self._collection and page.content.strip():
            self._vector_upsert(page)

        self._save_to_disk()
        return page.id

    def get_page(self, page_id: str) -> Optional[WikiPage]:
        page = self._pages.get(page_id)
        if page:
            page.access()
        return page

    def get_page_by_title(self, title: str) -> Optional[WikiPage]:
        pid = self._title_index.get(title.lower())
        if pid:
            return self.get_page(pid)
        return None

    def delete_page(self, page_id: str) -> bool:
        page = self._pages.pop(page_id, None)
        if not page:
            return False
        self._title_index.pop(page.title.lower(), None)
        for tag in page.tags:
            self._tag_index[tag].discard(page_id)
        if self._mode == "vector" and self._collection:
            try:
                self._collection.delete(ids=[page_id])
            except Exception:
                pass
        self._rebuild_backlinks()
        self._save_to_disk()
        return True

    def add_link(self, source_id: str, target_id: str, relation: str = "links_to") -> bool:
        """Add a directed link between two existing pages."""
        if source_id not in self._pages or target_id not in self._pages:
            return False
        if target_id not in self._pages[source_id].links:
            self._pages[source_id].links.append(target_id)
            self._backlinks[target_id].add(source_id)
            self._save_to_disk()
        return True

    def remove_link(self, source_id: str, target_id: str) -> bool:
        page = self._pages.get(source_id)
        if not page:
            return False
        if target_id in page.links:
            page.links.remove(target_id)
            self._backlinks[target_id].discard(source_id)
            self._save_to_disk()
            return True
        return False

    # ----------------------------------------------------------------
    # retrieval
    # ----------------------------------------------------------------
    def semantic_search(self, query: str, k: int = 10,
                        page_type: Optional[str] = None,
                        min_similarity: float = 0.35) -> List[Tuple[WikiPage, float]]:
        """全页面语义检索（优先向量，降级 keyword）。返回 [(page, score)]。"""
        if self._mode == "vector" and self._collection:
            return self._vector_search(query, k, page_type, min_similarity)
        return self._keyword_search(query, k, page_type)

    def find_pages(self, title: Optional[str] = None, tag: Optional[str] = None,
                   page_type: Optional[str] = None) -> List[WikiPage]:
        results = list(self._pages.values())
        if title:
            results = [p for p in results if title.lower() in p.title.lower()]
        if tag:
            ids = self._tag_index.get(tag, set())
            results = [p for p in results if p.id in ids]
        if page_type:
            results = [p for p in results if p.page_type == page_type]
        return results

    def get_related(self, page_id: str, depth: int = 2,
                    relation_filter: Optional[str] = None) -> List[Tuple[WikiPage, str]]:
        """沿链接扩撒 depth 层，返回 (page, incoming_relation_parent) 对。"""
        visited = {page_id}
        frontier = {page_id}
        results: List[Tuple[WikiPage, str]] = []

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for pid in frontier:
                page = self._pages.get(pid)
                if not page:
                    continue
                for linked_id in page.links:
                    if linked_id not in visited:
                        visited.add(linked_id)
                        linked = self._pages.get(linked_id)
                        if linked:
                            results.append((linked, "outgoing"))
                        next_frontier.add(linked_id)
                # incoming (backlinks)
                for incoming_id in self._backlinks.get(pid, set()):
                    if incoming_id not in visited:
                        visited.add(incoming_id)
                        incoming = self._pages.get(incoming_id)
                        if incoming:
                            results.append((incoming, "incoming"))
                        next_frontier.add(incoming_id)
            frontier = next_frontier
            if not frontier:
                break
        return results

    def shortest_path(self, source_id: str, target_id: str,
                      max_hops: int = 5) -> Optional[List[Tuple[WikiPage, str]]]:
        if source_id not in self._pages or target_id not in self._pages:
            return None
        if source_id == target_id:
            return [(self._pages[source_id], "self")]

        from collections import deque
        queue: deque = deque([(source_id, [])])
        visited = {source_id}

        for _ in range(max_hops):
            if not queue:
                break
            cur, path = queue.popleft()
            page = self._pages[cur]
            for linked_id in page.links:
                if linked_id == target_id:
                    return path + [(page, "outgoing"), (self._pages[linked_id], "arrived")]
                if linked_id not in visited:
                    visited.add(linked_id)
                    queue.append((linked_id, path + [(page, "outgoing")]))
        return None

    def suggest_related(self, page_id: str, k: int = 5) -> List[Tuple[WikiPage, float]]:
        candidates: Dict[str, float] = defaultdict(float)

        # graph: 1-hop
        page = self._pages.get(page_id)
        if page:
            for linked_id in page.links:
                candidates[linked_id] += 1.0
            for inc in self._backlinks.get(page_id, set()):
                candidates[inc] += 0.5

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self._pages[pid], score) for pid, score in ranked if pid in self._pages]

    # ----------------------------------------------------------------
    # maintenance
    # ----------------------------------------------------------------
    def prune_by_utility(self, threshold_days: float = 30.0,
                         keep_min: int = 50) -> int:
        if len(self._pages) <= keep_min:
            return 0
        cutoff = datetime.now() - timedelta(days=threshold_days)
        to_remove = []
        for pid, page in self._pages.items():
            last = datetime.fromisoformat(page.last_accessed)
            if (len(self._pages) - len(to_remove) > keep_min and
                    last < cutoff and len(page.links) < 2):
                to_remove.append(pid)
        for pid in to_remove:
            self.delete_page(pid)
        return len(to_remove)

    # ----------------------------------------------------------------
    # persistence
    # ----------------------------------------------------------------
    def _save_to_disk(self):
        data = {
            "pages": [self._page_to_dict(p) for p in self._pages.values()],
            "saved_at": datetime.now().isoformat(),
        }
        path = self.storage_path / "wiki_graph.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_from_disk(self):
        path = self.storage_path / "wiki_graph.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        self._pages.clear()
        self._title_index.clear()
        self._tag_index.clear()
        self._backlinks.clear()

        for item in data.get("pages", []):
            page = self._page_from_dict(item)
            self._pages[page.id] = page
            self._title_index[page.title.lower()] = page.id
            for tag in page.tags:
                self._tag_index[tag].add(page.id)

        # Note: ChromaDB collection is persistent; re-upserting every page at
        # startup would force-load the embedding model and block agent launch.
        # We only rebuild the backlink index from the in-memory graph here.
        self._rebuild_backlinks()

    @staticmethod
    def _page_to_dict(page: WikiPage) -> Dict:
        return {
            "id": page.id,
            "title": page.title,
            "content": page.content,
            "sections": page.sections,
            "links": page.links,
            "tags": page.tags,
            "page_type": page.page_type,
            "created_at": page.created_at,
            "last_accessed": page.last_accessed,
            "access_count": page.access_count,
        }

    @staticmethod
    def _page_from_dict(data: Dict) -> WikiPage:
        return WikiPage(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            sections=data.get("sections", {}),
            links=data.get("links", []),
            tags=data.get("tags", []),
            page_type=data.get("page_type", "entity"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_accessed=data.get("last_accessed", datetime.now().isoformat()),
            access_count=data.get("access_count", 0),
        )

    # ----------------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------------
    def _rebuild_backlinks(self):
        self._backlinks = defaultdict(set)
        for page in self._pages.values():
            for linked_id in page.links:
                if linked_id in self._pages:
                    self._backlinks[linked_id].add(page.id)

    def _vector_upsert(self, page: WikiPage):
        if not self._collection:
            return
        self._ensure_embedding_model()
        if self._mode != "vector":
            return
        page.access()
        doc = page.content
        embedding: Optional[List[float]] = None
        if self._embedding_model:
            try:
                embedding = self._embedding_model.encode(doc).tolist()
            except Exception:
                pass
        page.embedding = embedding
        try:
            self._collection.upsert(
                ids=[page.id],
                embeddings=[embedding] if embedding else None,
                documents=[doc],
                metadatas=[{
                    "title": page.title,
                    "page_type": page.page_type,
                    "created_at": page.created_at,
                    "last_accessed": page.last_accessed,
                    "tags": ",".join(page.tags),
                }],
            )
        except Exception:
            pass

    def _vector_search(self, query: str, k: int, page_type: Optional[str],
                       min_similarity: float) -> List[Tuple[WikiPage, float]]:
        self._ensure_embedding_model()
        if self._mode != "vector" or not self._collection or not self._embedding_model:
            return self._keyword_search(query, k, page_type)
        where = {"page_type": page_type} if page_type else None
        try:
            q_emb = self._embedding_model.encode(query).tolist()
        except Exception:
            return self._keyword_search(query, k, page_type)

        results = self._collection.query(
            query_embeddings=[q_emb], n_results=k * 2, where=where,
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        out: List[Tuple[WikiPage, float]] = []
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for i, doc in enumerate(docs):
            dist = dists[i] if i < len(dists) else 1.0
            sim = 1.0 - dist
            if sim < min_similarity:
                continue
            title = metas[i].get("title", "") if i < len(metas) else ""
            pid = self._title_index.get(title.lower())
            if pid and pid in self._pages:
                out.append((self._pages[pid], sim))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]

    def _keyword_search(self, query: str, k: int,
                        page_type: Optional[str]) -> List[Tuple[WikiPage, float]]:
        q = query.lower()
        scored: List[Tuple[WikiPage, float]] = []
        for page in self._pages.values():
            if page_type and page.page_type != page_type:
                continue
            text = f"{page.title} {page.content} {' '.join(page.tags)}".lower()
            score = 0.75 if q in page.title.lower() else 0.0
            score += len(re.findall(re.escape(q), text)) * 0.1
            score += sum(0.05 for w in q.split() if w in text)
            if score > 0:
                scored.append((page, min(score, 1.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ===================================================================
# 3. LLM-driven wiki helpers
# ===================================================================

# Prompt templates —— 高层级，不掺杂任何工具逻辑

SYSTEM_PROMPT_WIKI_WRITER = """\
You are an expert knowledge curator for an AI agent's long-term memory.
You maintain a set of wiki pages (one per entity/concept/experience),
similar to how Wikipedia maintains articles.

Output format rules:
1. Always start with the page title as a markdown H1: "# {title}"
2. Follow with section H2s: "## Summary", "## Key Facts", "## Relationships"
3. Be factual, concise, and information-dense — target 100-250 words.
4. Write for a future AI reader that needs to recall everything important.
5. Do NOT include conversational filler.
"""

SYSTEM_PROMPT_LINK_INFERENCE = """\
You are a knowledge graph relation analyst.
Given a short description of two wiki pages, output ONLY valid JSON:

{"relation": "short label", "confidence": 0.0-1.0}

Valid relation labels (pick the most accurate):
  is_a, part_of, used_by, depends_on, causes, related_to,
  instance_of, supersedes, conflicts_with, collaborates_with

Rules:
- Output only the JSON object, nothing else.
- confidence should reflect how certain you are."""

SYSTEM_PROMPT_PAGE_MERGE = """\
You are maintaining an AI agent's wiki memory.
You will receive an EXISTING wiki page (the "current article") and
a NEW information snippet (the "update").

Your job: write a merged version of the article that incorporates
all important new facts without losing existing content.

Output format: the FULL merged article in markdown, same structure as input.
Keep it under {max_words} words total.
Do not explain your changes, just output the merged article.
"""

SYSTEM_PROMPT_ENTITY_EXTRACT = """\
Extract all named entities from the text below.
Output valid JSON list only, nothing else:

[
  {"name": "...", "type": "person|organization|location|concept|tool|task", "confidence": 0.0-1.0}
]

Rules:
- Include only entities a future AI reader would want a dedicated wiki page for.
- Exclude proper nouns that are clearly not important (e.g., random words).
- Confidence should reflect how certain you are the entity exists in the text.
"""

SYSTEM_PROMPT_MORAL_SKELETON = """\
You are the "moral skeleton" extractor for an AI agent's long-term wiki memory.

Given one episode / interaction transcript, your job is to produce a compact
STRUCTURED SUMMARY of the underlying INTENTION, DECISION, and LESSON —
independent of surface details (names, numbers, specific tools).

Output ONLY the following JSON object, nothing else:

{
  "core_intention": "1 sentence — what was the agent trying to accomplish",
  "key_decision_point": "1 sentence — the single most consequential fork / choice",
  "lesson_learned": "1 sentence — the reusable principle or heuristic",
  "failure_mode": "one-phrase or empty string — if failed, the failure pattern",
  "success_pattern": "one-phrase or empty string — if succeeded, the verified pattern",
  "abstract_principle": "1 sentence — the broader strategy this instantiate",
  "applicable_contexts": ["list of short tags describing when this applies"],
  "confidence": 0.0-1.0
}

Rules:
- Be terse and generic. Replace specific names with role nouns (e.g. "the search tool",
  "the planning module") so the skeleton survives future tool renames.
- If the episode is purely conversational / trivia, set lesson_learned to
  "no_actionable_lesson" and leave other fields minimal.
- Confidence should reflect how much signal the text actually contains
  (ambiguous text → low confidence).
"""


def _call_llm(llm: Optional["LLMClient"], messages: List[Dict[str, str]],
              *, temperature: float = 0.3, max_tokens: int = 512, fallback: str = "") -> str:
    if llm is None:
        return fallback
    return llm.chat(messages, temperature=temperature, max_tokens=max_tokens)


def llm_generate_page(llm: Optional["LLMClient"], title: str,
                      context: str = "", page_type: str = "entity") -> Dict:
    """
    让 LLM 为一个新实体生成初始 wiki page 内容。
    返回 dict: {"title": str, "content": str, "sections": Dict[str, str], "tags": List[str], "page_type": str}
    """
    user_msg = (
        f"Create a wiki page for the entity: **{title}**.\n"
        f"Type hint: {page_type}.\n"
    )
    if context:
        user_msg += f"\nContext about this entity:\n{context[:2000]}\n"
    user_msg += (
        "\nGenerate the page now. Start with the markdown title line "
        f"# {title}, then sections. Do not include this instruction in the output."
    )

    text = _call_llm(
        llm,
        [{"role": "system", "content": SYSTEM_PROMPT_WIKI_WRITER},
         {"role": "user", "content": user_msg}],
        temperature=0.3,
        max_tokens=800,
        fallback="",
    )
    if not text.strip():
        return {
            "title": title,
            "content": f"# {title}\n\nNo additional information available.",
            "sections": {"Summary": f"No additional information available for {title}."},
            "tags": [title.lower().replace(" ", "_")],
            "page_type": page_type,
        }
    return _parse_markdown_page(text, page_type=page_type)


def llm_infer_relation(llm: Optional["LLMClient"], source_title: str,
                       target_title: str, source_snippet: str = "",
                       target_snippet: str = "") -> Tuple[str, float]:
    user_msg = (
        f"Source page: {source_title}\n"
        + (f"Source snippet: {source_snippet[:200]}\n" if source_snippet else "")
        + f"Target page: {target_title}\n"
        + (f"Target snippet: {target_snippet[:200]}\n" if target_snippet else "")
        + "What is the best relation label?"
    )
    raw = _call_llm(
        llm,
        [{"role": "system", "content": SYSTEM_PROMPT_LINK_INFERENCE},
         {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=64,
    )
    try:
        data = json.loads(raw)
        return data.get("relation", "related_to"), float(data.get("confidence", 0.5))
    except Exception:
        return "related_to", 0.3


def llm_merge_page(llm: Optional["LLMClient"], current_content: str,
                   update_text: str, max_words: int = 500) -> str:
    user_msg = (
        f"CURRENT ARTICLE:\n{current_content}\n\n"
        f"NEW INFORMATION TO INCORPORATE:\n{update_text}\n\n"
        f"Merge them into one article (max {max_words} words)."
        " Output ONLY the merged article."
    )
    return _call_llm(
        llm,
        [{"role": "system", "content": SYSTEM_PROMPT_PAGE_MERGE.format(max_words=max_words)},
         {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=max_words * 2,
    )


def llm_extract_entities(llm: Optional["LLMClient"], text: str,
                         max_entities: int = 10) -> List[Dict]:
    raw = _call_llm(
        llm,
        [{"role": "system", "content": SYSTEM_PROMPT_ENTITY_EXTRACT},
         {"role": "user", "content": text[:4000]}],
        temperature=0.1,
        max_tokens=512,
    )
    try:
        data = json.loads(raw)
        return data[:max_entities] if isinstance(data, list) else []
    except Exception:
        return []


def llm_extract_moral_skeleton(llm: Optional["LLMClient"],
                                text: str) -> Dict[str, Any]:
    """
    Extract the moral/meaning skeleton from an episode text.

    Returns a dict with keys:
      core_intention, key_decision_point, lesson_learned,
      failure_mode, success_pattern, abstract_principle,
      applicable_contexts (list), confidence
    Falls back to a deterministic heuristic when no LLM is available.
    """
    fallback = _heuristic_moral_skeleton(text)
    raw = _call_llm(
        llm,
        [{"role": "system", "content": SYSTEM_PROMPT_MORAL_SKELETON},
         {"role": "user", "content": text[:6000]}],
        temperature=0.1,
        max_tokens=256,
        fallback="",
    )
    if not raw.strip():
        return fallback
    try:
        data = json.loads(raw)
        for k, v in fallback.items():
            if k not in data or data[k] == "":
                data[k] = v
        return data
    except Exception:
        return fallback


def _heuristic_moral_skeleton(text: str) -> Dict[str, Any]:
    """No-LLM fallback: cheap rule-based skeleton."""
    lower = text.lower()
    failure_signals = ["error", "failed", "exception", "timeout",
                        "失败", "错误", "异常"]
    success_signals = ["success", "completed", "result", "success",
                        "成功", "完成", "结果"]
    failed = any(s in lower for s in failure_signals)
    succeeded = any(s in lower for s in success_signals) and not failed

    # crude task detection
    task_match = re.search(r"(?:task|任务|目标)[\s:：]+(.+?)(?:[.\n]|$)",
                           text, re.IGNORECASE)
    core = task_match.group(1).strip()[:120] if task_match else text[:120].strip()
    return {
        "core_intention": core or "unknown",
        "key_decision_point": "",
        "lesson_learned": ("no_actionable_lesson"
                           if not failed and not succeeded else
                           ("avoid repeating the failure pattern"
                            if failed else "replicate the success pattern")),
        "failure_mode": "episode_failed" if failed else "",
        "success_pattern": "verified_success" if succeeded else "",
        "abstract_principle": "",
        "applicable_contexts": [],
        "confidence": 0.2,
    }


def _parse_markdown_page(text: str, page_type: str = "entity") -> Dict:
    """
    把 LLM 输出的 markdown page 解析成 {content, sections, tags}。
    这是轻量的启发式解析器，不追求完美。
    """
    lines = text.strip().splitlines()
    title = ""
    sections: Dict[str, str] = {}
    current_key = "content"
    buf: List[str] = []

    for line in lines:
        m = re.match(r"^#{1,3}\s+(.+)$", line)
        if m:
            if buf:
                sections[current_key] = "\n".join(buf).strip()
                buf = []
            current_key = m.group(1).strip().rstrip(":")
            if not title:
                # first heading stripped of # is the title
                title = re.sub(r"^#+\s*", "", current_key).strip()
            continue
        buf.append(line)

    if buf:
        sections[current_key] = "\n".join(buf).strip()

    # infer tags from title + section names
    tags = [t.strip().lower() for t in title.split()
            if len(t.strip()) > 1 and not t.strip().isdigit()]
    tags = list(dict.fromkeys(tags))[:8]   # preserve order, dedup

    content_str = "\n\n".join(f"## {k}\n{v}" for k, v in sections.items() if v)
    return {"title": title, "content": content_str,
            "sections": sections, "tags": tags, "page_type": page_type}


# ===================================================================
# 4. 高层记忆管理器
# ===================================================================

class EpisodicWikiEntry:
    """Episode 也是一种 page，但属于 episode 类型。"""

    def __init__(self, task: str, trajectory: Dict[str, Any],
                 outcome: str, success: bool):
        self.task = task
        self.trajectory = trajectory
        self.outcome = outcome
        self.success = success
        self.timestamp = datetime.now().isoformat()

    def to_page_update(self) -> str:
        """转成 wiki 可合并的文本片段。"""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"### Episode  {self.timestamp}\n"
            f"Task: {self.task}\n"
            f"Outcome: {self.outcome}\n"
            f"Status: {status}\n"
        )


class LLMWikiManager:
    """
    对外统一接口，替代原始的 MemoryManager。

    核心能力
    -------
    - remember(text, context)   : 把一段交互存入 memory -> 生成/更新 wiki pages
    - recall(query, k=)          : 语义召回相关 pages + linked pages
    - forge_link(a, b, llm)      : 在两个页面之间建 link
    - iterate_episode(task, ...) : 存 episode，更新对应 entity pages
    - augment_prompt(base, query, k=) : 把记忆拼回 prompt

    向后兼容
    --------
    - 需要兼容旧接口: store_interaction / get_relevant_context /
                     augment_prompt_with_memory / extract_and_store_entities
    """

    def __init__(self, storage_path: str = "./data/wiki_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_client: Optional["LLMClient"] = None,
                 device: str = "cpu"):
        self.graph = WikiGraph(storage_path=storage_path,
                               embedding_model=embedding_model, device=device)
        self.llm = llm_client
        self._llm_enabled = llm_client is not None

    # ---- memory write API ----

    def remember(self, text: str, *, entity_names: Optional[List[str]] = None,
                 auto_link: bool = True, context: str = "") -> List[str]:
        """
        将一段文本（对话/观察/经验）写入 memory。

        流程
        1) 如果提供了 entity_names，直接 get_page / create page；
           否则调用 llm_extract_entities 自动抽取。
        2) 对每个新 entity，调用 llm_generate_page 生成页面内容。
        3) 把 text 以 episode 形式追加到对应页面的 "Episodes" section。
        4) 如果有 auto_link 且 ≥2 个实体，调用 llm_infer_relation 建 link。

        返回: 实际创建或更新的 page id 列表。
        """
        page_ids: List[str] = []

        # 1) determine entities
        if entity_names:
            entity_titles = entity_names
        elif self._llm_enabled:
            extracted = llm_extract_entities(self.llm, text, max_entities=8)
            entity_titles = [e["name"] for e in extracted if e.get("confidence", 0) > 0.5]
        else:
            # no LLM: use title-cased proper nouns via regex
            entity_titles = list(dict.fromkeys(
                re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b", text)
            ))[:6]

        if not entity_titles:
            # fallback: create a generic episode page
            entity_titles = [f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]

        # 2) get or create pages
        pages: Dict[str, WikiPage] = {}
        for title in entity_titles:
            page = self.graph.get_page_by_title(title)
            if not page:
                page = WikiPage(title=title)
                if self._llm_enabled:
                    gen = llm_generate_page(self.llm, title, context=text)
                    page.content = gen.get("content", "")
                    page.sections = gen.get("sections", {})
                    page.tags = gen.get("tags", page.tags)
                    page.page_type = gen.get("page_type", "entity")
                else:
                    page.set_section("Summary", text[:500])
                    page.set_section("Raw Context", text[:2000])
                self.graph.upsert_page(page)
            else:
                # append new episodic info
                episode_snippet = _build_episode_snippet(text)
                if episode_snippet:
                    old = page.get_section("Episodes") or ""
                    page.set_section("Episodes", f"{old}\n\n{episode_snippet}" if old else episode_snippet)
                # optionally merge top-level summary
                if self._llm_enabled and len(text) > 50:
                    new_summary = llm_merge_page(self.llm,
                                                 page.get_section("Summary") or page.content,
                                                 text[:800], max_words=400)
                    page.set_section("Summary", new_summary)
                self.graph.upsert_page(page)
            pages[title] = page
            page_ids.append(page.id)

        # 3) auto-link pairs
        if auto_link and len(pages) >= 2:
            titles = list(pages.keys())
            for i in range(len(titles)):
                for j in range(i + 1, len(titles)):
                    src_title, tgt_title = titles[i], titles[j]
                    src_id = pages[src_title].id
                    tgt_id = pages[tgt_title].id
                    if self._llm_enabled:
                        relation, conf = llm_infer_relation(
                            self.llm, src_title, tgt_title,
                            pages[src_title].get_section("Summary"),
                            pages[tgt_title].get_section("Summary"),
                        )
                        if conf < 0.3:
                            relation = "related_to"
                    else:
                        relation = "related_to"
                    self.graph.add_link(src_id, tgt_id, relation=relation)
                    self.graph.add_link(tgt_id, src_id, relation=relation)

        return page_ids

    # ---- episode API (backward compat) ----

    def store_interaction(self, task: str, trajectory: Dict[str, Any],
                          outcome: str, success: bool,
                          entities: Optional[List[str]] = None) -> str:
        episode = EpisodicWikiEntry(task=task, trajectory=trajectory,
                                    outcome=outcome, success=success)
        # 1) create task page if none supplied
        if not entities:
            entities = [task[:60]]
        page_ids: List[str] = []
        for title in entities:
            page = self.graph.get_page_by_title(title)
            if not page:
                page = WikiPage(title=title, page_type="task")
                snippet = (
                    f"Task: {task}\n"
                    f"Outcome: {outcome}\n"
                    f"Success: {success}\n"
                )
                page.set_section("Summary", snippet)
                page.set_section("Episodes", episode.to_page_update())
                self.graph.upsert_page(page)
            else:
                page.append_section("Episodes", episode.to_page_update())
                page.set_section("Outcome", outcome)
                page.set_section("Success", str(success))
                self.graph.upsert_page(page)
            page_ids.append(page.id)
        return page_ids[0]

    def extract_and_store_entities(self, text: str,
                                   entity_types: Optional[List[str]] = None) -> List[str]:
        """Legacy compat: extract entities and create page stubs."""
        titles: List[str] = []
        if self._llm_enabled:
            extracted = llm_extract_entities(self.llm, text, max_entities=6)
            for e in extracted:
                title = e["name"]
                etype = e.get("type", "concept")
                page = WikiPage(title=title, page_type=etype,
                                tags=[etype, title.lower().replace(" ", "_")])
                page.set_section("Summary", text[:500])
                page.set_section("Raw Context", text[:2000])
                self.graph.upsert_page(page)
                titles.append(page.id)
        else:
            # regex fallback
            matches = list(dict.fromkeys(
                re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b", text)
            ))[:5]
            for title in matches:
                page = WikiPage(title=title)
                page.set_section("Summary", text[:500])
                page.tags = ["auto_extracted"]
                self.graph.upsert_page(page)
                titles.append(page.id)
        return titles

    # ---- retrieval (backward compat + new) ----

    def get_relevant_context(self, query: str, max_pages: int = 5,
                             include_related: bool = True) -> Dict[str, Any]:
        """
        返回与 query 最相关的 pages + related pages。
        backward compat: 兼容旧 MemoryManager.get_relevant_context 返回结构
        """
        hits = self.graph.semantic_search(query, k=max_pages)
        pages: List[WikiPage] = [p for p, _ in hits]
        links_accum: List[Dict] = []

        if include_related:
            seen: Set[str] = {p.id for p in pages}
            for page in pages:
                for related, direction in self.graph.get_related(page.id, depth=1):
                    if related.id not in seen:
                        seen.add(related.id)
                        pages.append(related)
                        links_accum.append({
                            "via": page.title,
                            "direction": direction,
                            "page": {"id": related.id,
                                     "title": related.title,
                                     "label": related.page_type,
                                     "snippet": related.content[:200]},
                        })

        episodes_field = [{"summary": p.get_section("Summary") or p.content[:200],
                           "task_type": p.page_type,
                           "id": p.id,
                           "title": p.title}
                          for p in pages[:max_pages]]
        knowledge_field = [{"title": p.title, "label": p.page_type,
                            "properties": p.sections,
                            "id": p.id,
                            "score": 1.0}
                           for p in pages[:max_pages]]

        return {
            "episodes": episodes_field,
            "knowledge_nodes": knowledge_field,
            "related_nodes": links_accum,
        }

    def augment_prompt_with_memory(self, base_prompt: str, query: str,
                                   max_tokens: int = 1024) -> str:
        """Backward compat: 把相关记忆拼回 prompt。"""
        ctx = self.get_relevant_context(query, max_pages=3)
        parts = [base_prompt]
        if ctx.get("episodes"):
            parts.append("\n## Relevant Past Experiences:")
            for ep in ctx["episodes"][:3]:
                parts.append(f"\n- [{ep.get('title','')}] {ep.get('summary','')[:200]}")
        if ctx.get("knowledge_nodes"):
            parts.append("\n## Relevant Knowledge:")
            for node in ctx["knowledge_nodes"][:3]:
                snippet = node.get("properties", {}).get("Summary", "") or node.get("title", "")
                parts.append(f"\n- [{node.get('label','')}] {snippet[:200]}")
        return "\n".join(parts)

    def compress_and_store(self, text: str, task: str,
                           max_tokens: int = 256) -> str:
        """Backward compat stub: compress + store episode."""
        if self._llm_enabled:
            compressed = llm_merge_page(self.llm, text, "", max_words=max_tokens // 2)
        else:
            # extractive fallback
            sentences = re.split(r"[.!?\n]+", text)
            compressed = " ".join(sentences[:3]).strip() or text[:max_tokens]
        self.store_interaction(task=task, trajectory={"text": text},
                               outcome=compressed, success=True)
        return compressed

    # ---- new: page-centric API ----

    def create_page(self, title: str, *, page_type: str = "entity",
                    initial_text: str = "", llm: Optional["LLMClient"] = None
                    ) -> WikiPage:
        page = WikiPage(title=title, page_type=page_type)
        if initial_text:
            page.set_section("Summary", initial_text)
        elif (self._llm_enabled or llm) and title:
            gen = llm_generate_page(llm or self.llm, title)
            page.content = gen.get("content", "")
            page.sections = gen.get("sections", {})
            page.tags = gen.get("tags", page.tags)
            page.page_type = gen.get("page_type", page_type)
        self.graph.upsert_page(page)
        return page

    def update_page(self, page_id: str, *, section: Optional[str] = None,
                    text: Optional[str] = None, links_add: Optional[List[str]] = None
                    ) -> Optional[WikiPage]:
        page = self.graph.get_page(page_id)
        if not page:
            return None
        if section and text:
            if self._llm_enabled:
                merged = llm_merge_page(self.llm,
                                        page.get_section(section) or "",
                                        text,
                                        max_words=500)
                page.set_section(section, merged)
            else:
                page.append_section(section, text)
        if text and not section:
            page._rebuild_content()
        if links_add:
            for target in links_add:
                tgt_id = self.graph.get_page_by_title(target)
                if tgt_id:
                    self.graph.add_link(page.id, tgt_id.id or tgt_id)
        self.graph.upsert_page(page)
        return page

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        新检索接口，返回可直接渲染的 page dicts。
        """
        hits = self.graph.semantic_search(query, k=k)
        return [
            {
                "id": p.id,
                "title": p.title,
                "label": p.page_type,
                "snippet": (p.get_section("Summary") or p.content)[:400],
                "score": score,
                "tags": p.tags,
                "links": p.links,
            }
            for p, score in hits
        ]

    def get_graph_snapshot(self) -> Dict:
        """图谱快照：节点 + 边，用于可视化或调试。"""
        nodes = [{"id": p.id, "title": p.title, "type": p.page_type,
                  "snippet": p.content[:120]}
                 for p in self.graph._pages.values()]
        edges = []
        seen: Set[Tuple[str, str]] = set()
        for p in self.graph._pages.values():
            for linked in p.links:
                if linked in self.graph._pages:
                    key = (p.id, linked)
                    if key not in seen:
                        seen.add(key)
                        edges.append({"source": p.id, "target": linked})
        return {"nodes": nodes, "edges": edges, "stats": {
            "nodes": len(nodes), "edges": len(edges),
            "timestamp": datetime.now().isoformat(),
        }}


# ===================================================================
# 5. Episode segment helper
# ===================================================================

def _build_episode_snippet(text: str) -> str:
    # put the first ~400 chars in an episode block
    snippet = text.strip()[:500]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"- [{ts}] {snippet}"


# ===================================================================
# 6. 向后兼容的工厂
# ===================================================================

def create_memory_manager(kg_storage: str = "./data/kg_store",
                          episodic_storage: str = "./data/episodic_store",
                          embedding_model: str = "all-MiniLM-L6-v2",
                          llm_client: Optional["LLMClient"] = None,
                          file_memory_path: str = "./data/memory.md",
                          user_memory_path: str = "./data/user.md",
                          sqlite_session_path: str = "./data/sessions.db",
                          project_root: Optional[str] = None) -> "LLMWikiManager":
    """
    工厂函数，签名保持与原 create_memory_manager 兼容。
    llm_client 可选传，不传则走 keyword-only 降级模式。
    file/session memory arguments are accepted for API compatibility but ignored
    by the wiki backend; use memory.MemoryManager for those layers.
    """
    return LLMWikiManager(
        storage_path=kg_storage,
        embedding_model=embedding_model,
        llm_client=llm_client,
    )


def quick_store(task: str, content: str, summary: str = None,
                success: bool = True):
    """Backward compat stub."""
    mgr = create_memory_manager()
    mgr.store_interaction(
        task=task,
        trajectory={"content": content},
        outcome=summary or content[:100],
        success=success,
    )


# ===================================================================
# 7. CLI demo
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM Wiki Memory — demo (no LLM client, keyword mode)")
    print("=" * 60)

    manager = create_memory_manager()

    # ingest some text
    texts = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "Guido van Rossum worked at Dropbox after leaving Google.",
        "Dropbox is a cloud file storage service founded by Drew Houston.",
    ]
    for text in texts:
        page_ids = manager.remember(text, auto_link=True)
        print(f"remember → pages: {[pid[:30] for pid in page_ids]}")

    # query
    query = "Guido"
    hits = manager.search(query, k=3)
    print(f"\nsearch '{query}':")
    for h in hits:
        print(f"  [{h['label']}] {h['title']}: {h['snippet'][:80]}…")

    # graph snapshot
    snap = manager.get_graph_snapshot()
    print(f"\ngraph: {snap['stats']['nodes']} nodes, {snap['stats']['edges']} edges")

    print("\n\033[2m  Done\033[0m")
