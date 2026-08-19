"""
ContextEngine - Unified memory & context management for a world-class agent.

Layers
------
1. Working Memory       : current session events (user/assistant/tools/observations)
2. Episodic Memory      : past task trajectories (ExperienceBuffer)
3. Semantic Memory      : entity/concept graph (LLMWikiManager)
4. User Profile Memory  : user preferences, habits, facts
5. Context Compressor   : token-aware summarization when context grows

Design principles
-----------------
- Single facade: agent only talks to ContextEngine.
- Async consolidation: memory writes happen in background threads.
- Token-aware: every retrieval method accepts a token budget.
- Graceful degradation: works without vector deps and without LLM.
"""

import json
import re
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

try:
    from .experience import ExperienceBuffer, Lesson
except Exception:
    ExperienceBuffer = None  # type: ignore
    Lesson = None  # type: ignore

try:
    from .wiki_memory import create_memory_manager, LLMWikiManager, WikiPage
    _WIKI_AVAILABLE = True
except Exception:
    _WIKI_AVAILABLE = False

from .terminal import style as _style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Cheap token estimator (CJK ~1.5 chars/token, Latin ~4 chars/token)."""
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return max(1, cjk // 2 + (len(text) - cjk) // 4)


def _truncate_to_budget(parts: List[str], budget: int, header: str = "") -> str:
    """Join parts, dropping oldest items until the token budget is met."""
    # Always keep header if provided.
    used = _estimate_tokens(header)
    keep = []
    for part in reversed(parts):
        cost = _estimate_tokens(part)
        if used + cost <= budget:
            keep.append(part)
            used += cost
        else:
            break
    keep.reverse()
    return header + "\n".join(keep)


# ---------------------------------------------------------------------------
# 1. Working Memory
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemoryEvent:
    role: str  # user | assistant | tool | observation | system
    content: str
    event_type: str = "message"  # message | tool_call | tool_result | plan | thought
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """Short-term, in-session memory. Fast, lossless, token-budgeted."""

    def __init__(self, max_events: int = 200):
        self.events: List[WorkingMemoryEvent] = []
        self.max_events = max_events
        self._lock = threading.Lock()

    def add(self, role: str, content: str, event_type: str = "message",
            metadata: Optional[Dict[str, Any]] = None):
        with self._lock:
            self.events.append(WorkingMemoryEvent(
                role=role,
                content=content,
                event_type=event_type,
                metadata=metadata or {}
            ))
            if len(self.events) > self.max_events:
                # Drop oldest quarter to avoid linear growth.
                drop = self.max_events // 4
                self.events = self.events[drop:]

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        self.add("assistant", f"{tool_name}({arguments})", "tool_call",
                 {"tool_name": tool_name, "args": arguments})

    def add_tool_result(self, tool_name: str, result: str, success: bool):
        self.add("tool", result, "tool_result",
                 {"tool_name": tool_name, "success": success})

    def get_events(self, event_types: Optional[List[str]] = None,
                   since: Optional[float] = None,
                   limit: Optional[int] = None) -> List[WorkingMemoryEvent]:
        with self._lock:
            out = list(self.events)
        if since is not None:
            out = [e for e in out if e.timestamp >= since]
        if event_types:
            out = [e for e in out if e.event_type in event_types]
        if limit:
            out = out[-limit:]
        return out

    def recent_tool_history(self, n: int = 10) -> str:
        """Formatted recent tool calls/results for self-correction / reflection."""
        events = self.get_events(event_types=["tool_call", "tool_result"], limit=n)
        lines = []
        for e in events:
            if e.event_type == "tool_call":
                lines.append(f"→ {e.content}")
            else:
                status = "OK" if e.metadata.get("success") else "FAIL"
                summary = e.content[:120].replace("\n", " ")
                lines.append(f"  [{status}] {summary}")
        return "\n".join(lines)

    def format_for_prompt(self, max_tokens: int = 1500,
                          include_tool_history: bool = True) -> str:
        """Render recent working memory as prompt text, respecting token budget.

        Strategy: keep the most recent user/assistant turn intact (so references
        like '这个榜单' can be resolved), then include earlier turns with a soft
        truncation so the topic history is not lost.
        """
        user_assistant = self.get_events(event_types=["message"], limit=20)
        parts = []
        # Reserve budget for the latest turn so pronouns / references resolve.
        latest_turn_budget = max_tokens // 3
        current_budget = max_tokens
        for idx, e in enumerate(user_assistant):
            prefix = "User" if e.role == "user" else "Assistant"
            is_latest_turn = idx >= len(user_assistant) - 2
            if is_latest_turn:
                # Keep latest turn fuller; only hard-cap at 1200 chars.
                content = e.content[:1200]
                # If this would blow the budget, still keep at least a snippet.
                snippet = f"{prefix}: {content}"
                if _estimate_tokens("\n".join(parts) + "\n" + snippet) > current_budget:
                    content = e.content[:300]
            else:
                content = e.content[:300]
            parts.append(f"{prefix}: {content}")

        out = _truncate_to_budget(parts, current_budget,
                                  header="## Current Conversation:\n")

        if include_tool_history:
            tool_history = self.recent_tool_history(n=8)
            if tool_history:
                tool_block = f"\n\n## Recent Tool History:\n{tool_history}"
                if _estimate_tokens(out + tool_block) <= max_tokens:
                    out += tool_block
        return out

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "events": [asdict(e) for e in self.events],
                "max_events": self.max_events,
            }

    def restore(self, data: Dict[str, Any]):
        with self._lock:
            self.events = [WorkingMemoryEvent(**e) for e in data.get("events", [])]
            self.max_events = data.get("max_events", self.max_events)


# ---------------------------------------------------------------------------
# 2. User Profile Memory
# ---------------------------------------------------------------------------

class UserProfileMemory:
    """Persistent user preferences, habits, and facts across sessions."""

    def __init__(self, storage_path: str = "./data/user_profile.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._profile: Dict[str, Any] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "preferences": {},
            "facts": {},
            "habits": {},
            "last_updated": datetime.now().isoformat(),
        }

    def _save(self):
        try:
            self.storage_path.write_text(
                json.dumps(self._profile, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def set_preference(self, key: str, value: Any):
        with self._lock:
            self._profile.setdefault("preferences", {})[key] = {
                "value": value,
                "updated_at": datetime.now().isoformat(),
            }
            self._save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        with self._lock:
            pref = self._profile.get("preferences", {}).get(key)
            return pref["value"] if isinstance(pref, dict) else pref or default

    def add_fact(self, key: str, value: str, confidence: float = 1.0):
        with self._lock:
            self._profile.setdefault("facts", {})[key] = {
                "value": value,
                "confidence": confidence,
                "updated_at": datetime.now().isoformat(),
            }
            self._save()

    def update_from_interaction(self, task: str, final_answer: str,
                                llm_client: Optional[Any] = None):
        """Background update: infer user preferences/facts from a completed turn."""
        text = f"Task: {task}\nAnswer: {final_answer}"
        # Simple heuristic updates first.
        lower = text.lower()
        if any(k in lower for k in ["简洁", "简短", "concise", "brief"]):
            self.set_preference("response_style", "concise")
        if any(k in lower for k in ["详细", "详尽", "verbose", "detailed"]):
            self.set_preference("response_style", "detailed")
        if any(k in lower for k in ["中文", "chinese"]):
            self.set_preference("language", "zh")
        if any(k in lower for k in ["english", "英文"]):
            self.set_preference("language", "en")

        # LLM-powered extraction if available.
        if llm_client:
            try:
                raw = llm_client.chat([
                    {"role": "system", "content": (
                        "Extract any user preference, habit, or factual statement from the text. "
                        "Output valid JSON only: {\"preferences\":{...},\"facts\":{...}}. "
                        "If nothing extractable, output {}."
                    )},
                    {"role": "user", "content": text[:2000]},
                ], temperature=0.1, max_tokens=256)
                data = json.loads(raw)
                for k, v in data.get("preferences", {}).items():
                    self.set_preference(k, v)
                for k, v in data.get("facts", {}).items():
                    self.add_fact(k, str(v))
            except Exception:
                pass

    def format(self, max_tokens: int = 400) -> str:
        with self._lock:
            profile = self._profile
        parts = []
        prefs = profile.get("preferences", {})
        if prefs:
            items = [f"- {k}: {v['value'] if isinstance(v, dict) else v}" for k, v in prefs.items()]
            parts.append("User Preferences:\n" + "\n".join(items))
        facts = profile.get("facts", {})
        if facts:
            items = [f"- {k}: {v['value'] if isinstance(v, dict) else v}" for k, v in facts.items()]
            parts.append("Known Facts:\n" + "\n".join(items))
        if not parts:
            return ""
        text = "\n\n".join(parts)
        if _estimate_tokens(text) > max_tokens:
            text = text[:max_tokens * 4]
        # 显式指令: 让模型真正按用户偏好行动, 而不只是"看到"偏好
        return f"## User Profile:\n{text}\n\n重要: 严格遵守上述用户偏好(语言/风格/事实)来组织你的回复。"


# ---------------------------------------------------------------------------
# 3b. Memory Consolidator (显式记忆整合层)
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """显式记忆整合器: 决定"本次交互值不值得进入长期记忆".

    借鉴 Omni 把 memory 单独成模块的设计, 在"短期工作记忆 → 长期语义记忆"
    之间加一道**价值闸门**:

    - 高价值交互(成功 + 含明确事实/偏好/工作流/实体) → 写入长期记忆
    - 低价值交互(闲聊/失败/重复/无实质内容) → 只留在短期, 不污染长期库
    - 附带重要性评分与衰减: 长期记忆按 importance 排序召回, 低价值旧记忆降权

    这样长期记忆始终是高信噪比的"知识", 而非对话流水账。
    """

    def __init__(
        self,
        semantic_memory: Optional[Any] = None,
        user_profile: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        importance_threshold: float = 0.45,
    ):
        self.semantic_memory = semantic_memory
        self.user_profile = user_profile
        self.llm_client = llm_client
        self.importance_threshold = importance_threshold

    # ---- 重要性打分 ----

    def compute_importance(self, task: str, final_answer: str, success: bool,
                           metadata: Optional[Dict[str, Any]] = None) -> float:
        """0-1 重要性评分. 分值越高越值得进长期记忆."""
        if not final_answer:
            return 0.0
        score = 0.0
        meta = metadata or {}

        # 1. 成功是基础加分(失败交互一般不沉淀为"知识")
        if success:
            score += 0.3
        else:
            score += 0.1  # 失败的教训也可能有价值(但低)

        t = f"{task}\n{final_answer}"
        tl = t.lower()

        # 2. 含明确事实/偏好/工作流 → 高价值
        fact_signals = [
            "偏好", "喜欢", "习惯", "以后", "希望", "记住", "remem", "prefer", "habit",
            "是", "叫", "位于", "成立于", "开发", "发布", "融资", "价格", "版本",
            "工作流", "流程", "方法", "做法", "步骤",
            "用中文", "中文回答", "用英文", "英文回答", "简洁", "详细",
            "language", "chinese", "english", "concise", "detailed",
        ]
        score += min(0.35, sum(1 for s in fact_signals if s in tl) * 0.06)

        # 2b. 明确的用户偏好表达 → 额外加权(即使回答很短)
        pref_signals = ["用中文", "用英文", "中文回答", "英文回答", "以后", "希望", "偏好", "喜欢", "简洁", "详细"]
        if any(s in tl for s in pref_signals):
            score += 0.25

        # 3. 含具体实体(数字/专名/路径)→ 加分
        if re.search(r'\d+[%万亿美元元GBMB年]', t):
            score += 0.1
        if re.search(r'[\w\u4e00-\u9fff\-\.]+\.(md|txt|py|json|yaml|yml|csv|html)', t):
            score += 0.1
        if re.search(r'https?://\S+', t):
            score += 0.05

        # 4. 回答长度体现实质内容(太短=敷衍/闲聊)
        if len(final_answer) >= 60:
            score += 0.1
        elif len(final_answer) < 15:
            score -= 0.1

        # 5. 闲聊/无实质内容降权(仅当任务本身是闲聊, 不误伤"用中文回答我"这类偏好指令)
        chat_markers = ["你好", "谢谢", "再见", "hi", "thanks", "hello", "嗨"]
        if any(m in tl for m in chat_markers) and len(task) < 20 and not any(
            s in tl for s in ("用中文", "用英文", "以后", "希望", "偏好", "记住")
        ):
            score -= 0.25

        return max(0.0, min(1.0, score))

    # ---- 关键信息提取 ----

    def extract_key_facts(self, task: str, final_answer: str) -> List[str]:
        """从交互中提取"值得记住的关键事实/偏好"短句(最多5条)."""
        facts = []
        t = task.strip()
        a = final_answer.strip()

        # 偏好类: 用户明确表达偏好
        for m in re.finditer(r'(?:以后|之后|希望|请|记得)?\s*(?:用中文|用英文|中文回答|英文回答|简洁|简短|详细|详细一点|不要用emoj?i)', t + "\n" + a, re.IGNORECASE):
            fact = m.group(0).strip()
            if fact and fact not in facts:
                facts.append(f"偏好: {fact}")

        # 事实类: "X是/叫/位于/成立于/发布/融资了 Y"
        for m in re.finditer(r'([\u4e00-\u9fffA-Za-z][\u4e00-\u9fff\w\- ]{1,40})\s*(?:是|叫|位于|成立于|属于)\s*([\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fff\w\- ]{1,40})', a):
            if m.group(1).strip() and m.group(2).strip():
                facts.append(f"事实: {m.group(1).strip()} 是 {m.group(2).strip()}")

        # 关键数字事实: 融资额/价格/版本/比例 (支持 "融资7500万" 和 "完成7500万融资" 两种语序)
        for m in re.finditer(r'([\u4e00-\u9fffA-Za-z][\u4e00-\u9fff\w\-]{1,30})[^\d]{0,6}(?:完成|获得|拿到)?[^\d]{0,6}(\d+[.\d]*)\s*([%万美元亿元倍MBGB]+)', a):
            subj, num, unit = m.group(1), m.group(2), m.group(3)
            if subj and num and unit:
                facts.append(f"数据: {subj} {num}{unit}")

        return list(dict.fromkeys(facts))[:5]

    # ---- 整合入口 ----

    def consolidate(self, task: str, trajectory: Dict[str, Any]):
        """异步整合: 打分 → 过滤 → 写入长期记忆."""
        final_answer = str(trajectory.get("final_answer", "") or "")
        success = bool(trajectory.get("success", False))
        importance = self.compute_importance(task, final_answer, success, trajectory.get("metadata"))

        if importance >= self.importance_threshold and final_answer:
            # 1. 写入语义记忆(带重要性标注)
            if self.semantic_memory and hasattr(self.semantic_memory, "remember"):
                try:
                    text_to_remember = (
                        f"Task: {task}\nOutcome: {final_answer[:800]}\n"
                        f"Success: {success}\nImportance: {importance:.2f}"
                    )
                    self.semantic_memory.remember(text_to_remember, auto_link=True)
                except Exception:
                    pass

            # 2. 提取关键事实 → 用户画像(长期偏好/事实)
            if self.user_profile:
                try:
                    facts = self.extract_key_facts(task, final_answer)
                    for f in facts:
                        if f.startswith("偏好"):
                            k, v = f.split(":", 1)
                            self.user_profile.set_preference("_auto_" + k.strip(), v.strip())
                        elif f.startswith("事实"):
                            k, v = f.split(":", 1)
                            self.user_profile.add_fact("_auto_" + k.strip(), v.strip(), confidence=0.7)
                        elif f.startswith("数据"):
                            # 关键数字事实: 以"主体 数值"形式存入 facts
                            _, v = f.split(":", 1)
                            v = v.strip()
                            # 用主体作为 key 片段, 数值作为 value
                            parts = v.split(" ", 1)
                            if len(parts) == 2:
                                key = "_auto_data_" + re.sub(r'\s+', '_', parts[0])[:30]
                                self.user_profile.add_fact(key, parts[1], confidence=0.7)
                            else:
                                self.user_profile.add_fact("_auto_data_" + str(len(self.user_profile._profile.get("facts", {}))), v, confidence=0.6)
                except Exception:
                    pass

        # 3. 始终更新用户画像基础偏好(启发式)
        if self.user_profile:
            try:
                self.user_profile.update_from_interaction(task, final_answer, llm_client=self.llm_client)
            except Exception:
                pass

    # ---- 记忆查询辅助 ----

    def filter_recall(self, results: List[Dict[str, Any]], max_items: int = 5) -> List[Dict[str, Any]]:
        """按 importance 降序 + 时间新鲜度, 从召回结果中挑最值得看的."""
        scored = []
        for r in results:
            imp = float(r.get("importance", 0.3))
            created = r.get("created_at", "")
            try:
                age_days = max(0.0, (datetime.now() - datetime.fromisoformat(created)).total_seconds() / 86400.0)
            except Exception:
                age_days = 0.0
            freshness = max(0.0, 1.0 - age_days / 30.0)  # 30天内新鲜度线性衰减
            scored.append((imp * 0.7 + freshness * 0.3, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:max_items]]


# ---------------------------------------------------------------------------
# 3. Context Compressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """Token-aware compression of conversation history and observations."""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def compress_events(self, events: List[WorkingMemoryEvent],
                        target_tokens: int) -> str:
        """Compress a list of working-memory events into a summary."""
        if not events:
            return ""
        full_text = "\n".join(
            f"{e.role}: {e.content}" for e in events
        )
        current = _estimate_tokens(full_text)
        if current <= target_tokens:
            return full_text

        # Extractive fallback: keep first user request + last few exchanges.
        first_user = next((e for e in events if e.role == "user"), None)
        recent = events[-6:]
        parts = []
        if first_user:
            parts.append(f"Original request: {first_user.content}")
        parts.append("Recent exchanges:")
        for e in recent:
            prefix = "User" if e.role == "user" else e.role.capitalize()
            parts.append(f"- {prefix}: {e.content[:200]}")
        extractive = "\n".join(parts)
        if _estimate_tokens(extractive) <= target_tokens:
            return extractive

        # LLM abstractive summary.
        if self.llm_client:
            try:
                summary = self.llm_client.chat([
                    {"role": "system", "content": (
                        "Summarize the conversation so far in under 200 words. "
                        "Preserve facts the assistant learned and any user preferences."
                    )},
                    {"role": "user", "content": full_text[:4000]},
                ], temperature=0.2, max_tokens=target_tokens // 2)
                return f"[Compressed conversation summary]\n{summary.strip()}"
            except Exception:
                pass

        # Hard truncate.
        return extractive[:target_tokens * 4]

    def compress_observations(self, observations: List[str],
                              target_tokens: int) -> str:
        if not observations:
            return ""
        joined = "\n".join(observations)
        if _estimate_tokens(joined) <= target_tokens:
            return joined
        # Keep most recent observations.
        kept = []
        used = 0
        for obs in reversed(observations):
            cost = _estimate_tokens(obs)
            if used + cost > target_tokens:
                break
            kept.append(obs)
            used += cost
        kept.reverse()
        return "\n".join(kept)


# ---------------------------------------------------------------------------
# 4. Context Engine
# ---------------------------------------------------------------------------

class ContextEngine:
    """
    Unified facade for agent context & memory.

    Usage
    -----
    engine = ContextEngine(config, backend=backend)
    context = engine.recall(task)                       # before generation
    engine.observe_tool_call("file_ops", {...})         # during execution
    engine.consolidate(task, trajectory)                # after turn ends
    """

    def __init__(self, config: Any, backend: Optional[Any] = None,
                 episodic_memory: Optional[Any] = None,
                 semantic_memory: Optional[Any] = None):
        self.config = config
        self.backend = backend
        self.enabled = getattr(config, "memory", None) and config.memory.enabled

        # Token budgets
        self.max_total_context = getattr(config.memory, "max_context_tokens", 6000)
        self.working_budget = int(self.max_total_context * 0.25)
        self.episodic_budget = int(self.max_total_context * 0.20)
        self.semantic_budget = int(self.max_total_context * 0.20)
        self.profile_budget = int(self.max_total_context * 0.10)
        self.compressor_budget = int(self.max_total_context * 0.25)

        # Subsystems
        self.working_memory = WorkingMemory(max_events=200)
        self.user_profile = UserProfileMemory(
            storage_path=getattr(config.memory, "user_memory_path", "./data/user.md")
        )
        self.compressor = ContextCompressor(llm_client=self._llm_client())

        # 显式记忆整合层: 在短期→长期之间做价值过滤
        self.consolidator = MemoryConsolidator(
            semantic_memory=None,
            user_profile=self.user_profile,
            llm_client=self._llm_client(),
            importance_threshold=getattr(config.memory, "importance_threshold", 0.45),
        )

        self.episodic_memory: Optional[Any] = episodic_memory
        self.semantic_memory: Optional[Any] = semantic_memory

        if self.enabled:
            if self.episodic_memory is None and ExperienceBuffer is not None:
                try:
                    self.episodic_memory = ExperienceBuffer(config)
                except Exception as e:
                    print(_style(f"  ContextEngine: episodic memory init failed ({e})", "2"))

            if self.semantic_memory is None and _WIKI_AVAILABLE:
                try:
                    llm_client = self._llm_client()
                    self.semantic_memory = create_memory_manager(
                        kg_storage=config.memory.kg_storage_path,
                        episodic_storage=config.memory.episodic_storage_path,
                        llm_client=llm_client,
                    )
                except Exception as e:
                    print(_style(f"  ContextEngine: semantic memory init failed ({e})", "2"))

    def _llm_client(self) -> Optional[Any]:
        """Return a lightweight LLM client for memory operations if available."""
        if self.backend is None:
            return None
        # Try to wrap backend in PassthroughBackendClient if possible.
        try:
            from .wiki_memory import PassthroughBackendClient
            return PassthroughBackendClient(self.backend)
        except Exception:
            return None

    # ---- observation API ----

    def observe_user(self, task: str):
        self.working_memory.add("user", task, "message")

    def observe_assistant(self, text: str):
        self.working_memory.add("assistant", text, "message")

    def seed_history(self, turns: List[Dict[str, Any]]) -> None:
        """从持久化对话历史回填工作记忆, 让进程重启后仍能记住跨会话对话.

        工作记忆是内存态的, 重启即空; 若不回填, 重启后第一问就"失忆",
        尽管 data/conversation_history.json 里历史其实都在。
        """
        if not turns:
            return
        seeded = 0
        for turn in turns:
            user = (turn.get("user") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            if user:
                self.working_memory.add("user", user[:1000], "message")
                seeded += 1
            if assistant:
                self.working_memory.add("assistant", assistant[:1000], "message")
        if seeded:
            import logging
            logging.getLogger("context_engine").info(
                f"seeded {seeded} history turns into working memory"
            )

    def observe_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        self.working_memory.add_tool_call(tool_name, arguments)

    def observe_tool_result(self, tool_name: str, result: str, success: bool):
        self.working_memory.add_tool_result(tool_name, result, success)

    def observe_plan(self, plan_text: str):
        self.working_memory.add("system", plan_text, "plan")

    def observe_thought(self, thought_text: str):
        self.working_memory.add("assistant", thought_text, "thought")

    # ---- recall API ----

    def recall(self, task: str, k: int = 3) -> Dict[str, str]:
        """
        Retrieve all relevant context for a task, returned as formatted strings.
        Keys: working, episodic, semantic, profile, lessons.
        """
        if not self.enabled:
            return {"working": self.working_memory.format_for_prompt(self.working_budget)}

        # 1. Working memory (most important, freshest)
        working_ctx = self.working_memory.format_for_prompt(self.working_budget)

        # 2. User profile
        profile_ctx = self.user_profile.format(self.profile_budget)

        # 3. Semantic memory (wiki/entities)
        semantic_ctx = ""
        if self.semantic_memory:
            try:
                ctx = self.semantic_memory.get_relevant_context(task, max_pages=k)
                seen = set()
                parts = []
                if ctx.get("episodes"):
                    for ep in ctx["episodes"][:k]:
                        title = ep.get("title", "")
                        summary = (ep.get("summary") or "")[:180]
                        key = f"{title}:{summary}".lower()
                        if summary and key not in seen:
                            seen.add(key)
                            parts.append(f"- [{title}] {summary}")
                if ctx.get("knowledge_nodes"):
                    for node in ctx["knowledge_nodes"][:k]:
                        title = node.get("title", "")
                        snippet = node.get("properties", {}).get("Summary", "")[:180]
                        key = f"{title}:{snippet}".lower()
                        if snippet and key not in seen:
                            seen.add(key)
                            parts.append(f"- [{title}] {snippet}")
                if parts:
                    semantic_ctx = _truncate_to_budget(
                        parts, self.semantic_budget,
                        header="## Relevant Knowledge:\n"
                    )
            except Exception as e:
                print(_style(f"  ContextEngine semantic recall failed: {e}", "2"))

        # 4. Episodic memory (similar past tasks)
        episodic_ctx = ""
        if self.episodic_memory:
            try:
                similar = self.episodic_memory.get_similar(task, k=k, success_only=True)
                seen = set()
                parts = []
                for exp in similar[:k]:
                    traj = exp.trajectory or {}
                    outcome = str(traj.get("final_answer", traj.get("outcome", "")))[:180]
                    key = f"{exp.task[:60]}:{outcome}".lower()
                    if outcome and key not in seen:
                        seen.add(key)
                        parts.append(f"- [{exp.task[:60]}] {outcome}")
                if parts:
                    episodic_ctx = _truncate_to_budget(
                        parts, self.episodic_budget,
                        header="## Similar Past Tasks:\n"
                    )
            except Exception as e:
                print(_style(f"  ContextEngine episodic recall failed: {e}", "2"))

        # 5. Lessons
        lessons_ctx = ""
        if self.episodic_memory:
            try:
                lessons = self.episodic_memory.get_lessons(task, k=k)
                if lessons:
                    seen = set()
                    parts = []
                    for lesson in lessons[:k]:
                        marker = "DO" if lesson.success else "DO NOT"
                        text = f"{lesson.condition}: {lesson.action}"
                        key = text.lower()
                        if key not in seen:
                            seen.add(key)
                            parts.append(f"- [{marker}] {text}")
                    lessons_ctx = _truncate_to_budget(
                        parts, self.episodic_budget,
                        header="## Learned Lessons:\n"
                    )
            except Exception as e:
                print(_style(f"  ContextEngine lessons recall failed: {e}", "2"))

        return {
            "working": working_ctx,
            "profile": profile_ctx,
            "semantic": semantic_ctx,
            "episodic": episodic_ctx,
            "lessons": lessons_ctx,
        }

    def build_system_context(self, task: str, include_history: bool = True,
                             k: int = 3, mode: str = "normal") -> str:
        """Build the full context block to prepend to the system prompt.

        mode:
          - "turbo": minimal context for simple Q&A (avoid memory noise)
          - "normal": standard balanced recall
          - "deep": maximum context for complex tasks
        """
        if mode == "turbo":
            # Only the most recent working memory turns; no long-term memory.
            return self.working_memory.format_for_prompt(
                max_tokens=min(800, self.working_budget),
                include_tool_history=False,
            )

        # Adaptive budgets based on task length and mode.
        task_tokens = _estimate_tokens(task)
        scale = 1.5 if mode == "deep" else 1.0
        self.working_budget = int(min(self.max_total_context * 0.30 * scale, 2000))
        self.episodic_budget = int(self.max_total_context * 0.15 * scale)
        self.semantic_budget = int(self.max_total_context * 0.15 * scale)
        self.profile_budget = int(self.max_total_context * 0.08 * scale)
        self.compressor_budget = int(self.max_total_context * 0.32 * scale)

        ctx = self.recall(task, k=k)
        blocks = []
        if ctx.get("profile"):
            blocks.append(ctx["profile"])
        if include_history and ctx.get("working"):
            blocks.append(ctx["working"])
        if ctx.get("semantic"):
            blocks.append(ctx["semantic"])
        if ctx.get("episodic"):
            blocks.append(ctx["episodic"])
        if ctx.get("lessons"):
            blocks.append(ctx["lessons"])

        # Global budget guard: compress working memory if total exceeds budget.
        total = sum(_estimate_tokens(b) for b in blocks)
        if total > self.max_total_context and ctx.get("working"):
            events = self.working_memory.get_events(event_types=["message"], limit=50)
            # Always keep the current user request and the last assistant turn.
            preserved = [e for e in events if e.role in ("user", "assistant")][-4:]
            older = [e for e in events if e not in preserved]
            compressed = self.compressor.compress_events(
                older, self.compressor_budget
            )
            preserved_text = "\n".join(
                f"{'User' if e.role == 'user' else 'Assistant'}: {e.content[:300]}"
                for e in preserved
            )
            compressed_block = f"## Current Conversation (compressed):\n{preserved_text}\n\n[Older history]\n{compressed}"
            blocks = [b for b in blocks if not b.startswith("## Current Conversation")]
            blocks.insert(1, compressed_block)

        return "\n\n".join(blocks)

    # ---- consolidation API ----

    def consolidate(self, task: str, trajectory: Dict[str, Any]):
        """
        Persist the completed interaction into long-term memory (via MemoryConsolidator).

        The consolidator applies an importance gate: only high-value interactions
        (successful + containing facts/preferences/workflows) enter long-term memory.
        Runs in a background thread so response latency is not affected.
        """
        if not self.enabled:
            return

        # 同步 consolidator 的 semantic_memory 引用(初始化为 None)
        if self.consolidator is not None and self.consolidator.semantic_memory is None:
            self.consolidator.semantic_memory = self.semantic_memory

        def _work():
            try:
                if self.consolidator is not None:
                    self.consolidator.consolidate(task, trajectory)
                else:
                    # 无整合器时退化为旧行为
                    success = trajectory.get("success", False)
                    final_answer = str(trajectory.get("final_answer", "") or "")
                    if self.semantic_memory and final_answer:
                        try:
                            text_to_remember = f"Task: {task}\nOutcome: {final_answer}\nSuccess: {success}"
                            self.semantic_memory.remember(text_to_remember, auto_link=True)
                        except Exception:
                            pass
                    self.user_profile.update_from_interaction(task, final_answer,
                                                              llm_client=self._llm_client())
            except Exception as e:
                print(_style(f"  ContextEngine consolidation failed: {e}", "2"))

        threading.Thread(target=_work, daemon=True).start()

    # ---- compatibility shims ----

    def remember(self, text: str, *, context: str = "", auto_link: bool = True,
                 entity_names: Optional[List[str]] = None) -> List[str]:
        """Backward-compatible interface: extract entities and build wiki pages."""
        if self.semantic_memory and hasattr(self.semantic_memory, "remember"):
            try:
                return self.semantic_memory.remember(
                    text,
                    context=context,
                    auto_link=auto_link,
                    entity_names=entity_names,
                )
            except Exception:
                pass
        return []

    def store_interaction(self, task: str, trajectory: Dict[str, Any],
                          outcome: str, success: bool,
                          entities: Optional[List[str]] = None) -> str:
        """Backward-compatible interface used by agent.py."""
        if self.semantic_memory and hasattr(self.semantic_memory, "store_interaction"):
            try:
                return self.semantic_memory.store_interaction(
                    task=task, trajectory=trajectory,
                    outcome=outcome, success=success, entities=entities
                )
            except Exception:
                pass
        if self.episodic_memory:
            try:
                return self.episodic_memory.add_episode(
                    task=task, trajectory=trajectory,
                    task_type=trajectory.get("task_type", "unknown"),
                )
            except Exception:
                pass
        return ""

    def get_relevant_context(self, query: str, max_pages: int = 3) -> Dict[str, Any]:
        """Backward-compatible interface."""
        ctx = self.recall(query, k=max_pages)
        return {
            "episodes": [{"title": "Recent", "summary": ctx.get("episodic", "")[:500]}],
            "knowledge_nodes": [{"title": "Relevant", "properties": {"Summary": ctx.get("semantic", "")[:500]}}],
            "related_nodes": [],
        }

    def augment_prompt_with_memory(self, base_prompt: str, query: str,
                                   max_tokens: int = 1024) -> str:
        """Backward-compatible interface."""
        ctx = self.build_system_context(query, include_history=True, k=2)
        if ctx:
            return f"{base_prompt}\n\n{ctx}"
        return base_prompt

    def get_user_profile(self) -> str:
        return self.user_profile.format()

    def stats(self) -> Dict[str, Any]:
        return {
            "working_events": len(self.working_memory.events),
            "episodes": self.episodic_memory.count() if self.episodic_memory else 0,
            "semantic_pages": len(self.semantic_memory.graph._pages) if self.semantic_memory else 0,
            "profile_facts": len(self.user_profile._profile.get("facts", {})),
            "profile_preferences": len(self.user_profile._profile.get("preferences", {})),
        }
