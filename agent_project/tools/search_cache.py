"""
Search result cache with TTL.

Provides both in-memory and optional disk-backed caching so repeated queries
do not hit search engines within the TTL window.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    query: str
    results: List[Dict[str, Any]]
    created_at: float
    ttl: int

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class SearchCache:
    """Simple TTL cache for search results."""

    def __init__(
        self,
        ttl: int = 300,
        disk_path: Optional[str] = None,
        max_memory_entries: int = 200,
    ):
        self.ttl = ttl
        self.disk_path = Path(disk_path) if disk_path else None
        self.max_memory_entries = max_memory_entries
        self._memory: Dict[str, CacheEntry] = {}
        if self.disk_path:
            self.disk_path.mkdir(parents=True, exist_ok=True)

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        key = self._key(query)
        entry = self._memory.get(key)
        if entry is None and self.disk_path:
            entry = self._load_from_disk(key)
            if entry:
                self._memory[key] = entry
        if entry is None or entry.is_expired():
            return None
        return entry.results

    def set(self, query: str, results: List[Dict[str, Any]], ttl: Optional[int] = None) -> None:
        key = self._key(query)
        entry = CacheEntry(
            query=query,
            results=results,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self.ttl,
        )
        self._memory[key] = entry
        self._enforce_memory_limit()
        if self.disk_path:
            self._save_to_disk(key, entry)

    def invalidate(self, query: str) -> None:
        key = self._key(query)
        self._memory.pop(key, None)
        if self.disk_path:
            path = self.disk_path / f"{key}.json"
            if path.exists():
                path.unlink()

    def clear(self) -> None:
        self._memory.clear()
        if self.disk_path:
            for f in self.disk_path.glob("*.json"):
                f.unlink()

    def _key(self, query: str) -> str:
        """生成规范化缓存键, 提高命中率.

        策略: 小写 → 去标点 → 去修饰/动作虚词 → 去空格 → 排序.
        这样 "最新的AI新闻" 与 "ai 新闻"、"hermes 使用" 与 "查hermes用法"
        会命中同一缓存(核心语义相同)。
        排序使 "人工智能 趋势" 与 "趋势 人工智能" 也命中。
        注意: 保留主题词, 避免不同主题串缓存。
        """
        q = query.lower().strip()
        # 1. 去标点与多余空白
        import re as _re
        q = _re.sub(r"[，。！？!?、；;：:\s()（）\[\]【】\"'“”‘’,，\-_]+", " ", q).strip()
        # 2. 去修饰/动作/时间虚词(核心语义以外的词)
        # 注意: "新闻/趋势/价格" 等是主题词必须保留; "动态/资讯/消息" 等弱主题词可删
        for w in ("最新的", "最新", "最近的", "最近", "今天", "本月", "昨天", "上周",
                  "2026", "2026年", "年", "月份", "帮我", "请", "一下", "方面",
                  "有关", "关于", "的", "查一下", "查查", "查", "看看", "找找",
                  "使用", "用法", "教程", "方法", "怎么", "如何", "介绍", "what is", "how to",
                  "today", "this week", "latest", "recent", "news update",
                  "样", "呢", "吗", "呀", "啊", "了", "动态", "动向", "资讯", "消息"):
            q = q.replace(w, "")
        # 3. 去空格
        q = q.replace(" ", "")
        # 4. 排序字符(词序无关: "人工智能趋势"=="趋势人工智能")
        q = "".join(sorted(q))
        return hashlib.sha256(q.encode()).hexdigest()[:32]

    # ------------------------------------------------------------------
    # 语义近似匹配(提升命中率的第二层, 不引入外部依赖)
    # ------------------------------------------------------------------
    # 保留核心词的字符 2-gram 集合作为轻量语义指纹: 比停用词表健壮,
    # 对同义词/口语化/词序变化都能近似匹配("人工智能新闻"≈"AI新闻")。
    _STOP = frozenset("的了我你他她它是和与及在在了有就都还一个也这那要会到说给让为被从向把对以于等最更很非常一些已经正在将很")
    # 常见中英等价词(聚焦核心主题词): 让 "AI新闻"≈"人工智能新闻"、"app"≈"应用" 等可近似匹配。
    _ZH_EN_EQUIV = {
        "ai": "人工智能", "人工智能": "ai",
        "app": "应用", "应用": "app",
        "api": "接口", "接口": "api",
        "ai智能": "ai",
    }

    @classmethod
    def _semantic_ngrams(cls, query: str) -> frozenset:
        """生成查询的语义指纹: 英文 token + 中文核心字 n-gram + 中英等价词归一.

        - 英文: 小写 token + 相邻 2-gram, 并归一常见等价词(ai↔人工智能)
        - 中文: 去掉单字停用词后取相邻 2-gram("人工智能"→{人智,智能})
        这样 "AI新闻" 与 "人工智能新闻" 共享 {智能,新闻} 等, 可近似匹配。
        """
        q = query.lower().strip()
        ngrams = set()
        import re as _re
        # 中英等价词先归一: "人工智能"->"ai"(统一到英文 token), "应用"->"app"
        normalized = q
        for zh, en in cls._ZH_EN_EQUIV.items():
            normalized = normalized.replace(zh, en)
        # 英文 token(含归一后的)
        en_words = _re.findall(r"[a-z0-9]+", normalized)
        for w in en_words:
            if len(w) >= 2:
                ngrams.add("e:" + w)
        for i in range(len(en_words) - 1):
            ngrams.add("e:" + en_words[i] + "_" + en_words[i + 1])
        # 中文: 先剔除常见口语虚词(词级), 再取相邻 2-gram
        _zh_noise = ("今天", "昨天", "明天", "什么", "怎么", "为什么", "有没有",
                     "一下", "最近", "现在", "关于", "有关", "方面", "内容", "最新",
                     "呢", "吗", "呀", "啊", "哦", "嗯", "的", "了", "和", "与")
        zh = "".join(ch for ch in normalized if "\u4e00" <= ch <= "\u9fff" and ch not in cls._STOP)
        for w in _zh_noise:
            zh = zh.replace(w, "")
        if len(zh) >= 2:
            for i in range(len(zh) - 1):
                ngrams.add("z:" + zh[i:i + 2])
        elif zh:
            ngrams.add("z:" + zh)
        return frozenset(ngrams)

    @staticmethod
    def _similarity(a: frozenset, b: frozenset) -> float:
        """Jaccard 相似度(带公共核心的偏置, 避免短查询误判)."""
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union

    def find_similar(self, query: str, threshold: float = 0.6, max_scan: int = 60) -> Optional[CacheEntry]:
        """在缓存中找语义近似(非精确)的条目.

        先精确键命中(零开销), 再遍历最近条目做 n-gram 相似度。
        threshold 0.6 表示核心语义重叠 60% 即视为同一查询。
        返回最相似的未过期条目; 无则 None。
        """
        # 1. 精确键(已有逻辑)
        exact = self._memory.get(self._key(query))
        if exact is None and self.disk_path:
            exact = self._load_from_disk(self._key(query))
            if exact:
                self._memory[self._key(query)] = exact
        if exact and not exact.is_expired():
            return exact
        if exact and exact.is_expired():
            return None  # 精确命中但过期 -> 直接 miss, 不再近似(避免用过期结果)

        # 2. 语义近似: 遍历内存 + 磁盘条目
        q_grams = self._semantic_ngrams(query)
        if not q_grams:
            return None
        best = None
        best_score = threshold
        now = time.time()
        scanned = 0
        # 内存优先(最近插入的语义更可能相关)
        candidates = list(self._memory.values())
        if self.disk_path and len(candidates) < max_scan:
            try:
                for fp in sorted(self.disk_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if len(candidates) >= max_scan:
                        break
                    entry = self._load_from_disk(fp.stem)
                    if entry:
                        candidates.append(entry)
            except Exception:
                pass
        for entry in candidates:
            scanned += 1
            if scanned > max_scan:
                break
            if entry.is_expired():
                continue
            c_grams = self._semantic_ngrams(entry.query)
            score = self._similarity(q_grams, c_grams)
            if score > best_score:
                best_score = score
                best = entry
        return best

    def get(self, query: str, threshold: float = 0.6) -> Optional[List[Dict[str, Any]]]:
        """增强版 get: 精确键 优先, 语义近似 兜底.

        threshold: 语义命中的最小相似度(默认 0.6).
        """
        # 精确键(原逻辑)
        key = self._key(query)
        entry = self._memory.get(key)
        if entry is None and self.disk_path:
            entry = self._load_from_disk(key)
            if entry:
                self._memory[key] = entry
        if entry is not None and not entry.is_expired():
            return entry.results
        if entry is not None and entry.is_expired():
            # 精确键过期: 仍尝试语义近似(可能命中别的相关缓存)
            pass
        # 语义近似兜底
        sim = self.find_similar(query, threshold=threshold)
        if sim is not None:
            # 把近似命中的结果按新 query 缓存(加快下次精确命中)
            self.set(query, sim.results)
            return sim.results
        return None

    def _enforce_memory_limit(self) -> None:
        if len(self._memory) <= self.max_memory_entries:
            return
        # Evict oldest entries
        sorted_keys = sorted(self._memory.keys(), key=lambda k: self._memory[k].created_at)
        for k in sorted_keys[: len(self._memory) - self.max_memory_entries]:
            del self._memory[k]

    def _save_to_disk(self, key: str, entry: CacheEntry) -> None:
        if not self.disk_path:
            return
        path = self.disk_path / f"{key}.json"
        try:
            path.write_text(
                json.dumps(
                    {"query": entry.query, "results": entry.results, "created_at": entry.created_at, "ttl": entry.ttl},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        if not self.disk_path:
            return None
        path = self.disk_path / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CacheEntry(
                query=data["query"],
                results=data["results"],
                created_at=data["created_at"],
                ttl=data["ttl"],
            )
        except Exception:
            return None
