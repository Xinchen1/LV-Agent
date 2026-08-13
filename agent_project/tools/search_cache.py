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
