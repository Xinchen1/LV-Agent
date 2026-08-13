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
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]

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
