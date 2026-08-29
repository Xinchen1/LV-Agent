"""缓存层（从 agent.py 抽出的独立单元，去 God Object 第一步）。

# ToolResultCache: 工具结果 LRU 缓存,命中复用、容量上限、淘汰最久未用。
# MemoCache: 通用记忆化缓存,按 (namespace, *key) 惰性生成并缓存 factory() 结果。

# 两把锁，线程安全；不再与 agent.py 共享类级锁。
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple

_DEFAULT_CAPACITY = 512


class ToolResultCache:
    """工具调用结果缓存: 同一轮内重复调用相同工具时直接复用已存结果"""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY):
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._capacity = capacity
        self._lock = Lock()

    @property
    def store(self) -> "OrderedDict[str, Any]":
        """Underlying dict, read directly by the execution engine (preserving existing behavior without bypassing cache semantics)"""
        return self._store

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            val = self._store.get(key)
            if val is not None:
                self._store.move_to_end(key)
            return val

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            if len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store


class MemoCache:
    """Universal memoization: cache factory() results keyed by (namespace, *key)"""

    def __init__(self):
        self._store: Dict[Tuple, Any] = {}
        self._lock = Lock()

    def get_or_set(self, key: Tuple, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._store:
                self._store[key] = factory()
            return self._store[key]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
