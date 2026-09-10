"""缓存层（从 agent.py 抽出的独立单元，去 God Object 第一步）。

# ToolResultCache: 工具结果 LRU 缓存,命中复用、容量上限、淘汰最久未用。
# MemoCache: 通用记忆化缓存,按 (namespace, *key) 惰性生成并缓存 factory() 结果。
# PromptPrefixCache: 静态 prompt 前缀缓存(对标 Hermes prompt_caching):
#   按输入哈希去重组装开销; 命中/未命中计数供观测。注意: 真正省 token 的
#   服务端缓存(Anthropic cache_control)见 model_backends.AnthropicBackend。

# 锁均为实例级，线程安全；不再与 agent.py 共享类级锁。
"""

from __future__ import annotations

import hashlib
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


class PromptPrefixCache:
    """静态 prompt 前缀缓存: 相同输入直接复用组装结果, 省 CPU/正则开销.

    key = sha256(namespace + 各段文本); 命中计数 hits/misses 供观测.
    线程安全, LRU 淘汰. 不负责服务端 token 计费缓存, 那是 provider 的事.
    """

    def __init__(self, capacity: int = 64):
        self._store: "OrderedDict[str, str]" = OrderedDict()
        self._capacity = capacity
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(namespace: str, *parts: str) -> str:
        h = hashlib.sha256()
        h.update(namespace.encode("utf-8"))
        for p in parts:
            h.update(b"\x00")
            h.update((p or "").encode("utf-8"))
        return h.hexdigest()

    def get_or_build(self, namespace: str, parts: Tuple[str, ...],
                     builder: Callable[[], str]) -> str:
        """命中返回缓存, 未命中调 builder() 组装并缓存."""
        key = self.make_key(namespace, *parts)
        with self._lock:
            if key in self._store:
                self.hits += 1
                self._store.move_to_end(key)
                return self._store[key]
            self.misses += 1
        value = builder()
        with self._lock:
            self._store[key] = value
            if len(self._store) > self._capacity:
                self._store.popitem(last=False)
        return value

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
