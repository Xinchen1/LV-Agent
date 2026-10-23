"""Lightweight embedding client for semantic memory recall.

Talks to any OpenAI-compatible /v1/embeddings endpoint (Ollama:
http://localhost:11434/v1 with e.g. nomic-embed-text). Designed to be
best-effort: any failure returns None and callers fall back to lexical
scoring, so memory never hard-depends on a running embedding service.

Failure containment:
- short timeout (never stall an agent turn)
- circuit breaker: N consecutive failures disable the client for the session
- in-process cache keyed by sha256(text) to avoid re-embedding
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Callable, List, Optional

_log = logging.getLogger(__name__)

PROXY_BYPASS_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class EmbeddingClient:
    """OpenAI-compatible embeddings client with circuit breaker + cache."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
        timeout: float = 4.0,
        max_failures: int = 3,
        api_key: str = "",
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = max(0.5, float(timeout))
        self.max_failures = max(1, int(max_failures))
        self.api_key = api_key or ""
        self._lock = threading.RLock()
        self._cache: dict[str, Optional[List[float]]] = {}
        self._failures = 0
        self._broken = False
        # 可注入测试用: 覆盖后 embed() 走它
        self.embed_fn: Optional[Callable[[str], Optional[List[float]]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        with self._lock:
            return not self._broken and bool(self.base_url) and bool(self.model)

    def embed(self, text: str) -> Optional[List[float]]:
        """Return the embedding vector, or None if unavailable/failed."""
        if not text:
            return None
        if not self.is_available():
            return None

        key = hashlib.sha256(self.model.encode("utf-8") + b"\x00" + text.encode("utf-8", "ignore")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        try:
            if self.embed_fn is not None:
                vec = self.embed_fn(text)
            else:
                vec = self._embed_http(text)
        except Exception as e:
            self._note_failure(e)
            return None

        if vec:
            with self._lock:
                self._failures = 0
                self._cache[key] = vec
            return vec

        self._note_failure("empty embedding")
        return None

    def warmup(self) -> bool:
        """Probe the endpoint once. Call from a background thread at init."""
        return self.embed("warmup") is not None

    def reset_breaker(self) -> None:
        """Re-enable the client (e.g. after a manual model pull)."""
        with self._lock:
            self._failures = 0
            self._broken = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _note_failure(self, e: Exception | str) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.max_failures:
                self._broken = True
                _log.warning(
                    "embedding service unreachable (%s failures: %s) → 向量召回禁用, 退回词面匹配",
                    self._failures, e,
                )

    def _embed_http(self, text: str) -> Optional[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        # 直连: 经系统代理访问 localhost 会被 Clash 类代理拒掉(502)
        with PROXY_BYPASS_OPENER.open(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [{}])[0].get("embedding")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity with zero-vector/degenerate guards."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def vector_score(cos: float, floor: float = 0.42, ceil: float = 0.95) -> float:
    """把 cosine 压缩到与词面 coverage 同尺度的 0-1 分。

    nomic-embed-text 这类模型上: 无关文本 cosine ≈0.3-0.5 → 几乎 0 分,
    相关改写 ≈0.6-0.75 → 0.35-0.63, 高度相似 ≥0.95 → 1.0。
    """
    if cos <= floor:
        return 0.0
    return max(0.0, min(1.0, (cos - floor) / (ceil - floor)))
