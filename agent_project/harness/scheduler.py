"""Effect scheduler -- safe concurrency with deterministic lanes.

Lanes (assigned by the kernel):

* ``parallel``   -- reads/pure/net effects run concurrently, bounded by a
  semaphore so a burst of 50 reads can't exhaust file descriptors.
* ``write:<path>`` -- effects mutating the same path serialize on a
  per-path lock; different paths still run concurrently.
* ``serial``     -- unclassified exec effects serialize globally.

Identical effects (same idempotency key) inside one session are executed
once and replayed from an in-memory cache with a TTL -- the agent re-reading
the same file five times in a turn costs one syscall, not five.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from .effects import Effect
from .errors import ToolTimeoutError


@dataclass(frozen=True)
class Outcome:
    """Result of one scheduled effect."""

    effect: Effect
    ok: bool
    output: str = ""
    error: Optional[BaseException] = None
    from_cache: bool = False
    latency_ms: float = 0.0


@dataclass
class _CacheEntry:
    output: str
    expires_at: float


class Scheduler:
    """Async lane scheduler with dedup cache.

    ``submit`` is coroutine-based; the underlying executor is sync
    (legacy tools), so work runs in the default thread-pool via
    ``asyncio.to_thread`` and the event loop stays responsive.
    """

    def __init__(
        self,
        max_parallel: int = 8,
        cache_ttl_s: float = 30.0,
        default_timeout_s: float = 120.0,
    ):
        self._sem = asyncio.Semaphore(max_parallel)
        self._path_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._serial_lock = asyncio.Lock()
        self._cache: Dict[str, _CacheEntry] = {}
        self._cache_ttl = cache_ttl_s
        self._default_timeout = default_timeout_s
        self._inflight: Dict[str, asyncio.Task] = {}

    # ---------- public API ----------

    async def submit(
        self,
        effect: Effect,
        lane: str,
        run: Callable[[Effect], str],
    ) -> Outcome:
        """Admit *effect* into its lane and await the outcome.

        Mutating effects bypass the cache; read/pure effects deduplicate
        and share one in-flight task for identical concurrent intents.
        """

        key = effect.idempotency_key
        if effect.is_mutating:
            return await self._execute(effect, lane, run)

        hit = self._cache_get(key)
        if hit is not None:
            return Outcome(effect, ok=True, output=hit, from_cache=True)

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._execute_and_cache(effect, lane, run))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        outcome = await asyncio.shield(task)
        if outcome.from_cache or not outcome.ok:
            return outcome
        return Outcome(effect, ok=True, output=outcome.output, from_cache=False,
                       latency_ms=outcome.latency_ms)

    async def submit_many(
        self,
        items: list[Tuple[Effect, str]],
        run: Callable[[Effect], str],
    ) -> list[Outcome]:
        """Schedule a batch; per-path writes serialize, the rest overlaps."""

        return list(await asyncio.gather(*(self.submit(e, lane, run) for e, lane in items)))

    # ---------- internals ----------

    async def _execute_and_cache(
        self, effect: Effect, lane: str, run: Callable[[Effect], str]
    ) -> Outcome:
        outcome = await self._execute(effect, lane, run)
        if outcome.ok:
            self._cache_put(effect.idempotency_key, outcome.output)
        return outcome

    async def _execute(
        self, effect: Effect, lane: str, run: Callable[[Effect], str]
    ) -> Outcome:
        started = time.monotonic()
        try:
            if lane == "parallel":
                async with self._sem:
                    output = await self._call(effect, run)
            elif lane == "serial":
                async with self._serial_lock:
                    output = await self._call(effect, run)
            else:  # write:<paths> -> lock each canonical path in sorted order
                paths = sorted(lane.removeprefix("write:").split(","))
                locks = [self._path_locks[p] for p in paths]
                for lk in locks:
                    await lk.acquire()
                try:
                    output = await self._call(effect, run)
                finally:
                    for lk in reversed(locks):
                        lk.release()
            return Outcome(
                effect, ok=True, output=output,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in Outcome
            return Outcome(
                effect, ok=False, error=exc,
                latency_ms=(time.monotonic() - started) * 1000,
            )

    async def _call(self, effect: Effect, run: Callable[[Effect], str]) -> str:
        timeout = effect.timeout_s or self._default_timeout
        try:
            return await asyncio.wait_for(asyncio.to_thread(run, effect), timeout)
        except asyncio.TimeoutError as exc:
            raise ToolTimeoutError(
                f"effect '{effect.tool_name}' timed out after {timeout}s",
                detail={"timeout_s": timeout},
            ) from exc

    # ----- TTL cache -----

    def _cache_get(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.monotonic():
            self._cache.pop(key, None)
            return None
        return entry.output

    def _cache_put(self, key: str, output: str) -> None:
        self._cache[key] = _CacheEntry(output, time.monotonic() + self._cache_ttl)

    def clear_cache(self) -> None:
        self._cache.clear()
