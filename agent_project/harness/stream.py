"""Typed event stream -- frontends are subscribers, nothing more.

The runtime emits :class:`~agent_project.harness.events.Event` objects and
fine-grained delta signals (token chunks, status lines). The bus fans each
emission out to every subscriber with per-subscriber filtering.

Subscribers can be sync callables, async callables, or ``asyncio.Queue``
objects (for TUI/ACP-style consumers that already own a render loop).
A slow subscriber never blocks the runtime: queue delivery is bounded and
drops are counted, callable delivery is dispatched as a task.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set

from .events import Event


@dataclass(frozen=True)
class Delta:
    """High-frequency signal that is NOT journaled (tokens, status).

    Journal events define *what happened*; deltas define *what is happening*
    right now -- rendering detail that must never be replayed.
    """

    kind: str          # "token" | "reasoning" | "status"
    text: str
    turn_index: int = -1


Predicate = Callable[[Any], bool]
Sink = Callable[[Any], Any]


@dataclass
class _Subscriber:
    sink: Any                       # callable or asyncio.Queue
    predicate: Optional[Predicate]
    drops: int = 0


class EventBus:
    """Process-wide fan-out for one session's emissions."""

    def __init__(self, queue_bound: int = 256):
        self._subs: Set[int] = set()
        self._registry: dict[int, _Subscriber] = {}
        self._queue_bound = queue_bound
        self._next_id = 0
        self.total_drops = 0

    # ---------- subscription ----------

    def subscribe(
        self,
        sink: Any,
        predicate: Optional[Predicate] = None,
    ) -> Callable[[], None]:
        """Register *sink*; returns an unsubscribe handle."""

        sub_id = self._next_id
        self._next_id += 1
        self._registry[sub_id] = _Subscriber(sink, predicate)
        self._subs.add(sub_id)

        def _unsub() -> None:
            self._subs.discard(sub_id)
            self._registry.pop(sub_id, None)

        return _unsub

    def subscribe_events(self, sink: Any) -> Callable[[], None]:
        return self.subscribe(sink, lambda m: isinstance(m, Event))

    def subscribe_deltas(self, sink: Any) -> Callable[[], None]:
        return self.subscribe(sink, lambda m: isinstance(m, Delta))

    # ---------- emission ----------

    def emit(self, message: Any) -> None:
        """Fan out one message (Event or Delta) to all matching sinks."""

        for sub_id in list(self._subs):
            sub = self._registry.get(sub_id)
            if sub is None:
                continue
            if sub.predicate is not None and not sub.predicate(message):
                continue
            self._deliver(sub, message)

    def emit_event(self, event: Event) -> None:
        self.emit(event)

    def emit_delta(self, kind: str, text: str, turn_index: int = -1) -> None:
        self.emit(Delta(kind=kind, text=text, turn_index=turn_index))

    # ---------- delivery ----------

    def _deliver(self, sub: _Subscriber, message: Any) -> None:
        sink = sub.sink
        if isinstance(sink, asyncio.Queue):
            try:
                sink.put_nowait(message)
            except asyncio.QueueFull:
                sub.drops += 1
                self.total_drops += 1
            return
        try:
            result = sink(message)
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_swallow(result))
                except RuntimeError:
                    asyncio.run(_swallow(result))
        except Exception:  # noqa: BLE001 - a broken frontend must not kill the agent
            sub.drops += 1
            self.total_drops += 1


async def _swallow(awaitable: Any) -> None:
    try:
        await awaitable
    except Exception:  # noqa: BLE001
        pass
