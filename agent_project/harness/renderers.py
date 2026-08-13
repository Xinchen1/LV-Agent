"""Renderers -- adapt the typed bus to existing frontends, unchanged.

The legacy CLI/Telegram frontends consume ``stream_callback(kind, text)``
with kinds like status/reasoning/tool_call/tool_result/content. The harness
emits typed Events and Deltas instead. This adapter is the translation
layer: attach it to a session's bus and any legacy frontend renders a
harness run without modification.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .events import (
    CircuitTripped,
    ContextCompacted,
    EffectCompleted,
    EffectDenied,
    EffectFailed,
    EffectRequested,
    Event,
    ModelResponded,
)
from .stream import Delta, EventBus

_PREVIEW = 240


class StreamCallbackAdapter:
    """Convert bus messages into legacy ``callback(kind, text)`` calls."""

    def __init__(
        self,
        callback: Callable[[str, str], Any],
        token_callback: Optional[Callable[[int], Any]] = None,
    ):
        self.callback = callback
        self.token_callback = token_callback

    def attach(self, bus: EventBus) -> Callable[[], None]:
        return bus.subscribe(self)

    def __call__(self, message: Any) -> None:
        if isinstance(message, Delta):
            self._on_delta(message)
        elif isinstance(message, Event):
            self._on_event(message)

    # ---------- internals ----------

    def _on_delta(self, delta: Delta) -> None:
        kind = {"token": "content", "reasoning": "reasoning", "status": "status"}.get(
            delta.kind
        )
        if kind:
            self.callback(kind, delta.text)

    def _on_event(self, event: Event) -> None:
        if isinstance(event, EffectRequested):
            args = str(event.arguments)
            if len(args) > _PREVIEW:
                args = args[:_PREVIEW] + "…"
            self.callback("tool_call", f"{event.tool_name} {args}")
        elif isinstance(event, EffectCompleted):
            self.callback("tool_result", event.output[:_PREVIEW])
        elif isinstance(event, EffectFailed):
            self.callback("tool_result", f"error: {event.error.message}")
        elif isinstance(event, EffectDenied):
            self.callback("tool_result", f"denied: {event.reason}")
        elif isinstance(event, ContextCompacted):
            self.callback(
                "status", f"context compacted ({event.dropped_messages} messages folded)"
            )
        elif isinstance(event, CircuitTripped):
            self.callback("status", f"circuit breaker: {event.breaker}")
        elif isinstance(event, ModelResponded) and self.token_callback:
            total = event.prompt_tokens + event.completion_tokens
            if total:
                self.token_callback(total)


class ConsoleRenderer:
    """Minimal headless renderer: prints a compact trace to stdout."""

    def attach(self, bus: EventBus) -> Callable[[], None]:
        return bus.subscribe(self)

    def __call__(self, message: Any) -> None:
        if isinstance(message, Delta):
            if message.kind == "token":
                print(message.text)
            elif message.kind == "status":
                print(f"\033[2m· {message.text}\033[0m")
        elif isinstance(message, EffectRequested):
            print(f"\033[34m→ {message.tool_name}\033[0m")
        elif isinstance(message, EffectCompleted):
            print(f"\033[32m✓ {message.output[:_PREVIEW]}\033[0m")
        elif isinstance(message, (EffectFailed, EffectDenied)):
            print(f"\033[31m✗ {getattr(message, 'reason', None) or message.error.message}\033[0m")
