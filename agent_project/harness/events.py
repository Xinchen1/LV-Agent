"""
Immutable event taxonomy -- the single source of truth for session state.

Design rule: **every state change in a session is an event**. The session is
nothing more than a left fold over its event stream, which gives us replay,
fork, resume-after-crash and auditability for free.

Events are frozen pydantic models. They carry:

* ``seq``  -- monotonically increasing sequence number *within a session*
  (assigned by the journal on append; -1 while in-flight).
* ``ts``   -- wall-clock seconds (informational only; ordering is by seq).
* ``kind`` -- discriminator for fold/replay, derived from the class name.

New event types must remain backward compatible: old journals must always
replay. Add fields with defaults; never remove or rename.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .errors import ErrorRecord


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Event(BaseModel):
    """Base of every journal entry."""

    model_config = {"frozen": True}

    seq: int = Field(default=-1, description="Journal-assigned sequence number")
    ts: float = Field(default_factory=time.time)
    event_id: str = Field(default_factory=_new_id)

    @property
    def kind(self) -> str:
        return type(self).__name__

    def with_seq(self, seq: int) -> "Event":
        return self.model_copy(update={"seq": seq})


# ---------- session lifecycle ----------

class SessionStarted(Event):
    task: str
    parent_event_seq: Optional[int] = None  # set when this session is a fork
    config_digest: Dict[str, Any] = Field(default_factory=dict)


class SessionFinished(Event):
    status: Literal["completed", "failed", "cancelled", "budget_exhausted"]
    summary: str = ""


# ---------- model interaction ----------

class ModelRequested(Event):
    """A sampling call was dispatched to the model backend."""

    call_id: str
    n_messages: int
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class ModelResponded(Event):
    """The model produced a full (non-streamed) response."""

    call_id: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    reasoning: str = ""  # chain-of-thought channel, if the backend exposes one


class ModelFailed(Event):
    call_id: str
    error: ErrorRecord


# ---------- tool interaction (effect lifecycle) ----------

class EffectRequested(Event):
    """An effect intent was admitted by the kernel (post policy check)."""

    effect_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    lane: str = "default"  # scheduler lane; writes are path-scoped


class EffectCompleted(Event):
    effect_id: str
    output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    from_cache: bool = False


class EffectFailed(Event):
    effect_id: str
    error: ErrorRecord


class EffectDenied(Event):
    """Policy engine refused the effect before execution."""

    effect_id: str
    tool_name: str
    reason: str


# ---------- context management ----------

class ContextAssembled(Event):
    n_messages: int
    approx_tokens: int


class ContextCompacted(Event):
    """Older turns were summarized/dropped to fit the window."""

    dropped_messages: int
    kept_messages: int
    summary: str = ""


# ---------- budget ----------

class BudgetConsumed(Event):
    """Point-in-time ledger snapshot after consumption."""

    tokens_total: int
    tool_calls_total: int
    elapsed_s: float


class CircuitTripped(Event):
    breaker: str  # e.g. "consecutive_errors", "stagnation"
    reason: str


# ---------- loop control (continuations) ----------

class TurnStarted(Event):
    turn_index: int


class TurnFinished(Event):
    turn_index: int
    stop_reason: Literal[
        "final_answer", "budget", "circuit", "paused", "error", "max_turns", "continued"
    ]


class LoopPaused(Event):
    """The continuation was serialized; the loop can resume later."""

    continuation_id: str


class CheckpointWritten(Event):
    """Workspace snapshot reference for undo/rewind."""

    checkpoint_id: str
    description: str = ""


class FinalAnswer(Event):
    text: str


# ---------- fold ----------

#: Events that mutate the logical message list during a fold.
_MESSAGE_EVENTS = (ModelResponded, EffectCompleted, EffectFailed, EffectDenied)


class SessionState(BaseModel):
    """The logical state obtained by folding an event stream.

    Pure data: no I/O, no clocks. Rebuilding a session after a crash is
    ``state = fold(journal.read())`` and nothing else.
    """

    task: str = ""
    finished: bool = False
    finish_status: Optional[str] = None
    final_text: Optional[str] = None
    turn_index: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    denials: int = 0
    errors: List[ErrorRecord] = Field(default_factory=list)
    breakers: List[str] = Field(default_factory=list)
    checkpoints: List[str] = Field(default_factory=list)
    paused_continuation: Optional[str] = None


def fold(events: List[Event], state: Optional[SessionState] = None) -> SessionState:
    """Left-fold events into :class:`SessionState`. Pure function."""

    s = state or SessionState()
    for ev in events:
        s = _apply(s, ev)
    return s


def _apply(s: SessionState, ev: Event) -> SessionState:
    upd: Dict[str, Any] = {}
    if isinstance(ev, SessionStarted):
        upd["task"] = ev.task
        upd["finished"] = False
    elif isinstance(ev, TurnStarted):
        upd["turn_index"] = ev.turn_index
    elif isinstance(ev, ModelResponded):
        upd["model_calls"] = s.model_calls + 1
        upd["total_tokens"] = s.total_tokens + ev.prompt_tokens + ev.completion_tokens
    elif isinstance(ev, ModelFailed):
        upd["errors"] = [*s.errors, ev.error]
    elif isinstance(ev, EffectCompleted):
        upd["tool_calls"] = s.tool_calls + (0 if ev.from_cache else 1)
    elif isinstance(ev, EffectFailed):
        upd["tool_calls"] = s.tool_calls + 1
        upd["errors"] = [*s.errors, ev.error]
    elif isinstance(ev, EffectDenied):
        upd["denials"] = s.denials + 1
    elif isinstance(ev, CircuitTripped):
        upd["breakers"] = [*s.breakers, ev.breaker]
    elif isinstance(ev, CheckpointWritten):
        upd["checkpoints"] = [*s.checkpoints, ev.checkpoint_id]
    elif isinstance(ev, LoopPaused):
        upd["paused_continuation"] = ev.continuation_id
    elif isinstance(ev, FinalAnswer):
        upd["final_text"] = ev.text
    elif isinstance(ev, SessionFinished):
        upd["finished"] = True
        upd["finish_status"] = ev.status
    if not upd:
        return s
    return s.model_copy(update=upd)


_EVENT_TYPES = (
    SessionStarted,
    SessionFinished,
    ModelRequested,
    ModelResponded,
    ModelFailed,
    EffectRequested,
    EffectCompleted,
    EffectFailed,
    EffectDenied,
    ContextAssembled,
    ContextCompacted,
    BudgetConsumed,
    CircuitTripped,
    TurnStarted,
    TurnFinished,
    LoopPaused,
    CheckpointWritten,
    FinalAnswer,
)

#: kind-string -> class, for journal deserialization.
EVENT_REGISTRY: Dict[str, type] = {cls.__name__: cls for cls in _EVENT_TYPES}

__all__ = [
    "Event",
    "SessionState",
    "fold",
    "EVENT_REGISTRY",
    *[cls.__name__ for cls in _EVENT_TYPES],
]
