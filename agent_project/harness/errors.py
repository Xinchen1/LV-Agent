"""
Typed error taxonomy for the harness.

Every failure inside the runtime is classified along two axes:

* ``kind``    -- what went wrong (model, tool, policy, budget, ...).
* ``retriable`` -- whether retrying the *same* operation could succeed.

The loop, the scheduler and frontends all branch on these two fields only,
which keeps error policy in exactly one place instead of smearing
``except Exception`` ladders across the codebase.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorKind(str, enum.Enum):
    """Coarse classification of runtime failures."""

    MODEL_TRANSIENT = "model_transient"      # 429/5xx, timeouts, resets
    MODEL_FATAL = "model_fatal"              # 401/403, invalid request
    MODEL_EXHAUSTED = "model_exhausted"      # quota / context overflow
    TOOL_FAILED = "tool_failed"              # tool ran and reported failure
    TOOL_CRASHED = "tool_crashed"            # tool raised unexpectedly
    TOOL_TIMEOUT = "tool_timeout"
    POLICY_DENIED = "policy_denied"          # capability kernel refused
    BUDGET_EXHAUSTED = "budget_exhausted"    # ledger hit a hard limit
    STAGNATION = "stagnation"                # loop detector fired
    CANCELLED = "cancelled"                  # caller asked to stop
    INTERNAL = "internal"                    # harness bug; never retriable


class HarnessError(Exception):
    """Base exception carrying a machine-readable classification."""

    kind: ErrorKind = ErrorKind.INTERNAL
    retriable: bool = False

    def __init__(self, message: str, *, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.detail: Dict[str, Any] = detail or {}

    def to_record(self) -> "ErrorRecord":
        return ErrorRecord(
            kind=self.kind,
            retriable=self.retriable,
            message=self.message,
            detail=self.detail,
        )


class ErrorRecord(BaseModel):
    """Serializable view of a failure; safe to persist in the journal."""

    model_config = {"frozen": True}

    kind: ErrorKind
    retriable: bool
    message: str
    detail: Dict[str, Any] = Field(default_factory=dict)


# ---------- concrete errors ----------

class ModelTransientError(HarnessError):
    kind = ErrorKind.MODEL_TRANSIENT
    retriable = True


class ModelFatalError(HarnessError):
    kind = ErrorKind.MODEL_FATAL
    retriable = False


class ModelExhaustedError(HarnessError):
    kind = ErrorKind.MODEL_EXHAUSTED
    retriable = False


class ToolFailedError(HarnessError):
    kind = ErrorKind.TOOL_FAILED
    retriable = False


class ToolCrashedError(HarnessError):
    kind = ErrorKind.TOOL_CRASHED
    retriable = True  # crash may be input-independent; one retry is sane


class ToolTimeoutError(HarnessError):
    kind = ErrorKind.TOOL_TIMEOUT
    retriable = True


class PolicyDeniedError(HarnessError):
    kind = ErrorKind.POLICY_DENIED
    retriable = False


class BudgetExhaustedError(HarnessError):
    kind = ErrorKind.BUDGET_EXHAUSTED
    retriable = False


class StagnationError(HarnessError):
    kind = ErrorKind.STAGNATION
    retriable = False


class CancelledError(HarnessError):
    kind = ErrorKind.CANCELLED
    retriable = False


def classify_exception(exc: BaseException) -> ErrorRecord:
    """Map an arbitrary exception to an :class:`ErrorRecord`.

    Already-classified errors pass through; everything else becomes
    ``INTERNAL``/non-retriable so unknown failures stop the loop loudly
    instead of being silently retried forever.
    """

    if isinstance(exc, HarnessError):
        return exc.to_record()
    if isinstance(exc, TimeoutError):
        return ErrorRecord(
            kind=ErrorKind.TOOL_TIMEOUT, retriable=True, message=str(exc) or "timeout"
        )
    if isinstance(exc, (KeyboardInterrupt, asyncio_cancelled())):
        return ErrorRecord(
            kind=ErrorKind.CANCELLED, retriable=False, message=str(exc) or "cancelled"
        )
    return ErrorRecord(
        kind=ErrorKind.INTERNAL,
        retriable=False,
        message=f"{type(exc).__name__}: {exc}",
    )


def asyncio_cancelled() -> type[BaseException]:
    """Return the asyncio CancelledError type without importing at module top.

    Kept behind a function so this module stays importable in contexts where
    asyncio policy tinkering happens later (tests, embedded interpreters).
    """

    import asyncio

    return asyncio.CancelledError
