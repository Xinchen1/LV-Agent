"""Budget ledger and circuit breakers -- resources as first-class citizens.

A run consumes four scarce resources: tokens, wall-clock time, tool calls
and (optionally) dollars. The ledger tracks all four against hard limits
and raises :class:`BudgetExhaustedError` the moment one is crossed, so a
runaway loop dies *deterministically* instead of after a surprise bill.

Circuit breakers are orthogonal: they watch *failure dynamics* (consecutive
errors, identical repeated effects = stagnation) and trip the loop even when
budget remains. Budgets bound quantity; breakers bound futility.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from .errors import BudgetExhaustedError, StagnationError


@dataclass(frozen=True)
class Limits:
    """Hard caps for a single run. ``None`` disables that dimension."""

    max_tokens: Optional[int] = None
    max_seconds: Optional[float] = None
    max_tool_calls: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_turns: Optional[int] = None

    #: consecutive tool/model errors before the breaker trips
    max_consecutive_errors: int = 5
    #: same idempotency key seen this many times -> stagnation
    max_identical_effects: int = 3


@dataclass
class Ledger:
    """Mutable per-run consumption record; one per session run."""

    limits: Limits = field(default_factory=Limits)
    started_at: float = field(default_factory=time.monotonic)
    tokens: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    consecutive_errors: int = 0
    _recent_effects: Deque[str] = field(default_factory=lambda: deque(maxlen=32))

    # ---------- consumption ----------

    def consume_tokens(self, n: int) -> None:
        self.tokens += n
        self._check()

    def consume_tool_call(self) -> None:
        self.tool_calls += 1
        self._check()

    def consume_cost(self, usd: float) -> None:
        self.cost_usd += usd
        self._check()

    def consume_turn(self) -> None:
        self.turns += 1
        self._check()

    # ---------- failure dynamics ----------

    def record_success(self) -> None:
        self.consecutive_errors = 0

    def record_error(self) -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.limits.max_consecutive_errors:
            raise StagnationError(
                f"circuit breaker: {self.consecutive_errors} consecutive errors",
                detail={"breaker": "consecutive_errors"},
            )

    def record_effect(self, idempotency_key: str) -> None:
        """Detect a loop re-issuing the identical effect without progress."""

        self._recent_effects.append(idempotency_key)
        window = list(self._recent_effects)[-self.limits.max_identical_effects :]
        if len(window) == self.limits.max_identical_effects and len(set(window)) == 1:
            raise StagnationError(
                f"circuit breaker: identical effect repeated {len(window)}x "
                f"(key={idempotency_key})",
                detail={"breaker": "stagnation", "key": idempotency_key},
            )

    # ---------- introspection ----------

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_tokens(self) -> Optional[int]:
        if self.limits.max_tokens is None:
            return None
        return max(0, self.limits.max_tokens - self.tokens)

    def snapshot(self) -> dict:
        return {
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "cost_usd": round(self.cost_usd, 6),
            "turns": self.turns,
            "elapsed_s": round(self.elapsed, 3),
            "consecutive_errors": self.consecutive_errors,
        }

    # ---------- internals ----------

    def _check(self) -> None:
        lim = self.limits
        checks: Tuple[Tuple[bool, str], ...] = (
            (lim.max_tokens is not None and self.tokens > lim.max_tokens,
             f"token budget exhausted ({self.tokens}/{lim.max_tokens})"),
            (lim.max_seconds is not None and self.elapsed > lim.max_seconds,
             f"time budget exhausted ({self.elapsed:.1f}s/{lim.max_seconds}s)"),
            (lim.max_tool_calls is not None and self.tool_calls > lim.max_tool_calls,
             f"tool-call budget exhausted ({self.tool_calls}/{lim.max_tool_calls})"),
            (lim.max_cost_usd is not None and self.cost_usd > lim.max_cost_usd,
             f"cost budget exhausted (${self.cost_usd:.4f}/${lim.max_cost_usd})"),
            (lim.max_turns is not None and self.turns > lim.max_turns,
             f"turn budget exhausted ({self.turns}/{lim.max_turns})"),
        )
        for hit, msg in checks:
            if hit:
                raise BudgetExhaustedError(msg, detail=self.snapshot())
