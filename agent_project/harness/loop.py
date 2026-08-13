"""Continuation-based agent loop -- an explicit, serializable state machine.

The classic agent loop is a while-loop buried inside a god object: invisible,
unpausable, unforkable. Here the loop is a state machine whose entire state
lives in the journal:

* messages are **derived** -- rebuilt from ModelResponded / Effect* events,
* the only mutable cursor is ``turn_index``,
* pausing writes a continuation marker; resuming re-enters the same function.

That gives pause/resume/fork/time-travel with no extra state to corrupt,
and makes the loop testable with a scripted fake sampler.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .budget import Ledger
from .context import ContextAssembler
from .effects import Effect, make_effect
from .errors import (
    BudgetExhaustedError,
    ErrorRecord,
    StagnationError,
    classify_exception,
)
from .events import (
    BudgetConsumed,
    CircuitTripped,
    ContextAssembled,
    ContextCompacted,
    EffectCompleted,
    EffectDenied,
    EffectFailed,
    EffectRequested,
    Event,
    FinalAnswer,
    LoopPaused,
    ModelFailed,
    ModelRequested,
    ModelResponded,
    SessionFinished,
    SessionStarted,
    TurnFinished,
    TurnStarted,
    fold,
)
from .journal import Journal
from .kernel import Kernel
from .scheduler import Scheduler
from .stream import EventBus


# ---------- sampler protocol (model boundary) ----------

@dataclass(frozen=True)
class SampleResult:
    """Normalized model reply, backend-agnostic."""

    text: str
    tool_calls: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Sampler(Protocol):
    """Anything that can map messages -> SampleResult (async)."""

    async def __call__(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> SampleResult: ...


@dataclass(frozen=True)
class Continuation:
    """Serializable loop cursor. Messages are NOT stored -- they fold."""

    continuation_id: str
    turn_index: int
    task: str


# ---------- message derivation (the fold that feeds the model) ----------

def rebuild_messages(events: List[Event], system_prompt: str) -> List[Dict[str, Any]]:
    """Reconstruct the conversation sent to the model, purely from events."""

    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    pending: Dict[str, EffectRequested] = {}
    for ev in events:
        if isinstance(ev, SessionStarted):
            msgs.append({"role": "user", "content": ev.task})
        elif isinstance(ev, ModelResponded):
            if ev.text:
                msgs.append({"role": "assistant", "content": ev.text})
        elif isinstance(ev, EffectRequested):
            pending[ev.effect_id] = ev
        elif isinstance(ev, EffectCompleted):
            src = pending.pop(ev.effect_id, None)
            name = src.tool_name if src else "tool"
            msgs.append({"role": "user", "content": f"[obs:{name}] {ev.output}"})
        elif isinstance(ev, EffectFailed):
            src = pending.pop(ev.effect_id, None)
            name = src.tool_name if src else "tool"
            msgs.append({"role": "user", "content": f"[obs:{name}] ERROR {ev.error.message}"})
        elif isinstance(ev, EffectDenied):
            msgs.append(
                {"role": "user", "content": f"[obs:{ev.tool_name}] DENIED {ev.reason}"}
            )
    return msgs


# ---------- the loop ----------

@dataclass
class AgentLoop:
    """Drives one session to a terminal state.

    All side effects flow through the injected collaborators; the loop
    itself holds no hidden state, so two loops over the same journal
    observe the same world.
    """

    kernel: Kernel
    scheduler: Scheduler
    ledger: Ledger
    journal: Journal
    bus: EventBus
    sampler: Sampler
    system_prompt: str = "You are a capable agent. Use tools when useful."
    tools_schema: List[Dict[str, Any]] = field(default_factory=list)
    max_model_retries: int = 3
    model_retry_base_s: float = 0.5
    pause_check: Optional[Callable[[], bool]] = None
    context: Optional[ContextAssembler] = None
    verify_final: bool = True          # 最终答案 LLM 核验(准确)
    max_verification_rounds: int = 2   # 核验-修正轮数
    converge_on_stable: bool = True    # 连续稳定则提前收敛(高效)

    async def run(self, task: str) -> str:
        """Fresh run; returns the final answer text."""

        self._emit(SessionStarted(task=task))
        return await self._drive(task, start_turn=0)

    async def resume(self) -> str:
        """Resume after LoopPaused: fold the journal and keep driving."""

        events = self.journal.read_all()
        state = fold(events)
        if state.paused_continuation is None:
            raise RuntimeError("journal has no paused continuation to resume")
        task = state.task
        return await self._drive(task, start_turn=state.turn_index + 1)

    # ---------- internals ----------

    def _emit(self, event: Event) -> Event:
        """Journal durably, then publish to subscribers -- one atomic fact."""

        stamped = self.journal.append(event)
        self.bus.emit_event(stamped)
        return stamped

    async def _drive(self, task: str, start_turn: int) -> str:
        turn = start_turn
        try:
            while True:
                self.ledger._check()
                self._emit(TurnStarted(turn_index=turn))
                self.ledger.consume_turn()

                if self.pause_check and self.pause_check():
                    cid = uuid.uuid4().hex[:12]
                    self._emit(LoopPaused(continuation_id=cid))
                    return self._paused_result(cid)

                sample = await self._sample_with_retry(task, turn)

                if not sample.tool_calls:
                    final_text = sample.text
                    if self.verify_final and self.max_verification_rounds > 0:
                        final_text = await self._verify_and_finalize(task, sample.text)
                    self._emit(FinalAnswer(text=final_text))
                    self._emit(
                        TurnFinished(turn_index=turn, stop_reason="final_answer")
                    )
                    self._emit(
                        SessionFinished(status="completed", summary=final_text[:200])
                    )
                    return final_text

                await self._dispatch_all(sample.tool_calls, turn)
                self._emit_budget(turn)
                self._emit(TurnFinished(turn_index=turn, stop_reason="continued"))
                turn += 1
        except BudgetExhaustedError as exc:
            self._emit(
                SessionFinished(status="budget_exhausted", summary=str(exc)[:200])
            )
            raise

    # ----- model step -----

    async def _sample_with_retry(self, task: str, turn: int) -> SampleResult:
        events = self.journal.read_all()
        if self.context is not None:
            assembled = self.context.assemble(events, rebuild_messages)
            messages = assembled.messages
            self._emit(
                ContextAssembled(
                    n_messages=len(messages), approx_tokens=assembled.approx_tokens
                )
            )
            if assembled.compacted:
                self._emit(
                    ContextCompacted(
                        dropped_messages=assembled.dropped_messages,
                        kept_messages=len(messages),
                        summary=assembled.summary,
                    )
                )
        else:
            messages = rebuild_messages(events, self.system_prompt)
        call_id = uuid.uuid4().hex[:12]
        self._emit(ModelRequested(call_id=call_id, n_messages=len(messages)))
        attempt = 0
        while True:
            try:
                result = await self.sampler(messages, self.tools_schema)
            except Exception as exc:  # noqa: BLE001 - classified below
                record = classify_exception(exc)
                self._emit(ModelFailed(call_id=call_id, error=record))
                self.ledger.record_error()
                attempt += 1
                if record.retriable and attempt < self.max_model_retries:
                    await asyncio.sleep(self.model_retry_base_s * (2 ** (attempt - 1)))
                    continue
                self._finish_with_failure(record, turn)
                raise
            else:
                self.ledger.record_success()
                self._emit(
                    ModelResponded(
                        call_id=call_id,
                        text=result.text,
                        reasoning=result.reasoning,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    )
                )
                self.ledger.consume_tokens(result.prompt_tokens + result.completion_tokens)
                self.bus.emit_delta("token", result.text, turn_index=turn)
                return result

    # ----- tool step -----

    async def _dispatch_all(
        self, tool_calls: Tuple[Tuple[str, Dict[str, Any]], ...], turn: int
    ) -> None:
        admitted: List[Tuple[Effect, str]] = []
        for name, args in tool_calls:
            effect = make_effect(name, args)
            admission = self.kernel.evaluate(effect)
            if admission.decision.value == "deny":
                self._emit(
                    EffectDenied(
                        effect_id=effect.idempotency_key,
                        tool_name=name,
                        reason=admission.reason,
                    )
                )
                continue
            self._emit(
                EffectRequested(
                    effect_id=effect.idempotency_key,
                    tool_name=name,
                    arguments=args,
                    idempotency_key=effect.idempotency_key,
                    lane=admission.lane,
                )
            )
            admitted.append((effect, admission.lane))

        outcomes = await self.scheduler.submit_many(admitted, self.kernel.executor)
        for outcome in outcomes:
            self.ledger.consume_tool_call()
            try:
                self.ledger.record_effect(outcome.effect.idempotency_key)
            except StagnationError as exc:
                self._emit(CircuitTripped(breaker="stagnation", reason=str(exc)))
                self._emit(SessionFinished(status="failed", summary="stagnation"))
                raise
            if outcome.ok:
                self.ledger.record_success()
                self._emit(
                    EffectCompleted(
                        effect_id=outcome.effect.idempotency_key,
                        output=outcome.output,
                        latency_ms=outcome.latency_ms,
                        from_cache=outcome.from_cache,
                    )
                )
            else:
                record = classify_exception(outcome.error or RuntimeError("unknown"))
                self._emit(
                    EffectFailed(effect_id=outcome.effect.idempotency_key, error=record)
                )
                try:
                    self.ledger.record_error()
                except StagnationError as exc:
                    self._emit(
                        CircuitTripped(breaker="consecutive_errors", reason=str(exc))
                    )
                    self._emit(SessionFinished(status="failed", summary="error storm"))
                    raise

    # ----- verification (accuracy) -----

    async def _verify_and_finalize(self, task: str, answer: str) -> str:
        """最终答案 LLM 核验: 基于工具证据复核, 错误则修正, 稳定则提前收敛."""
        events = self.journal.read_all()
        tool_names: Dict[str, str] = {}
        for ev in events:
            if isinstance(ev, EffectRequested):
                tool_names[ev.effect_id] = ev.tool_name
        tool_evidence = [
            ev for ev in events
            if isinstance(ev, (EffectCompleted, EffectFailed))
        ][-6:]
        if not tool_evidence:
            return answer  # 纯对话无工具证据, 无需核验

        obs_lines = []
        for ev in tool_evidence:
            content = getattr(ev, "output", None) or str(getattr(ev, "error", ""))
            name = tool_names.get(ev.effect_id, "tool")
            obs_lines.append(f"[{name}] {str(content)[:300]}")
        obs_text = "\n".join(obs_lines)

        current = answer
        for _r in range(self.max_verification_rounds):
            msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
                {"role": "assistant", "content": current},
                {"role": "user", "content": (
                    "你是一名严格的质检员。请核验候选答案是否与工具证据一致、完整、无编造。\n"
                    f"任务: {task}\n\n工具证据:\n{obs_text or '(无)'}\n\n候选答案:\n{current}\n\n"
                    "若正确: 输出 'VERIFIED: <答案>'\n"
                    "若错误或不完整: 输出 'REVISED: <修正后的完整答案>'\n"
                    "只输出以上格式。"
                )},
            ]
            call_id = uuid.uuid4().hex[:12]
            self._emit(ModelRequested(call_id=call_id, n_messages=len(msgs)))
            try:
                result = await self.sampler(msgs, self.tools_schema)
            except Exception as exc:  # noqa: BLE001 - verification is best-effort
                self._emit(ModelFailed(call_id=call_id, error=classify_exception(exc)))
                break
            self.ledger.record_success()
            self._emit(
                ModelResponded(
                    call_id=call_id,
                    text=result.text,
                    reasoning=result.reasoning,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
            )
            self.ledger.consume_tokens(result.prompt_tokens + result.completion_tokens)

            text = (result.text or "").strip()
            lowered = text.lower()
            if lowered.startswith(("revised", "revise")):
                revised = text.split(":", 1)[1].strip() if ":" in text else text
                if not revised:
                    break
                if self.converge_on_stable and revised == current:
                    return revised
                current = revised
                continue
            if lowered.startswith("verified"):
                final = text.split(":", 1)[1].strip() if ":" in text else current
                return final or current
            return text or current
        return current

    # ----- housekeeping -----

    def _emit_budget(self, turn: int) -> None:
        snap = self.ledger.snapshot()
        self._emit(
            BudgetConsumed(
                tokens_total=snap["tokens"],
                tool_calls_total=snap["tool_calls"],
                elapsed_s=snap["elapsed_s"],
            )
        )

    def _finish_with_failure(self, record: ErrorRecord, turn: int) -> None:
        self._emit(TurnFinished(turn_index=turn, stop_reason="error"))
        self._emit(SessionFinished(status="failed", summary=record.message[:200]))

    @staticmethod
    def _paused_result(continuation_id: str) -> str:
        return f"[paused: continuation={continuation_id}]"
