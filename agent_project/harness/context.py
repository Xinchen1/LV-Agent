"""Context assembly and compaction -- the window is a budgeted resource.

The assembler turns journaled events into the model's message list under a
token budget. When the derived conversation exceeds the budget, a compactor
rewrites the *middle* of the conversation (preserving system prompt, the
original task, and the most recent turns) and reports what it dropped.

Compaction is event-sourced too: the loop appends ContextAssembled and,
when rewriting happened, ContextCompacted, so later analysis can see
exactly when the agent forgot what -- and a fold can reconstruct the
context the model actually saw at any turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .events import Event


def approx_tokens(text: str) -> int:
    """Cheap heuristic (~4 chars/token) matching model_backends' estimator."""

    return max(1, len(text) // 4) if text else 0


def messages_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(approx_tokens(str(m.get("content", ""))) for m in messages)


class Compactor(Protocol):
    """Rewrites an over-budget message list; returns (messages, summary)."""

    def __call__(
        self, messages: List[Dict[str, Any]], budget_tokens: int
    ) -> Tuple[List[Dict[str, Any]], str]: ...


@dataclass(frozen=True)
class AssembledContext:
    messages: List[Dict[str, Any]]
    approx_tokens: int
    compacted: bool
    dropped_messages: int = 0
    summary: str = ""


@dataclass
class TruncatingCompactor:
    """Keep head + tail, fold the middle into a deterministic stub.

    The stub lists the tool names seen in the dropped span so the model
    keeps a factual trace of what it already did (prevents repeat loops)
    without paying for full transcripts.
    """

    keep_tail: int = 6

    def __call__(
        self, messages: List[Dict[str, Any]], budget_tokens: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        if len(messages) <= self.keep_tail + 2:
            return messages, ""
        head = messages[:2]  # system + original task
        tail = messages[-self.keep_tail :]
        middle = messages[2 : -self.keep_tail]

        tools_seen: List[str] = []
        for msg in middle:
            content = str(msg.get("content", ""))
            if content.startswith("[obs:"):
                name = content[5 : content.find("]")].strip()
                if name and name not in tools_seen:
                    tools_seen.append(name)
        summary = (
            f"[compacted {len(middle)} earlier messages"
            + (f"; tools already used: {', '.join(tools_seen)}" if tools_seen else "")
            + "]"
        )
        compacted = [*head, {"role": "user", "content": summary}, *tail]
        if messages_tokens(compacted) <= budget_tokens:
            return compacted, summary
        # Still over budget: shrink the tail until it fits.
        while len(tail) > 1 and messages_tokens([*head, *tail]) > budget_tokens:
            tail = tail[2:]
        return [*head, {"role": "user", "content": summary}, *tail], summary


@dataclass
class LLMCompactor:
    """Summarize the dropped middle with an LLM; fall back deterministically.

    ``summarize`` is any sync callable transcript -> summary (a backend
    generate(), a local model, a rule). Any failure -- or a summary that
    still overflows the budget -- degrades to :class:`TruncatingCompactor`,
    so compaction is never the thing that kills the run.
    """

    summarize: Callable[[str], str]
    keep_tail: int = 6
    max_summary_chars: int = 1200
    fallback: TruncatingCompactor = field(default_factory=TruncatingCompactor)

    _PROMPT = (
        "Summarize this agent transcript in <=6 bullet points. Preserve: "
        "files touched, commands run, key findings, and open sub-tasks. "
        "Be terse; no preamble.\n\n"
    )

    def __call__(
        self, messages: List[Dict[str, Any]], budget_tokens: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        if len(messages) <= self.keep_tail + 2:
            return messages, ""
        head = messages[:2]
        tail = messages[-self.keep_tail :]
        middle = messages[2 : -self.keep_tail]

        transcript = "\n".join(
            f"{m.get('role', '?')}: {str(m.get('content', ''))[:500]}" for m in middle
        )
        try:
            summary_text = self.summarize(self._PROMPT + transcript).strip()
        except Exception:  # noqa: BLE001 - summarizer failure must not kill the run
            return self.fallback(messages, budget_tokens)
        if not summary_text:
            return self.fallback(messages, budget_tokens)

        stub = {"role": "user", "content": f"[summary of earlier work]\n{summary_text[: self.max_summary_chars]}"}
        compacted = [*head, stub, *tail]
        if messages_tokens(compacted) > budget_tokens:
            return self.fallback(messages, budget_tokens)
        return compacted, summary_text


@dataclass
class ContextAssembler:
    """Derives the model-visible context under a hard token budget."""

    system_prompt: str
    budget_tokens: Optional[int] = None
    compactor: Optional[Compactor] = None

    def assemble(
        self, events: List[Event], rebuild: Any
    ) -> AssembledContext:
        """``rebuild`` is loop.rebuild_messages (injected to avoid a cycle)."""

        messages = rebuild(events, self.system_prompt)
        total = messages_tokens(messages)
        if self.budget_tokens is None or total <= self.budget_tokens:
            return AssembledContext(messages, total, compacted=False)

        compactor = self.compactor or TruncatingCompactor()
        rewritten, summary = compactor(messages, self.budget_tokens)
        dropped = len(messages) - len(rewritten)
        return AssembledContext(
            messages=rewritten,
            approx_tokens=messages_tokens(rewritten),
            compacted=True,
            dropped_messages=max(0, dropped),
            summary=summary,
        )
