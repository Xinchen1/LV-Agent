"""Bridge -- run the *existing* OpenMythos stack on the harness, unchanged.

Two adapters, no rewrites:

* :class:`BackendSampler` wraps any legacy backend object exposing
  ``generate(prompt, ...) -> str`` and turns it into the harness
  :class:`~agent_project.harness.loop.Sampler` protocol, parsing the
  established ``[TOOL:name] {json} [/TOOL]`` text convention.
* :func:`kernel_guarded_executor` lets the legacy agent route its tool
  execution through the capability kernel (policy + audit) by swapping one
  callable -- an opt-in, reversible change.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .effects import make_effect
from .kernel import Executor, Kernel
from .loop import SampleResult

_TOOL_BLOCK = re.compile(
    r"\[TOOL:\s*(?P<name>[A-Za-z0-9_\-.]+)\s*\]\s*(?P<body>.*?)\s*\[/TOOL\]",
    re.DOTALL,
)


def parse_tool_blocks(text: str) -> Tuple[str, Tuple[Tuple[str, Dict[str, Any]], ...]]:
    """Split model output into (clean_text, tool_calls).

    Recognizes ``[TOOL:name] {...} [/TOOL]`` with JSON or key=value bodies;
    unparseable bodies stay in the text so the model can be re-prompted.
    """

    calls: List[Tuple[str, Dict[str, Any]]] = []
    spans: List[Tuple[int, int]] = []
    for match in _TOOL_BLOCK.finditer(text):
        args = _parse_body(match.group("body"))
        if args is None:
            continue
        calls.append((match.group("name"), args))
        spans.append(match.span())
    clean = text
    for start, end in reversed(spans):
        clean = clean[:start] + clean[end:]
    return clean.strip(), tuple(calls)


def _parse_body(body: str) -> Optional[Dict[str, Any]]:
    body = body.strip()
    if not body:
        return {}
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    kv: Dict[str, Any] = {}
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', body):
        kv[m.group(1)] = m.group(2)
    return kv or None


class BackendSampler:
    """Adapt a legacy ``.generate()`` backend to the harness Sampler."""

    def __init__(
        self,
        backend: Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_tokens: Optional[Callable[[int], None]] = None,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.on_tokens = on_tokens

    async def __call__(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> SampleResult:
        prompt = self._flatten(messages)
        usage = {"total": 0}

        def _token_cb(n: int) -> None:
            usage["total"] = n
            if self.on_tokens:
                self.on_tokens(n)

        text = await asyncio.to_thread(
            lambda: self.backend.generate(
                prompt,
                n_loops=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                token_callback=_token_cb,
            )
        )
        clean, calls = parse_tool_blocks(text or "")
        return SampleResult(
            text=clean,
            tool_calls=calls,
            prompt_tokens=usage["total"],
            completion_tokens=0,
        )

    @staticmethod
    def _flatten(messages: List[Dict[str, Any]]) -> str:
        """Legacy backends take one prompt string; fold roles into text."""

        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[previous answer]\n{content}")
            else:
                parts.append(str(content))
        return "\n\n".join(p for p in parts if p)


def kernel_guarded_executor(kernel: Kernel) -> Callable[[str, Dict[str, Any]], str]:
    """A drop-in ``(tool_name, arguments) -> output`` for the legacy agent.

    Every call becomes an Effect, passes policy, and is audited by the
    kernel. Denials raise PolicyDeniedError with the rule's reason.
    """

    def _run(tool_name: str, arguments: Dict[str, Any]) -> str:
        return kernel.run(make_effect(tool_name, arguments))

    return _run


def build_kernel_from_registry(registry: Any, **kernel_kwargs: Any) -> Kernel:
    """Convenience: a Kernel whose executor is the legacy TOOLS_REGISTRY."""

    from .kernel import registry_executor

    kernel_kwargs.setdefault("executor", registry_executor(registry))
    return Kernel(**kernel_kwargs)
