"""Legacy-compatible session runner.

One function, :func:`run_task`, that any existing entry point (super_agent
CLI, telegram bot, scripts) can call exactly like ``agent.run`` -- same
callbacks, compatible result dict -- while execution actually happens on
the event-sourced harness: policy kernel, budget ledger, lanes, journal.

The journal persists under ``data/harness_sessions/`` so every interactive
run is inspectable and resumable after the fact.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .bridge import BackendSampler
from .budget import Limits
from .kernel import Kernel, permissive_policy, registry_executor, safe_default_policy
from .renderers import StreamCallbackAdapter
from .session import Session


def _build_tool_prompt(registry: Any) -> str:
    """工具感知系统提示: 告诉模型有哪些工具、如何用 [TOOL:...] 调用.

    这是 harness 模式"真正有效"的关键 —— 之前模型根本不知道可用工具。
    """
    try:
        tools = registry.get_tools_dict()  # name -> description
    except Exception:
        tools = {}
    lines = [
        "You are Lv Super Agent, a capable assistant with tools.",
        "",
        "When a task needs external data, files, code execution or computation, "
        "call a tool. Use EXACTLY this format (each block on its own):",
        "  [TOOL:tool_name]",
        "  {json arguments}",
        "  [/TOOL]",
        "Wait for the tool result before continuing. "
        "You may call several tools across turns. "
        "When you have the final answer, output it plainly (no tool blocks).",
        "",
        "Available tools:",
    ]
    if tools:
        for name in sorted(tools):
            desc = (tools[name] or "").strip().split("\n")[0][:160]
            lines.append(f"- {name}: {desc}")
    else:
        lines.append("- (no tools registered)")
    lines.append("")
    lines.append("Be precise and grounded: state numbers/data only when a tool or prompt confirms them.")
    return "\n".join(lines)


def _tools_schema(registry: Any) -> list:
    """构造工具 schema(供未来函数调用型后端使用)."""
    try:
        tools = registry.get_tools_dict()
    except Exception:
        tools = {}
    return [{"name": n, "description": (d or "")[:200]} for n, d in tools.items()]


def make_legacy_runner(agent: Any, config: Any) -> Callable[..., Dict[str, Any]]:
    """Build an ``agent.run``-compatible callable backed by the harness."""

    harness_cfg = getattr(config, "harness", None)

    def runner(
        task: str,
        code_mode: bool = False,
        stream_callback: Optional[Callable] = None,
        token_callback: Optional[Callable] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        return run_task(
            task,
            agent.backend,
            harness_cfg=harness_cfg,
            stream_callback=stream_callback,
            token_callback=token_callback,
        )

    return runner


def run_task(
    task: str,
    backend: Any,
    *,
    harness_cfg: Optional[Any] = None,
    stream_callback: Optional[Callable] = None,
    token_callback: Optional[Callable] = None,
    limits: Optional[Limits] = None,
    context_budget_tokens: Optional[int] = None,
    session_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run *task* on the harness; return an ``agent.run``-shaped result."""

    from ..tools import TOOLS_REGISTRY

    hc = harness_cfg
    policy_name = getattr(hc, "policy", "safe") if hc else "safe"
    if policy_name == "permissive":
        policy = permissive_policy()
    else:
        root = (
            getattr(hc, "workspace_root", None)
            or str(Path(__file__).resolve().parents[2])
        )
        policy = safe_default_policy(root)

    kernel = Kernel(policy=policy, executor=registry_executor(TOOLS_REGISTRY))
    sampler = BackendSampler(backend, on_tokens=token_callback)
    root = session_root or (Path(__file__).resolve().parents[2] / "data" / "harness_sessions")

    # 预算与核验参数: 优先从 config.harness 读取
    if limits is None:
        limits = Limits(
            max_turns=getattr(hc, "max_turns", 12) if hc else 12,
            max_seconds=getattr(hc, "max_seconds", 600.0) if hc else 600.0,
            max_tokens=getattr(hc, "max_tokens", None) if hc else None,
            max_tool_calls=getattr(hc, "max_tool_calls", None) if hc else None,
        )

    system_prompt = _build_tool_prompt(TOOLS_REGISTRY)

    session = Session(
        root=root,
        sampler=sampler,
        kernel=kernel,
        limits=limits,
        system_prompt=system_prompt,
        tools_schema=_tools_schema(TOOLS_REGISTRY),
        context_budget_tokens=context_budget_tokens,
        verify_final=getattr(hc, "verify_final_answer", True) if hc else True,
        max_verification_rounds=getattr(hc, "max_verification_rounds", 2) if hc else 2,
        converge_on_stable=getattr(hc, "converge_on_stable", True) if hc else True,
        max_model_retries=getattr(hc, "max_model_retries", 3) if hc else 3,
    )
    detach = None
    if stream_callback is not None:
        detach = StreamCallbackAdapter(stream_callback, token_callback).attach(session.bus)

    started = time.monotonic()
    status = "completed"
    try:
        answer = asyncio.run(session.run(task))
    except Exception as exc:  # noqa: BLE001 - surfaced in the result dict
        status = "failed"
        answer = f"harness run failed: {type(exc).__name__}: {exc}"
    finally:
        if detach is not None:
            detach()
    duration_ms = (time.monotonic() - started) * 1000

    state = session.state
    return {
        "final_answer": answer,
        "mode": "harness",
        "outer_loops": max(1, state.turn_index + 1),
        "thinking_steps": state.model_calls,
        "metadata": {
            "duration_ms": duration_ms,
            "status": status,
            "session_id": session.session_id,
            "journal": str(session.journal.path),
            "denials": state.denials,
            "tool_calls": state.tool_calls,
        },
        "session_token_usage": {"last_call_tokens": state.total_tokens},
    }
