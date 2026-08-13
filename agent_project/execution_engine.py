"""
Unified Execution Engine for OpenMythos Agent.

Consolidates the previously duplicated loop logic from:
  - OpenMythosAgent._run_traditional
  - ReasoningEngine.reason / _reason_react / _reason_super

The engine is strategy-agnostic: a `ThinkingPolicy` decides what prompt to send
and how to interpret the model output, while the engine handles the
think -> act -> observe -> converge loop.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .checkpoint import CheckpointManager
from .tools import TOOLS_REGISTRY, ToolResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRequest:
    tool_name: str
    arguments: Dict[str, Any]
    display_key: str = ""

    def __post_init__(self):
        if not self.display_key:
            self.display_key = f"{self.tool_name}({self._fmt_args(self.arguments)})"

    @staticmethod
    def _fmt_args(args: Dict[str, Any]) -> str:
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 80:
                v = v[:77] + "..."
            parts.append(f"{k}={v!r}")
        return ", ".join(parts)


@dataclass
class PolicyOutput:
    """Result of parsing one model output."""

    reasoning: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    final_answer: Optional[str] = None
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    step_number: int
    prompt: str
    output: str
    reasoning: str
    tool_calls: List[ToolCallRequest]
    observations: List[str]
    final_answer: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExecutionTrace:
    """Lightweight trace produced by the engine."""

    task: str
    strategy: str
    steps: List[StepRecord] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    quality_score: float = 0.0
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "strategy": self.strategy,
            "total_steps": len(self.steps),
            "tools_used": self.tools_used,
            "final_answer": (self.final_answer or "")[:200],
            "success": self.success,
            "quality_score": self.quality_score,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ExecutionContext:
    """Mutable context carried through a single execution."""

    task: str
    available_tools: Optional[Dict[str, Any]]
    config: Any
    max_steps: int = 16
    stream_callback: Optional[Callable[[str, str], None]] = None
    token_callback: Optional[Callable[[int], None]] = None
    code_mode: bool = False
    extra_context: str = ""
    history_context: str = ""

    # State accumulated during execution
    steps: List[StepRecord] = field(default_factory=list)
    executed_calls: Dict[str, str] = field(default_factory=dict)
    call_counts: Dict[str, int] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Execute parsed tool calls with caching, dedup, timeout and harness gate."""

    def __init__(
        self,
        harness_kernel: Optional[Any] = None,
        per_turn_cache: Optional[Dict[str, ToolResult]] = None,
        max_workers: int = 4,
        tool_timeout: float = 90.0,
    ):
        self.harness_kernel = harness_kernel
        self.per_turn_cache = per_turn_cache or {}
        self.max_workers = max_workers
        self.tool_timeout = tool_timeout
        self.logger = logging.getLogger("ToolExecutor")

    def execute_calls(
        self,
        calls: List[ToolCallRequest],
        ctx: ExecutionContext,
    ) -> List[Tuple[ToolCallRequest, str, bool]]:
        """Execute calls in parallel; return (call, observation_text, ok)."""
        results: List[Tuple[ToolCallRequest, str, bool]] = []
        if not calls:
            return results

        deduped = self._deduplicate(calls)
        if len(deduped) < len(calls):
            self.logger.debug(f"deduplicated {len(calls)} -> {len(deduped)} tool calls")

        pending = []
        cache_hits = []
        for call in deduped:
            cache_key = f"{call.tool_name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
            if cache_key in self.per_turn_cache:
                res = self.per_turn_cache[cache_key]
                cache_hits.append((call, res.output or res.error or "", res.success))
                continue
            pending.append((call, cache_key))

        if pending:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(pending), self.max_workers))
            try:
                futures = {executor.submit(self._execute_one, call, ctx): (call, cache_key) for call, cache_key in pending}
                done, not_done = concurrent.futures.wait(futures, timeout=self.tool_timeout)
                # 超时项: 标记失败但不中断整体, 已完成结果继续保留
                for future in not_done:
                    future.cancel()
                    call, cache_key = futures[future]
                    ok = False
                    obs = (
                        f"SYSTEM STOP: Tool '{call.tool_name}' timed out after {self.tool_timeout}s. "
                        "Do NOT retry this call; use a different tool or answer with what you have."
                    )
                    self.per_turn_cache[cache_key] = ToolResult(success=ok, output=obs)
                    results.append((call, obs, ok))
                for future in done:
                    call, cache_key = futures[future]
                    try:
                        ok, obs = future.result()
                    except Exception as e:
                        ok = False
                        obs = f"Tool execution error: {e}"
                    self.per_turn_cache[cache_key] = ToolResult(success=ok, output=obs)
                    results.append((call, obs, ok))
            finally:
                # 不等待慢线程结束, 立即返回(后台线程自然结束)
                executor.shutdown(wait=False, cancel_futures=True)

        results.extend(cache_hits)
        # Preserve input order for deterministic prompts
        order = {id(c): i for i, c in enumerate(deduped)}
        results.sort(key=lambda x: order.get(id(x[0]), 0))
        return results

    def _deduplicate(self, calls: List[ToolCallRequest]) -> List[ToolCallRequest]:
        seen = set()
        unique = []
        for call in calls:
            key = json.dumps({"name": call.tool_name, "args": call.arguments}, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                unique.append(call)
        return unique

    @staticmethod
    def _patch_default_args(tool_name: str, args: Dict[str, Any]) -> None:
        """轻量参数兜底: 补缺省必填参数, 避免空调用 TypeError(如 glob:{} / search_files:{})."""
        if not isinstance(args, dict):
            return
        if tool_name == "glob":
            pat = args.get("pattern") or args.get("query")
            if not isinstance(pat, str) or not pat.strip():
                args["pattern"] = "**"
            if not args.get("path"):
                args["path"] = "."
        elif tool_name == "search_files":
            pat = args.get("pattern") or args.get("query")
            if not isinstance(pat, str) or not pat.strip():
                args["pattern"] = ""
            if not args.get("path"):
                args["path"] = "."
        elif tool_name == "file_ops":
            if not isinstance(args.get("path"), str) or not args.get("path", "").strip():
                args["path"] = "."
        elif tool_name == "bash_exec":
            cmd = args.get("command") or args.get("cmd")
            if not isinstance(cmd, str):
                args["command"] = str(cmd or "")

    def _execute_one(self, call: ToolCallRequest, ctx: ExecutionContext) -> Tuple[bool, str]:
        tool_name = call.tool_name
        args = dict(call.arguments or {})

        # 轻量参数兜底: 补缺省必填参数, 避免空调用 TypeError(如 glob:{} / search_files:{})
        self._patch_default_args(tool_name, args)

        # Harness policy gate
        if self.harness_kernel is not None:
            try:
                from .harness.effects import make_effect
                from .harness.kernel import Decision

                admission = self.harness_kernel.evaluate(make_effect(tool_name, args))
                if admission.decision is not Decision.ALLOW:
                    return False, f"SYSTEM STOP: harness blocked {tool_name}: {admission.reason}"
            except Exception as e:
                self.logger.warning(f"harness admission failed: {e}")

        tool = TOOLS_REGISTRY.get(tool_name)
        if not tool:
            return False, f"Tool not found: {tool_name}"

        try:
            if args:
                result = tool.execute(**args)
            else:
                result = tool.execute()
        except Exception as e:
            return False, f"Tool execution error: {e}"

        if result.success:
            return True, (result.output or "").strip()[:2500]
        return False, f"Tool error: {result.error or 'unknown error'}"


# ---------------------------------------------------------------------------
# Observation manager
# ---------------------------------------------------------------------------

class ObservationManager:
    """Format execution history into compact prompt context."""

    @staticmethod
    def format_history(steps: List[StepRecord], max_obs_chars: int = 2000) -> str:
        lines = []
        for step in steps:
            if step.reasoning:
                lines.append(f"Thought: {step.reasoning[:300]}")
            for obs in step.observations:
                lines.append(f"Observation: {obs[:max_obs_chars]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convergence checker
# ---------------------------------------------------------------------------

class ConvergenceChecker:
    """Decide whether the loop should stop early."""

    def __init__(self, min_steps: int = 2):
        self.min_steps = min_steps

    def should_stop(
        self,
        ctx: ExecutionContext,
        policy_output: PolicyOutput,
        step_number: int,
    ) -> bool:
        if policy_output.done:
            return True
        if step_number >= ctx.max_steps:
            return True
        if step_number < self.min_steps:
            return False

        # 连续 SYSTEM STOP -> 判断是真卡住还是良性去重:
        # - "already executed" (去重) 是良性, 模型应换工具继续, 不算卡住
        # - "timed out" / "harness blocked" 是真失败, 连续出现才是卡住
        recent_obs = ctx.observations[-3:]
        hard_stops = [o for o in recent_obs
                      if "SYSTEM STOP" in o
                      and "already executed" not in o
                      and "You already have this result" not in o]
        dedup_stops = [o for o in recent_obs if "already executed" in o]
        # 真失败连续 3 个 → 卡住
        if len(hard_stops) >= 3 and step_number >= 5:
            return True
        # 全部是去重拦截且 ≥3 个, 且步骤足够多 → 模型在绕圈, 停止并让其反思
        if len(dedup_stops) >= 3 and step_number >= 6 and len(hard_stops) == 0:
            return True

        # Repeated identical tool calls 3+ times
        for call in policy_output.tool_calls:
            key = f"{call.tool_name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
            if ctx.call_counts.get(key, 0) >= 3:
                return True

        return False


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

class _LiveStreamFilter:
    """把后端流式原始输出清洗成用户可见内容并实时透出.

    - [TOOL:...[/TOOL] 工具调用块可跨行, 用状态机丢弃
    - <think>...</think> 思考块同样丢弃
    - Thought:/Action:/Observation: 等 ReAct 标签行不下发到 content
    - 其余文本在行/缓冲边界即时下发, 让输出逐字流畅显示而不是整段蹦出
    """

    _LABEL_RE = re.compile(r"^\s*(Thought|Action|Observation)\s*:.*$", re.IGNORECASE)
    _MAX_HOLD = 120

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._buf = ""
        self._in_tool = False
        self._in_think = False

    def reset(self) -> None:
        self._buf = ""
        self._in_tool = False
        self._in_think = False

    def feed(self, text: str) -> None:
        if not text:
            return
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle(line + "\n")
        # 超长且不含块起始的单行, 提前下发, 避免长时间无输出
        if len(self._buf) > self._MAX_HOLD and "[TOOL:" not in self._buf and "<think" not in self._buf:
            self._handle(self._buf)
            self._buf = ""

    def flush(self) -> None:
        if self._buf:
            self._handle(self._buf)
            self._buf = ""

    def _handle(self, chunk: str) -> None:
        while True:
            if self._in_tool:
                idx = chunk.find("[/TOOL]")
                if idx == -1:
                    return  # 工具块未闭合, 保持状态丢弃后续
                chunk = chunk[idx + len("[/TOOL]"):]
                self._in_tool = False
                continue
            if self._in_think:
                close = chunk.find("</think")  # 同时匹配 </think> 与 </thinking>
                if close == -1:
                    return
                end = chunk.find(">", close)
                if end == -1:
                    return
                chunk = chunk[end + 1:]
                self._in_think = False
                continue
            ti = chunk.find("[TOOL:")
            ki = chunk.find("<think")  # 同时匹配 <think> 与 <thinking>
            opens = [i for i in (ti, ki) if i != -1]
            if not opens:
                break
            first = min(opens)
            if first > 0:
                self._emit_clean(chunk[:first])
            chunk = chunk[first:]
            if chunk.startswith("[TOOL:"):
                chunk = chunk[len("[TOOL:"):]
                self._in_tool = True
                end = chunk.find("[/TOOL]")
                if end == -1:
                    return
                chunk = chunk[end + len("[/TOOL]"):]
                self._in_tool = False
            else:  # <think> 或 <thinking>
                chunk = chunk[len("<think"):]
                self._in_think = True
                close = chunk.find("</think")
                if close == -1:
                    return
                end = chunk.find(">", close)
                if end == -1:
                    return
                chunk = chunk[end + 1:]
                self._in_think = False
        self._emit_clean(chunk)

    def _emit_clean(self, text: str) -> None:
        kept = []
        for ln in text.split("\n"):
            stripped = ln.strip()
            if self._LABEL_RE.match(stripped):
                continue
            # 去掉 "Final Answer:" 前缀, 保留答案正文, 让输出更干净
            if stripped.startswith("Final Answer:"):
                ln = stripped[len("Final Answer:"):].lstrip()
            kept.append(ln)
        out = "\n".join(kept)
        if out.strip():
            self._emit(out)


class ExecutionEngine:
    """Unified think-act-observe loop."""

    def __init__(
        self,
        model_backend: Any,
        config: Any,
        harness_kernel: Optional[Any] = None,
        per_turn_cache: Optional[Dict[str, ToolResult]] = None,
    ):
        self.model = model_backend
        self.config = config
        self.tool_executor = ToolExecutor(
            harness_kernel=harness_kernel,
            per_turn_cache=per_turn_cache,
        )
        self.convergence = ConvergenceChecker(min_steps=2)
        self.obs_manager = ObservationManager()
        self.logger = logging.getLogger("ExecutionEngine")

    def run(
        self,
        policy,
        ctx: ExecutionContext,
    ) -> ExecutionTrace:
        """Run the policy loop until done or convergence."""
        trace = ExecutionTrace(
            task=ctx.task,
            strategy=getattr(policy, "name", policy.__class__.__name__),
        )
        start = time.time()

        try:
            prompt = policy.first_prompt(ctx)
            step_number = 0
            while True:
                step_number += 1
                self._emit_status(ctx, f"thinking (step {step_number}/{ctx.max_steps})")

                output = self._generate(prompt, ctx, step_number)
                parsed = policy.parse_output(output, ctx)

                # Stream reasoning and tool calls
                # 实时流式已显示内容时, 不再重复 emit(避免与 live token 重复/抖动)
                if not getattr(ctx, "_live_streamed", False):
                    self._stream_reasoning(ctx, parsed.reasoning)
                for call in parsed.tool_calls:
                    self._emit(ctx, "tool_call", call.display_key)

                record = StepRecord(
                    step_number=step_number,
                    prompt=prompt,
                    output=output,
                    reasoning=parsed.reasoning,
                    tool_calls=parsed.tool_calls,
                    observations=[],
                )

                if parsed.done or not parsed.tool_calls:
                    if parsed.final_answer:
                        record.final_answer = parsed.final_answer
                        trace.final_answer = parsed.final_answer
                        trace.success = True
                        ctx.steps.append(record)
                        trace.steps.append(record)
                        break
                    # 只输出了思考/计划, 既没有工具调用也没有最终答案(如模型光说不做)
                    # -> 注入催促, 让它在预算内真正行动或作答, 而不是静默结束返回思考文字
                    if not parsed.done and step_number < ctx.max_steps:
                        ctx.steps.append(record)
                        trace.steps.append(record)
                        nudge = (
                            "\n\n注意: 你刚才只进行了思考, 没有做出下一步动作。\n"
                            "· 若任务需要查文件/搜索/执行, 必须立即输出: Action: [TOOL:工具名] {参数} [/TOOL]\n"
                            "· 若已能回答, 必须输出: Final Answer: <完整答案>\n"
                            "不要重复思考, 直接行动或作答。"
                        )
                        next_prompt = policy.next_prompt(ctx, output)
                        prompt = (next_prompt or "") + nudge
                        continue
                    ctx.steps.append(record)
                    trace.steps.append(record)
                    break

                # Execute tools
                exec_results = self._execute_tool_calls(parsed.tool_calls, ctx)
                for call, obs, ok in exec_results:
                    record.observations.append(obs)
                    ctx.observations.append(obs)
                    trace.observations.append(obs)
                    trace.tools_used.append(call.tool_name)
                    self._emit(ctx, "tool_result", obs)

                ctx.steps.append(record)
                trace.steps.append(record)

                # 动态扩展预算: 接近上限时, 若模型仍产出新的工具调用(有进展),
                # 且未超过硬上限, 则扩大 max_steps 继续(自适应思考深度)。
                # 必须在 convergence.should_stop 之前执行, 否则到上限会被先判定停止。
                if step_number >= ctx.max_steps:
                    hard_limit = getattr(self.config, "max_thinking_loops", 32)
                    if ctx.max_steps < hard_limit and parsed.tool_calls:
                        new_max = min(hard_limit, ctx.max_steps + 4)
                        self.logger.info(
                            f"dynamic loop extension: step {step_number} → max_steps {ctx.max_steps} → {new_max} "
                            f"(model still producing tool calls)"
                        )
                        ctx.max_steps = new_max
                        if ctx.stream_callback:
                            ctx.stream_callback("status", f"extending loops → up to {new_max} (still progressing)")

                if self.convergence.should_stop(ctx, parsed, step_number):
                    break

                next_prompt = policy.next_prompt(ctx, output)
                if next_prompt is None:
                    break
                prompt = next_prompt

            if not trace.final_answer or self._is_truncated_fragment(trace.final_answer):
                # final_answer 为空或为截断碎片(如 "We"/"The")时,
                # 基于已有观察重新生成完整答案, 避免残缺回答直接返回给用户
                trace.final_answer = self._force_final_answer(ctx)
                trace.success = bool(trace.final_answer)

        except Exception as e:
            self.logger.error(f"ExecutionEngine failed: {type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
            trace.success = False
            trace.metadata["error"] = f"{type(e).__name__}: {e}"
            try:
                rollback = CheckpointManager().rollback_latest()
                if rollback:
                    trace.metadata["rollback"] = rollback
            except Exception:
                pass

        finally:
            trace.duration_ms = int((time.time() - start) * 1000)
            trace.quality_score = self._score_trace(trace)

        return trace

    def _generate(self, prompt: str, ctx: ExecutionContext, step_number: int) -> str:
        max_tokens = 8192 if ctx.code_mode else 2048
        if step_number == 1 and len(prompt) > 8000:
            max_tokens = 4096
        text, _streamed = self._streaming_generate(
            prompt, ctx, getattr(self.config, "temperature", 0.7), max_tokens
        )
        return text

    def _streaming_generate(self, prompt: str, ctx: ExecutionContext, temperature: float, max_tokens: int) -> Tuple[str, bool]:
        """调用后端生成; 提供 stream_callback 时透传后端做 token 级实时流式.

        返回 (完整文本, 是否真实流式). 真实流式时, 原始输出经 _LiveStreamFilter
        清洗(去掉工具/思考块/ReAct 标签)后逐字显示, 避免生成期间"卡住无输出",
        也避免整段结束后一次性蹦出.
        """
        if not ctx.stream_callback:
            setattr(ctx, "_live_streamed", False)
            text = self.model.generate(
                prompt,
                n_loops=1,
                temperature=temperature,
                max_tokens=max_tokens,
                token_callback=ctx.token_callback,
            )
            return text, False

        live = {"active": False}
        live_filter = _LiveStreamFilter(lambda text: self._emit(ctx, "content", text))

        def on_stream(kind: str, text: str):
            if not text:
                return
            live["active"] = True
            if kind == "reasoning":
                self._emit(ctx, "reasoning", text)
            elif kind == "content":
                live_filter.feed(text)

        text = self.model.generate(
            prompt,
            n_loops=1,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_callback=on_stream,
            token_callback=ctx.token_callback,
        )
        live_filter.flush()
        setattr(ctx, "_live_streamed", live["active"])
        return text, live["active"]

    def _execute_tool_calls(
        self,
        calls: List[ToolCallRequest],
        ctx: ExecutionContext,
    ) -> List[Tuple[ToolCallRequest, str, bool]]:
        # Update call counts and detect exact duplicates against this session
        pending = []
        duplicate_results = []
        for call in calls:
            key = json.dumps({"name": call.tool_name, "args": call.arguments}, sort_keys=True, ensure_ascii=False)
            ctx.call_counts[key] = ctx.call_counts.get(key, 0) + 1
            if key in ctx.executed_calls:
                obs = (
                    f"SYSTEM STOP: You already executed {call.display_key}. "
                    f"Result was: {ctx.executed_calls[key][:800]}\n"
                    "You already have this result. Do NOT call the same tool with the same arguments again. "
                    "Either call a DIFFERENT tool with different arguments, or output 'Final Answer:' now."
                )
                ctx.executed_calls[key] = obs
                duplicate_results.append((call, obs, False))
            else:
                pending.append(call)

        results = self.tool_executor.execute_calls(pending, ctx)
        for call, obs, ok in results:
            key = json.dumps({"name": call.tool_name, "args": call.arguments}, sort_keys=True, ensure_ascii=False)
            ctx.executed_calls[key] = obs
        return results + duplicate_results

    def _force_final_answer(self, ctx: ExecutionContext) -> str:
        if not ctx.observations:
            if ctx.steps:
                return ctx.steps[-1].reasoning or "No result produced."
            return "No result produced."

        recent = "\n\n".join(ctx.observations[-3:])
        prompt = (
            "Based ONLY on the tool results below, provide a complete final answer to the task.\n"
            "Do NOT include <think> tags, reasoning traces, or tool-call tags.\n\n"
            f"## Task:\n{ctx.task}\n\n"
            f"## Tool Results:\n{recent}\n\n"
            "## Final Answer:\n"
        )
        answer, _streamed = self._streaming_generate(prompt, ctx, 0.3, 2048)
        cleaned = self._clean_final_text(answer or "")
        # 二次兜底: 生成仍为空/截断时, 用最近一步思考或观察作为答案
        if self._is_truncated_fragment(cleaned):
            if ctx.steps:
                reasoning = (ctx.steps[-1].reasoning or "").strip()
                if reasoning and not self._is_truncated_fragment(reasoning):
                    return reasoning
            if ctx.observations:
                obs = ctx.observations[-1].strip()
                if obs and len(obs) > 6:
                    return f"根据工具结果:\n{obs[:800]}"
        return cleaned

    @staticmethod
    def _is_truncated_fragment(text: str) -> bool:
        """判断最终答案是否为流截断碎片(过短/残缺, 如 "We"/"The"/"And").

        正常短答("好"/"ok"/"完成")不算; 仅当纯 ASCII、无标点、很短时视为截断。
        """
        t = (text or "").strip()
        if not t:
            return True
        if t in {"好", "好的", "ok", "okay", "OK", "嗯", "对", "是", "完成", "done", "yes", "no", "y", "n"}:
            return False
        if len(t) <= 6 and all(ord(c) < 128 for c in t) and not any(c in ".,;:!?()\"'" for c in t):
            return True
        return False

    @staticmethod
    def _clean_final_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\[TOOL:\w+\].*?\[/TOOL\]", "", text, flags=re.DOTALL)
        text = re.sub(r"\[TOOL:\w+\]\s*\{[^}]*\}", "", text)
        text = re.sub(r"\[/?TOOL:?\w*\]", "", text)
        text = re.sub(r"^Thought:\s*.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Action:\s*.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Final Answer:\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _score_trace(self, trace: ExecutionTrace) -> float:
        score = 0.0
        if trace.success:
            score += 0.4
        final = (trace.final_answer or "").strip()
        if final:
            score += 0.15
            if len(final) > 80:
                score += 0.05
        steps = len(trace.steps)
        if 2 <= steps <= 8:
            score += 0.15
        elif steps > 16:
            score -= 0.15
        if trace.tools_used:
            unique = len(set(trace.tools_used))
            score += min(unique * 0.05, 0.15)
            most_common = max(set(trace.tools_used), key=trace.tools_used.count)
            ratio = trace.tools_used.count(most_common) / len(trace.tools_used)
            if ratio > 0.7 and len(trace.tools_used) >= 3:
                score -= 0.15
        stop_count = sum(1 for o in trace.observations if "SYSTEM STOP" in o)
        if trace.observations and stop_count / len(trace.observations) > 0.5:
            score -= 0.2
        if trace.duration_ms < 10000:
            score += 0.1
        elif trace.duration_ms > 60000:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _stream_reasoning(self, ctx: ExecutionContext, reasoning: str):
        if not reasoning or not ctx.stream_callback:
            return
        # Emit in small chunks to mimic streaming
        for i in range(0, len(reasoning), 3):
            ctx.stream_callback("reasoning", reasoning[i : i + 3])

    def _emit(self, ctx: ExecutionContext, kind: str, text: str):
        if ctx.stream_callback and text:
            try:
                ctx.stream_callback(kind, text)
            except Exception:
                pass

    def _emit_status(self, ctx: ExecutionContext, text: str):
        self._emit(ctx, "status", text)
