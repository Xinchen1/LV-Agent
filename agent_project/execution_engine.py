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

    # 动态扩展状态: 连续无进展步数(达到阈值则停止并提示失败, 替代硬上限)
    no_progress_streak: int = 0

    # 分支监控 agent 状态(旁路感知主 agent 执行)
    monitor_enabled: bool = True
    monitor_hints: List[str] = field(default_factory=list)  # 注入给主 agent 的简明提示
    monitor_rounds: int = 0  # 监控检查轮数(限制频率)


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
                        f"SYSTEM SKIP: Tool '{call.tool_name}' timed out after {self.tool_timeout}s. "
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
        """轻量参数兜底: 补缺省必填参数, 避免空调用 TypeError(如 glob:{} / search_files:{}).

        同时把模型常用别名参数迁移到规范键(command/cmd、pattern/query), 否则执行会因缺键失败。
        """
        if not isinstance(args, dict):
            return
        if tool_name == "glob":
            # 迁移 query -> pattern
            if "pattern" not in args or not isinstance(args.get("pattern"), str) or not args["pattern"].strip():
                if args.get("query"):
                    args["pattern"] = args["query"]
            pat = args.get("pattern")
            if not isinstance(pat, str) or not pat.strip():
                args["pattern"] = "**"
            if not args.get("path"):
                args["path"] = "."
        elif tool_name == "search_files":
            if "pattern" not in args or not isinstance(args.get("pattern"), str) or not args["pattern"].strip():
                if args.get("query"):
                    args["pattern"] = args["query"]
            pat = args.get("pattern")
            if not isinstance(pat, str) or not pat.strip():
                args["pattern"] = ""
            if not args.get("path"):
                args["path"] = "."
        elif tool_name == "file_ops":
            if not isinstance(args.get("path"), str) or not args.get("path", "").strip():
                args["path"] = "."
        elif tool_name == "bash_exec":
            # 迁移 cmd -> command
            if not args.get("command"):
                if args.get("cmd"):
                    args["command"] = args["cmd"]
            cmd = args.get("command")
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

                effect = make_effect(tool_name, args)
                admission = self.harness_kernel.evaluate(effect)
                if admission.decision is Decision.ASK:
                    # 让用户选择确认/拒绝, 而非直接拦截; 非交互环境回退为拒绝
                    granted = False
                    try:
                        import sys as _sys
                        _interactive = bool(getattr(_sys, "stdin", None) and _sys.stdin.isatty())
                        if _interactive:
                            granted = self.harness_kernel.ask(effect, admission.reason)
                    except Exception as e:
                        self.logger.warning(f"harness ask failed: {e}")
                    if granted:
                        # 用户同意: 放行并记住
                        try:
                            self.harness_kernel.allowlist_add(effect)
                        except Exception:
                            pass
                    else:
                        return False, f"SYSTEM SKIP: harness blocked {tool_name}: {admission.reason}"
                elif admission.decision is not Decision.ALLOW:
                    return False, f"SYSTEM SKIP: harness blocked {tool_name}: {admission.reason}"
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

        # SYSTEM SKIP / SYSTEM STOP → 判断是真卡住还是良性去重:
        # - "already executed" (去重) 是良性, 模型应换工具继续, 不算卡住
        # - "timed out" / "harness blocked" / "permission denied" 是真失败, 连续出现才是卡住
        recent_obs = ctx.observations[-3:]
        hard_stops = [o for o in recent_obs
                      if ("SYSTEM SKIP" in o or "SYSTEM STOP" in o or "timed out" in o.lower())
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

    # ------------------------------------------------------------------
    # 分支监控 agent: 旁路感知主 agent 执行, 发现异常注入简明提示
    # ------------------------------------------------------------------
    def _rule_monitor(self, ctx: ExecutionContext) -> Optional[str]:
        """规则层监控(零成本): 检测明显异常信号, 返回简明提示或 None."""
        obs = ctx.observations[-3:] if len(ctx.observations) > 3 else ctx.observations
        hints = []
        # 1. 连续工具失败
        fails = [o for o in obs if "tool error" in o.lower() or "failed" in o.lower()
                 or "timed out" in o.lower() or "error:" in o.lower()]
        if len(fails) >= 2:
            hints.append("提示: 连续工具失败, 换一种工具或方法, 检查参数/路径。")
        # 2. 重复调用同一工具(去重拦截)
        dedups = [o for o in obs if "already executed" in o]
        if len(dedups) >= 2:
            hints.append("提示: 你重复调用了相同工具, 请换新工具或直接总结。")
        # 3. 偏离任务(工具结果与任务主题无关) - 简化: 观察里大量权限错误
        perm_errors = sum(1 for o in obs if "Operation not permitted" in o or "permission denied" in o.lower())
        if perm_errors >= 3:
            hints.append("提示: 正在全盘扫描遇到权限错误, 缩小搜索范围(如指定目录)。")
        return "\n".join(hints) if hints else None

    def _llm_monitor(self, ctx: ExecutionContext) -> Optional[str]:
        """协作分支 agent: 感知主 agent 轨迹, 判断问题并决策是否主动补位.

        返回: (monitor_hint, backfill_action) 或 None
        """
        recent_steps = ctx.steps[-4:] if len(ctx.steps) > 4 else ctx.steps
        recent_obs = ctx.observations[-4:] if len(ctx.observations) > 4 else ctx.observations
        lines = []
        for st in recent_steps:
            tools = ", ".join(c.display_key for c in st.tool_calls) or "(思考)"
            lines.append(f"步{st.step_number}: {tools}")
        obs_snippet = " | ".join(o[:80] for o in recent_obs)
        prompt = (
            "你是协作分支 agent, 与主 agent 共同完成任务。感知主 agent 执行轨迹。\n"
            "主 agent 任务: " + (ctx.task[:200]) + "\n"
            "最近步骤: " + "; ".join(lines) + "\n"
            "最近观察: " + obs_snippet[:400] + "\n"
            "若发现异常(搜索失败/写入失败/卡住/走偏), 你应主动补位完成子任务。\n"
            "只输出 JSON: {\"problem\": \"一句话描述问题或OK\", \"backfill\": "
            "\"要补位的工具调用, 如web_search(query=...), 若无补位输出none\"}\n"
            "JSON:"
        )
        try:
            raw = self.model.generate(prompt, n_loops=1, temperature=0.2, max_tokens=120)
            text = str(raw or "").strip()
            if not text:
                return None
            import json as _json, re as _re
            m = _re.search(r'\{.*\}', text, _re.DOTALL)
            if not m:
                return None
            payload = _json.loads(m.group(0))
            problem = str(payload.get("problem", "") or "").strip()
            if not problem or problem.upper() == "OK":
                return None
            backfill = str(payload.get("backfill", "") or "").strip()
            if backfill and backfill.lower() != "none":
                # 主动补位: 子 agent 执行补位动作, 把结果返回
                result = self._execute_backfill(ctx, backfill)
                if result:
                    return f"[协作补位] {problem}。我已主动补位完成: {result}"
            return problem[:160]
        except Exception as e:
            self.logger.debug(f"llm monitor failed: {e}")
            return None

    def _execute_backfill(self, ctx: ExecutionContext, backfill: str) -> Optional[str]:
        """协作分支 agent 主动补位: 解析工具调用并执行, 返回结果摘要.

        补位动作如 "web_search(query=AI新闻)" / "file_ops(action=write, path=..., content=...)"
        """
        from .policies import ToolCallParser
        try:
            calls = ToolCallParser.parse_all(backfill)
            if not calls:
                return None
            tool_name, args = calls[0]
            from .tools import TOOLS_REGISTRY
            tool = TOOLS_REGISTRY.get(tool_name)
            if tool is None:
                return None
            result = tool.execute(**args)
            out_src = result.output or result.error or "executed"
            out = out_src[:200]
            status = "OK" if result.success else f"FAIL: {(result.error or '')[:80]}"
            ctx.observations.append(f"[协作补位] {tool_name}: {status} | {out[:100]}")
            return f"{tool_name} → {out[:100]}"
        except Exception as e:
            self.logger.debug(f"backfill failed: {e}")
            return None

    def _monitor_and_inject(self, ctx: ExecutionContext, output: str) -> None:
        """监控入口: 规则层+LLM 分支 agent, 发现异常注入简明提示到 monitor_hints."""
        if not ctx.monitor_enabled:
            return
        ctx.monitor_rounds += 1
        # 频率控制: 每 2 步检查一次, 避免过度干预
        if ctx.monitor_rounds % 2 != 0:
            return
        hint = self._rule_monitor(ctx)
        if hint is None and ctx.max_steps >= 8 and len(ctx.steps) >= 3:
            # 规则没发现, 但复杂任务已有多个步骤时, 用 LLM 分支 agent 感知
            # (降低频率: 每 4 轮才做一次 LLM 感知, 省 token)
            if ctx.monitor_rounds % 4 == 0:
                hint = self._llm_monitor(ctx)
        if hint:
            ctx.monitor_hints.append(hint)
            if ctx.stream_callback:
                ctx.stream_callback("status", f"monitor: {hint[:60]}")

    def _attach_monitor_hints(self, prompt: str, ctx: ExecutionContext) -> str:
        """把监控提示注入主 agent 的下一轮 prompt."""
        if not ctx.monitor_hints:
            return prompt
        joined = "\n".join(ctx.monitor_hints[-2:])
        return prompt + f"\n\n【监控提示】{joined}\n请根据提示调整你的下一步(可调整计划/换方法/继续)。"

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
                if not output:
                    output = ""
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

                # ---- 阶段 1: 有最终答案 → 完成 ----
                if parsed.final_answer:
                    record.final_answer = parsed.final_answer
                    trace.final_answer = parsed.final_answer
                    trace.success = True
                    ctx.steps.append(record)
                    trace.steps.append(record)
                    # 最终答案通过 content 通道透出(让 UI 高亮 markdown),
                    # 而不是只由 _finish_result 纯文本打印。
                    if not getattr(ctx, "_live_streamed", False):
                        self._emit(ctx, "content", parsed.final_answer)
                    break

                # ---- 阶段 2: 无动作(既无工具调用也无答案) ----
                if not parsed.tool_calls:
                    ctx.steps.append(record)
                    trace.steps.append(record)
                    if step_number >= ctx.max_steps:
                        break  # 预算已耗尽, 交给下方 force_final_answer 兜底
                    wants_more = self._wants_to_continue(output)
                    if wants_more:
                        # 模型明确想继续内省 → 扩展预算并放行
                        if self._extend_loops(ctx):
                            self._emit_status(ctx, f"↑ loop {ctx.max_steps} (增加思考预算)")
                        prompt = self._attach_monitor_hints(
                            (policy.next_prompt(ctx, output) or "")
                            + "\n\n你表示想继续/确认, 请继续下一步——可调用工具, 或直接给出 Final Answer。",
                            ctx,
                        )
                        continue
                    # 无明确继续意图 → 催促行动
                    prompt = self._attach_monitor_hints(
                        (policy.next_prompt(ctx, output) or "")
                        + "\n\n注意: 你刚才只进行了思考, 没有做出下一步动作。\n"
                          "· 需要查文件/搜索/执行则立即输出: [TOOL:工具名] {参数}\n"
                          "· 已能回答则输出: Final Answer: <完整答案>\n不要重复思考, 直接行动或作答。",
                        ctx,
                    )
                    continue

                # ---- 阶段 3: 执行工具 ----
                exec_results = self._execute_tool_calls(parsed.tool_calls, ctx)
                for call, obs, ok in exec_results:
                    record.observations.append(obs)
                    ctx.observations.append(obs)
                    trace.observations.append(obs)
                    trace.tools_used.append(call.tool_name)
                    self._emit(ctx, "tool_result", obs)

                ctx.steps.append(record)
                trace.steps.append(record)

                # ---- 阶段 4: 无进展约束 + 动态扩展 + 收敛 ----
                if not self._has_real_progress(ctx):
                    ctx.no_progress_streak += 1
                    if ctx.no_progress_streak >= 3:
                        self.logger.warning(
                            f"loop stopped after {ctx.no_progress_streak} consecutive no-progress steps (step {step_number})"
                        )
                        self._emit_status(ctx, f"连续 {ctx.no_progress_streak} 步无进展, 已停止(请检查工具/参数或换个思路)")
                        break
                else:
                    ctx.no_progress_streak = 0

                self._monitor_and_inject(ctx, output)

                if step_number >= ctx.max_steps:
                    if self._extend_loops(ctx):
                        self._emit_status(ctx, f"↑ loop {ctx.max_steps} (增加思考预算, 仍在推进)")

                if self.convergence.should_stop(ctx, parsed, step_number):
                    break

                next_prompt = policy.next_prompt(ctx, output)
                if next_prompt is None:
                    break
                # 把分支监控 agent 的提示注入主 agent 的下一轮
                prompt = self._attach_monitor_hints(next_prompt, ctx)

            # 项目/代码分析任务: 最终答案若为空/截断/过短(如只提一个文件)则强制重生成完整分析。
            # 覆盖用户场景: 模型调 project_context 拿到结构后只回一句"docs/goai/作品简介"就结束。
            _analysis_needed = bool(
                re.search(r"(分析|架构|剖析|调研|概述|概览|报告|overview|analy[sz]e|architectur|structure|analyze)", ctx.task, re.IGNORECASE)
            )
            _answer = (trace.final_answer or "").strip()
            # 敷衍判定: 长度很短(如只提一个文件名/一句概述) 且 不含结论性/结构化内容。
            # 避免误伤"这是分析结论"这类合理的短结论——用"实质内容"而非单纯长度判断。
            _has_substance = bool(
                _answer and (
                    len(_answer) >= 200
                    or re.search(r"(?:结论|综上|核心|架构|模块|要点|结构|亮点|总结|整体)", _answer)
                )
            )
            _short_answer = _analysis_needed and 0 < len(_answer) < 200 and not _has_substance
            # 兜底条件: 有观察 或 至少走过多步(模型全程只思考没调工具时 observations 为空,
            # 但应基于已有思考/任务重生成答案, 不能空手而归)。
            _can_regenerate = bool(ctx.observations or ctx.steps)
            if (not trace.final_answer or self._is_truncated_fragment(trace.final_answer) or _short_answer) and _can_regenerate:
                # 基于已有观察重新生成完整答案, 避免残缺/过短回答直接返回给用户
                hint = ""
                if _analysis_needed:
                    hint = (
                        "\n\n这是对目标项目/代码的完整分析, 请覆盖以下结构:\n"
                        "- 项目定位与核心模块(按实际观察到的结构)\n"
                        "- 每个关键目录/文件的职责(读到的内容)\n"
                        "- 技术栈/架构特点(从结构推断)\n"
                        "- 核心逻辑与亮点(基于读取的文件内容)\n"
                        "给出结构化 Markdown 分析, 不要只说一句话。"
                    )
                trace.final_answer = self._force_final_answer(ctx, extra=hint, max_obs=12)
                # 二次审核: 分析报告若仍过短/敷衍, 基于全部有效观察再强制一次
                if _analysis_needed:
                    _again = (trace.final_answer or "").strip()
                    if not _again or self._is_truncated_fragment(_again) or len(_again) < 120:
                        self.logger.info("analysis report still too short; forcing with full observations")
                        trace.final_answer = self._force_final_answer(ctx, extra=hint, max_obs=16)
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
        # 分析/报告类任务需要更大输出空间, 避免长报告被 max_tokens 截断
        if re.search(r"(分析|报告|架构|剖析|调研|总结|评估|overview|analy[sz]e|report|architectur|summar)", ctx.task, re.IGNORECASE):
            max_tokens = max(max_tokens, 8192)
        if step_number == 1 and len(prompt) > 8000:
            max_tokens = 4096
        # 原生 Function Calling 优先: 后端若支持 generate_native, 直接拿结构化 tool_calls,
        # 转成文本协议格式交给下游解析(复用现有逻辑), 避免模型"猜格式"导致的解析失败。
        native = self._generate_native(prompt, ctx, step_number)
        if native is not None:
            return native
        text, _streamed = self._streaming_generate(
            prompt, ctx, getattr(self.config, "temperature", 0.7), max_tokens
        )
        return text

    def _generate_native(self, prompt: str, ctx: ExecutionContext, step_number: int) -> Optional[str]:
        """尝试原生 Function Calling; 不支持/失败时返回 None 回退文本协议."""
        backend = getattr(self, "model", None)
        if backend is None or not hasattr(backend, "generate_native"):
            return None
        if not ctx.available_tools:
            return None
        try:
            from .tools import TOOLS_REGISTRY
            tools = TOOLS_REGISTRY.get_openai_tools()
            if not tools:
                return None
            max_tokens = 8192 if ctx.code_mode else 2048
            # 分析/报告类任务: 更大输出空间, 避免长报告被截断
            if re.search(r"(分析|报告|架构|剖析|调研|总结|评估|overview|analy[sz]e|report|architectur|summar)", ctx.task, re.IGNORECASE):
                max_tokens = max(max_tokens, 8192)
            result = backend.generate_native(
                prompt,
                tools=tools,
                n_loops=1,
                temperature=getattr(self.config, "temperature", 0.7),
                max_tokens=max_tokens,
            )
            tcs = result.get("tool_calls") or []
            if tcs:
                parts = []
                for tc in tcs:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    parts.append(f"[TOOL:{name}] {json.dumps(args, ensure_ascii=False)} [/TOOL]")
                if parts:
                    # 有工具调用: 返回文本协议格式, 由 parse_output 复用现有解析
                    reasoning = (result.get("content") or "").strip()
                    if reasoning:
                        return reasoning + "\n" + " ".join(parts)
                    return " ".join(parts)
            content = (result.get("content") or "").strip()
            if content:
                return content
            return None
        except Exception as e:
            self.logger.debug(f"native FC unavailable, falling back: {e}")
            return None

    def _extend_loops(self, ctx: ExecutionContext) -> bool:
        """在安全上限内扩展 loop 预算. 返回是否真的扩展了.

        极简设计: 不再用 max_thinking_loops 作为硬上限截断——只要模型持续产出
        新工具调用且有真实进展就继续扩展; safety_cap 仅防极端死循环。
        """
        safety_cap = max(
            int(getattr(self.config, "max_thinking_loops", 32)) * 4,
            64,
        )
        if ctx.max_steps >= safety_cap:
            return False
        # 扩展次数上限: 最多扩展 3 次(每次 +4), 避免无限膨胀烧 token
        max_extends = getattr(self.config, "loop_extension_limit", 3)
        if getattr(ctx, "_loop_extends", 0) >= max_extends:
            return False
        ctx._loop_extends = getattr(ctx, "_loop_extends", 0) + 1
        ctx.max_steps = min(safety_cap, ctx.max_steps + 4)
        return True

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
                    f"⊙ Smart Dedup: search for \"{call.display_key}\" already completed. "
                    f"Reusing prior results to maximize research coverage.\n"
                    f"  Cached: {ctx.executed_calls[key][:400]}\n"
                    "  → Pivoting to a new angle, or synthesizing a final answer from existing evidence."
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

    def _force_final_answer(self, ctx: ExecutionContext, extra: str = "", max_obs: int = 3) -> str:
        """基于观察重生成完整答案.

        max_obs: 喂给模型的观察条数上限。分析/报告任务应传较大的值(如 12),
        避免只取最后几条(最后几条常是 git log/目录列表, 丢失前面读到的文档内容)。
        """
        # 无观察但有思考记录(模型全程只思考没调工具): 用思考内容兜底
        if not ctx.observations and ctx.steps:
            reasonings = [st.reasoning for st in ctx.steps if st.reasoning]
            if reasonings:
                recent = "\n\n".join(r[-500:] for r in reasonings[-3:])
                prompt = (
                    "根据你对以下任务的思考过程, 给出最终回答。\n"
                    "不要包含 <think> 标签或推理痕迹, 直接给出答案。\n\n"
                    f"## Task:\n{ctx.task}\n\n"
                    f"## 你的思考:\n{recent}\n\n"
                    "## Final Answer:\n"
                )
                if extra:
                    prompt = prompt.rstrip() + extra + "\n"
                answer, _streamed = self._streaming_generate(prompt, ctx, 0.3, 2048)
                cleaned = self._clean_final_text(answer or "")
                if cleaned and not self._is_truncated_fragment(cleaned):
                    return cleaned
        if not ctx.observations:
            if ctx.steps:
                return ctx.steps[-1].reasoning or "No result produced."
            return "No result produced."

        # 过滤掉 SYSTEM SKIP 去重拦截(不提供新信息, 且会占掉观察配额)
        useful = [o for o in ctx.observations if "SYSTEM SKIP" not in o and "already executed" not in o]
        if not useful:
            useful = ctx.observations
        recent = "\n\n".join(useful[-max_obs:])
        prompt = (
            "Based ONLY on the tool results below, provide a complete final answer to the task.\n"
            "Do NOT include <think> tags, reasoning traces, or tool-call tags.\n\n"
            f"## Task:\n{ctx.task}\n\n"
            f"## Tool Results:\n{recent}\n\n"
            "## Final Answer:\n"
        )
        if extra:
            prompt = prompt.rstrip() + extra + "\n"
        # 分析/报告任务: 更大输出空间, 避免兜底报告被截断
        _mt = 8192 if re.search(r"(分析|报告|架构|剖析|调研|总结|评估|overview|analy[sz]e|report|architectur|summar)", ctx.task, re.IGNORECASE) else 3072
        answer, _streamed = self._streaming_generate(prompt, ctx, 0.3, _mt)
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
    def _has_real_progress(ctx: ExecutionContext) -> bool:
        """评估最近几步是否产生真实进展(而非原地打转).

        依据:
        - 模型仍在产生新/不同的工具调用 → 视为有进展(执行层面失败≠停滞)
        - 同一调用反复执行(≥3次)且观察未变 → 原地打转
        - 至少 1 个非空、非错误的成功观察才视为有进展
        """
        import json as _json
        # 工具调用签名格式与 call_counts key 格式一致 (line 940):
        #   json.dumps({"name": tool_name, "args": arguments})
        # 这样 call_counts.get(top_sig) 才能命中
        recent_steps = getattr(ctx, "steps", [])[-3:]
        recent_sigs = []
        for st in recent_steps:
            for c in st.tool_calls:
                sig = _json.dumps(
                    {"name": c.tool_name, "args": c.arguments},
                    sort_keys=True, ensure_ascii=False
                )
                recent_sigs.append(sig)
        if recent_sigs:
            if len(set(recent_sigs)) >= 2:
                return True  # 多种不同调用 = 在换思路
            if getattr(ctx, "call_counts", {}).get(recent_sigs[0], 0) >= 3:
                return False  # 连续≥3次同一调用
            return True  # 开始尝试，短时间不判为打转

        obs = ctx.observations[-6:] if len(ctx.observations) > 6 else ctx.observations
        if not obs:
            return False
        _fail_markers = (
            "tool error", "tool execution error", "failed", "timed out",
            "error:", "denied", "blocked", "not found", "no results",
            "already executed", "SYSTEM SKIP", "SYSTEM STOP",
            "no output", "(no output)",
        )
        successful = 0
        total = 0
        for o in obs:
            ol = str(o).lower()
            if "SYSTEM SKIP" in ol or "SYSTEM STOP" in ol or "already executed" in ol:
                continue
            total += 1
            if any(m.lower() in ol for m in _fail_markers):
                continue
            if ol.strip():
                successful += 1
        if total == 0:
            return False
        return successful >= max(1, total // 2)

    @staticmethod
    def _wants_to_continue(output: str) -> bool:
        """精准检测模型是否明确表达'还想继续/补充/再多做'的意图.

        用于动态扩展 loop 预算: 模型主动要求继续时, 即使还没到预算上限也扩展。

        识别信号(任一命中即为继续意图):
        - 明确的继续/补充动词 + 动作对象: "再搜索一下/补充一下/还需要/让我继续/接着做/多找找/再查查"
        - "还/再" 前缀 + 动词: "还要看看/再试试/再来一轮"
        - 收尾性但含未完成语义: "暂时先这些, 不过我还想…"

        排除误判(不当作继续):
        - 已完成/收尾: "已完成/不需要了/就这些/不用了/这是最终/到此为止/以上是全部"
        - 只是描述现状而非主动继续: "我已经搜索了/我们已经有了"
        """
        t = (output or "").strip()
        if not t:
            return False

        # 强排除: 明确完成/收尾/否定继续
        _done = (
            "已完成", "完成了", "不需要了", "不用了", "就这些", "到此为止", "以上是全部",
            "这是最终", "最终答案", "全部找到", "已经够了", "这就够了", "不用继续",
            "done", "that's all", "no more", "finished", "complete", "this is all",
            "总结如下", "综上所述", "最后总结",
        )
        for d in _done:
            if d in t.lower():
                return False

        # 强肯定: 明确的继续/补充意图
        _continue = (
            "再搜索", "再搜", "再查", "再找", "再试", "再来", "再补充", "再深入",
            "继续搜索", "继续查", "继续找", "继续分析", "继续研究", "继续查看",
            "还要", "还想", "还需要", "让我继续", "我继续", "接着", "接下来",
            "补充一下", "补充搜索", "多找找", "多搜", "多查", "再看看",
            "还没有", "尚未", "不够全面", "再完善", "继续探索",
            "search more", "keep going", "continue", "let me continue",
            "more results", "dig deeper", "further", "one more",
        )
        # 排除"描述现状"("我已经搜了/已经查过了/已有结果")—— 陈述而非主动继续
        _state_desc = ("我已经", "我已", "已经搜索", "已经查", "已经找到", "已经看了",
                       "我已经找", "已经研究", "i already", "i've already", "already searched",
                       "already found", "we already", "已经完成")
        for s in _state_desc:
            if s in t.lower():
                return False
        # 通用模式: "再/继续/接着" + 具体动作(至少2字符) → 高置信继续意图
        _generic_continue = re.compile(
            r"(?:再|继续|接着|再多|再来)\s*[^。！？!?\n]{0,20}?(?:看|查|找|搜|试|做|写|读|执行|补充|分析|研究|探索|搜索|更新|深入)",
            re.IGNORECASE,
        )
        if _generic_continue.search(t) and not any(d in t.lower() for d in ("但已经", "不过已经", "已经够了", "不需要", "不用继续")):
            return True
        for c in _continue:
            if c in t.lower():
                if any(d in t.lower() for d in ("但已经", "不过已经", "已经够了", "不需要")):
                    return False
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
        stop_count = sum(1 for o in trace.observations if "SYSTEM SKIP" in o)
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
