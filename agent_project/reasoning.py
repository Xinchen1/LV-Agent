"""
Reasoning Engine - Deep thinking framework with adaptive control.

This module now delegates the actual think-act-observe loop to the unified
ExecutionEngine, while keeping the public ReasoningEngine.reason() API
unchanged for backward compatibility.
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .checkpoint import CheckpointManager
from .execution_engine import ExecutionContext, ExecutionEngine
from .policies import (
    CoTPolicy,
    DirectPolicy,
    ReActPolicy,
    SuperAgentPolicy,
    VerifyPolicy,
)


# ============ Enums and Dataclasses ============

class ReasoningStrategy(Enum):
    """Available reasoning strategies"""

    CHAIN_OF_THOUGHT = "cot"
    REACT = "react"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHTS = "tot"
    MONTE_CARLO = "mcts"
    VERIFICATION = "verify"
    ZERO_SHOT = "zero_shot"
    SUPER_AGENT = "super_agent"


@dataclass
class ThoughtStep:
    """A single step in the reasoning process"""

    step_number: int
    content: str
    confidence: float = 0.8
    requires_tool: bool = False
    suggested_tool: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process"""

    task: str
    strategy: ReasoningStrategy
    steps: List[ThoughtStep] = field(default_factory=list)
    total_loops: int = 0
    outer_loops: int = 0
    tools_used: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    quality_score: float = 0.0
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ThoughtStep):
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "strategy": self.strategy.value,
            "steps": [
                {
                    "step_number": s.step_number,
                    "content": s.content[:200],
                    "confidence": s.confidence,
                    "requires_tool": s.requires_tool,
                    "suggested_tool": s.suggested_tool,
                }
                for s in self.steps
            ],
            "total_loops": self.total_loops,
            "outer_loops": self.outer_loops,
            "tools_used": self.tools_used,
            "final_answer": self.final_answer[:200] if self.final_answer else None,
            "success": self.success,
            "quality_score": self.quality_score,
            "duration_ms": self.duration_ms,
        }


class LoopController:
    """
    Adaptive loop depth controller.
    Dynamically adjusts thinking depth based on task complexity and progress.
    """

    def __init__(
        self,
        min_loops: int = 2,
        max_loops: int = 16,
        default_loops: int = 2,
        convergence_threshold: float = 0.95,
    ):
        self.min_loops = min_loops
        self.max_loops = max_loops
        self.default_loops = default_loops
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger("LoopController")

    def determine_loops(
        self,
        task: str,
        strategy: ReasoningStrategy,
        similarity: Optional[float] = None,
        estimated_complexity: float = 0.5,
    ) -> int:
        base = self.default_loops
        strategy_multipliers = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 1.0,
            ReasoningStrategy.REACT: 1.0,
            ReasoningStrategy.SELF_CONSISTENCY: 1.2,
            ReasoningStrategy.TREE_OF_THOUGHTS: 1.5,
            ReasoningStrategy.MONTE_CARLO: 2.0,
            ReasoningStrategy.VERIFICATION: 1.1,
            ReasoningStrategy.ZERO_SHOT: 0.5,
            ReasoningStrategy.SUPER_AGENT: 1.2,
        }
        multiplier = strategy_multipliers.get(strategy, 1.0)
        complexity_multiplier = 0.5 + estimated_complexity
        if similarity is not None and similarity > 0.8:
            similarity_multiplier = 0.8
        elif similarity is not None and similarity < 0.3:
            similarity_multiplier = 1.1
        else:
            similarity_multiplier = 1.0

        loops = base * multiplier * complexity_multiplier * similarity_multiplier
        loops = int(round(loops))
        loops = max(self.min_loops, min(loops, self.max_loops))
        self.logger.debug(
            f"Loop calculation: base={base} * strat={multiplier:.2f} "
            f"* comp={complexity_multiplier:.2f} * sim={similarity_multiplier:.2f} = {loops}"
        )
        return loops


class ReasoningEngine:
    """
    Core reasoning engine supporting multiple strategies.

    Internally uses the unified ExecutionEngine so the think-act-observe loop
    is maintained in one place. The public API remains unchanged.
    """

    def __init__(
        self,
        model_backend: Any,
        tokenizer: Any,
        config: Any,
        loop_controller: Optional[LoopController] = None,
        harness_kernel: Optional[Any] = None,
        per_turn_cache: Optional[Any] = None,
    ):
        self.model = model_backend
        self.tokenizer = tokenizer
        self.config = config

        if hasattr(config, "loop_controller_min_loops"):
            min_loops = config.loop_controller_min_loops
            max_loops = config.loop_controller_max_loops
            default_loops = config.loop_controller_default_loops
        else:
            min_loops = getattr(config, "default_thinking_loops", 4)
            max_loops = getattr(config, "max_thinking_loops", 32)
            default_loops = getattr(config, "default_thinking_loops", 8)

        self.loop_controller = loop_controller or LoopController(
            min_loops=min_loops,
            max_loops=max_loops,
            default_loops=default_loops,
        )
        self.logger = logging.getLogger("ReasoningEngine")
        self.execution_engine = ExecutionEngine(
            model_backend=model_backend,
            config=config,
            harness_kernel=harness_kernel,
            per_turn_cache=per_turn_cache,
        )

    def reason(
        self,
        task: str,
        available_tools: Optional[Dict[str, Any]] = None,
        strategy: ReasoningStrategy = ReasoningStrategy.REACT,
        custom_loops: Optional[int] = None,
        similarity_score: Optional[float] = None,
        context: Optional[str] = None,
        stream_callback: Optional[Callable] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        code_mode: bool = False,
        plan: Optional[Any] = None,
    ) -> ReasoningTrace:
        self.logger.info(f"Reasoning: strategy={strategy.value}, task={task[:50]}...")

        trace = ReasoningTrace(
            task=task,
            strategy=strategy,
            metadata={"context": context, "code_mode": code_mode},
        )
        start_time = time.time()

        try:
            if custom_loops is not None:
                n_loops = custom_loops
            else:
                estimated_complexity = self._estimate_task_complexity(task)
                n_loops = self.loop_controller.determine_loops(
                    task, strategy, similarity_score, estimated_complexity
                )
                # 动态预算保证: 明确的多步任务(搜索+写入/读取+修改等)需要足够步骤,
                # 防止因预算不足在多步任务中提前截断(未完成就结束)。
                multi_step = any(k in task for k in
                    ("搜索", "查找", "搜", "写", "创建", "修改", "新建", "下载",
                     "search", "find", "write", "create", "modify", "download",
                     "分析", "总结", "调研", "研究",
                     "加", "加上", "加入", "添加", "增加", "优化", "重构", "修复", "实现"))
                if multi_step and n_loops < 6:
                    n_loops = 6
            trace.total_loops = n_loops

            policy = self._create_policy(strategy)

            ctx = ExecutionContext(
                task=task,
                available_tools=available_tools,
                config=self.config,
                max_steps=n_loops,
                stream_callback=stream_callback,
                token_callback=token_callback,
                code_mode=code_mode,
                extra_context=context or "",
            )

            # 规划驱动: 将 plan 节点透传给执行上下文, 使循环可按节点完成度提前收敛。
            if plan is not None and getattr(plan, "nodes", None):
                ctx.plan_node_ids = [n.id for n in plan.nodes.values()
                                     if getattr(n, "id", None)]
                # 同时保留 live Plan 对象, 供运行时动态重规划修改 DAG。
                ctx.plan = plan

            # SuperAgentPolicy handles its own meta-loop;
            # 搜索型策略(SELF_CONSISTENCY / TREE_OF_THOUGHTS / MONTE_CARLO)走专门实现
            if isinstance(policy, SuperAgentPolicy):
                exec_trace = policy.run_meta_loop(self.execution_engine, ctx)
            elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
                exec_trace = self._reason_self_consistency(ctx)
            elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
                exec_trace = self._reason_tree_of_thoughts(ctx)
            elif strategy == ReasoningStrategy.MONTE_CARLO:
                exec_trace = self._reason_best_of_n(ctx)
            else:
                exec_trace = self.execution_engine.run(policy, ctx)

            trace = self._adapt_trace(exec_trace, trace)
            trace.success = True

        except Exception as e:
            self.logger.error(f"Reasoning failed: {type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
            trace.success = False
            trace.metadata["error"] = f"{type(e).__name__}: {e}"
            try:
                rollback_info = CheckpointManager().rollback_latest()
                if rollback_info:
                    trace.metadata["rollback"] = rollback_info
            except Exception as rollback_err:
                self.logger.error(f"rollback failed: {rollback_err}")
        finally:
            trace.duration_ms = int((time.time() - start_time) * 1000)

        return trace

    def _create_policy(self, strategy: ReasoningStrategy):
        if strategy == ReasoningStrategy.REACT:
            return ReActPolicy()
        if strategy == ReasoningStrategy.SUPER_AGENT:
            return SuperAgentPolicy(self.model)
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            return CoTPolicy()
        if strategy == ReasoningStrategy.VERIFICATION:
            return VerifyPolicy()
        if strategy == ReasoningStrategy.ZERO_SHOT:
            return DirectPolicy()
        # 其余策略(含搜索型 TOT/MCTS/SC, 它们在 reason() 中单独分发)统一用 ReAct
        return ReActPolicy()

    def _clone_ctx(self, ctx: ExecutionContext) -> ExecutionContext:
        """克隆执行上下文, 每次 rollout 使用独立状态, 避免共享 steps/observations."""
        return ExecutionContext(
            task=ctx.task,
            available_tools=ctx.available_tools,
            config=ctx.config,
            max_steps=ctx.max_steps,
            stream_callback=ctx.stream_callback,
            token_callback=ctx.token_callback,
            code_mode=ctx.code_mode,
            extra_context=ctx.extra_context,
            history_context=getattr(ctx, "history_context", None),
        )

    # ===================== Self-Consistency =====================

    def _run_multiple_rollouts(
        self,
        ctx: ExecutionContext,
        n: int,
        policy,
        post_process,
        status_msg: str,
    ) -> Any:
        """Run N independent ReAct rollouts with cloned contexts, then apply post-processing.

        每个 rollout 使用独立的上下文状态(避免共享 steps/observations), 中间 rollout 不流式输出.
        """
        traces = []
        for i in range(n):
            sub = self._clone_ctx(ctx)
            if i < n - 1:
                sub.stream_callback = None
            traces.append(self.execution_engine.run(policy, sub))
        if ctx.stream_callback:
            ctx.stream_callback("status", status_msg.format(n=n))
        return post_process(traces)

    # ===================== Self-Consistency =====================

    def _reason_self_consistency(self, ctx: ExecutionContext, n_samples: int = 3) -> Any:
        """Self-Consistency: 同一任务独立采样多次, 对最终答案做多数投票.

        每个 sample 是一次完整的 ReAct 执行(含工具). 中间采样不流式输出,
        投票胜出的答案平票时取质量分最高者. 返回胜出的 ExecutionTrace.
        """
        return self._run_multiple_rollouts(
            ctx,
            n_samples,
            ReActPolicy(),
            self._vote_by_answer,
            "self-consistency: {n} samples, majority vote",
        )

    def _vote_by_answer(self, traces: List[Any]) -> Any:
        """按最终答案多数投票, 平票取 quality_score 最高者."""
        answers = [t.final_answer for t in traces if t.final_answer]
        if not answers:
            return max(traces, key=lambda t: t.quality_score)
        counts = Counter(a.strip() for a in answers)
        top_answer = counts.most_common(1)[0][0]
        winners = [t for t in traces if t.final_answer and t.final_answer.strip() == top_answer]
        return max(winners, key=lambda t: t.quality_score)

    # ===================== Best-of-N (config label: "mcts") =====================
    # 说明: 这不是真正的蒙特卡洛树搜索(无树扩展 / 反向传播 / UCT 选择),
    # 而是 N 次独立完整 ReAct rollout 取 quality_score 最高者。

    def _reason_best_of_n(self, ctx: ExecutionContext, n_rollouts: int = 3) -> Any:
        """Best-of-N: 运行 N 次独立完整 ReAct, 返回 quality_score 最高的轨迹.

        每个 rollout 是一次完整 ReAct 执行(带工具调用), 以 trace.quality_score
        作为该路径的回报, 返回回报最高的轨迹. 中间 rollout 不流式输出.
        """
        return self._run_multiple_rollouts(
            ctx,
            n_rollouts,
            ReActPolicy(),
            lambda traces: max(traces, key=lambda t: t.quality_score),
            "monte-carlo: {n} rollouts, best score={best:.2f}",
        )

    # ===================== Tree-of-Thoughts =====================

    def _reason_tree_of_thoughts(self, ctx: ExecutionContext, beam_width: int = 2, branches: int = 2, max_depth: int = 3) -> Any:
        """Tree-of-Thoughts 束搜索: 生成候选思路 -> 自评分 -> 保留 Top-B -> 引导执行.

        每层: 对束内每条路径用 LLM 生成 branches 个不同下一步思路,
        再用自批判 prompt 打分(0-1), 保留得分最高的 beam_width 条路径;
        出现 final answer 或达到 max_depth 后, 用最佳路径作为 extra_context
        执行一次 ReAct 收敛出最终答案.
        """
        nodes = [("", 1.0)]
        best_path = ""
        reached = 0
        for depth in range(max_depth):
            candidates = []
            for path, _score in nodes:
                for text in self._tot_generate_branches(ctx, path, branches):
                    score = self._tot_score_thought(ctx, path, text)
                    candidates.append((path, text, score))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[2], reverse=True)
            keep = candidates[:beam_width]
            nodes = [(p + "\n" + t, s) for p, t, s in keep]
            reached = depth + 1
            for p, text, _s in keep:
                if re.search(r"(?:final answer|最终答案|总结)", text, re.IGNORECASE):
                    best_path = p
                    break
            if best_path:
                break
        if not best_path and nodes:
            best_path = nodes[0][0]

        sub = self._clone_ctx(ctx)
        sub.extra_context = (
            f"{ctx.extra_context or ''}\n\n## Tree-of-Thoughts 探索路径(用于指导最终执行):\n"
            f"{best_path[:2000]}"
        )
        if ctx.stream_callback:
            ctx.stream_callback("status", f"tree-of-thoughts: beam={beam_width} branches={branches} depth={reached}")
        return self.execution_engine.run(ReActPolicy(), sub)

    def _tot_generate_branches(self, ctx: ExecutionContext, path: str, branches: int) -> List[str]:
        """让 LLM 基于当前路径生成 branches 个不同的下一步思路."""
        prompt = (
            "你是解决任务的推理器。基于现有推理路径, 给出 "
            f"{branches} 个不同的下一步思考方向(每个都是一种独立思路)。\n\n"
            f"任务: {ctx.task}\n\n"
            f"已有推理:\n{path or '(尚无)'}\n\n"
            f"请输出 {branches} 个编号的候选思路(1. 2. ...):"
        )
        try:
            raw = self.model.generate(prompt, n_loops=1, temperature=0.7, max_tokens=512)
        except Exception as e:
            self.logger.warning(f"ToT branch generation failed: {e}")
            return []
        texts = [ln.strip() for ln in raw.splitlines() if ln.strip() and re.match(r"^\s*\d+[.、]", ln.strip())]
        return texts[:branches] or [raw.strip()[:300]]

    def _tot_score_thought(self, ctx: ExecutionContext, path: str, text: str) -> float:
        """用自批判 prompt 给候选思路打分(0.0-1.0)."""
        prompt = (
            "评估下面这个下一步思路对解决任务的价值, 只输出 0.0 到 1.0 之间的一个数字。\n\n"
            f"任务: {ctx.task}\n\n已有推理:\n{path or '(尚无)'}\n\n候选思路:\n{text}\n\n分数:"
        )
        try:
            raw = self.model.generate(prompt, n_loops=1, temperature=0.2, max_tokens=8)
        except Exception:
            return 0.5
        m = re.search(r"\d+(?:\.\d+)?", raw or "")
        if not m:
            return 0.5
        score = float(m.group(0))
        return min(max(score, 0.0), 1.0)

    def _adapt_trace(self, exec_trace: Any, trace: ReasoningTrace) -> ReasoningTrace:
        trace.final_answer = exec_trace.final_answer
        trace.tools_used = exec_trace.tools_used
        trace.observations = exec_trace.observations
        trace.outer_loops = len(exec_trace.steps)
        trace.quality_score = exec_trace.quality_score
        trace.metadata.update(exec_trace.metadata)

        for i, step in enumerate(exec_trace.steps, start=1):
            thought = step.reasoning
            tool_calls = step.tool_calls
            suggested_tool = None
            tool_arguments = None
            if tool_calls:
                suggested_tool = tool_calls[0].tool_name
                tool_arguments = tool_calls[0].arguments
            # 动态置信度: 基于工具成功/结论关键词/步骤质量(取代硬编码 0.8)
            step_conf = 0.8
            step_obs = list(getattr(step, "observations", []) or [])
            if step_obs:
                joined = "\n".join(step_obs).lower()
                if any(k in joined for k in ("error", "timed out", "system stop", "失败", "错误", "blocked")):
                    step_conf -= 0.25  # 工具失败/错误 → 降置信
                elif any(k in joined for k in ("done", "success", "ok", "完成", "成功", "found", "存在")):
                    step_conf += 0.1   # 工具成功 → 升置信
            thought_low = (thought or "").lower()
            # 结论/总结性关键词 → 置信度上调(说明推理收敛)
            if re.search(r"(因此|所以|综上所述|结论是|最终|综上|总结|因此可以确定|这意味着)", thought_low):
                step_conf += 0.1
            # 最后一步是最终答案 → 置信度参考整体质量分
            if i == len(exec_trace.steps) and exec_trace.final_answer:
                step_conf = step_conf * 0.6 + max(exec_trace.quality_score, 0.0) * 0.4
            step_conf = max(0.0, min(1.0, round(step_conf, 2)))
            ts = ThoughtStep(
                step_number=i,
                content=thought[:1000] or step.output[:300],
                confidence=step_conf,
                requires_tool=bool(tool_calls),
                suggested_tool=suggested_tool,
                tool_arguments=tool_arguments,
            )
            trace.steps.append(ts)

        return trace

    @staticmethod
    def _estimate_task_complexity(task: str) -> float:
        words = task.split()
        complexity = 0.0
        if len(words) > 50:
            complexity += 0.3
        elif len(words) > 20:
            complexity += 0.15
        complex_keywords = [
            "calculate", "compute", "derive", "analyze", "compare",
            "synthesize", "design", "optimize", "implement", "deploy",
            "research", "investigate", "critique", "evaluate",
        ]
        # 中文复杂任务关键词
        cn_complex_keywords = [
            "分析", "剖析", "解析", "比较", "对比", "评估", "评价",
            "设计", "实现", "开发", "优化", "重构", "调试", "修复",
            "研究", "调研", "综合", "总结", "推导", "计算", "规划",
            "构建", "编写", "测试", "部署", "撰写", "归纳",
        ]
        task_lower = task.lower()
        matches = sum(1 for kw in complex_keywords if kw in task_lower)
        cn_matches = sum(1 for kw in cn_complex_keywords if kw in task)
        # 中文长句(>40字)也视为复杂
        if len(task) > 40:
            complexity += 0.15
        complexity += min(matches * 0.1, 0.4)
        complexity += min(cn_matches * 0.1, 0.4)
        if any(word in task_lower for word in ["search", "find", "lookup", "file", "api", "web", "项目", "文件夹", "代码", "数据库", "文件"]):
            complexity += 0.2
        return min(complexity, 1.0)
