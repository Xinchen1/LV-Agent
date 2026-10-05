"""回归测试: 2026-10-05 全量修复.

覆盖:
- 生成退化(原句复读)检测与折叠
- 阶段2纯文本空转熔断(此前绕过无进展检查, 导致同一句话循环几十次)
- 工具参数按 execute() 签名过滤(模型多生成 description 等字段不再 TypeError)
- ConvergenceChecker 重复调用 key 与 call_counts 写入格式一致
- 重复调用≥3次时显式输出"无进展"停止提示
"""

import json
import logging

from agent_project.config import AgentConfig
from agent_project.execution_engine import (
    ConvergenceChecker,
    ExecutionContext,
    ExecutionEngine,
    ToolCallRequest,
    PolicyOutput,
)
from agent_project.policies import ReActPolicy
from agent_project.tools import TOOLS_REGISTRY, ToolResult


# ---------------------------------------------------------------------------
# 生成退化检测
# ---------------------------------------------------------------------------

def test_degenerate_repetition_single_line():
    lines = "\n".join(["我需要查看桌面目录的详细内容来查找截图文件。让我执行一次命令。"] * 40)
    assert ExecutionEngine._is_degenerate_repetition(lines)
    collapsed = ExecutionEngine._collapse_repetition(lines)
    assert collapsed.count("我需要查看桌面目录") == 2, "折叠后应只保留前2次"


def test_degenerate_repetition_alternating_pair():
    block = "\n".join(["让我查看桌面目录的详细内容。", "我需要执行命令。"] * 20)
    assert ExecutionEngine._is_degenerate_repetition(block)


def test_normal_output_not_degenerate():
    normal = "\n".join(f"第{i}步: 读取文件 config-{i}.yaml 并解析。" for i in range(12))
    assert not ExecutionEngine._is_degenerate_repetition(normal)
    assert ExecutionEngine._collapse_repetition(normal) == normal.strip()


# ---------------------------------------------------------------------------
# 工具参数签名过滤
# ---------------------------------------------------------------------------

def test_filter_args_drops_unknown_description():
    from agent_project.execution_engine import ToolExecutor
    from agent_project.tools.bash_exec import BashExecTool
    te = ToolExecutor()
    tool = BashExecTool()
    args = {"command": "ls -la", "description": "列出文件", "cwd": ""}
    kept = te._filter_args_to_schema("bash_exec", tool, args)
    assert "description" not in kept, "schema 外参数应被剔除"
    assert kept["command"] == "ls -la" and "cwd" in kept


def test_execute_one_tolerates_extra_description_kwarg():
    """模型给 bash_exec 额外传 description 不应报废整次调用."""
    from agent_project.execution_engine import ToolExecutor
    te = ToolExecutor(harness_kernel=None)
    te.logger = logging.getLogger("test")
    call = ToolCallRequest(
        tool_name="bash_exec",
        arguments={"command": "echo ok", "description": "测试用"},
        display_key="bash_exec",
    )
    ok, obs = te._execute_one(call, ExecutionContext(
        task="t", available_tools={}, config=AgentConfig(),
    ))
    assert ok, f"应执行成功, 实际: {obs}"
    assert "ok" in obs


def test_drop_unexpected_kwargs_from_type_error():
    from agent_project.execution_engine import ToolExecutor
    err = TypeError("BashExecTool.execute() got an unexpected keyword argument 'description'")
    out = ToolExecutor._drop_unexpected_kwargs({"command": "ls", "description": "x"}, err)
    assert out == {"command": "ls"}
    assert ToolExecutor._drop_unexpected_kwargs({"command": "ls"}, err) is None


# ---------------------------------------------------------------------------
# ConvergenceChecker key 一致性
# ---------------------------------------------------------------------------

def test_convergence_dedup_key_matches_call_counts():
    """call_counts 用 json.dumps({"name","args"}) 写入, 查询必须同格式才能命中."""
    cc = ConvergenceChecker(min_steps=2)
    ctx = ExecutionContext(task="t", available_tools={}, config=AgentConfig(), max_steps=8)
    call = ToolCallRequest(tool_name="web_search", arguments={"query": "x"})
    key = json.dumps({"name": "web_search", "args": {"query": "x"}},
                     sort_keys=True, ensure_ascii=False)
    ctx.call_counts[key] = 2  # 旧格式查表永远 miss → 死代码; 新格式应能命中
    out = PolicyOutput(reasoning="r", tool_calls=[call], final_answer=None, done=False)
    assert not cc.should_stop(ctx, out, 3), "计数=2 尚未到阈值3, 应允许重试"

    ctx.call_counts[key] = 3
    assert cc.should_stop(ctx, out, 3), "同一调用计数≥3 应收敛"


# ---------------------------------------------------------------------------
# 阶段2纯文本空转熔断
# ---------------------------------------------------------------------------

def test_text_only_loop_stops_within_three_steps():
    """模型连续只输出文本(无工具/无答案) → 3步内必须熔断, 不再无限催促."""
    cfg = AgentConfig()

    class FB_TextOnly:
        def __init__(self):
            self.c = 0

        def generate(self, prompt, **kw):
            self.c += 1
            return "让我查看桌面目录的详细内容来查找截图文件。让我执行一次命令。"

    eng = ExecutionEngine(model_backend=FB_TextOnly(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="找截图", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                           config=cfg, max_steps=30)
    ctx.monitor_enabled = False
    tr = eng.run(ReActPolicy(), ctx)
    assert len(tr.steps) <= 3, f"纯文本空转应在≤3步熔断, 实际 {len(tr.steps)} 步"
    assert eng.model.c <= 4, f"不应反复催促模型, 实际调用 {eng.model.c} 次"


def test_repetitive_output_collapsed_then_stops():
    """复读输出应被折叠, 且整轮在少数步骤内收敛."""
    cfg = AgentConfig()

    class FB_Loop:
        def __init__(self):
            self.c = 0

        def generate(self, prompt, **kw):
            self.c += 1
            if self.c <= 2:
                return "\n".join(["我需要查看桌面目录的详细内容来查找截图文件。让我执行一次命令。"] * 40)
            return '{"final_answer": "未找到截图文件"}'

    eng = ExecutionEngine(model_backend=FB_Loop(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="清理截图", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                           config=cfg, max_steps=20)
    ctx.monitor_enabled = False
    tr = eng.run(ReActPolicy(), ctx)
    assert tr.final_answer, "应基于兜底生成最终答案"
    assert len(tr.steps) <= 4, f"复读应在少数步骤内收敛, 实际 {len(tr.steps)} 步"


# ---------------------------------------------------------------------------
# 重复调用≥3次: 显式"无进展"提示
# ---------------------------------------------------------------------------

def _install_fake_web_search(behavior):
    from agent_project.tools import TOOLS_REGISTRY as _reg

    class _Fake:
        name = "web_search"
        description = "fake"
        parameters = {"type": "object", "properties": {}}

        def execute(self, query="", **kw):
            if behavior == "fail":
                return ToolResult(success=False, output="", error="fake search failed")
            return ToolResult(success=True, output='[{"title": "ok"}]')

    old = _reg.get("web_search")
    _reg._tools["web_search"] = _Fake()

    def _restore():
        _reg._tools["web_search"] = old

    return _Fake(), _restore


def test_repeated_failing_call_stops_with_no_progress_status():
    cfg = AgentConfig()
    fake, restore = _install_fake_web_search("fail")
    try:
        class FB_Fail:
            def __init__(self):
                self.c = 0

            def generate(self, prompt, **kw):
                self.c += 1
                if self.c <= 12:
                    return '{"action": "web_search", "args": {"query": "fail"}}'
                return '{"final_answer": "done"}'

        status_msgs = []
        eng = ExecutionEngine(model_backend=FB_Fail(), config=cfg)
        eng.logger = logging.getLogger("test")
        ctx = ExecutionContext(task="搜", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                               config=cfg, max_steps=30)
        ctx.monitor_enabled = False
        ctx.stream_callback = lambda kind, text: status_msgs.append((kind, text))
        tr = eng.run(ReActPolicy(), ctx)
        assert len(tr.tools_used) < 12, f"应提前停止, 实际 {len(tr.tools_used)} 次"
        assert any(k == "status" and "无进展" in str(t) for k, t in status_msgs), \
            f"应有无进展停止提示: {status_msgs}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# 超时放弃的同参调用不重复起线程
# ---------------------------------------------------------------------------

def test_inflight_skip_prevents_duplicate_thread_pileup():
    import time as _time
    from agent_project.execution_engine import ToolExecutor

    te = ToolExecutor(max_workers=2, tool_timeout=0.3)
    te.logger = logging.getLogger("test")
    call = ToolCallRequest(tool_name="slow_tool", arguments={"x": 1}, display_key="slow")

    class _FakeSlowTool:
        def execute(self, x=0):
            _time.sleep(0.5)
            return ToolResult(success=True, output="late")

    TOOLS_REGISTRY._tools["slow_tool"] = _FakeSlowTool()
    try:
        # 第一次: 超时放弃(线程仍在后台)
        r1 = te.execute_calls([call], ExecutionContext(task="t", available_tools={}, config=AgentConfig()))
        assert any("timed out" in o for _c, o, _ok in r1)
        # 第二次(缓存命中 SYSTEM SKIP, 不再起新线程)
        r2 = te.execute_calls([call], ExecutionContext(task="t", available_tools={}, config=AgentConfig()))
        assert len(r2) == 1
        # 清理: 等后台线程结束
        _time.sleep(0.4)
    finally:
        TOOLS_REGISTRY._tools.pop("slow_tool", None)
