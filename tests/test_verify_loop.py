"""验证闭环测试: apply_diff/write 自动语法验证 + 失败定位 + python fallback."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.tools.file_ops import FileOpsTool


def make_tool():
    return FileOpsTool()


def test_apply_diff_auto_verify_ok():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "good.py")
    Path(py).write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    tool = make_tool()
    diff = (
        "<<<<<<< SEARCH\ndef hello():\n    return 'hi'\n"
        "=======\ndef hello():\n    return 'hello world'\n>>>>>>> REPLACE"
    )
    r = tool.execute(action="apply_diff", path=py, diff=diff)
    assert r.success, f"正常 diff 应成功: {r.error}"
    assert "syntax OK" in r.output.lower() or "Verify" not in r.output, r.output


def test_apply_diff_detects_syntax_error():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "bad.py")
    Path(py).write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    tool = make_tool()
    diff = (
        "<<<<<<< SEARCH\ndef hello():\n    return 'hi'\n"
        "=======\ndef broken(:\n    return 'x'\n>>>>>>> REPLACE"
    )
    r = tool.execute(action="apply_diff", path=py, diff=diff)
    assert r.success, "diff 应应用成功(语法错误作为 warning)"
    assert "SYNTAX VERIFICATION FAILED" in r.output.upper(), r.output


def test_apply_diff_not_found_has_hint():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "x.py")
    Path(py).write_text("import os\n\ndef main():\n    return os.getcwd()\n", encoding="utf-8")
    tool = make_tool()
    diff = (
        "<<<<<<< SEARCH\nthis line does not exist at all\n"
        "=======\nreplacement\n>>>>>>> REPLACE"
    )
    r = tool.execute(action="apply_diff", path=py, diff=diff)
    assert not r.success, "不匹配的 block 应失败"
    assert "not found" in (r.error or "").lower(), r.error
    assert "whitespace" in (r.error or "").lower(), "应有修复提示"


def test_write_auto_verify():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "new.py")
    tool = make_tool()
    r = tool.execute(action="write", path=py, content="def broken(:\n    pass\n")
    assert r.success
    assert "SYNTAX VERIFICATION FAILED" in r.output.upper(), r.output


def test_verify_action():
    td = tempfile.mkdtemp()
    good = os.path.join(td, "good.py")
    bad = os.path.join(td, "bad.py")
    Path(good).write_text("print(1)\n", encoding="utf-8")
    Path(bad).write_text("def broken(:\n", encoding="utf-8")
    tool = make_tool()
    rg = tool.execute(action="verify", path=good)
    assert rg.success, rg.error
    rb = tool.execute(action="verify", path=bad)
    assert not rb.success, "坏代码 verify 应失败"
    assert rb.error and "error" in rb.error.lower() or "invalid" in (rb.error or "").lower()


def test_python_fallback_verify_syntax():
    """后缀比较 bug 回归: .py 文件必须被正确识别."""
    tool = make_tool()
    td = tempfile.mkdtemp()
    bad = os.path.join(td, "bad.py")
    r = tool._python_verify_syntax(Path(bad), "def broken(:\n    pass\n")
    assert r, "坏 Python 应返回语法错误, 而不是空(后缀 lstrip bug)"
    good = os.path.join(td, "good.py")
    r2 = tool._python_verify_syntax(Path(good), "print(1)\n")
    assert r2 == "", "好代码应返回空"


def test_main_loop_json_final_answer_extraction():
    """主循环: 模型输出 {thoughts, final_answer} JSON 时, 应提取纯文本而非整个 JSON."""
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.policies import ReActPolicy
    from agent_project.policies import ToolCallParser

    # 直接验证 _route_react_output: final_answer JSON 提取
    policy = ReActPolicy()
    out = '{"thoughts": "done", "final_answer": "目录已创建成功"}'
    parsed = policy._route_react_output(out)
    assert parsed.final_answer == "目录已创建成功", f"应提取纯文本, 实际: {parsed.final_answer!r}"
    assert parsed.done, "应标记完成"

    # answer 字段变体
    out2 = '{"reasoning": "r", "answer": "好的, 完成了"}'
    parsed2 = policy._route_react_output(out2)
    assert parsed2.final_answer == "好的, 完成了"


def test_main_loop_full_execution_with_json_tool_calls():
    """主循环端到端: JSON action+args → 工具 → 观察 → JSON final_answer."""
    import tempfile, os
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY

    class FakeBackend:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, **kw):
            self.calls += 1
            if self.calls == 1:
                return '{"thoughts": "t", "action": "bash_exec", "args": {"command": "echo loop_ok"}}'
            if self.calls == 2:
                return '{"thoughts": "t", "tool_calls": [{"action": "bash_exec", "args": {"command": "echo done2"}}]}'
            return '{"thoughts": "t", "final_answer": "loop 执行完成"}'

    cfg = AgentConfig()
    eng = ExecutionEngine(model_backend=FakeBackend(), config=cfg)
    ctx = ExecutionContext(task="测试", available_tools=TOOLS_REGISTRY.get_tools_dict(), config=cfg, max_steps=6)
    trace = eng.run(ReActPolicy(), ctx)
    assert trace.success, "loop 应成功"
    assert trace.final_answer == "loop 执行完成", f"应提取纯文本 final_answer, 实际: {trace.final_answer!r}"
    assert len(trace.steps) >= 3, f"应有多次迭代, 实际 {len(trace.steps)}"
    assert "bash_exec" in trace.tools_used


def test_task_complexity_estimation_chinese():
    """中文复杂任务应获得更高的复杂度估算(动态 loop 调整)."""
    from agent_project.reasoning import LoopController, ReasoningEngine, ReasoningStrategy

    simple = ReasoningEngine._estimate_task_complexity("你好")
    complex_cn = ReasoningEngine._estimate_task_complexity("分析一下桌面那个文件夹,看看里面每个爬虫的用途和区别")
    assert simple < complex_cn, f"中文复杂任务应比简单问候复杂度高: {simple} vs {complex_cn}"

    lc = LoopController(min_loops=2, max_loops=16, default_loops=4)
    loops_simple = lc.determine_loops("你好", ReasoningStrategy.REACT, estimated_complexity=simple)
    loops_complex = lc.determine_loops("分析文件夹", ReasoningStrategy.REACT, estimated_complexity=complex_cn)
    assert loops_complex > loops_simple, f"复杂任务 loop 数应更多: {loops_simple} vs {loops_complex}"

    # 中文重构关键词
    c = ReasoningEngine._estimate_task_complexity("帮我重构一下这个项目的代码,优化性能并编写单元测试")
    assert c > 0.3, f"中文重构任务应有较高复杂度: {c}"


def test_file_modify_intent_routes_to_main_loop():
    """'给XX加功能'等修改意图应走主循环(simple=False)并识别为 file_ops 操作."""
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)
    a._method_cache = {}; a._code_mode_override = False; a._current_task = ""

    # 修改意图 → 不走 fast path
    for t in [
        "给贪吃蛇加一个变速功能",
        "给贪食蛇加入变速功能",
        "修改一下 snake_game.py",
        "在游戏里加音效",
    ]:
        assert not a._is_simple_query(t), f"{t!r} 不应走 fast path"
        cls = a._classify_intent(t)
        assert cls is not None, f"{t!r} 应被意图分类"
        assert cls[0] == "file_ops", f"{t!r} 应识别为 file_ops, 实际 {cls[0]}"

    # 带文件名 → 应 read 目标文件
    cls = a._classify_intent("修改一下 snake_game.py")
    assert cls[1].get("path") == "snake_game.py"
    assert cls[1].get("action") == "read"

    # 问候仍走 fast path
    assert a._is_simple_query("你好")
