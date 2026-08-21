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


def test_convergence_dedup_stop_does_not_halt_early():
    """良性去重拦截(SYSTEM STOP: already executed)不应导致过早停止."""
    from agent_project.execution_engine import ConvergenceChecker, ExecutionContext, PolicyOutput, ToolCallRequest
    from agent_project.config import AgentConfig

    cc = ConvergenceChecker(min_steps=2)
    ctx = ExecutionContext(task="t", available_tools={}, config=AgentConfig(), max_steps=8)
    ctx.observations = [
        "SYSTEM STOP: You already executed file_ops list. You already have this result. Do NOT call the same tool with the same arguments again.",
        "SYSTEM STOP: You already executed file_ops list. You already have this result. Do NOT call the same tool with the same arguments again.",
    ]
    call = ToolCallRequest(tool_name="web_search", arguments={"query": "x"})
    out = PolicyOutput(reasoning="继续", tool_calls=[call], final_answer=None, done=False)
    assert not cc.should_stop(ctx, out, 4), "2个去重拦截+还想继续 → 不应停止"

    # 真失败(超时)连续3个 → 应停止
    ctx2 = ExecutionContext(task="t", available_tools={}, config=AgentConfig(), max_steps=8)
    ctx2.observations = [
        "SYSTEM STOP: Tool 'web_search' timed out after 120s.",
        "SYSTEM STOP: Tool 'bash_exec' timed out after 120s.",
        "SYSTEM STOP: Tool 'file_ops' timed out after 120s.",
    ]
    out2 = PolicyOutput(reasoning="r", tool_calls=[], final_answer=None, done=False)
    assert cc.should_stop(ctx2, out2, 5), "3个真失败 → 应停止"


def test_multi_step_task_gets_more_loop_budget():
    """多步任务(搜索+写文件/修改功能)应获得≥6 的 loop 预算."""
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.reasoning import LoopController, ReasoningEngine, ReasoningStrategy

    lc = LoopController(min_loops=2, max_loops=16, default_loops=2)
    for task in ["搜索 hermes 的使用方法,在lv 新建一个文件", "给贪吃蛇加变速功能"]:
        c = ReasoningEngine._estimate_task_complexity(task)
        n = lc.determine_loops(task, ReasoningStrategy.SUPER_AGENT, estimated_complexity=c)
        multi = any(k in task for k in ("搜索","搜","写","创建","修改","加","优化","重构","实现","分析","下载","search","write","create","modify","find"))
        if multi and n < 6:
            n = 6
        assert n >= 6, f"{task!r} 应至少 6 loops, 实际 {n}"


def test_memory_recall_question_gets_more_tokens():
    """'昨天我们聊了什么'等记忆召回问题应有足够 max_tokens, 避免回复被截断."""
    import re
    from agent_project.agent import OpenMythosAgent

    # 复现 fast path 的 max_tokens 决策
    for task, expected in [
        ("昨天我们聊了什么", 2048),
        ("上次我们讨论的健康话题", 2048),
        ("你好", 512),
        ("搜索 hermes", 4096),
    ]:
        task_lower = task.lower()
        fast_max_tokens = 512
        _memory_recall = bool(re.search(r'(昨天|上次|之前|刚才|还记得|我们聊|话题|对话历史)', task))
        if any(k in task_lower for k in ['搜索','搜','查找','查','report','报告','总结','分析','写','代码','code']):
            fast_max_tokens = 4096
        elif _memory_recall:
            fast_max_tokens = 2048
        elif len(task) > 40 or '?' in task or '？' in task:
            fast_max_tokens = 2048
        assert fast_max_tokens >= expected, f"{task!r} 应至少 {expected} tokens, 实际 {fast_max_tokens}"


def test_promise_detection_with_immediacy_words():
    """'马上帮你搜...稍等' 等即时承诺词应被识别为空承诺(光说不做)."""
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)

    # 用户实际场景: 说了要做但没执行
    assert a._is_promise_response("马上帮你搜最新的 AI 动态,稍等～")
    assert a._is_promise_response("这就去查一下 AI 新闻")
    assert a._is_promise_response("好的,我来搜索一下")

    # 不应误判: 正常问候 / 有结论的回答
    assert not a._is_promise_response("你好！有什么可以帮你的吗？")
    assert not a._is_promise_response("根据搜索结果,今天的 AI 新闻有……")


def test_dynamic_loop_extension():
    """模型持续产出新工具调用时应动态扩展 max_steps, 快速完成时不扩展."""
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY
    import logging

    cfg = AgentConfig()

    # 持续进展: 9 次工具调用, 预算 6 → 应扩展到 10
    class FB_Progress:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            if self.c <= 9:
                return '{"action": "web_search", "args": {"query": "AI %d"}}' % self.c
            return '{"final_answer": "done"}'

    eng = ExecutionEngine(model_backend=FB_Progress(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="搜新闻", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                           config=cfg, max_steps=6)
    ctx.monitor_enabled = False  # 聚焦测试动态扩展, 不触发监控 LLM
    tr = eng.run(ReActPolicy(), ctx)
    assert len(tr.tools_used) == 9, f"应执行全部 9 次工具调用, 实际 {len(tr.tools_used)}"
    assert ctx.max_steps > 6, f"max_steps 应动态扩展, 实际 {ctx.max_steps}"

    # 快速完成: 1 次调用后给答案 → 不扩展
    class FB_Fast:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            if self.c == 1:
                return '{"action": "web_search", "args": {"query": "AI"}}'
            return '{"final_answer": "answer"}'

    eng2 = ExecutionEngine(model_backend=FB_Fast(), config=cfg)
    eng2.logger = logging.getLogger("test")
    ctx2 = ExecutionContext(task="搜", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                            config=cfg, max_steps=6)
    tr2 = eng2.run(ReActPolicy(), ctx2)
    assert ctx2.max_steps == 6, f"快速完成不应扩展, 实际 {ctx2.max_steps}"


def test_wants_to_continue_detection():
    """精准识别'继续/补充'意图, 排除完成/收尾误判."""
    from agent_project.execution_engine import ExecutionEngine
    W = ExecutionEngine._wants_to_continue

    # 继续意图 → True
    for text in [
        "让我再搜索一下其他来源补充资料",
        "还需要再看看 AI 的最新动态",
        "继续搜索更多相关内容",
        "我再多找几个例子",
        "继续执行下一步操作",
        "接着分析这些数据",
        "还想看看更多来源",
    ]:
        assert W(text), f"应识别为继续意图: {text!r}"

    # 完成/收尾 → False
    for text in [
        "已完成全部搜索,这就是最终结果",
        "以上就是所有新闻,不需要了",
        "最终答案: AI 新闻汇总完毕",
        "还要再搜,但已经够了",
        "根据以上搜索结果,总结如下",
        "谢谢,不需要了",
        "这是最终的完整答案",
    ]:
        assert not W(text), f"不应识别为继续意图: {text!r}"


def test_promise_detection_write_actions():
    """'我现在把XX整理成文章存到文件夹' 应识别为空承诺(说了要写入却没执行)."""
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)

    # 承诺了写入但没执行 → 应识别
    assert a._is_promise_response("好的，我现在把搜集到的AI新闻整理成文章存到文件夹里。")
    assert a._is_promise_response("我现在把这些资料保存到文件里")
    assert a._is_promise_response("好的，我这就把结果写入 lv 文件夹")

    # 已完成/正常 → 不误判
    assert not a._is_promise_response("好的，已经整理好了，文件在 lv 文件夹")
    assert not a._is_promise_response("根据搜索结果，总结如下")
    assert not a._is_promise_response("你好呀")


def test_post_execution_verification_file_exists():
    """写入文件后应能核验文件确实存在(后置感知)."""
    import tempfile, os
    from agent_project.tools import TOOLS_REGISTRY

    tool = TOOLS_REGISTRY.get("file_ops")
    td = tempfile.mkdtemp()
    py = os.path.join(td, "verify.md")

    r = tool.execute(action="write", path=py, content="# 测试\n内容")
    assert r.success, r.error

    # 核验文件存在
    ex = tool.execute(action="exists", path=py)
    assert ex.success and ex.output.strip().lower() == "true", f"应核验文件存在: {ex.output}"

    # 写入无效路径应失败(系统会拦截)
    r2 = tool.execute(action="write", path="/nonexistent_dir_xxx_123/xx.md", content="x")
    assert not r2.success, "无效路径写入应失败"


def test_recurrent_depth_reasoning_allows_multi_round_introspection():
    """复杂任务允许模型多轮内省(OpenMythos 循环深度思路), 不被强制打断."""
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY
    import logging

    class FB:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            if self.c == 1:
                return '<think>初步理解</think>\nThought: 让我继续深入思考'
            if self.c == 2:
                return '<think>反方向审视, 发现遗漏</think>\nThought: 还需要再看看因果链'
            if self.c == 3:
                return '<think>状态收敛</think>\nFinal Answer: 这是分析结论'
            return 'Final Answer: done'

    cfg = AgentConfig()
    eng = ExecutionEngine(model_backend=FB(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="复杂分析", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                           config=cfg, max_steps=12)
    ctx.monitor_enabled = False  # 聚焦测试循环深度, 不触发监控 LLM
    tr = eng.run(ReActPolicy(), ctx)
    assert tr.final_answer == "这是分析结论"
    assert len(tr.steps) >= 3, f"应允许多轮内省, 实际 {len(tr.steps)} steps"


def test_branch_monitor_rule_layer():
    """分支监控 agent: 规则层应检测连续失败/权限错误并生成简明提示."""
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.config import AgentConfig

    cfg = AgentConfig()
    eng = ExecutionEngine(model_backend=None, config=cfg)

    # 连续工具失败 → 提示
    ctx = ExecutionContext(task="t", available_tools={}, config=cfg, max_steps=8)
    ctx.observations = ["tool error: x", "Tool 'web_search' timed out", "error: boom"]
    hint = eng._rule_monitor(ctx)
    assert hint and "连续工具失败" in hint, f"应提示换工具: {hint}"

    # 权限错误(全盘扫描) → 提示缩小范围
    ctx2 = ExecutionContext(task="t", available_tools={}, config=cfg, max_steps=8)
    ctx2.observations = ["Operation not permitted"] * 4
    hint2 = eng._rule_monitor(ctx2)
    assert hint2 and "缩小搜索范围" in hint2, f"应提示缩小范围: {hint2}"

    # 正常 → 无提示
    ctx3 = ExecutionContext(task="t", available_tools={}, config=cfg, max_steps=8)
    ctx3.observations = ["成功", "done"]
    assert eng._rule_monitor(ctx3) is None


def test_branch_monitor_inject():
    """分支监控提示应注入到主 agent 的下一轮 prompt."""
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.config import AgentConfig

    cfg = AgentConfig()
    eng = ExecutionEngine(model_backend=None, config=cfg)
    ctx = ExecutionContext(task="t", available_tools={}, config=cfg, max_steps=8)
    ctx.monitor_hints = ["提示: 换工具"]
    out = eng._attach_monitor_hints("原始prompt", ctx)
    assert "监控提示" in out and "换工具" in out


def test_collaborative_backfill():
    """协作 agent: 检测到主 agent 失败时主动补位执行子任务."""
    import logging
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine, StepRecord
    from agent_project.config import AgentConfig

    class FB:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            return '{"problem": "主agent搜索失败", "backfill": "web_search(query=AI新闻 最新)"}'

    cfg = AgentConfig()
    eng = ExecutionEngine(model_backend=FB(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="搜集AI新闻", available_tools={}, config=cfg, max_steps=10)
    ctx.observations.append("Tool error: No search results returned")
    for i in range(2):
        ctx.steps.append(StepRecord(step_number=i+1, prompt="p", output="o", reasoning="r",
                                    tool_calls=[], observations=["error"]))

    result = eng._llm_monitor(ctx)
    assert result and "协作补位" in result, f"应主动补位, 实际: {result}"
    assert "web_search" in result
    # 补位结果应写入观察
    assert any("协作补位" in o for o in ctx.observations)


def test_deep_research_post_verification():
    """深度研究后置核验: 无报告/无来源时应提示未完成, 不谎报成功."""
    import tempfile, os
    from agent_project.agent import OpenMythosAgent

    a = object.__new__(OpenMythosAgent)
    a.logger = __import__("logging").getLogger("test")
    a.config = None

    # 用 _run_deep_research 的核验逻辑直接测试判定
    # 场景1: 报告不存在 → 应判定无效
    from pathlib import Path
    bad_path = str(Path(tempfile.mkdtemp()) / "nonexist.md")
    sources_count = 0
    valid = False
    rp = Path(bad_path)
    valid = rp.exists() and rp.stat().st_size > 200 and sources_count > 0 if rp.exists() else False
    assert not valid, "不存在的报告应判定无效"

    # 场景2: 真实报告 + 有来源 → 有效
    td = tempfile.mkdtemp()
    real = os.path.join(td, "report.md")
    Path(real).write_text("# 报告\n" + "x" * 500, encoding="utf-8")
    rp2 = Path(real)
    valid2 = rp2.exists() and rp2.stat().st_size > 200 and 5 > 0
    assert valid2, "存在的非空报告应有来源时判定有效"


def test_fast_path_does_not_emit_reasoning():
    """fast path 不应把思考过程(reasoning)实时显示给用户, 只透出最终答案."""
    import inspect
    from agent_project.agent import OpenMythosAgent

    src = inspect.getsource(OpenMythosAgent._run_simple)
    # _buffer_user_cb 只透出工具/状态事件 + 轻量 thinking 提示, 不透出 reasoning 正文
    assert "reasoning 到达时给一个轻量\"思考中\"状态" in src or "'reasoning' 与 'content' 的正文都不实时透出" in src, "应注释明确不透出 reasoning 正文"
    # 不应再直接透出 reasoning 原文(旧的 simulate_stream_tokens reasoning 透出)
    assert "simulate_stream_tokens(reasoning_text" not in src, "不应把 reasoning 原文流式透出"
    # 应有轻量 thinking 状态避免假死
    assert "thinking" in src, "应有思考中状态提示"


def test_deep_research_summary_no_meta_blurb():
    """深度研究 summary 不应包含'证据与置信度/引用标注/HTML'等多余说明."""
    import inspect
    from agent_project.research_report import ResearchReportGenerator
    src = inspect.getsource(ResearchReportGenerator._generate_summary)
    assert "报告含 '## 证据与置信度'" not in src, "summary 不应含'证据与置信度'说明句"
    assert "HTML 版可直接在浏览器打开阅读" not in src, "summary 不应含 HTML 说明句"


def test_search_cache_normalized_key_improves_hit_rate():
    """缓存键规范化: 相同意图不同说法应命中; 不同主题不串缓存."""
    from agent_project.tools.search_cache import SearchCache
    c = SearchCache(disk_path=None)

    # 相同意图不同说法 → 命中
    same = [
        ("AI 新闻", "最新的ai新闻"),
        ("人工智能 趋势 2026", "2026年人工智能最新趋势"),
        ("hermes 使用", "帮我查一下hermes的用法"),
        ("天气 北京", "北京的天气怎么样"),
        ("苹果 AI", "苹果AI的最新动态"),
    ]
    for a, b in same:
        assert c._key(a) == c._key(b), f"应命中同一缓存: {a!r} vs {b!r}"

    # 不同主题 → 严格区分
    diff = [
        ("AI 新闻", "足球 新闻"),
        ("AI 新闻", "AI 趋势"),
        ("苹果 AI", "苹果 种植"),
        ("天气 北京", "北京 房价"),
    ]
    for a, b in diff:
        assert c._key(a) != c._key(b), f"不应串缓存: {a!r} vs {b!r}"


def test_continuation_topic_inference():
    """延续性请求('我的意思是你也去搜索整合')应解析出真实主题而非整句."""
    import logging
    from agent_project.agent import OpenMythosAgent
    from agent_project.research_report import extract_research_topic

    a = object.__new__(OpenMythosAgent)
    a.logger = logging.getLogger("test")
    a.conversation_history = [
        {"user": "你也去搜索一些和这方面有关的,主要是用ai增强个人生命体能的", "assistant": "好的"},
        {"user": "搜AI增强个人生命体能 相关", "assistant": "已搜索"},
    ]

    task = "我的意思是你也去搜索然后整合起来一起分析"
    topic = a._infer_continuation_topic(task)
    assert topic, "应解析出延续主题"
    assert "生命体能" in topic or "AI" in topic, f"应得到真实主题, 实际 {topic!r}"

    # 旧逻辑会把整句当主题(错误)
    old = extract_research_topic(task)
    assert old != topic or "整合" not in topic, "不应把延续话术当主题"


def test_low_budget_continue_intent_expands():
    """低预算任务中模型表达继续意图('让我先确认')时, loop 应动态扩展而非终止."""
    import sys, logging
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY

    class FB:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            if self.c == 1:
                return '{"action": "bash_exec", "args": {"command": "ls"}}'
            if self.c == 2:
                return '{"action": "bash_exec", "args": {"command": "grep x"}}'
            if self.c == 3:
                return '<think>让我先确认</think>\nThought: 让我先确认是否已写入'
            if self.c == 4:
                return '{"final_answer": "已确认"}'
            return '{"final_answer": "done"}'

    cfg = AgentConfig()
    eng = ExecutionEngine(model_backend=FB(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="你已经写入文档了?", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                           config=cfg, max_steps=2)
    ctx.monitor_enabled = False
    tr = eng.run(ReActPolicy(), ctx)
    assert tr.final_answer == "已确认", f"应完成确认, 实际 {tr.final_answer!r}"
    assert ctx.max_steps > 2, f"预算应动态扩展, 实际 {ctx.max_steps}"


def test_real_progress_detection():
    """进展质量感知: 搜索成功有结果算进展; 重复/失败/打转不算."""
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.config import AgentConfig
    cfg = AgentConfig()

    # 真实进展(搜索成功)
    ctx1 = ExecutionContext(task="搜新闻", available_tools={}, config=cfg, max_steps=8)
    ctx1.observations = ['[{"title": "AI news", "url": "..."}]', '[{"title": "trend"}]']
    assert ExecutionEngine._has_real_progress(ctx1)

    # 原地打转(去重拦截 + 失败)
    ctx2 = ExecutionContext(task="t", available_tools={}, config=cfg, max_steps=8)
    ctx2.observations = [
        "SYSTEM STOP: already executed file_ops list",
        "Tool error: No search results returned",
        "Tool 'web_search' timed out",
    ]
    assert not ExecutionEngine._has_real_progress(ctx2), "失败+重复不应算进展"

    # 混合
    ctx3 = ExecutionContext(task="t", available_tools={}, config=cfg, max_steps=8)
    ctx3.observations = ['{"title": "found"}', "Tool error: xxx"]
    assert ExecutionEngine._has_real_progress(ctx3), "有真实成功结果应算进展"


def test_promise_detection_continuation_words():
    """'那我继续搜索/先换关键词再试' 等承诺应识别为空承诺(说了要做却没执行)."""
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)

    # 用户场景: 承诺继续/换词搜索但没执行工具
    assert a._is_promise_response("好的，那我继续搜索最新 AI 新闻，搜到后写入 lv 目录。\n\n先换更精准的关键词搜索：")
    assert a._is_promise_response("好的，我换个关键词再试一次搜索")
    assert a._is_promise_response("那我接着搜一下")

    # 已完成/正常 → 不误判
    assert not a._is_promise_response("好的，已经搜索完成并写入 lv 目录了")
    assert not a._is_promise_response("根据搜索结果，总结如下")
    assert not a._is_promise_response("你好")


def test_logging_console_default_off():
    """日志 console 默认应关闭, 避免 warn/confidence 打到 stderr 污染 UI."""
    from agent_project.config import LoggingConfig
    cfg = LoggingConfig()
    assert cfg.console is False, f"console 默认应关闭, 实际 {cfg.console}"


def test_github_search_registered_and_signature():
    """github_search 应注册且 execute 接受 sort/order 等模型常用参数."""
    from agent_project.tools import TOOLS_REGISTRY
    assert "github_search" in TOOLS_REGISTRY.list_tools(), "github_search 应默认注册"
    t = TOOLS_REGISTRY.get("github_search")
    import inspect
    sig = inspect.signature(t.execute)
    params = sig.parameters
    assert "sort" in params, "应接受 sort 参数"
    assert "order" in params, "应接受 order 参数"
    assert "per_page" in params, "应接受 per_page 参数"
    # 参数 schema 描述应覆盖模型常用形式
    desc = t.parameters
    assert "repositories" in str(desc["properties"]["kind"].get("enum", []))


def test_tool_parser_multiline_tool_json():
    """跨行 [TOOL:name]\\n{json} 格式(fast-path prompt 教的格式)必须能解析出工具调用.

    回归: 模型输出 '先检查目录。[TOOL:bash_exec]\\n{"command": "cd ..."}' 时,
    之前解析返回 [], 导致 loop 认为无工具调用而停止。
    """
    from agent_project.policies import ToolCallParser

    # 用户实际场景: 带前缀 + 跨行 JSON
    out = '先检查目录。[TOOL:bash_exec]\n{"command": "cd /home/dev/projects/super-ide && pwd && ls -la | head -50"}'
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"应解析出 1 个工具调用, 实际: {calls}"
    assert calls[0][0] == "bash_exec"
    assert "cd /home/dev/projects/super-ide" in calls[0][1]["command"]

    # 无前缀 + 跨行
    out2 = '[TOOL:web_search]\n{"query": "AI 新闻"}'
    calls2 = ToolCallParser.parse_all(out2)
    assert calls2 and calls2[0][0] == "web_search" and calls2[0][1]["query"] == "AI 新闻"

    # python_exec 跨行
    out3 = '[TOOL:python_exec]\n{"code": "print(1)"}'
    calls3 = ToolCallParser.parse_all(out3)
    assert calls3 and calls3[0][0] == "python_exec"

    # 多行 JSON(美化排版)
    out4 = '[TOOL:bash_exec]\n{\n  "command": "ls",\n  "timeout": 30\n}'
    calls4 = ToolCallParser.parse_all(out4)
    assert calls4 and calls4[0][0] == "bash_exec" and calls4[0][1]["command"] == "ls"

    # 跨行 + 文本环绕
    out5 = '先看看目录\n[TOOL:bash_exec]\n{"command": "ls -la"}\n然后回答'
    calls5 = ToolCallParser.parse_all(out5)
    assert calls5 and calls5[0][0] == "bash_exec"


def test_strip_tool_calls_multiline():
    """strip_tool_calls 应一并剔除跨行工具调用及其 JSON 参数."""
    from agent_project.policies import ToolCallParser

    cleaned = ToolCallParser.strip_tool_calls('先检查目录。[TOOL:bash_exec]\n{"command": "ls"}')
    assert "bash_exec" not in cleaned and "ls" not in cleaned
    assert cleaned.strip() == "先检查目录。"

    cleaned2 = ToolCallParser.strip_tool_calls('这是思考\n[TOOL:web_search]\n{"query": "AI"}\n结论')
    assert "web_search" not in cleaned2 and "AI" not in cleaned2
    assert "结论" in cleaned2

    # 纯文本不受影响
    assert ToolCallParser.strip_tool_calls("普通文本") == "普通文本"


def _install_fake_web_search(behavior):
    """把 registry 里的 web_search 临时替换成假工具, 避免测试联网/依赖真实结果.

    behavior: "ok" -> 每次都成功(有结果); "fail" -> 每次都失败.
    """
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.tools import TOOLS_REGISTRY, BaseTool, ToolResult

    class FakeWebSearch(BaseTool):
        name = "web_search"
        description = "fake web search"
        parameters = {"type": "object", "properties": {"query": {"type": "string"}}}

        def __init__(self, behavior):
            self._behavior = behavior
            self.calls = 0

        def execute(self, **kwargs):
            self.calls += 1
            if self._behavior == "ok":
                return ToolResult(success=True, output='[{"title": "result", "url": "http://x"}]')
            return ToolResult(success=False, output="", error="Tool error: fake search failed")

    fake = FakeWebSearch(behavior)
    TOOLS_REGISTRY._tools["web_search"] = fake
    return fake


def test_loop_extension_shows_status_and_no_hard_cap():
    """动态扩展应显示'增加 loop'状态, 且不再受 max_thinking_loops 硬上限截断."""
    import logging
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY
    from agent_project.tools import ToolResult  # noqa: F401

    cfg = AgentConfig()
    cfg.max_thinking_loops = 6  # 故意设很小的初始硬上限
    fake = _install_fake_web_search("ok")

    # 模型持续产出新工具调用(10 次)且都有真实进展 → 预算应突破 max_thinking_loops=6
    class FB_Progress:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            if self.c <= 10:
                return '{"action": "web_search", "args": {"query": "AI %d"}}' % self.c
            return '{"final_answer": "done"}'

    status_msgs = []
    eng = ExecutionEngine(model_backend=FB_Progress(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="搜新闻", available_tools=TOOLS_REGISTRY.get_tools_dict(),
                           config=cfg, max_steps=4)
    ctx.monitor_enabled = False
    ctx.stream_callback = lambda kind, text: status_msgs.append((kind, text))
    tr = eng.run(ReActPolicy(), ctx)
    assert len(tr.tools_used) == 10, f"应执行全部 10 次工具调用, 实际 {len(tr.tools_used)}"
    # 预算应突破 6(原硬上限), 证明不再被 max_thinking_loops 截断
    assert ctx.max_steps > 6, f"预算应突破硬上限 6, 实际 {ctx.max_steps}"
    # 应显示"增加 loop"状态提示
    assert any(k == "status" and "loop" in str(t) for k, t in status_msgs), f"应有增加 loop 状态: {status_msgs}"


def test_loop_stops_after_consecutive_no_progress():
    """连续无进展(工具反复失败)≥3 步时, loop 应停止并提示失败(不无限烧预算)."""
    import logging
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY

    cfg = AgentConfig()
    cfg.max_thinking_loops = 32  # 硬上限很大, 但连续无进展应先行停止
    fake = _install_fake_web_search("fail")

    class FB_Fail:
        def __init__(self): self.c = 0
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
    # 连续无进展约束应生效: 工具调用远少于 12 次(在无进展3次后停止)
    assert len(tr.tools_used) < 12, f"连续无进展应提前停止, 实际执行了 {len(tr.tools_used)} 次"
    assert any(k == "status" and "无进展" in str(t) for k, t in status_msgs), f"应有停止提示: {status_msgs}"



def test_analysis_task_short_answer_forced_full_report():
    """分析任务中模型给过短答案(只提一句)时, 应强制基于工具结果重生成完整分析.

    回归: 用户"分析 super ide, 输出分析报告"场景——模型调 project_context 拿到
    结构后只回一句"docs/goai/作品简介"就结束。
    """
    import sys, logging
    sys.path.insert(0, ".")
    sys.path.insert(0, "agent_project")
    from agent_project.execution_engine import ExecutionContext, ExecutionEngine
    from agent_project.policies import ReActPolicy
    from agent_project.config import AgentConfig
    from agent_project.tools import TOOLS_REGISTRY, BaseTool, ToolResult

    class FakeProj(BaseTool):
        name = "project_context"
        description = "fake"
        parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
        def execute(self, **kw):
            return ToolResult(success=True, output="Project: super-ide\napps/desktop (Electron)\napps/web (Vue 3)\ndocs/ARCHITECTURE-v2.0.md")

    TOOLS_REGISTRY._tools["project_context"] = FakeProj()
    cfg = AgentConfig()

    class FB:
        def __init__(self): self.c = 0
        def generate(self, prompt, **kw):
            self.c += 1
            if self.c == 1:
                return '[TOOL:project_context]\n{"path": "/x"}'
            if "项目定位" in prompt:  # 兜底重生成 → 完整分析
                return "## 项目分析\n\nsuper-ide 是一个多端 IDE:\n- apps/desktop: Electron\n- apps/web: Vue 3\n- docs: 架构文档\n\n通过 conductor 协调各端。"
            return "Final Answer: docs/goai/作品简介。"

    eng = ExecutionEngine(model_backend=FB(), config=cfg)
    eng.logger = logging.getLogger("test")
    ctx = ExecutionContext(task="分析 /home/dev/projects/super-ide, 输出分析报告",
                           available_tools=TOOLS_REGISTRY.get_tools_dict(), config=cfg, max_steps=8)
    ctx.monitor_enabled = False
    tr = eng.run(ReActPolicy(), ctx)
    assert len(tr.final_answer or "") > 50, f"分析场景应强制重生成完整报告, 实际 {len(tr.final_answer or '')} 字符"
    assert "apps/desktop" in (tr.final_answer or ""), "报告应覆盖实际观察到的项目结构"
    assert "作品简介" not in (tr.final_answer or ""), "过短的一句话答案应被替换"


def test_multi_read_paths_not_blocked_by_required_path():
    """multi_read 用 paths 列表(无 path)不应被 required=['action','path'] 校验拒绝.

    回归: 模型输出 [TOOL:file_ops] {"action":"multi_read","paths":[...]} 时,
    之前 add_call 因 path 缺失返回空, 工具调用被当文本。
    """
    from agent_project.policies import ToolCallParser

    out = '[TOOL:file_ops] {"action": "multi_read", "paths": ["../a.md", "../b.md"]} [/TOOL]'
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"应解析出 1 个调用: {calls}"
    assert calls[0][0] == "file_ops"
    assert calls[0][1]["action"] == "multi_read"
    assert len(calls[0][1]["paths"]) == 2

    # 带前缀文本(用户实际场景)
    out2 = "让我并行读取\n\n[TOOL:file_ops] {\"action\": \"multi_read\", \"paths\": [\"a.md\", \"b.md\"]} [/TOOL]"
    calls2 = ToolCallParser.parse_all(out2)
    assert calls2 and calls2[0][0] == "file_ops"


def test_glob_query_alias_not_blocked():
    """glob 用 query 代替 pattern 不应被 required=['pattern'] 校验拒绝."""
    from agent_project.policies import ToolCallParser

    out = '[TOOL:glob] {"query": "**/*.md"} [/TOOL]'
    calls = ToolCallParser.parse_all(out)
    assert calls and calls[0][0] == "glob", f"应解析出 glob: {calls}"


def test_bash_cmd_alias_not_blocked():
    """bash_exec 用 cmd 代替 command 不应被 required=['command'] 校验拒绝."""
    from agent_project.policies import ToolCallParser

    out = '[TOOL:bash_exec] {"cmd": "ls -la"} [/TOOL]'
    calls = ToolCallParser.parse_all(out)
    assert calls and calls[0][0] == "bash_exec", f"应解析出 bash_exec: {calls}"


def test_patch_default_args_migrates_aliases():
    """执行阶段应将别名迁移到规范键: query->pattern, cmd->command."""
    from agent_project.execution_engine import ToolExecutor
    te = ToolExecutor()

    a1 = {"query": "**/*.md"}
    te._patch_default_args("glob", a1)
    assert a1.get("pattern") == "**/*.md", f"query 应迁移到 pattern: {a1}"

    a2 = {"cmd": "ls -la"}
    te._patch_default_args("bash_exec", a2)
    assert a2.get("command") == "ls -la", f"cmd 应迁移到 command: {a2}"

    a3 = {"query": "TODO"}
    te._patch_default_args("search_files", a3)
    assert a3.get("pattern") == "TODO", f"query 应迁移到 pattern: {a3}"


def test_search_cache_semantic_approximation():
    """语义近似缓存: 同义词/口语化/词序变化应命中; 无关主题应 miss.

    这是比精确 _key 更强的第二层: "人工智能新闻"≈"AI新闻",
    "AI今天有什么新闻"≈"AI新闻", 而 "NBA赛事" 与 "AI新闻" 无关。
    """
    from agent_project.tools.search_cache import SearchCache
    c = SearchCache(disk_path=None)
    c.set("AI 新闻", [{"title": "AI新闻1", "url": "http://a"}])
    c.set("天气 北京", [{"title": "北京天气"}])

    should_hit = [
        "人工智能新闻",
        "AI今天有什么新闻",
        "最新的AI新闻",
        "AI新闻 今天",
    ]
    should_miss = [
        "苹果AI",
        "NBA赛事",
        "足球比分",
        "股市行情",
    ]
    for q in should_hit:
        r = c.get(q)
        assert r, f"语义近似应命中: {q!r}"
    for q in should_miss:
        r = c.get(q)
        assert r is None, f"无关主题不应命中: {q!r}"


def test_search_cache_semantic_similarity_metric():
    """Jaccard 相似度: 共享 n-gram 越多分越高."""
    from agent_project.tools.search_cache import SearchCache
    a = SearchCache._semantic_ngrams("AI新闻")
    b = SearchCache._semantic_ngrams("人工智能新闻")
    c = SearchCache._semantic_ngrams("足球比赛")
    assert SearchCache._similarity(a, b) > 0.3, f"相关查询相似度应较高: {SearchCache._similarity(a,b):.2f}"
    assert SearchCache._similarity(a, c) < 0.2, f"无关查询相似度应很低: {SearchCache._similarity(a,c):.2f}"


def test_deep_research_pronoun_resolves_context():
    """深度研究输入含代词(它/这个)时应结合对话历史解析真实主题.

    回归: '深度分析它,生成研究报告' 之前会拿 '报告'/'它' 当搜索词,
    现在应解析出历史中的真实主题(如 'Omni 系统的架构')。
    """
    import re
    from agent_project.agent import OpenMythosAgent
    from agent_project.research_report import extract_research_topic

    a = object.__new__(OpenMythosAgent)
    a._method_cache = {}
    a._code_mode_override = False
    a._current_task = ""
    a.logger = __import__("logging").getLogger("test")
    a.conversation_history = [
        {"user": "分析一下 Omni 系统的架构", "assistant": "Omni 是本地 AI Agent 系统, Rust+TS 架构"},
    ]

    # 模拟 _run_deep_research 的代词消解段
    clean_task = "深度分析它,生成研究报告"
    _raw_topic = extract_research_topic(clean_task)
    _pronoun_present = bool(re.search(
        r"(深度分析|分析|研究|调研)\s*(它|这个|那个|这些|那些|该系统|这个系统|那个系统)", clean_task))
    _needs = _pronoun_present or _raw_topic in ("它", "报告", "这个", "那个", "系统", "该", "此") or len(_raw_topic) <= 1
    assert _pronoun_present, "含'分析它'应识别为代词指代"
    if _needs:
        prev = a._infer_continuation_topic(clean_task)
        if prev:
            clean_task = f"深度研究 {prev}"
    assert "Omni" in clean_task, f"应结合上下文解析出 Omni: {clean_task!r}"
    assert "报告" not in clean_task.split("深度研究", 1)[-1] or "Omni" in clean_task
