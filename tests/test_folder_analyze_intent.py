#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research





import re
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.agent import OpenMythosAgent
from agent_project.tools import ToolCall
from agent_project.cache import MemoCache


def make_agent():
    a = object.__new__(OpenMythosAgent)
    a._method_cache = MemoCache()
    a._code_mode_override = False
    a._current_task = ""
    return a


def test_folder_read_intent_analyze_variants():
    a = make_agent()
    assert a._is_folder_read_intent("grok-build 分析下")
    assert a._is_folder_read_intent("grok-build 分析")
    assert a._is_folder_read_intent("grok-build 剖析一下")
    assert a._is_folder_read_intent("分析下 grok-build")
    assert a._is_folder_read_intent("分析 grok-build 项目")
    assert a._is_folder_read_intent("帮我分析下 Desktop 的 grok-build 文件夹")
    assert not a._is_folder_read_intent("你好")
    assert not a._is_folder_read_intent("今天的天气怎么样")


def test_analyze_intent_classifies_to_file_ops_list():
    a = make_agent()
    cls = a._classify_intent("grok-build 分析下")
    assert cls is not None
    tool_name, args, conf, _ = cls
    assert tool_name == "file_ops"
    assert args.get("action") == "list"
    assert args.get("path") == "grok-build"

    cls2 = a._classify_intent("分析 grok-build 项目")
    assert cls2 is not None
    _, args2, conf2, _ = cls2
    assert args2.get("path") == "grok-build"


def test_analyze_is_simple_query_fast():
    a = make_agent()
    assert a._is_simple_query("grok-build 分析下")
    assert a._is_simple_query("分析 grok-build 项目")
    assert not a._is_simple_query("今天的天气怎么样")


def test_bash_ls_path_extraction_for_folder_analyze():
    a = make_agent()
    task = "grok-build 分析下"
    if not a._is_folder_read_intent(task):
        assert False, "task should be folder-read intent"

    for cmd, expected in [
        ("ls -la ~/Desktop/grok-build", "~/Desktop/grok-build"),
        ("ls ~/Desktop/grok-build", "~/Desktop/grok-build"),
        ("ls -la", None),
        ("ls grok-build 2>/dev/null", "grok-build"),
    ]:
        seg = re.split(r"[;&|]", cmd)[0]
        after_ls = seg.split("ls", 1)[1] if "ls" in seg else ""
        path = None
        for tok in after_ls.split():
            if tok.startswith("-") or tok == "2>" or tok.endswith("/dev/null"):
                continue
            path = tok.strip().strip('"\'')
            break
        assert path == expected, f"cmd={cmd!r}: got {path!r}, expected {expected!r}"


def test_tool_calls_array_json_parsing():
    a = make_agent()
    import logging
    a.logger = logging.getLogger("test")

    out = (
        '{\n  "thoughts": "locate the grok-build directory",\n'
        '  "final_answer": "",\n  "tool_calls": [\n'
        '    {"action": "bash_exec", "args": {"command": "ls -la ~/Desktop/grok-build"}}\n'
        "  ]\n}"
    )
    actions = a._parse_all_output_actions(out)
    assert len(actions) == 1
    assert actions[0].tool_name == "bash_exec"
    assert actions[0].arguments.get("command") == "ls -la ~/Desktop/grok-build"

    first = a._parse_output_for_action(out)
    assert first is not None and first.tool_name == "bash_exec"


def test_read_folder_markdown_is_instance_method():
    """_read_folder_markdown 必须是实例方法(有 self), 否则实例调用会报缺参."""
    import inspect
    from agent_project.agent import OpenMythosAgent
    m = getattr(OpenMythosAgent, "_read_folder_markdown")
    assert not isinstance(m, staticmethod), "不应是 staticmethod(签名含 self 会导致缺参)"
    sig = inspect.signature(m)
    assert "self" in sig.parameters, "签名应包含 self(实例方法)"

    # 实例调用不应报 TypeError
    a = make_agent()
    a.conversation_history = []
    try:
        out = a._read_folder_markdown("/tmp/nonexistent-grok-build")
        assert out == "" or isinstance(out, str)
    except TypeError as e:
        assert False, f"实例调用 _read_folder_markdown 报错: {e}"


def test_tool_call_parser_json_string_action():
    """policies.ToolCallParser 应支持 action 字符串 + 同级 args 的 JSON 格式."""
    from agent_project.policies import ToolCallParser

    # 用户实际场景: {thoughts, action: "bash_exec", args: {...}}
    out = (
        '{\n  "thoughts": "create folder",\n'
        '  "action": "bash_exec",\n'
        '  "args": {"command": "mkdir snake_game"}\n}'
    )
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"应解析出 1 个工具调用, 实际 {calls}"
    assert calls[0][0] == "bash_exec"
    assert calls[0][1].get("command") == "mkdir snake_game"

    # reasoning + action + arguments 变体
    out2 = (
        '{\n  "reasoning": "list dir",\n'
        '  "action": "file_ops",\n'
        '  "arguments": {"action": "list", "path": "."}\n}'
    )
    calls2 = ToolCallParser.parse_all(out2)
    assert len(calls2) == 1
    assert calls2[0][0] == "file_ops"
    assert calls2[0][1]["path"] == "."


def test_tool_call_parser_json_tool_calls_array():
    """policies.ToolCallParser 应支持 tool_calls 数组内 action 字符串格式."""
    from agent_project.policies import ToolCallParser

    out = (
        '{\n  "thoughts": "locate",\n'
        '  "tool_calls": [\n'
        '    {"action": "bash_exec", "args": {"command": "ls"}}\n'
        "  ]\n}"
    )
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"应解析出 1 个工具调用, 实际 {calls}"
    assert calls[0][0] == "bash_exec"
    assert calls[0][1].get("command") == "ls"


def test_truncated_fragment_detection():
    """execution_engine 应识别截断碎片(final answer 为 'We'/'The' 等时兜底重建)."""
    from agent_project.execution_engine import ExecutionEngine

    assert ExecutionEngine._is_truncated_fragment("We")
    assert ExecutionEngine._is_truncated_fragment("The")
    assert ExecutionEngine._is_truncated_fragment("")
    assert ExecutionEngine._is_truncated_fragment("  ")
    assert not ExecutionEngine._is_truncated_fragment("好")
    assert not ExecutionEngine._is_truncated_fragment("ok")
    assert not ExecutionEngine._is_truncated_fragment("完成")
    assert not ExecutionEngine._is_truncated_fragment("We should improve this project")
    assert not ExecutionEngine._is_truncated_fragment("首先, 需要分析文件")


def test_unified_tool_parser_all_formats():
    """统一解析器后, agent.py 应委托 ToolCallParser 且支持全部格式(含 XML)."""
    import logging
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)
    a.logger = logging.getLogger("test")

    cases = {
        "closed_tag": ('[TOOL:bash_exec] {"command": "ls"} [/TOOL]', "bash_exec", "ls"),
        "json_string_action": ('{"thoughts": "t", "action": "bash_exec", "args": {"command": "mkdir x"}}', "bash_exec", "mkdir x"),
        "tool_calls_array": ('{"tool_calls": [{"action": "bash_exec", "args": {"command": "ls"}}]}', "bash_exec", "ls"),
        "func_single_str": ('bash_exec("echo hi; ls")', "bash_exec", "echo hi; ls"),
        "xml": ('<tool_call><function=bash_exec><parameter=command>echo x</parameter></function></tool_call>', "bash_exec", "echo x"),
        "opencode": ('<|message_model|>bash_exec<|content_invoke_tool_json|>{"args":{"command":"ls"}}<|end_message|>', "bash_exec", "ls"),
    }
    for name, (text, tool, arg_val) in cases.items():
        r = a._parse_all_output_actions(text)
        assert len(r) == 1, f"{name}: 应解析出 1 个, 实际 {r}"
        assert r[0].tool_name == tool, f"{name}: 工具名错误 {r[0].tool_name}"
        assert r[0].arguments.get("command") == arg_val, f"{name}: 参数错误 {r[0].arguments}"


def test_tool_parser_handles_chinese_quotes_in_json():
    """content 里含中文引号/markdown反引号时, TOOL 调用应能正确解析并写入."""
    from agent_project.policies import ToolCallParser
    from agent_project.tools import TOOLS_REGISTRY

    # 中文引号 + markdown 反引号混合
    out = (
        '[TOOL:file_ops] {"action": "write", "path": "lv/test.md", '
        '"content": "希望 AI 能“记住自己”的用户\\n使用 `hermes` 命令"} [/TOOL]'
    )
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"应解析出 1 个, 实际 {calls}"
    assert calls[0][0] == "file_ops"
    args = calls[0][1]
    assert args.get("action") == "write"
    assert "记住自己" in args.get("content", ""), "中文引号应保留在 content 里"
    assert "`hermes`" in args.get("content", ""), "markdown 反引号应保留"

    # 反引号包 JSON 键
    out2 = '[TOOL:file_ops] {`action`: `write`, `path`: `lv/x.md`, `content`: `test`} [/TOOL]'
    calls2 = ToolCallParser.parse_all(out2)
    assert len(calls2) == 1 and calls2[0][1].get("action") == "write"

    # 完整长内容(类似真实场景)能真正写入
    import tempfile, os
    out3 = (
        '[TOOL:file_ops] {"action": "write", "path": "P", '
        '"content": "Hermes Agent 使用说明\\n\\n**核心特点:**\\n- 持久记忆“记住自己”\\n- 使用 `hermes` 命令\\n"} [/TOOL]'
    )
    calls3 = ToolCallParser.parse_all(out3)
    assert calls3, f"长内容应能解析: {calls3}"
    tool = TOOLS_REGISTRY.get("file_ops")
    td = tempfile.mkdtemp()
    args3 = dict(calls3[0][1])
    args3["path"] = os.path.join(td, "hermes.md")
    r = tool.execute(**args3)
    assert r.success, f"应能实际写入: {r.error}"
    assert os.path.exists(args3["path"])


def test_tool_parser_list_prefixed_unclosed():
    """带列表前缀(- / 1. / *)且无 [/TOOL] 闭合的调用应能解析."""
    from agent_project.policies import ToolCallParser

    # 用户场景: 正文列出多个无闭合调用, 带 "- " 前缀
    out = (
        "Let me execute:\n"
        '- [TOOL:web_search] {"query": "Hermes Agent 使用教程"}\n'
        '- [TOOL:bash_exec] {"command": "ls -la lv"}'
    )
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 2, f"应解析出 2 个, 实际 {calls}"
    assert calls[0][0] == "web_search"
    assert calls[1][0] == "bash_exec"

    # 数字列表 + * 列表
    assert len(ToolCallParser.parse_all('1. [TOOL:web_search] {"query": "x"}')) == 1
    assert len(ToolCallParser.parse_all('* [TOOL:bash_exec] {"command": "ls"}')) == 1

    # 行首无前缀仍正常
    assert len(ToolCallParser.parse_all('[TOOL:web_search] {"query": "hermes"}')) == 1


def test_tool_parser_full_matrix_case_insensitive():
    """解析器应覆盖全部格式 + 大小写不敏感, 工具调用绝不漏解析."""
    from agent_project.policies import ToolCallParser

    cases = {
        "TOOL闭合标准": '[TOOL:bash_exec] {"command": "ls"} [/TOOL]',
        "TOOL闭合中文引号": '[TOOL:file_ops] {"action": "write", "path": "a.md", "content": "记住“自己”"} [/TOOL]',
        "TOOL闭合大小写": '[TOOL:File_Ops] {"action": "list", "path": "."} [/TOOL]',
        "TOOL未闭合列表前缀": '- [TOOL:web_search] {"query": "hermes"}',
        "TOOL未闭合数字": '1. [TOOL:bash_exec] {"command": "ls"}',
        "TOOL未闭合带think": '<think>x</think>\n[TOOL:bash_exec] {"command": "ls"}',
        "JSON action字符串": '{"action": "bash_exec", "args": {"command": "ls"}}',
        "JSON tool_calls数组": '{"tool_calls": [{"action": "bash_exec", "args": {"command": "ls"}}]}',
        "JSON 大写": '{"action": "BASH_EXEC", "args": {"command": "ls"}}',
        "JSON 带think": '<think>t</think> {"action": "bash_exec", "args": {"command": "ls"}}',
        "函数 单字符串": 'bash_exec("ls -la")',
        "函数 key=value": 'file_ops(action="list", path=".")',
        "函数 大写": 'BASH_EXEC("ls")',
        "函数 带think": '<think>t</think>\nfile_ops(action="list", path=".")',
        "XML": '<tool_call><function=bash_exec><parameter=command>ls</parameter></function></tool_call>',
        "XML 大写": '<tool_call><function=BASH_EXEC><parameter=command>ls</parameter></function></tool_call>',
        "XML 带think": '<think>t</think>\n<tool_call><function=bash_exec><parameter=command>ls</parameter></function></tool_call>',
        "OpenCode": '<|message_model|>bash_exec<|content_invoke_tool_json|>{"args":{"command":"ls"}}<|end_message|>',
        "OpenCode 大写": '<|message_model|>BASH_EXEC<|content_invoke_tool_json|>{"args":{"command":"ls"}}<|end_message|>',
        "混合多工具": '<think>计划</think>\n- [TOOL:web_search] {"query": "hermes"}\n- [TOOL:bash_exec] {"command": "ls lv"}',
    }
    for name, out in cases.items():
        r = ToolCallParser.parse_all(out)
        assert r, f"{name}: 工具调用被漏解析! 输出={out[:80]}"


def test_tool_parser_flat_json_fileops_and_toolname():
    """模型省略工具名/用平铺参数时, 应能推断出工具调用(不把执行当文本)."""
    from agent_project.policies import ToolCallParser

    # file_ops 子动作平铺
    out = '{"action": "write", "path": "lv/健身计划.md", "content": "# 减脂健身计划\\n\\n目标: 减脂"}'
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"file_ops 平铺应解析, 实际 {calls}"
    assert calls[0][0] == "file_ops"
    assert calls[0][1]["action"] == "write"
    assert calls[0][1]["path"] == "lv/健身计划.md"
    assert "减脂" in calls[0][1]["content"]

    # 工具名 + 平铺参数
    out2 = '{"action": "web_search", "query": "hermes", "max_results": 5}'
    calls2 = ToolCallParser.parse_all(out2)
    assert len(calls2) == 1 and calls2[0][0] == "web_search"
    assert calls2[0][1]["query"] == "hermes"

    # 带 thoughts 字段
    out3 = '{"thoughts": "先搜", "action": "web_search", "query": "AI 新闻"}'
    calls3 = ToolCallParser.parse_all(out3)
    assert len(calls3) == 1 and calls3[0][0] == "web_search"
    assert "thoughts" not in calls3[0][1], "thoughts 不应作为参数"


def test_intent_mismatch_overrides_wrong_tool():
    """模型生成与意图冲突的工具(find)时, 意图分类器应覆盖为 web_search."""
    import logging
    from agent_project.agent import OpenMythosAgent
    from agent_project.tools import ToolCall
    a = object.__new__(OpenMythosAgent)
    a.logger = logging.getLogger("test")

    action = ToolCall(tool_name="bash_exec", arguments={"command": "find /home/dev -name '*ai*'"})
    classified = a._classify_intent("查下ai 方面新闻")
    assert classified and classified[0] == "web_search"
    c_tool, c_args, c_conf, c_reason = classified
    mismatch = (c_tool == "web_search" and action.tool_name in ("bash_exec", "find", "glob", "search_files"))
    assert mismatch, "查新闻用 find 应判定为意图冲突"


def test_location_fast_path_excludes_news_search():
    """'查ai新闻' 不应触发 find 全盘扫描, 应走 web_search."""
    import re
    # 复现 _try_location_fast_path 的触发判断
    def should_find(task):
        task_lower = task.lower()
        locate_verbs = ["查找", "找一下", "搜索", "搜一下", "看看有没有", "看看", "看下", "在哪里", "在哪", "位于", "找", "查", "搜"]
        has_verb = any(v in task for v in locate_verbs)
        has_suffix = any(s in task for s in ["项目", "文件夹", "目录", "文件"])
        if not has_verb:
            return False
        if not has_suffix and not re.search(r'[a-zA-Z_\-0-9]+', task):
            return False
        info_search_markers = ["新闻", "资讯", "消息", "动态", "最新", "信息", "资料", "教程",
                               "怎么", "如何", "教程", "介绍", "是什么", "怎么做", "天气",
                               "news", "update", "info", "how to", "what is", "weather",
                               "股票", "行情", "价格", "比分", "比赛"]
        if has_suffix is False and any(m in task_lower for m in info_search_markers):
            return False
        return True

    assert not should_find("查下最新的 ai 新闻")
    assert not should_find("查一下AI新闻")
    assert not should_find("搜索AI相关新闻")
    assert should_find("找一下 project.py 文件")
    assert should_find("看看桌面的报告文件")


def test_llm_intent_classify_fallback():
    """规则未命中且非简单任务时, LLM 意图分类兜底应生效; 简单任务/规则命中不触发."""
    import logging
    from agent_project.agent import OpenMythosAgent

    calls = {"n": 0}
    class FakeBackend:
        def generate(self, prompt, **kw):
            calls["n"] += 1
            return '{"tool": "web_search", "args": {"query": "x"}, "reason": "r"}'

    a = object.__new__(OpenMythosAgent)
    a.logger = logging.getLogger("test")
    a.backend = FakeBackend()
    a._method_cache = MemoCache(); a._code_mode_override = False; a._current_task = ""

    # 规则未命中 + 非简单 → LLM 兜底
    calls["n"] = 0
    r1 = a._classify_intent("给我找点AI方面的资料")
    assert r1 and calls["n"] == 1, f"应触发 LLM 兜底, 实际调用 {calls['n']}"

    # 简单问候 → 不触发
    calls["n"] = 0
    assert a._classify_intent("你好啊") is None
    assert calls["n"] == 0

    # 规则命中(新闻) → 不触发
    calls["n"] = 0
    r3 = a._classify_intent("查下最新的 ai 新闻")
    assert r3 and r3[0] == "web_search"
    assert calls["n"] == 0, "规则命中不应调 LLM"


def test_tool_parser_bare_quotes_in_content():
    """content 里含裸英文双引号(未转义)时, 平铺 JSON 仍应能解析并写入."""
    from agent_project.policies import ToolCallParser

    # 用户实际场景: content 含（"这部分太虚"）裸引号
    out = (
        '{"action": "write", "path": "lv/提高使用AI的水平.md", '
        '"content": "好的提问（"这部分太虚，换成具体案例"）\\n效率 = 明确目标"}'
    )
    calls = ToolCallParser.parse_all(out)
    assert len(calls) == 1, f"应解析出 1 个, 实际 {calls}"
    assert calls[0][0] == "file_ops"
    args = calls[0][1]
    assert args["action"] == "write"
    assert args["path"] == "lv/提高使用AI的水平.md"
    assert "这部分太虚" in args["content"], "裸引号应保留在 content 里"
    assert "效率 = 明确目标" in args["content"]

    # 标准 JSON 仍正常
    ok = ToolCallParser.parse_all('{"action": "write", "path": "lv/a.md", "content": "测试"}')
    assert ok


def test_pronoun_resolves_relative_path_to_absolute():
    """'分析它' 中 '它' 应解析历史里的相对路径(../IDE/super-ide)为绝对路径."""
    import os
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)
    a._method_cache = MemoCache(); a._code_mode_override = False; a._current_task = ""
    os.chdir(os.path.expanduser("~"))

    a.conversation_history = [
        {"user": "分析super ide", "assistant": "已用 project_context 查看 ../projects/super-ide, 结构如下: docs/goai/作品简介。"},
    ]
    recent = "\n".join(t.get("user", "") + "\n" + t.get("assistant", "") for t in a.conversation_history)
    r = a._resolve_pronoun_to_entity("要你 分析它,输出分析报告", recent)
    assert "super-ide" in r, f"应解析出包含 super-ide 的绝对路径: {r!r}"
    assert "-build" not in r, f"不应匹配到 super-ide-build: {r!r}"


def test_pronoun_resolves_absolute_path():
    """历史里是绝对路径时直接使用."""
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)
    a._method_cache = MemoCache(); a._code_mode_override = False; a._current_task = ""
    a.conversation_history = [
        {"user": "分析super ide", "assistant": "project_context path=/home/dev/projects/super-ide OK"},
    ]
    recent = "\n".join(t.get("user", "") + "\n" + t.get("assistant", "") for t in a.conversation_history)
    r = a._resolve_pronoun_to_entity("分析它", recent)
    assert r and "/home/dev/projects/super-ide" in r, f"应使用绝对路径: {r!r}"


def test_pronoun_resolves_entity_name_via_dir_locate(tmp_path):
    """历史只有实体名时, 应通过 _locate_project_dir 定位真实目录."""
    import os
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)
    a._method_cache = MemoCache(); a._code_mode_override = False; a._current_task = ""
    # 在临时目录下建一个 super-ide 目录, 使 _locate_project_dir(cwd=tmp_path) 能定位到
    os.chdir(tmp_path)
    (tmp_path / "super-ide").mkdir()
    a.conversation_history = [
        {"user": "分析super ide", "assistant": "用 project_context 查看了 super-ide"},
    ]
    recent = "\n".join(t.get("user", "") + "\n" + t.get("assistant", "") for t in a.conversation_history)
    r = a._resolve_pronoun_to_entity("要你 分析它", recent)
    assert r and "super-ide" in r, f"应定位到真实目录: {r!r}"


def test_resolved_analysis_task_not_simple():
    """指代解析后的 '分析 <路径>,输出分析报告' 应走主循环(非 fast path)."""
    from agent_project.agent import OpenMythosAgent
    a = object.__new__(OpenMythosAgent)
    a._method_cache = MemoCache(); a._code_mode_override = False; a._current_task = ""
    resolved = "要你 分析 '/home/dev/super-ide', 输出分析报告"
    assert not a._is_simple_query(resolved), "含'分析'+路径+报告的任务不应走 fast path"
