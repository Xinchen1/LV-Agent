"""意图分类器对"分析 X 文件夹"场景的识别与解析测试."""

import re
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.agent import OpenMythosAgent
from agent_project.tools import ToolCall


def make_agent():
    a = object.__new__(OpenMythosAgent)
    a._method_cache = {}
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
        out = a._read_folder_markdown("/Users/mac/Desktop/grok-build")
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
