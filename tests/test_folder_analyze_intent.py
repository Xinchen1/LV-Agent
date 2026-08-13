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
