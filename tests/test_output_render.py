"""输出美化回归: 工具调用参数 JSON 化、错误只出现一次、[TOOL:] 不泄漏到正文."""

from types import SimpleNamespace

from agent_project.agent import OpenMythosAgent
from agent_project.tools import ToolCall, ToolResult
from agent_project.stream_adapters import is_error_text, is_policy_text
from agent_project.response_filter import strip_tool_tags
import super_agent


def _agent() -> OpenMythosAgent:
    return OpenMythosAgent.__new__(OpenMythosAgent)


def test_tool_args_rendered_as_json_not_python_dict():
    text = OpenMythosAgent._format_tool_args({"action": "read", "path": "笔记.md"})
    assert text == '{"action": "read", "path": "笔记.md"}'
    assert "'" not in text, "不能用 Python dict 的单引号 repr"
    long = OpenMythosAgent._format_tool_args({"content": "x" * 500})
    assert len(long) <= 200 and long.endswith("…")


def test_failed_tool_emits_box_only_once():
    events = []
    agent = _agent()
    action = ToolCall(tool_name="file_ops", arguments={"action": "grep", "path": "."})
    err = "工具调用失败: grep 已移除 -> 用 search_files(query=..., path=...)"
    result = ToolResult(success=False, output="", error="grep 已移除 -> 用 search_files")
    agent._stream_tool_observation(action, result, err, lambda k, t: events.append((k, t)))

    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result"], f"失败不应再作为正文重播: {kinds}"
    assert events[0][1] == 'file_ops: {"action": "grep", "path": "."}'
    assert is_error_text(events[1][1]), "结果框需能被识别为失败(✗)"


def test_successful_tool_strips_tool_tags_from_content():
    events = []
    agent = _agent()
    action = ToolCall(tool_name="bash_exec", arguments={"command": "ls"})
    raw = "file list\n[TOOL:file_ops] {\"action\": \"read\"} [/TOOL]\ndone"
    result = ToolResult(success=True, output=raw, error=None)
    agent._stream_tool_observation(action, result, raw, lambda k, t: events.append((k, t)))

    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "content"]
    content = dict(events)["content"]
    assert "[TOOL:" not in content and "[/TOOL]" not in content
    assert "file list" in content and "done" in content


def test_error_and_policy_classification():
    assert is_error_text("工具调用失败: boom")
    assert is_error_text("tool execution error: boom")
    assert is_error_text("timed out after 30s")
    assert not is_error_text("这是正常答案")
    assert is_policy_text("denied by harness: rm -rf /")
    assert not is_policy_text("文件已写入")


def test_content_cleaner_drops_protocol_markers():
    leaked = 'answer.\n[TOOL:file_ops] {"action": "read", "path": "a"} [/TOOL]\ntail.'
    assert super_agent._clean_content_text(leaked) == "answer.\n\ntail."
    assert strip_tool_tags(leaked) == "answer.\n\ntail."


def test_finish_result_does_not_reprint_tool_error(capsys):
    fake = SimpleNamespace(
        format_result=lambda r: "FOOTER",
        _maybe_open_report=lambda r: None,
    )
    super_agent.SuperAgentCLI._finish_result(
        fake, {"final_answer": "工具调用失败: File not found: /nope"})
    out = capsys.readouterr().out
    assert "File not found" not in out, "错误已由 ✗ 结果框展示, 不该再打印一遍"
    assert "FOOTER" in out

    super_agent.SuperAgentCLI._finish_result(fake, {"final_answer": "这是最终答案"})
    out2 = capsys.readouterr().out
    assert "这是最终答案" in out2, "正常答案仍需兜底打印"


def test_long_observation_only_is_echoed_as_content():
    from agent_project.execution_engine import ExecutionEngine
    assert not ExecutionEngine._should_echo_observation("a.md\nb.md\nc.md")
    assert not ExecutionEngine._should_echo_observation("工具调用失败: boom")
    assert not ExecutionEngine._should_echo_observation("   ")
    assert ExecutionEngine._should_echo_observation("x" * 900)
