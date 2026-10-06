"""file_ops 精简回归: grep/find/analyze/backup/diff 已拆分到 search_files/glob/bash_exec.

三工具分工:
  file_ops    读写/列表/编辑/校验 (不做搜索)
  search_files 内容搜索; glob 按文件名找文件
  bash_exec   diff / backup / analyze 类系统操作
"""

from agent_project.tools.file_ops import FileOpsTool
from agent_project.policies import ToolCallParser

REMOVED_HINTS = {
    "grep": "search_files",
    "find": "glob",
    "analyze": "bash_exec",
    "backup": "bash_exec",
    "diff": "bash_exec",
}

KEPT_ACTIONS = ["read", "multi_read", "fast_read", "write", "list", "exists",
                "apply_diff", "verify", "open", "mkdir", "delete", "move", "copy", "stat", "tree", "du"]


def test_removed_actions_return_redirect_hint():
    tool = FileOpsTool()
    for action, hint in REMOVED_HINTS.items():
        r = tool.execute(action=action, path="/tmp", pattern="x")
        assert not r.success, f"{action} 应返回纠偏提示而不是执行"
        assert hint in (r.error or ""), f"{action} 提示应指向 {hint}: {r.error}"


def test_schema_exposes_only_kept_actions():
    props = FileOpsTool.parameters["properties"]
    assert props["action"]["enum"] == KEPT_ACTIONS
    assert "pattern" not in props, "grep/find 已移除, pattern 参数不应再出现在 schema"


def test_removed_actions_never_touch_disk(tmp_path):
    tool = FileOpsTool()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    r = tool.execute(action="backup", path=str(victim))
    assert not r.success
    assert victim.read_text() == "keep me"
    assert [p.name for p in tmp_path.iterdir()] == ["victim.txt"]


def test_kept_list_action_still_works(tmp_path):
    (tmp_path / "a.md").write_text("x")
    r = FileOpsTool().execute(action="list", path=str(tmp_path))
    assert r.success, r.error
    assert "a.md" in r.output


def test_flat_json_removed_action_still_parses_to_file_ops():
    """平铺 {"action":"grep"} 仍要解析为 file_ops 调用(否则模型调用被静默丢弃),
    由 execute() 给出纠偏提示。"""
    calls = ToolCallParser.parse_all(
        '[TOOL:file_ops] {"action": "grep", "path": "/tmp", "pattern": "x"} [/TOOL]'
    )
    assert calls, "移除的 action 仍应解析出 file_ops 调用"
    assert calls[0][0] == "file_ops"
    assert calls[0][1].get("action") == "grep"
