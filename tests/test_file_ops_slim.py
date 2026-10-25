"""file_ops 核心动作回归.

三工具分工:
  file_ops     核心文件动作: read/write/list/apply_diff/delete/move/copy/mkdir/exists
  search_files 内容搜索; glob 按文件名找文件
  bash_exec    通用系统操作 (open/stat/tree/du/diff/backup)
"""

from agent_project.tools.file_ops import FileOpsTool
from agent_project.policies import ToolCallParser

KEPT_ACTIONS = ["read", "write", "list", "exists", "apply_diff",
                "mkdir", "delete", "move", "copy"]

REMOVED_ACTIONS = {
    "multi_read": "read",
    "fast_read": "read",
    "verify": "bash_exec",
    "open": "bash_exec",
    "stat": "bash_exec",
    "tree": "bash_exec",
    "du": "bash_exec",
    "grep": "search_files",
    "find": "glob",
    "analyze": "bash_exec",
    "backup": "bash_exec",
    "diff": "bash_exec",
}


def test_removed_actions_return_redirect_hint():
    tool = FileOpsTool()
    for action, replacement in REMOVED_ACTIONS.items():
        r = tool.execute(action=action, path="/tmp")
        assert not r.success, action
        assert action in (r.error or ""), r.error
        assert replacement in (r.error or ""), r.error


def test_schema_exposes_only_core_actions():
    props = FileOpsTool.parameters["properties"]
    assert props["action"]["enum"] == KEPT_ACTIONS
    assert "paths" not in props
    assert "query" not in props


def test_kept_list_action_still_works(tmp_path):
    (tmp_path / "a.md").write_text("x")
    r = FileOpsTool().execute(action="list", path=str(tmp_path))
    assert r.success, r.error
    assert "a.md" in r.output


def test_core_file_actions_roundtrip(tmp_path):
    tool = FileOpsTool()
    src = tmp_path / "src.txt"
    r = tool.execute(action="write", path=str(src), content="line1\nline2\n")
    assert r.success, r.error
    assert src.read_text() == "line1\nline2\n"

    r = tool.execute(action="exists", path=str(src))
    assert r.success and "true" in r.output.lower()

    r = tool.execute(action="read", path=str(src), line_numbers=False)
    assert r.success, r.error
    assert "line1" in r.output

    dst = tmp_path / "dst.txt"
    r = tool.execute(action="copy", path=str(src), destination=str(dst))
    assert r.success, r.error
    assert dst.read_text() == src.read_text()

    moved = tmp_path / "moved.txt"
    r = tool.execute(action="move", path=str(dst), destination=str(moved))
    assert r.success, r.error
    assert moved.exists() and not dst.exists()

    r = tool.execute(action="mkdir", path=str(tmp_path / "sub" / "dir"))
    assert r.success, r.error
    assert (tmp_path / "sub" / "dir").is_dir()

    r = tool.execute(action="delete", path=str(moved))
    assert r.success, r.error
    assert not moved.exists()


def test_flat_json_removed_action_still_parses_to_file_ops():
    """平铺 {"action":"grep"} 仍要解析为 file_ops 调用, 由 execute 返回重定向提示."""
    calls = ToolCallParser.parse_all(
        '[TOOL:file_ops] {"action": "grep", "path": "/tmp", "pattern": "x"} [/TOOL]'
    )
    assert calls, "grep 仍应解析出 file_ops 调用"
    assert calls[0][0] == "file_ops"
    assert calls[0][1].get("action") == "grep"
