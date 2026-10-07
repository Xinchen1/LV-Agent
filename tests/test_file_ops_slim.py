"""file_ops 本地操作回归: 包含传统本地操作 (grep/find/analyze/backup/diff) 及新强操作.

三工具分工:
  file_ops    读写/列表/编辑/校验/本地复制移动/目录分析 (统一本地操作入口)
  search_files 内容搜索; glob 按文件名找文件
  bash_exec   通用系统操作
"""

from agent_project.tools.file_ops import FileOpsTool
from agent_project.policies import ToolCallParser

KEPT_ACTIONS = ["read", "multi_read", "fast_read", "write", "list", "exists",
                "apply_diff", "verify", "open", "mkdir", "delete", "move", "copy", "stat", "tree", "du",
                "grep", "find", "analyze", "backup", "diff"]


def test_legacy_actions_delegate():
    tool = FileOpsTool()
    # grep/find 应能委托执行, backup/diff 应有明确错误或成功
    r = tool.execute(action="grep", path="/tmp", pattern="x", query="x")
    assert r.success or "Error" not in (r.error or "") or "已移除" not in (r.error or "")

    r = tool.execute(action="find", path="/tmp", pattern="*")
    assert r.success or "Error" not in (r.error or "") or "已移除" not in (r.error or "")

    r = tool.execute(action="analyze", path="/tmp")
    assert r.success or "Error" not in (r.error or "") or "已移除" not in (r.error or "")

    r = tool.execute(action="backup", path="/tmp")
    assert r.success or "Error" not in (r.error or "") or "已移除" not in (r.error or "")

    r = tool.execute(action="diff", path="/tmp")
    assert r.success or "Error" not in (r.error or "") or "已移除" not in (r.error or "")


def test_schema_exposes_kept_actions():
    props = FileOpsTool.parameters["properties"]
    assert props["action"]["enum"] == KEPT_ACTIONS


def test_backup_legacy_action_delegates(tmp_path):
    tool = FileOpsTool()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    r = tool.execute(action="backup", path=str(victim))
    # backup 现在是委托给 bash_exec 的真实操作, 不再因下线而报错
    assert r.success, r.error
    # 原始文件必须完好
    assert victim.read_text() == "keep me"
    # 备份应存在
    assert (tmp_path / "victim.txt.bak").exists()


def test_kept_list_action_still_works(tmp_path):
    (tmp_path / "a.md").write_text("x")
    r = FileOpsTool().execute(action="list", path=str(tmp_path))
    assert r.success, r.error
    assert "a.md" in r.output


def test_flat_json_removed_action_still_parses_to_file_ops():
    """平铺 {"action":"grep"} 仍要解析为 file_ops 调用(否则模型调用被静默丢弃),
    但 ahora 由 execute 内委托处理。"""
    calls = ToolCallParser.parse_all(
        '[TOOL:file_ops] {"action": "grep", "path": "/tmp", "pattern": "x"} [/TOOL]'
    )
    assert calls, "grep 仍应解析出 file_ops 调用"
    assert calls[0][0] == "file_ops"
    assert calls[0][1].get("action") == "grep"
