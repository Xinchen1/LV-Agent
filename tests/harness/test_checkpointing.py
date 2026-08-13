"""Checkpoint middleware: snapshot-before-mutate, result recording, events, undo."""

import asyncio

from agent_project.harness import events as ev
from agent_project.harness.checkpointing import CheckpointMiddleware
from agent_project.harness.effects import make_effect
from agent_project.harness.session import Session
from agent_project.harness.loop import SampleResult
from agent_project.checkpoint import CheckpointManager


def _effect_write(path, content="new"):
    return make_effect("file_ops", {"action": "write", "path": str(path), "content": content})


def test_snapshot_taken_before_write_and_result_recorded(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")
    manager = CheckpointManager(session_id="t", root_dir=tmp_path / "cps")
    seen = []
    mw = CheckpointMiddleware(manager, on_checkpoint=lambda cid, desc: seen.append(cid))

    def executor(effect):
        target.write_text("modified", encoding="utf-8")
        return "written"

    out = mw.wrap(executor)(_effect_write(target))
    assert out == "written"
    assert len(seen) == 1

    recent = manager.list_recent()
    assert len(recent) == 1 and recent[0]["success"] is True
    assert recent[0]["path"] == str(target.resolve())

    # undo restores the pre-write content
    result = manager.rollback(recent[0]["id"])
    assert result["restored"] is True
    assert target.read_text(encoding="utf-8") == "original"


def test_failed_mutation_records_failure(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    manager = CheckpointManager(session_id="t2", root_dir=tmp_path / "cps")
    mw = CheckpointMiddleware(manager)

    def bad_executor(effect):
        raise RuntimeError("disk full")

    import pytest

    with pytest.raises(RuntimeError):
        mw.wrap(bad_executor)(_effect_write(target))
    recent = manager.list_recent()
    assert recent[0]["success"] is False and "disk full" in recent[0]["error"]


def test_read_effects_are_not_snapshotted(tmp_path):
    manager = CheckpointManager(session_id="t3", root_dir=tmp_path / "cps")
    mw = CheckpointMiddleware(manager)
    mw.wrap(lambda e: "ok")(make_effect("grep", {"pattern": "x"}))
    assert manager.list_recent() == []


def test_session_journals_checkpoint_events_and_rolls_back(tmp_path):
    target = tmp_path / "work.txt"
    target.write_text("before", encoding="utf-8")
    manager = CheckpointManager(session_id="s", root_dir=tmp_path / "cps")

    class Script:
        def __init__(self):
            self.items = [
                SampleResult(text="", tool_calls=(
                    ("file_ops", {"action": "write", "path": str(target), "content": "after"}),)),
                SampleResult(text="done"),
            ]

        async def __call__(self, messages, tools):
            return self.items.pop(0)

    def executor(effect):
        target.write_text(effect.arguments["content"], encoding="utf-8")
        return "written"

    from agent_project.harness.kernel import Kernel, permissive_policy

    session = Session(
        root=tmp_path / "journals",
        sampler=Script(),
        kernel=Kernel(permissive_policy(), executor=executor),
        checkpoint_manager=manager,
    )
    assert asyncio.run(session.run("change the file")) == "done"

    kinds = [e.kind for e in session.events()]
    assert "CheckpointWritten" in kinds
    assert target.read_text(encoding="utf-8") == "after"

    # 按事件里的 checkpoint_id 回滚(rollback(None) 只回滚失败写入, 成功写入须指定 id)
    cp_id = next(e.checkpoint_id for e in session.events() if e.kind == "CheckpointWritten")
    result = session.rollback(cp_id)
    assert result["restored"] is True
    assert target.read_text(encoding="utf-8") == "before"


def test_session_without_checkpoints_rollback_is_none(tmp_path):
    class Script:
        async def __call__(self, messages, tools):
            return SampleResult(text="ok")

    session = Session(root=tmp_path, sampler=Script())
    assert session.rollback() is None
