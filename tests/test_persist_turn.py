"""_persist_turn 单测: 快慢共用落盘 + 闲聊跳图谱(无网络)."""
from agent_project.agent import OpenMythosAgent


class Cfg:
    class memory:
        enabled = True


class FakeCE:
    def __init__(self):
        self.consolidated = []

    def consolidate(self, task, traj):
        self.consolidated.append(task)


class FakeBuffer:
    def __init__(self):
        self.episodes = []

    def add_episode(self, task, trajectory, task_type):
        self.episodes.append(task)
        return "ep1"

    def add_lesson(self, lesson):
        pass

    def count(self):
        return len(self.episodes)


class FakeWiki:
    def __init__(self):
        self.saved = []

    def remember(self, text, context=None, auto_link=True):
        self.saved.append(text)
        return ["p1"]


def make_persist_agent(**kw):
    import logging
    a = OpenMythosAgent.__new__(OpenMythosAgent)
    a.logger = logging.getLogger("persist_test")
    a.config = Cfg()
    a.context_engine = kw.get("ce", FakeCE())
    a.experience_buffer = kw.get("buf", FakeBuffer())
    a.memory_manager = kw.get("wiki", FakeWiki())
    a.memskill_engine = None
    a.skill_engine = None
    a.episodes_completed = 0
    a._log_to_file = lambda *args, **k: None
    a._extract_lessons = lambda traj: []
    return a


def tool_traj():
    return {"task": "列目录", "final_answer": "有 a.md",
            "success": True, "actions": [{"tool_name": "file_ops"}],
            "observations": [], "metadata": {"mode": "fast", "fast_path": True}}


def test_persist_writes_all_channels():
    a = make_persist_agent()
    a._persist_turn("列目录", tool_traj())
    assert a.context_engine.consolidated == ["列目录"], "consolidate 应被调"
    assert a.experience_buffer.episodes == ["列目录"], "经验应落盘(含快路)"
    assert len(a.memory_manager.saved) == 1, "有工具证据应写 wiki"
    assert a.episodes_completed == 1


def test_chitchat_skips_wiki_but_keeps_experience():
    a = make_persist_agent()
    traj = {"task": "你好", "final_answer": "你好呀", "success": True,
            "actions": [], "observations": [],
            "metadata": {"mode": "fast", "fast_path": True}}
    a._persist_turn("你好", traj)
    assert a.memory_manager.saved == [], "纯闲聊不灌 wiki"
    assert a.experience_buffer.episodes == ["你好"], "闲聊仍记经验(反思不饥荒)"


def test_persist_tolerates_dead_backends():
    class BoomCE:
        def consolidate(self, *a, **k):
            raise RuntimeError("down")
    a = make_persist_agent(ce=BoomCE(), buf=None, wiki=None)
    a.experience_buffer = None
    a.memory_manager = None
    # 全坏也不应抛, 只记 episodes+1 与否不重要, 关键不崩
    a._persist_turn("t", tool_traj())
