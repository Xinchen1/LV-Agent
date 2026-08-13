"""Integration-style tests for context / continuation fixes."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent_project.agent import OpenMythosAgent
from agent_project.context_engine import WorkingMemory


def test_continuation_detection():
    """The OSWorld follow-up should be detected as a continuation."""
    fn = OpenMythosAgent._is_continuation_query
    # Pass None as self because the method does not use self.
    assert fn(None, "这个榜单很权威吗,真实性") is True
    assert fn(None, "该榜单真实性如何") is True
    assert fn(None, "上面说的对么") is True
    assert fn(None, "搜索 AI") is False
    print("PASSED: continuation detection covers Chinese pronouns")


def test_pronoun_resolution():
    """Pronoun resolver should match '这个榜单' style references."""
    class DummyAgent:
        _extract_most_recent_entity = OpenMythosAgent._extract_most_recent_entity
        _PRONOUN_SUB_RE = OpenMythosAgent._PRONOUN_SUB_RE
    fn = OpenMythosAgent._resolve_pronoun_to_entity
    recent_text = (
        "User: OSWorld 这个权威吗,其他上榜的还有什么\n"
        "Assistant: OSWorld 权威性较强。其他相关评测包括 OSWorld-Verified 和 OSWorld 2.0。"
    )
    resolved = fn(DummyAgent(), "这个榜单很权威吗,真实性", recent_text)
    print(f"resolved: {resolved}")
    assert resolved is not None
    assert "OSWorld" in resolved
    print("PASSED: pronoun resolution handles '这个榜单'")


def test_working_memory_retention():
    """Latest assistant turn must survive truncation so references resolve."""
    wm = WorkingMemory()
    q1 = "OSWorld 这个权威吗,其他上榜的还有什么"
    a1 = (
        "OSWorld 权威性较强，属于学术界广泛认可的基准测试。"
        "其他相关评测包括 OSWorld-Verified（29 个模型上榜）和 OSWorld 2.0（14 个模型上榜）。"
    )
    q2 = "这个榜单很权威吗,真实性"
    wm.add("user", q1, "message")
    wm.add("assistant", a1, "message")
    wm.add("user", q2, "message")

    ctx = wm.format_for_prompt(max_tokens=1200, include_tool_history=False)
    assert "OSWorld" in ctx
    assert "这个榜单很权威吗" in ctx
    assert "29 个模型" in ctx
    print("PASSED: working memory retains latest turn details")


if __name__ == "__main__":
    test_continuation_detection()
    test_pronoun_resolution()
    test_working_memory_retention()
