"""工具发现单一事实源回归: 注册表工具必须全部出现在 system prompt 文本协议表中."""
from agent_project.tools import TOOLS_REGISTRY, render_compact_tool_list


def test_collect_context_parts_labels():
    from agent_project.agent import OpenMythosAgent
    a = OpenMythosAgent.__new__(OpenMythosAgent)
    a._get_skill_context = lambda task: "SK"
    fast = a._collect_context_parts("t", "H", "M", "## Relevant Memory:\n")
    assert fast == ["H", "## Relevant Memory:\nM", "## Active Skill Instructions:\nSK"]
    deep = a._collect_context_parts("t", "H", "M")
    assert deep == ["H", "M", "## Active Skill Instructions:\nSK"]
    assert a._collect_context_parts("t") == ["## Active Skill Instructions:\nSK"]
    a._get_skill_context = lambda task: ""
    assert a._collect_context_parts("t") == []


def test_fast_prompt_keeps_backend_only_disciplines():
    """快路跳过后端 system 后, 后端独占纪律必须在快 prompt 内自给自足."""
    import logging
    from agent_project.agent import OpenMythosAgent
    a = OpenMythosAgent.__new__(OpenMythosAgent)
    a.logger = logging.getLogger("disc")
    a.context_engine = None
    a._get_skill_context = lambda t: ""
    a._get_skill_tool_hint = lambda: []
    a._recall_past_conversations = lambda *a_, **k: ""
    p = a._build_fast_prompt("总结报告", memory_context="", history_context="")
    for marker in ("MEMORY RECALL DISCIPLINE", "HONEST SUMMARY",
                   "READING A FILE", "<think>", "[TOOL:tool_name]"):
        assert marker in p, f"快 prompt 缺纪律: {marker}"


def test_compact_list_covers_registry():
    names = set(TOOLS_REGISTRY.list_tools())
    text = render_compact_tool_list()
    assert "{TOOL_LIST}" not in text
    missing = [n for n in names if f"- {n}(" not in text]
    assert not missing, f"这些工具在文本协议中不可见: {missing}"


def test_system_prompts_expose_all_tools():
    from agent_project.model_backends import AnthropicBackend, OpenAIBackend
    names = set(TOOLS_REGISTRY.list_tools())
    a = AnthropicBackend.__new__(AnthropicBackend)
    a._system_prompt_cache = {}
    s1 = a.get_system_prompt(1)
    assert "{TOOL_LIST}" not in s1
    o = OpenAIBackend.__new__(OpenAIBackend)
    o.model = "test"
    s2 = o._build_system_prompt(1)
    assert "{TOOL_LIST}" not in s2
    for s in (s1, s2):
        missing = [n for n in names if n not in s]
        assert not missing, f"system prompt 缺工具: {missing}"
