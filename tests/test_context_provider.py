"""ContextProvider ABC + PromptPrefixCache + Anthropic prompt caching 单测(无网络)."""
from agent_project.cache import PromptPrefixCache
from agent_project.context_engine import ContextEngine, ContextProvider


def test_context_engine_implements_provider():
    assert issubclass(ContextEngine, ContextProvider)
    for m in ("observe_user", "observe_assistant", "observe_tool_call",
              "observe_tool_result", "consolidate", "compress_working_memory",
              "seed_history", "build_system_context"):
        assert callable(getattr(ContextEngine, m, None)), f"缺方法 {m}"


def test_cannot_instantiate_bare_provider():
    try:
        ContextProvider()
    except TypeError:
        return
    raise AssertionError("裸 ABC 应不可实例化")


def test_prefix_cache_hit_and_stats():
    c = PromptPrefixCache(capacity=4)
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return "SYS"

    assert c.get_or_build("ns", ("a", "b"), builder) == "SYS"
    assert c.get_or_build("ns", ("a", "b"), builder) == "SYS"
    assert calls["n"] == 1, "第二次应命中不重建"
    assert c.stats() == {"hits": 1, "misses": 1, "size": 1}
    assert c.get_or_build("ns", ("c",), builder) == "SYS"
    assert c.stats()["misses"] == 2


def test_prefix_cache_evicts_lru():
    c = PromptPrefixCache(capacity=2)
    c.get_or_build("ns", ("1",), lambda: "one")
    c.get_or_build("ns", ("2",), lambda: "two")
    c.get_or_build("ns", ("3",), lambda: "three")
    assert c.stats()["size"] == 2


def test_anthropic_cache_blocks_shape():
    from agent_project.model_backends import AnthropicBackend
    blocks = AnthropicBackend.cached_system_blocks("hello")
    assert blocks == [{
        "type": "text", "text": "hello",
        "cache_control": {"type": "ephemeral"},
    }]


def test_anthropic_system_prompt_memoized():
    from agent_project.model_backends import AnthropicBackend
    b = AnthropicBackend.__new__(AnthropicBackend)
    b._system_prompt_cache = {}
    builds = {"n": 0}
    b._build_system_prompt = lambda n: (builds.__setitem__("n", builds["n"] + 1), f"SYS{n}")[1]
    assert b.get_system_prompt(1) == "SYS1"
    assert b.get_system_prompt(1) == "SYS1"
    assert builds["n"] == 1, "同 n_loops 只组装一次"
    assert b.get_system_prompt(2) == "SYS2"
    assert builds["n"] == 2
