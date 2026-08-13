"""Legacy-stack adapters: tool-block parsing, prompt flattening, BackendSampler."""

import asyncio

import pytest

from agent_project.harness.bridge import (
    BackendSampler,
    build_kernel_from_registry,
    kernel_guarded_executor,
    parse_tool_blocks,
)
from agent_project.harness.errors import PolicyDeniedError
from agent_project.harness.kernel import Decision, Rule


def test_parse_json_tool_block():
    text = 'I will read the file. [TOOL:file_ops] {"action": "read", "path": "a.txt"} [/TOOL] done'
    clean, calls = parse_tool_blocks(text)
    assert calls == (("file_ops", {"action": "read", "path": "a.txt"}),)
    assert "[TOOL:" not in clean and "I will read the file." in clean


def test_parse_key_value_tool_block():
    text = '[TOOL:web_search] query="openai news" limit="3" [/TOOL]'
    _clean, calls = parse_tool_blocks(text)
    assert calls == (("web_search", {"query": "openai news", "limit": "3"}),)


def test_unparseable_body_stays_in_text():
    text = "[TOOL:grep] not-valid-body [/TOOL]"
    clean, calls = parse_tool_blocks(text)
    assert calls == () and clean == text


def test_multiple_tool_blocks():
    text = ('[TOOL:grep] {"pattern": "a"} [/TOOL] mid '
            '[TOOL:glob] {"pattern": "*.py"} [/TOOL]')
    clean, calls = parse_tool_blocks(text)
    assert [c[0] for c in calls] == ["grep", "glob"]
    assert "mid" in clean


class _FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, n_loops=1, temperature=None, max_tokens=None,
                 tools=None, stream=False, stream_callback=None,
                 token_callback=None, **kwargs):
        self.prompts.append(prompt)
        if token_callback:
            token_callback(123)
        return self.reply


def test_backend_sampler_parses_reply_and_reports_tokens():
    backend = _FakeBackend('answer [TOOL:grep] {"pattern": "x"} [/TOOL]')
    sampler = BackendSampler(backend)
    seen_tokens = []
    sampler.on_tokens = seen_tokens.append

    result = asyncio.run(sampler(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}], []
    ))
    assert result.text == "answer"
    assert result.tool_calls == (("grep", {"pattern": "x"}),)
    assert result.prompt_tokens == 123 and seen_tokens == [123]
    assert "SYS" in backend.prompts[0] and "hi" in backend.prompts[0]


def test_backend_sampler_no_tools_plain_text():
    sampler = BackendSampler(_FakeBackend("plain answer"))
    result = asyncio.run(sampler([{"role": "user", "content": "q"}], []))
    assert result.text == "plain answer" and result.tool_calls == ()


def test_kernel_guarded_executor_denies():
    from agent_project.harness.kernel import Kernel

    kernel = Kernel(
        [Rule(Decision.DENY, tool="bash_exec", reason="no"),
         Rule(Decision.ALLOW, reason="rest")],
        executor=lambda e: "ok",
    )
    guarded = kernel_guarded_executor(kernel)
    with pytest.raises(PolicyDeniedError):
        guarded("bash_exec", {"command": "ls"})
    assert guarded("calculator", {"expression": "1+1"}) == "ok"


def test_build_kernel_from_registry_uses_real_tools():
    from agent_project.tools import TOOLS_REGISTRY

    kernel = build_kernel_from_registry(
        TOOLS_REGISTRY, policy=[Rule(Decision.ALLOW, reason="all")]
    )
    out = kernel.run_effect if hasattr(kernel, "run_effect") else kernel.run
    from agent_project.harness.effects import make_effect

    result = out(make_effect("calculator", {"expression": "2+3"}))
    assert "5" in result
