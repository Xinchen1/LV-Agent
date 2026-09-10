"""system_override 单测: 快路跳过后端默认 system(无网络, 假 client 验参)."""
from agent_project.model_backends import AnthropicBackend, OpenAIBackend


class FakeAnthropicMessages:
    def __init__(self):
        self.created = None

    def create(self, **kw):
        self.created = kw

        class Usage:
            input_tokens = 10
            output_tokens = 5

        class Block:
            type = "text"
            text = "hi"

        class Resp:
            usage = Usage()
            content = [Block()]
        return Resp()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeAnthropicMessages()


def test_anthropic_override_skips_system():
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model, b.max_tokens, b.temperature = "m", 64, 0.0
    b._system_prompt_cache = {}
    b._build_system_prompt = lambda n: "DEFAULT SYSTEM"
    b._client = FakeAnthropicClient()
    out = b.generate("hi", system_override="")
    assert out == "hi"
    assert "system" not in b._client.messages.created, "空覆盖应省略 system 字段"


def test_anthropic_default_keeps_system():
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model, b.max_tokens, b.temperature = "m", 64, 0.0
    b._system_prompt_cache = {}
    b._build_system_prompt = lambda n: "DEFAULT SYSTEM"
    b._client = FakeAnthropicClient()
    b.generate("hi")
    assert b._client.messages.created.get("system") == "DEFAULT SYSTEM"


def test_anthropic_custom_system_string():
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model, b.max_tokens, b.temperature = "m", 64, 0.0
    b._system_prompt_cache = {}
    b._build_system_prompt = lambda n: "DEFAULT SYSTEM"
    b._client = FakeAnthropicClient()
    b.generate("hi", system_override="CUSTOM")
    assert b._client.messages.created.get("system") == "CUSTOM"


def test_openai_override_drops_system_message():
    import agent_project.model_backends as MB

    captured = {}

    class FakeResp:
        class Choice:
            class Msg:
                content = "yo"
                reasoning_content = ""
            message = Msg()
            finish_reason = "stop"
        choices = [Choice()]
        usage = None

    class FakeClient:
        def __init__(self):
            self.chat = self

        class completions:
            @staticmethod
            def create(**kw):
                captured.update(kw)
                return FakeResp()

    b = OpenAIBackend.__new__(OpenAIBackend)
    b.model, b.max_tokens, b.temperature = "m", 64, 0.0
    b.top_p, b.timeout = 0.9, 30
    b.base_url = "http://x"
    b._consecutive_errors = 0
    b._client = FakeClient()
    b._build_system_prompt = lambda n: "DEFAULT SYSTEM"
    b._cb_check = lambda: None
    b._ensure_connection = lambda: None
    out = b.generate("hi", system_override="")
    assert out == "yo"
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user"], f"空覆盖应只剩 user 消息, 实际 {roles}"
