"""MCP server 单测: schema 适配 / dispatch 分发 / Bearer 鉴权 / 真实 Streamable HTTP 握手。

无网络、无 LLM: 全程用假工具 + ASGI 直连, 只验证协议面。
"""
import pytest

pytest.importorskip("mcp", reason="需要官方 MCP SDK: pip install 'mcp>=2.0'")

import anyio  # noqa: E402
from httpx2 import ASGITransport, AsyncClient  # noqa: E402

from agent_project.mcp_server import (  # noqa: E402
    AGENT_TOOL_NAME,
    BearerTokenMiddleware,
    HEALTH_PATH,
    _resolve_registry,
    collect_descriptors,
    create_app,
    dispatch_tool_call,
    normalize_input_schema,
    required_args,
)
from agent_project.tools import TOOLS_REGISTRY, BaseTool, ToolRegistry, ToolResult  # noqa: E402


# ============ Fakes ============

class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the payload back."
    parameters = {
        "type": "object",
        "properties": {"payload": {"type": "string", "description": "text to echo"}},
        "required": ["payload"],
    }

    def execute(self, **kwargs):
        return ToolResult(success=True, output=f"echo:{kwargs.get('payload')}")


class BrokenTool(BaseTool):
    name = "broken"
    description = "Always explodes."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        raise RuntimeError("boom")


def make_registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(BrokenTool())
    return reg


class StubAgent:
    def __init__(self, answer="stub answer"):
        self.answer = answer
        self.calls = []

    def run(self, task, mode="production", stream_callback=None, token_callback=None, code_mode=False):
        self.calls.append((task, code_mode))
        return {"success": True, "final_answer": self.answer, "trajectory": ["lots of noise"]}


# ============ Schema ============

def test_normalize_input_schema_fills_missing_shape():
    assert normalize_input_schema(None) == {"type": "object", "properties": {}, "required": []}
    assert normalize_input_schema({"properties": None, "required": "task"}) == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_normalize_input_schema_keeps_valid_schema():
    schema = normalize_input_schema(EchoTool.parameters)
    assert schema["type"] == "object"
    assert schema["required"] == ["payload"]
    assert "payload" in schema["properties"]


def test_required_args_reads_schema():
    assert required_args({"parameters": EchoTool.parameters}) == ["payload"]
    assert required_args({}) == []


def test_collect_descriptors_adds_agent_tool():
    descs = collect_descriptors(make_registry(), include_agent_tool=True)
    assert {"echo", "broken", AGENT_TOOL_NAME} <= set(descs)
    assert descs[AGENT_TOOL_NAME]["parameters"]["required"] == ["task"]

    assert AGENT_TOOL_NAME not in collect_descriptors(make_registry(), include_agent_tool=False)


def test_real_registry_tools_are_exposed():
    names = set(collect_descriptors(_resolve_registry(), include_agent_tool=False))
    assert "calculator" in names, "内置工具应当能被 MCP 看到"


# ============ Dispatch ============

def test_dispatch_runs_tool_and_renders_output():
    text, is_error = dispatch_tool_call(make_registry(), collect_descriptors(make_registry()), "echo", {"payload": "hi"})
    assert (text, is_error) == ("echo:hi", False)


def test_dispatch_unknown_tool_is_error():
    text, is_error = dispatch_tool_call(make_registry(), collect_descriptors(make_registry()), "nope", {})
    assert is_error and "Unknown tool" in text and "echo" in text


def test_dispatch_missing_required_arg_is_error():
    text, is_error = dispatch_tool_call(make_registry(), collect_descriptors(make_registry()), "echo", {})
    assert is_error and "payload" in text


def test_dispatch_tool_exception_becomes_error_text():
    text, is_error = dispatch_tool_call(make_registry(), collect_descriptors(make_registry()), "broken", {})
    assert is_error and "RuntimeError: boom" in text, "异常要转成可读文本而不是裸栈"


def test_dispatch_failed_result_carries_error_field():
    class FailingTool(BaseTool):
        name = "failing"
        description = "fails politely"
        parameters = {"type": "object", "properties": {}}

        def execute(self, **kwargs):
            return ToolResult(success=False, output="partial", error="nope")

    reg = ToolRegistry()
    reg.register(FailingTool())
    text, is_error = dispatch_tool_call(reg, collect_descriptors(reg), "failing", {})
    assert is_error and "nope" in text and "partial" in text


def test_dispatch_agent_tool_returns_final_answer_only():
    agent = StubAgent("结论在这里")
    reg = make_registry()
    text, is_error = dispatch_tool_call(
        reg,
        collect_descriptors(reg),
        AGENT_TOOL_NAME,
        {"task": "算一下 2+2", "code_mode": True},
        agent_factory=lambda: agent,
    )
    assert (text, is_error) == ("结论在这里", False)
    assert agent.calls == [("算一下 2+2", True)], "轨迹不该回传给对端"


def test_dispatch_agent_tool_agent_failure_is_error():
    class ExplodingAgent:
        def run(self, *a, **kw):
            raise ValueError("模型没配")

    reg = make_registry()
    text, is_error = dispatch_tool_call(
        reg, collect_descriptors(reg), AGENT_TOOL_NAME, {"task": "x"}, agent_factory=ExplodingAgent
    )
    assert is_error and "ValueError: 模型没配" in text


# ============ Auth ============

def test_bearer_middleware_allows_health_but_guards_mcp():
    inner_gut = []

    async def inner(scope, receive, send):
        inner_gut.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def call(path, auth):
        app = BearerTokenMiddleware(inner, token="s3cret")
        headers = [(b"authorization", auth.encode())] if auth else []
        scope = {"type": "http", "path": path, "headers": headers, "method": "POST"}
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request"}

        await app(scope, receive, send)
        return sent[0]["status"]

    async def flow():
        assert await call(HEALTH_PATH, None) == 200, "健康检查不该要 token"
        assert await call("/mcp", None) == 401
        assert await call("/mcp", "Bearer wrong") == 401
        assert await call("/mcp", "Bearer s3cret") == 200

    anyio.run(flow)
    assert inner_gut == [HEALTH_PATH, "/mcp"]


def test_custom_mcp_path_is_the_only_protected_one():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def status(path, auth):
        app = BearerTokenMiddleware(inner, token="t", protected_paths=("/lv/mcp",))
        headers = [(b"authorization", auth.encode())] if auth else []
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request"}

        await app({"type": "http", "path": path, "headers": headers, "method": "POST"}, receive, send)
        return sent[0]["status"]

    async def flow():
        assert await status("/lv/mcp", None) == 401
        assert await status("/mcp", None) == 200, "非受保护路径不受影响"

    anyio.run(flow)


# ============ End-to-end over Streamable HTTP ============

def test_health_endpoint_reports_tools_and_auth():
    app = create_app(make_registry(), host="127.0.0.1")

    async def hit():
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://health") as client:
                resp = await client.get(HEALTH_PATH)
                return resp.status_code, resp.json()

    status, body = anyio.run(hit)
    assert status == 200
    assert body["status"] == "ok" and body["tools"] == 3 and body["auth"] == "none"


def test_real_mcp_handshake_lists_and_calls_tools():
    # 用 0.0.0.0 建 app: 那是公网部署配置(SDK 的 DNS rebinding 防护自动关闭),
    # 否则 ASGI 直连的 Host 头会被本地白名单挡掉。
    app = create_app(make_registry(), host="0.0.0.0")

    async def session_flow():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:8788") as client:
                async with streamable_http_client("http://127.0.0.1:8788/mcp", http_client=client) as streams:
                    read_stream, write_stream = streams[0], streams[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        init = await session.initialize()
                        tools = await session.list_tools()
                        called = await session.call_tool("echo", {"payload": "ping"})
                        unknown = await session.call_tool("echo", {})
                        return init, tools, called, unknown

    init, tools, called, unknown = anyio.run(session_flow)
    assert init.server_info.name == "lv-agent"
    names = {tool.name for tool in tools.tools}
    assert {"echo", "broken", AGENT_TOOL_NAME} <= names
    echo_schema = next(t for t in tools.tools if t.name == "echo")
    assert echo_schema.input_schema["required"] == ["payload"]

    assert called.is_error is False
    assert called.content[0].text == "echo:ping"
    assert unknown.is_error is True and "payload" in unknown.content[0].text