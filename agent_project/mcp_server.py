"""Expose LV Agent as a remote MCP server (Streamable HTTP).

背景: 第三方 Agent 平台要「连我们的工具」时, 需要一个可远程访问的 MCP endpoint
(形如 https://host/mcp) + 可选 Bearer token。本项目此前只有 tools/mcp_client.py
这个 stdio **客户端**, 没有任何对外的 server, 所以那个框里没有地址可填。

本模块把 TOOLS_REGISTRY 里的内置工具原样暴露为 MCP tools, 并额外提供一个
`lv_agent` 工具把整个 Agent(含推理循环/记忆/反思)当作一个任务级工具调用。

    python -m agent_project.mcp_server                          # 127.0.0.1:8788/mcp
    python -m agent_project.mcp_server --host 0.0.0.0 --token S  # 对外暴露

环境变量: LV_MCP_HOST / LV_MCP_PORT / LV_MCP_PATH / LV_MCP_TOKEN /
          LV_MCP_STATELESS / LV_MCP_ALLOWED_HOSTS / LV_MCP_JSON_RESPONSE
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import anyio

try:
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.transport_security import TransportSecuritySettings
except ModuleNotFoundError as _mcp_missing:  # pragma: no cover - 只在未装 SDK 时触发
    raise ModuleNotFoundError(
        "MCP server 需要官方 SDK: pip install 'mcp>=2.0'"
    ) from _mcp_missing

from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

SERVER_NAME = "lv-agent"
SERVER_VERSION = "1.1.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788
DEFAULT_PATH = "/mcp"
HEALTH_PATH = "/healthz"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# 任务级工具: 把整个 Agent 包成一个 MCP tool, 供「只会调工具」的宿主 Agent 使用。
AGENT_TOOL_NAME = "lv_agent"
AGENT_TOOL_DESC: Dict[str, Any] = {
    "name": AGENT_TOOL_NAME,
    "description": (
        "Delegate a complete task to the LV Agent and get its final answer. "
        "The agent runs its own reasoning loop with memory, reflection and the "
        "full built-in tool suite, so prefer this over chaining individual tools "
        "when the task is open-ended."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The full natural-language task for the agent to complete.",
            },
            "code_mode": {
                "type": "boolean",
                "description": "Force coding/engineering workflow. Default false.",
            },
        },
        "required": ["task"],
    },
}

_AGENT_SINGLETON: Dict[str, Any] = {}
_AGENT_LOCK = threading.Lock()

# 调用观测: 判断「对面到底有没有在用」的唯一硬证据, /healthz 直接可读。
_CALL_STATS: Dict[str, Any] = {"calls": 0, "errors": 0, "last_tool": None, "last_call_at": None}
_STATS_LOCK = threading.Lock()


def _record_call(name: str, is_error: bool) -> None:
    with _STATS_LOCK:
        _CALL_STATS["calls"] += 1
        if is_error:
            _CALL_STATS["errors"] += 1
        _CALL_STATS["last_tool"] = name
        _CALL_STATS["last_call_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def call_stats() -> Dict[str, Any]:
    with _STATS_LOCK:
        return dict(_CALL_STATS)


def _fingerprint(value: str) -> str:
    """token 指纹: 够用来判断「客户端带的是不是同一把」, 又不至于泄密。"""
    import hashlib

    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


# ============ Schema 适配 ============

def normalize_input_schema(parameters: Any) -> Dict[str, Any]:
    """把 OpenAI 风格 tool schema 规整成 MCP 合法的 inputSchema。

    MCP 要求 inputSchema 是 object 且带 properties; 老工具的 parameters 可能缺字段
    或写成空 dict, 直接透传会让对端 tools/list 校验失败。
    """
    schema: Dict[str, Any] = dict(parameters) if isinstance(parameters, dict) else {}
    props = schema.get("properties")
    schema["properties"] = dict(props) if isinstance(props, dict) else {}
    schema["type"] = "object"
    required = schema.get("required")
    schema["required"] = [r for r in required if isinstance(r, str)] if isinstance(required, list) else []
    return schema


def required_args(desc: Dict[str, Any]) -> List[str]:
    return normalize_input_schema(desc.get("parameters")).get("required", [])


def _resolve_registry(registry: Any = None) -> Any:
    if registry is None:
        from .tools import TOOLS_REGISTRY

        return TOOLS_REGISTRY
    return registry


def collect_descriptors(registry: Any, include_agent_tool: bool = True) -> Dict[str, Dict[str, Any]]:
    """从 ToolRegistry 抽出 {name: desc}, 可选并入任务级 lv_agent 工具。"""
    descriptors: Dict[str, Dict[str, Any]] = {}
    for name in registry.list_tools():
        desc = registry.get_tool_schema(name) or {}
        descriptors[name] = {
            "name": name,
            "description": desc.get("description") or "",
            "parameters": desc.get("parameters") or {},
        }
    if include_agent_tool:
        descriptors[AGENT_TOOL_NAME] = dict(AGENT_TOOL_DESC)
    return descriptors


# ============ 调用分发 ============

def _render_tool_result(result: Any) -> Tuple[str, bool]:
    """把 BaseTool 的 ToolResult 压成 (文本, 是否错误)。"""
    success = bool(getattr(result, "success", True))
    output = getattr(result, "output", None)
    if output is None:
        text = json.dumps(result, ensure_ascii=False, default=str)
    elif isinstance(output, str):
        text = output
    else:
        text = json.dumps(output, ensure_ascii=False, default=str)
    error = getattr(result, "error", None)
    if error:
        text = f"{text}\n[error] {error}".strip()
        success = False
    return text, not success


def run_agent_task(task: str, code_mode: bool = False, agent_factory: Optional[Callable[[], Any]] = None) -> str:
    """跑一次完整 Agent, 只回传最终答案(丢弃中间轨迹, 免得撑爆对端上下文)。"""
    agent = agent_factory() if agent_factory is not None else get_agent_singleton()
    result = agent.run(task, code_mode=code_mode)
    if isinstance(result, dict):
        for key in ("final_answer", "answer", "response", "output"):
            value = result.get(key)
            if value:
                return str(value)
        trimmed = {k: v for k, v in result.items() if k in ("success", "task", "error")}
        return json.dumps(trimmed, ensure_ascii=False, default=str)
    return str(result)


def get_agent_singleton() -> Any:
    """惰性构建进程内唯一 Agent(构造 ~200ms 且要读配置, 不能在 import 期做)。"""
    agent = _AGENT_SINGLETON.get("agent")
    if agent is not None:
        return agent
    with _AGENT_LOCK:
        agent = _AGENT_SINGLETON.get("agent")
        if agent is None:
            from .agent import OpenMythosAgent
            from .config import AgentConfig

            agent = OpenMythosAgent(AgentConfig())
            _AGENT_SINGLETON["agent"] = agent
    return agent


def dispatch_tool_call(
    registry: Any,
    descriptors: Dict[str, Dict[str, Any]],
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    agent_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[str, bool]:
    """同步分发一次 tool call, 返回 (文本, is_error)。协议层只做线程卸载。"""
    args = dict(arguments or {})
    desc = descriptors.get(name)
    if desc is None:
        return f"Unknown tool '{name}'. Available: {', '.join(sorted(descriptors))}", True

    missing = [key for key in required_args(desc) if key not in args]
    if missing:
        return f"Missing required argument(s) for '{name}': {', '.join(missing)}", True

    if name == AGENT_TOOL_NAME:
        try:
            return run_agent_task(
                str(args.get("task", "")),
                code_mode=bool(args.get("code_mode", False)),
                agent_factory=agent_factory,
            ), False
        except Exception as exc:
            logger.exception("lv_agent tool call failed")
            return f"{type(exc).__name__}: {exc}", True

    tool = registry.get(name)
    if tool is None:  # pragma: no cover - descriptors 与 registry 同源, 仅防御
        return f"Unknown tool '{name}'. Available: {', '.join(sorted(descriptors))}", True

    try:
        # 复用解析层已有的 schema 纠偏: 宿主常把 int/array 传成字符串。
        from .policies import ToolCallParser

        coerced = ToolCallParser._coerce_args_schema(name, args)
        return _render_tool_result(tool.execute(**coerced))
    except Exception as exc:
        logger.exception("tool %s failed", name)
        return f"{type(exc).__name__}: {exc}", True


# ============ MCP server ============

def build_server(
    registry: Any = None,
    *,
    include_agent_tool: bool = True,
    agent_factory: Optional[Callable[[], Any]] = None,
) -> Server:
    """组装 MCP server: 注册表工具 + lv_agent 任务级工具。"""
    registry = _resolve_registry(registry)
    descriptors = collect_descriptors(registry, include_agent_tool=include_agent_tool)

    async def on_list_tools(_context: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=desc["name"],
                    description=desc["description"],
                    inputSchema=normalize_input_schema(desc["parameters"]),
                )
                for desc in descriptors.values()
            ]
        )

    async def on_call_tool(_context: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        # 工具执行是阻塞 IO(搜索/子进程/文件), 必须丢线程池, 否则会把事件循环卡死。
        started = time.perf_counter()
        text, is_error = await anyio.to_thread.run_sync(
            lambda: dispatch_tool_call(
                registry, descriptors, params.name, params.arguments, agent_factory=agent_factory
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        _record_call(params.name, is_error)
        preview = ", ".join(
            f"{key}={str(value)[:60]!r}" for key, value in list((params.arguments or {}).items())[:4]
        )
        logger.info(
            "mcp tool call: %s (%.0fms, is_error=%s) %s", params.name, elapsed_ms, is_error, preview
        )
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)

    return Server(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "LV Agent tools. Use `lv_agent` to delegate a whole task end-to-end; "
            "call the individual tools for narrow, single-step work."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


# ============ HTTP 层 ============

class BearerTokenMiddleware:
    """只保护 MCP 路径的 token 校验(常数时间比较)。

    裸 ASGI 中间件而非 Starlette BaseHTTPMiddleware: 后者会吃掉 SSE 流式响应。

    不同 MCP 客户端递 token 的写法不统一(Authorization: Bearer x / Authorization: x /
    x-api-key: x), 这里都收, 否则会出现「网络通但恒 401」的死局 —— 面板侧看不到原因。
    """

    AUTH_HEADERS = ("authorization", "x-api-key", "api-key", "mcp-bearer-token")

    def __init__(self, app: Any, token: str, protected_paths: Sequence[str] = (DEFAULT_PATH,)) -> None:
        self.app = app
        self.token = token
        self.protected_paths = tuple(protected_paths)

    def _needs_auth(self, path: str) -> bool:
        return any(path == p or path.startswith(p.rstrip("/") + "/") for p in self.protected_paths)

    @classmethod
    def _presented_token(cls, headers: Dict[str, str]) -> Tuple[Optional[str], List[str]]:
        """返回 (候选 token, 出现过的鉴权头名列表)。"""
        present = [name for name in cls.AUTH_HEADERS if headers.get(name)]
        for name in cls.AUTH_HEADERS:
            raw = (headers.get(name) or "").strip()
            if not raw:
                continue
            if raw.lower().startswith("bearer "):
                raw = raw[len("bearer "):].strip()
            return raw, present
        return None, present

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._needs_auth(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        presented, present = self._presented_token(headers)
        if presented is None or not hmac.compare_digest(presented, self.token):
            # 只记指纹不记明文: 日志会外流, token 不能落盘。
            logger.warning(
                "mcp auth failed from %s: headers=%s presented=%s",
                scope.get("client", ("?",))[0],
                present or "none",
                f"len={len(presented)} sha256={_fingerprint(presented)}" if presented else "absent",
            )
            response = JSONResponse(
                {"error": "unauthorized", "hint": "send 'Authorization: Bearer <token>'"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _transport_security(host: str, allowed_hosts: Optional[Sequence[str]]) -> Optional[TransportSecuritySettings]:
    if allowed_hosts:
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed_hosts),
            allowed_origins=[],
        )
    if host in LOCAL_HOSTS:
        # 交给 SDK 自动开启本地防护(它会放开 127.0.0.1/localhost/::1 三种写法)。
        return None
    # 绑 0.0.0.0/公网 IP 时 SDK 不自动开防护: Host 头由部署方域名决定, 硬编码白名单会误杀。
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def create_app(
    registry: Any = None,
    *,
    token: Optional[str] = None,
    host: str = DEFAULT_HOST,
    path: str = DEFAULT_PATH,
    stateless: bool = False,
    json_response: bool = False,
    include_agent_tool: bool = True,
    agent_factory: Optional[Callable[[], Any]] = None,
    allowed_hosts: Optional[Sequence[str]] = None,
) -> Any:
    """构造可直接交给 uvicorn 的 ASGI app。"""
    registry = _resolve_registry(registry)
    server = build_server(registry, include_agent_tool=include_agent_tool, agent_factory=agent_factory)
    tool_count = len(collect_descriptors(registry, include_agent_tool=include_agent_tool))

    async def health(_request: Any) -> JSONResponse:
        payload: Dict[str, Any] = {
            "status": "ok",
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "tools": tool_count,
            "auth": "bearer" if token else "none",
        }
        payload.update(call_stats())
        return JSONResponse(payload)

    app = server.streamable_http_app(
        streamable_http_path=path,
        json_response=json_response,
        stateless_http=stateless,
        host=host,
        transport_security=_transport_security(host, allowed_hosts),
        custom_starlette_routes=[Route(HEALTH_PATH, health, methods=["GET"])],
    )
    if token:
        app = BearerTokenMiddleware(app, token=token, protected_paths=(path,))
    return app


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str) -> Optional[List[str]]:
    raw = os.environ.get(name)
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _configure_logging() -> None:
    """给模块 logger 挂一个 handler。

    uvicorn 只配置自己那三个 logger, root 没有 handler, 于是本模块的 logger.info
    会被静默丢弃 —— 明明打了调用日志却什么都看不到。
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [mcp] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m agent_project.mcp_server",
        description="Serve LV Agent tools over MCP (Streamable HTTP).",
    )
    parser.add_argument("--host", default=os.environ.get("LV_MCP_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LV_MCP_PORT", DEFAULT_PORT)))
    parser.add_argument("--path", default=os.environ.get("LV_MCP_PATH", DEFAULT_PATH))
    parser.add_argument("--token", default=os.environ.get("LV_MCP_TOKEN") or None)
    parser.add_argument("--stateless", action="store_true", default=_env_bool("LV_MCP_STATELESS"))
    parser.add_argument("--json-response", action="store_true", default=_env_bool("LV_MCP_JSON_RESPONSE"))
    parser.add_argument("--no-agent-tool", action="store_true", help="不暴露任务级 lv_agent 工具")
    parser.add_argument("--check", action="store_true", help="只打印要填的 endpoint, 不起服务")
    args = parser.parse_args(argv)

    include_agent_tool = not args.no_agent_tool
    allowed_hosts = _env_list("LV_MCP_ALLOWED_HOSTS")

    if args.host not in LOCAL_HOSTS and not args.token:
        logger.warning(
            "对外暴露(%s)却没设 --token: 任何人都能调你的 shell/文件工具。生产请务必加 token。",
            args.host,
        )

    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    endpoint = f"http://{shown_host}:{args.port}{args.path}"
    banner = "\n".join(
        [
            "MCP endpoint（填进第三方面板的「MCP endpoint」框）:",
            f"  {endpoint}",
            f"工具名: 可留空, 自动发现（共 {len(collect_descriptors_from_registry(include_agent_tool))} 个）",
            f"鉴权: {'Bearer ' + args.token if args.token else '无（本机调试用）'}",
            f"健康检查: {endpoint.replace(args.path, HEALTH_PATH)}",
        ]
    )
    print(banner, flush=True)
    if args.check:
        return 0

    import uvicorn

    app = create_app(
        token=args.token,
        host=args.host,
        path=args.path,
        stateless=args.stateless,
        json_response=args.json_response,
        include_agent_tool=include_agent_tool,
        allowed_hosts=allowed_hosts,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def collect_descriptors_from_registry(include_agent_tool: bool = True) -> Dict[str, Dict[str, Any]]:
    return collect_descriptors(_resolve_registry(), include_agent_tool=include_agent_tool)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())