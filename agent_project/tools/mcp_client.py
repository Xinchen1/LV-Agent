"""
Lightweight MCP (Model Context Protocol) stdio client.
Implements JSON-RPC 2.0 over subprocess stdin/stdout without heavy SDK deps.
**UPGRADED: optional Rust bridge for sub-millisecond MCP tool calls.**
"""

import json
import logging
import os
import random
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import TOOLS_REGISTRY, BaseTool, ToolResult


def _find_rust_mcp_bridge() -> Optional[str]:
    """Locate the rust_mcp_bridge binary next to rust_file_ops or on PATH.

    Returns None if the binary exists but has a wrong CPU architecture
    (e.g. x86_64 binary on arm64), so the caller can skip the 6-second
    socket-wait timeout and fall back to the pure-Python stdio client
    immediately.
    """
    candidate = Path(__file__).parent.parent.parent / "rust_file_ops" / "target" / "release" / "rust_mcp_bridge"
    if candidate.exists() and os.access(candidate, os.X_OK):
        # Quick architecture compatibility check — avoids a 6s startup delay
        # when the binary was compiled for a different CPU (copy-to-new-machine).
        try:
            probe = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True, text=True, timeout=3,
            )
            if probe.returncode == 0:
                return str(candidate)
            return None  # binary exists but crashes → wrong arch
        except Exception:
            return None  # "Bad CPU type" or timeout → skip
    from_path = shutil.which("rust_mcp_bridge")
    return from_path


class RustMcpBridge:
    """
    High-performance MCP bridge backed by a Rust async daemon.

    The daemon keeps MCP server processes alive and multiplexes JSON-RPC
    requests over a Unix domain socket.  This avoids Python GIL/threading
    overhead and yields much lower per-call latency than the pure-Python
    stdio client.
    """

    def __init__(self, binary_path: Optional[str] = None, socket_path: Optional[str] = None):
        self.binary_path = binary_path or _find_rust_mcp_bridge()
        self.socket_path = socket_path or f"/tmp/lv-mcp-bridge-{os.getpid()}-{random.randint(1000, 9999)}.sock"
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        if not self.binary_path:
            raise RuntimeError("rust_mcp_bridge binary not found")
        # Remove stale socket.
        try:
            Path(self.socket_path).unlink(missing_ok=True)
        except Exception:
            pass
        self._proc = subprocess.Popen(
            [self.binary_path, "--socket", self.socket_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for the daemon to bind the socket.
        deadline = time.time() + 6
        while time.time() < deadline:
            if Path(self.socket_path).exists():
                # Verify by sending a status request.
                try:
                    self.status(timeout=1)
                    return
                except Exception:
                    pass
            time.sleep(0.05)
        raise RuntimeError(f"rust_mcp_bridge failed to start on {self.socket_path}")

    def _request(self, cmd: Dict[str, Any], timeout: float = 60) -> Dict[str, Any]:
        with self._lock:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect(self.socket_path)
                payload = json.dumps(cmd, ensure_ascii=False) + "\n"
                sock.sendall(payload.encode("utf-8"))
                # Read one JSON line back.
                chunks = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
                raw = b"".join(chunks).decode("utf-8").strip()
                if not raw:
                    raise RuntimeError("empty response from rust_mcp_bridge")
                return json.loads(raw)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def status(self, timeout: float = 5) -> Dict[str, Any]:
        return self._request({"cmd": "status"}, timeout=timeout)

    def add_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        init_timeout: int = 120,
    ) -> Dict[str, Any]:
        return self._request(
            {
                "cmd": "add_server",
                "name": name,
                "command": command,
                "args": args or [],
                "env": env or {},
                "timeout": timeout,
                "init_timeout": init_timeout,
            },
            timeout=init_timeout + 5,
        )

    def list_tools(self, name: str, timeout: float = 10) -> Dict[str, Any]:
        return self._request({"cmd": "list_tools", "name": name}, timeout=timeout)

    def call_tool(
        self,
        name: str,
        tool: str,
        arguments: Dict[str, Any],
        timeout: int = 60,
    ) -> Dict[str, Any]:
        return self._request(
            {
                "cmd": "call_tool",
                "name": name,
                "tool": tool,
                "arguments": arguments,
                "timeout": timeout,
            },
            timeout=timeout + 5,
        )

    def remove_server(self, name: str, timeout: float = 5) -> Dict[str, Any]:
        return self._request({"cmd": "remove_server", "name": name}, timeout=timeout)

    def stop(self):
        proc = self._proc
        self._proc = None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            Path(self.socket_path).unlink(missing_ok=True)
        except Exception:
            pass


class McpServerConnection:
    """
    Maintains a long-running stdio connection to one MCP server.
    Uses a background thread to read JSON-RPC responses.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        init_timeout: Optional[int] = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.timeout = timeout
        self.init_timeout = init_timeout
        self.tools: List[Dict[str, Any]] = []
        self._error: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._pending: Dict[str, threading.Event] = {}
        self._responses: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._shutdown = threading.Event()
        self._next_id = 0

        # Merge env with current environment
        self._env = dict(os.environ)
        if env:
            self._env.update(env)

    def _ensure_process(self):
        if self._process is None or self._process.poll() is not None:
            try:
                self._process = subprocess.Popen(
                    [self.command, *self.args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=self._env,
                )
                self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
                self._reader_thread.start()
                self._initialize()
            except Exception as e:
                self._error = str(e)
                self._ready.set()

    def _read_loop(self):
        """Continuously read JSON-RPC lines from server stdout."""
        try:
            while not self._shutdown.is_set() and self._process and self._process.stdout:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None:
                    with self._lock:
                        self._responses[str(msg_id)] = msg
                        event = self._pending.pop(str(msg_id), None)
                    if event:
                        event.set()
        except Exception:
            pass

    def _send(self, method: Optional[str], params: Any, msg_id: Optional[str] = None) -> Optional[str]:
        msg: Dict[str, Any] = {"jsonrpc": "2.0"}
        if method:
            msg["method"] = method
        if params is not None:
            msg["params"] = params
        if msg_id is not None:
            msg["id"] = msg_id
        else:
            msg["id"] = msg_id

        line = json.dumps(msg, ensure_ascii=False) + "\n"
        if self._process and self._process.stdin:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        return msg_id

    def _call(self, method: str, params: Any, timeout: Optional[int] = None) -> Any:
        timeout = timeout or self.timeout
        with self._lock:
            self._next_id += 1
            msg_id = str(self._next_id)
            event = threading.Event()
            self._pending[msg_id] = event

        self._send(method, params, msg_id)

        if not event.wait(timeout=timeout):
            with self._lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError(f"MCP call '{method}' timed out on server '{self.name}'")

        with self._lock:
            response = self._responses.pop(msg_id, {})

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error: {err.get('message', err)}")
        return response.get("result")

    def _initialize(self):
        """MCP handshake: initialize -> initialized -> list_tools."""
        try:
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lv-agent", "version": "0.1.0"},
            }
            # Longer timeout for npx-based servers that may need to download packages.
            init_timeout = self.init_timeout or max(60, self.timeout)
            self._call("initialize", init_params, timeout=init_timeout)
            self._send("notifications/initialized", {}, msg_id=None)
            result = self._call("tools/list", {}, timeout=30)
            self.tools = result.get("tools", []) if result else []
            self._ready.set()
        except Exception as e:
            self._error = str(e)
            self.tools = []
            self._ready.set()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return its text content."""
        self._ensure_process()
        # 首次调用懒加载: 等待服务器初始化(npx 拉包可能需要更久), 用 60s 超时
        if not self._ready.wait(timeout=60):
            raise RuntimeError(f"MCP server '{self.name}' not ready")
        if self._error:
            raise RuntimeError(f"MCP server '{self.name}' failed: {self._error}")

        result = self._call("tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", []) if result else []
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            else:
                texts.append(str(item))
        return "\n".join(texts)

    def disconnect(self):
        self._shutdown.set()
        try:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=2)
        except Exception:
            pass


class McpManager:
    """Discovers and manages all configured MCP servers.

    Uses the Rust bridge when available for sub-millisecond tool calls;
    falls back to the pure-Python stdio client otherwise.
    """

    def __init__(self, servers_config: Optional[Dict[str, Any]] = None):
        self.servers: Dict[str, McpServerConnection] = {}
        self._registered_tool_names: Dict[str, List[str]] = {}
        self._bridge: Optional[RustMcpBridge] = None
        self._bridge_configs: Dict[str, Dict[str, Any]] = {}
        self._use_bridge = False
        self._logger = logging.getLogger("McpManager")

        bridge_path = _find_rust_mcp_bridge()
        if bridge_path:
            try:
                self._bridge = RustMcpBridge(binary_path=bridge_path)
                self._use_bridge = True
                self._logger.info("rust mcp bridge ready")
            except Exception as e:
                self._logger.warning(f"rust mcp bridge unavailable: {e}")

        if not servers_config:
            return

        for name, cfg in servers_config.items():
            self._add_server_internal(name, cfg)

    @staticmethod
    def _resolve_path_placeholders(args: List[str]) -> List[str]:
        """Resolve __REPO_ROOT__ / __AGENT_DIR__ placeholders in MCP args.

        This keeps config.yaml portable when copying the source tree to a new
        machine (no more hardcoded /Users/<user>/... paths).
        """
        try:
            repo_root = str(Path(__file__).resolve().parents[3])
        except Exception:
            repo_root = str(Path(__file__).resolve().parents[2])
        agent_dir = str(Path(__file__).resolve().parents[2])
        resolved = []
        for a in args:
            if isinstance(a, str):
                a = a.replace("__REPO_ROOT__", repo_root).replace("__AGENT_DIR__", agent_dir)
            resolved.append(a)
        return resolved

    def _add_server_internal(self, name: str, cfg: Dict[str, Any]) -> Optional[Any]:
        if not isinstance(cfg, dict):
            return None
        if not cfg.get("enabled", True):
            return None
        command = cfg.get("command")
        if not command:
            return None

        # Resolve path placeholders so the config stays machine-independent.
        raw_args = cfg.get("args", []) or []
        args = self._resolve_path_placeholders(raw_args)

        if self._use_bridge and self._bridge:
            self._bridge_configs[name] = cfg
            resp = self._bridge.add_server(
                name=name,
                command=command,
                args=args,
                env=cfg.get("env"),
                timeout=cfg.get("timeout", 60),
                init_timeout=cfg.get("init_timeout", 120),
            )
            if not resp.get("success"):
                self._logger.warning(f"mcp server '{name}' failed: {resp.get('error')}")
                self._bridge_configs.pop(name, None)
                return None
            self._logger.info(f"mcp server '{name}' ready ({len(resp.get('metadata', {}).get('tools', []))} tools)")
            return resp

        conn = McpServerConnection(
            name=name,
            command=command,
            args=args,
            env=cfg.get("env"),
            timeout=cfg.get("timeout", 60),
            init_timeout=cfg.get("init_timeout"),
        )
        self.servers[name] = conn
        # 懒加载: 不立即启动进程, 首次调用工具时才启动(避免启动时 npx 拉包卡顿)
        return conn

    def add_server(self, name: str, cfg: Dict[str, Any]) -> Optional[Any]:
        """Add and start a server at runtime. Replaces existing connection."""
        old = self.servers.get(name)
        if old:
            old.disconnect()
        self.servers.pop(name, None)
        self._registered_tool_names.pop(name, None)
        return self._add_server_internal(name, cfg)

    def remove_server(self, name: str):
        """Disconnect and remove a server."""
        if self._use_bridge and self._bridge:
            self._bridge.remove_server(name)
        conn = self.servers.pop(name, None)
        if conn:
            conn.disconnect()
        self._bridge_configs.pop(name, None)
        self._registered_tool_names.pop(name, None)

    def discover_tools(self) -> List[Dict[str, Any]]:
        """Return all tools discovered from connected MCP servers."""
        discovered = []

        if self._use_bridge and self._bridge:
            for server_name, cfg in list(self._bridge_configs.items()):
                resp = self._bridge.list_tools(server_name, timeout=10)
                if not resp.get("success"):
                    self._logger.warning(f"mcp list_tools '{server_name}' failed: {resp.get('error')}")
                    continue
                tools = resp.get("metadata", {}).get("tools", [])
                for tool in tools:
                    adapted = dict(tool)
                    adapted["server"] = server_name
                    adapted["mcp_name"] = tool.get("name")
                    adapted["name"] = f"mcp_{server_name}_{tool.get('name')}"
                    adapted["description"] = f"[MCP:{server_name}] {tool.get('description', '')}"
                    params = tool.get("inputSchema") or tool.get("parameters") or {"type": "object", "properties": {}}
                    adapted["parameters"] = params
                    discovered.append(adapted)
            return discovered

        for server_name, conn in self.servers.items():
            # 懒加载: 服务器未就绪时跳过(不阻塞启动), 首次调用工具时才初始化
            if not conn._ready.is_set():
                continue
            if conn._error:
                print(f"\033[2m  MCP server '{server_name}' failed: {conn._error}\033[0m")
                continue
            for tool in conn.tools:
                adapted = dict(tool)
                adapted["server"] = server_name
                adapted["mcp_name"] = tool.get("name")
                adapted["name"] = f"mcp_{server_name}_{tool.get('name')}"
                adapted["description"] = (
                    f"[MCP:{server_name}] {tool.get('description', '')}"
                )
                params = tool.get("inputSchema") or tool.get("parameters") or {"type": "object", "properties": {}}
                adapted["parameters"] = params
                discovered.append(adapted)
        return discovered

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        if self._use_bridge and self._bridge:
            resp = self._bridge.call_tool(server_name, tool_name, arguments)
            if not resp.get("success"):
                raise RuntimeError(resp.get("error", "unknown MCP error"))
            return resp.get("output", "")

        conn = self.servers.get(server_name)
        if not conn:
            raise RuntimeError(f"MCP server '{server_name}' not found")
        return conn.call_tool(tool_name, arguments)

    def disconnect_all(self):
        if self._use_bridge and self._bridge:
            self._bridge.stop()
            self._bridge = None
            self._use_bridge = False
        for conn in self.servers.values():
            conn.disconnect()


class McpAdapterTool(BaseTool):
    """Wraps a single MCP tool into the project's BaseTool interface."""

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        display_name: str,
        description: str,
        parameters: Dict[str, Any],
        manager: McpManager,
    ):
        self.server_name = server_name
        self.tool_name = tool_name
        self.name = display_name
        self.description = description
        self.parameters = parameters
        self.manager = manager

    def execute(self, **kwargs) -> ToolResult:
        try:
            output = self.manager.call_tool(self.server_name, self.tool_name, kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def register_mcp_tools(manager: McpManager):
    """Register all discovered MCP tools into the global TOOLS_REGISTRY."""
    discovered = manager.discover_tools()

    # Group by server so re-registration replaces stale adapters.
    by_server: Dict[str, List[Dict[str, Any]]] = {}
    for tool in discovered:
        by_server.setdefault(tool["server"], []).append(tool)

    for server_name, tools in by_server.items():
        old_names = manager._registered_tool_names.pop(server_name, [])
        for old_name in old_names:
            TOOLS_REGISTRY.unregister(old_name)

        new_names = []
        for tool in tools:
            adapter = McpAdapterTool(
                server_name=tool["server"],
                tool_name=tool["mcp_name"],
                display_name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
                manager=manager,
            )
            TOOLS_REGISTRY.register(adapter)
            new_names.append(tool["name"])
        manager._registered_tool_names[server_name] = new_names

    if discovered:
        print(f"\033[2m  MCP tools registered: {len(discovered)}\033[0m")


class McpOrchestrator:
    """
    Lightweight orchestrator that recommends tool combinations for a task.
    Maps task keywords to relevant MCP/native tools so the agent does not
    have to search through the full tool registry on every step.
    """

    # Keyword -> recommended tool name patterns (supports native and MCP tools)
    _TOOL_MAP = {
        "time": ["mcp_time_get_current_time", "time"],
        "weather": ["mcp_weather_get_forecast", "weather", "web_search"],
        "search": ["web_search", "mcp_ddg_search", "mcp_google_search", "mcp_brave_search"],
        "read file": ["file_ops", "mcp_filesystem_read_file"],
        "write file": ["file_ops", "mcp_filesystem_write_file"],
        "list directory": ["file_ops", "mcp_filesystem_list_directory"],
        "fetch": ["mcp_fetch_fetch", "api_call"],
        "browse": ["mcp_puppeteer_navigate", "mcp_puppeteer_snapshot"],
        "memory": ["mcp_memory_read_graph", "mcp_memory_create_entities"],
        "calculate": ["python_exec"],
        "code": ["python_exec", "mcp_sequential_thinking"],
        "github": ["mcp_github_search_repositories", "mcp_github_get_file_contents"],
        "database": ["mcp_sqlite_query", "mcp_sqlite_execute"],
        "sequential thinking": ["mcp_sequential_thinking"],
    }

    def __init__(self, manager: Optional[McpManager] = None):
        self.manager = manager
        self.logger = logging.getLogger("McpOrchestrator")

    def recommend_tools(self, task: str, all_tools: Dict[str, Any], max_tools: int = 12) -> Dict[str, Any]:
        """
        Return a filtered tool dictionary containing the most relevant tools for the task.
        Falls back to the full registry if no strong match is found.
        """
        task_lower = task.lower()
        scores: Dict[str, int] = {}

        for keyword, candidates in self._TOOL_MAP.items():
            if keyword in task_lower:
                for tool_name in candidates:
                    scores[tool_name] = scores.get(tool_name, 0) + 2

        # Also boost tools whose descriptions contain task keywords
        for name, desc in all_tools.items():
            desc_lower = desc.lower()
            for word in task_lower.split():
                if len(word) > 3 and word in desc_lower:
                    scores[name] = scores.get(name, 0) + 1

        if not scores:
            return all_tools

        # Always include core tools
        for core in ["file_ops", "python_exec", "web_search"]:
            if core in all_tools:
                scores[core] = scores.get(core, 0) + 1

        ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        selected = ranked[:max_tools]

        # Ensure at least one fallback from each category if task is broad
        if len(selected) < max_tools:
            for name in all_tools:
                if name not in selected and len(selected) < max_tools:
                    selected.append(name)

        return {name: all_tools[name] for name in selected if name in all_tools}

    def suggest_combination(self, task: str) -> List[str]:
        """Suggest a canonical multi-tool sequence for common compound tasks."""
        task_lower = task.lower()
        suggestions = []
        if any(kw in task_lower for kw in ["analyze", "分析", "项目", "project", "folder"]):
            suggestions.extend(["file_ops", "python_exec"])
        if any(kw in task_lower for kw in ["search", "查找", "调研", "research"]):
            suggestions.extend(["web_search", "mcp_fetch_fetch"])
        if any(kw in task_lower for kw in ["weather", "天气", "forecast"]):
            suggestions.extend(["mcp_time_get_current_time", "mcp_weather_get_forecast"])
        if any(kw in task_lower for kw in ["code", "代码", "program", "script"]):
            suggestions.extend(["python_exec", "mcp_sequential_thinking"])
        return suggestions
