"""
MCP auto-configurator tool.

Lets the agent configure, test, and register MCP servers from a natural-language
request or explicit command/args. Discovered tools are saved to config.yaml and
registered at runtime without restarting the agent.
"""

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import BaseTool, TOOLS_REGISTRY, ToolResult
from .mcp_client import McpManager, McpServerConnection, register_mcp_tools


# Module-level runtime MCP manager so configured servers stay alive across calls.
_GLOBAL_MCP_MANAGER: Optional[McpManager] = None


def get_global_mcp_manager() -> McpManager:
    global _GLOBAL_MCP_MANAGER
    if _GLOBAL_MCP_MANAGER is None:
        _GLOBAL_MCP_MANAGER = McpManager()
    return _GLOBAL_MCP_MANAGER


class McpAutoConfigurator:
    """Infer, test, persist, and register MCP servers."""

    KNOWN = {
        "filesystem": {
            "keywords": ["filesystem", "file system", "文件系统", "文件操作"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "{path}"],
            "default_path": ".",
            "env": {},
        },
        "fetch": {
            "keywords": ["fetch", "http", "web fetch", "网页获取"],
            "runtime": "uvx",
            "fallback_runtime": "npx",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "fallback_command": "npx",
            "fallback_args": ["-y", "@modelcontextprotocol/server-fetch"],
            "env": {},
        },
        "brave-search": {
            "keywords": ["brave", "brave search", "search"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": ""},
        },
        "github": {
            "keywords": ["github", "gh", "git hub"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
        },
        "sqlite": {
            "keywords": ["sqlite", "sqlite3", "database"],
            "runtime": "uvx",
            "command": "uvx",
            "args": ["mcp-server-sqlite", "--db-path", "{path}"],
            "default_path": "./data/mcp.db",
            "env": {},
        },
        "puppeteer": {
            "keywords": ["puppeteer", "browser", "chrome", "浏览"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
            "env": {},
        },
        "memory": {
            "keywords": ["memory", "记忆"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "env": {},
        },
        "sequentialthinking": {
            "keywords": ["sequential thinking", "sequentialthinking", "思考链"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "env": {},
        },
        "time": {
            "keywords": ["time", "clock", "时间"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-time"],
            "env": {},
        },
        "slack": {
            "keywords": ["slack"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-slack"],
            "env": {"SLACK_TOKEN": "", "SLACK_TEAM_ID": ""},
        },
        "google-maps": {
            "keywords": ["google maps", "maps", "地图"],
            "runtime": "npx",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-google-maps"],
            "env": {"GOOGLE_MAPS_API_KEY": ""},
        },
    }

    TOKEN_PATTERNS = {
        "GITHUB_PERSONAL_ACCESS_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
        "BRAVE_API_KEY": re.compile(r"\bB[A-Za-z0-9_]{39,}\b"),  # heuristic
        "GOOGLE_MAPS_API_KEY": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    }

    def infer_server_type(self, request: str) -> Optional[str]:
        req_lower = request.lower()
        for name, meta in self.KNOWN.items():
            for kw in meta["keywords"]:
                if kw in req_lower:
                    return name
        return None

    def _extract_path(self, request: str, default: str = ".") -> str:
        # Try to capture an absolute/home/relative path.
        m = re.search(r"(~?/[A-Za-z0-9_\./\-\\]+)|(\bDesktop\b|\bDocuments\b|\bDownloads\b)", request)
        if m:
            path = m.group(0)
            if path in ("Desktop", "Documents", "Downloads"):
                path = f"~/{path}"
            return os.path.expanduser(path)
        return default

    def _extract_token(self, env_name: str, request: str) -> Optional[str]:
        value = os.getenv(env_name)
        if value:
            return value
        pattern = self.TOKEN_PATTERNS.get(env_name)
        if pattern:
            m = pattern.search(request)
            if m:
                return m.group(0)
        return None

    def _check_runtime(self, runtime: str) -> bool:
        return shutil.which(runtime) is not None

    def build_config(self, server_type: str, request: str = "") -> Optional[Dict[str, Any]]:
        meta = self.KNOWN.get(server_type)
        if not meta:
            return None

        command = meta["command"]
        args = list(meta["args"])

        # Fallback to npx if uvx is not installed.
        if meta.get("runtime") == "uvx" and not self._check_runtime("uvx"):
            if meta.get("fallback_runtime") == "npx" and self._check_runtime("npx"):
                command = meta.get("fallback_command", "npx")
                args = list(meta.get("fallback_args", []))
            else:
                raise RuntimeError(
                    f"MCP server '{server_type}' requires '{meta['runtime']}' but it is not installed."
                )

        if meta.get("runtime") == "npx" and not self._check_runtime("npx"):
            raise RuntimeError(
                f"MCP server '{server_type}' requires 'npx' (Node.js) but it is not installed."
            )

        # Replace path placeholder.
        if "{path}" in args:
            path = self._extract_path(request, meta.get("default_path", "."))
            args = [path if p == "{path}" else p for p in args]

        # Fill env vars when known.
        env: Dict[str, Optional[str]] = {}
        for env_name in meta.get("env", {}):
            token = self._extract_token(env_name, request)
            if token:
                env[env_name] = token

        return {
            "enabled": True,
            "command": command,
            "args": args,
            "env": env,
            "timeout": 60,
            "init_timeout": 120,
        }

    def infer_and_build(self, request: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        server_type = self.infer_server_type(request)
        if not server_type:
            return "", None
        config = self.build_config(server_type, request)
        return server_type, config

    def test_config(
        self, name: str, config: Dict[str, Any]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Spin up the server, perform MCP handshake, and return discovered tools."""
        conn = McpServerConnection(
            name=name,
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
            timeout=config.get("timeout", 60),
            init_timeout=config.get("init_timeout", 120),
        )
        try:
            conn._ensure_process()
            if not conn._ready.wait(timeout=130):
                return False, "MCP server did not become ready (timeout)", []
            if conn._error:
                return False, f"MCP server failed: {conn._error}", []
            return True, "", conn.tools
        finally:
            # Give the handshake a moment to settle before terminating the test process.
            time.sleep(0.2)
            conn.disconnect()

    def save_config(self, config_path: str, name: str, config: Dict[str, Any]):
        path = Path(config_path)
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            raw = {}
        raw.setdefault("mcp", {"enabled": False, "servers": {}})
        raw["mcp"]["enabled"] = True
        raw["mcp"]["servers"][name] = dict(config)
        path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False), encoding="utf-8")


class McpConfiguratorTool(BaseTool):
    """
    Configure and activate an MCP server from a user request.

    Examples:
      - "enable filesystem MCP for ~/Desktop"
      - "add fetch MCP"
      - "use github MCP"
      - command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."]
    """

    name = "mcp_setup"
    description = (
        "Configure, test, and register an MCP (Model Context Protocol) server. "
        "Pass a natural-language request like 'enable filesystem MCP for ~/Desktop' "
        "or provide explicit command/args. The tool tests the connection, saves the "
        "configuration, and registers discovered tools so they can be used immediately."
    )

    parameters = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "Natural-language request describing which MCP server to configure, e.g. 'enable filesystem MCP for ~/Desktop'.",
            },
            "server_name": {
                "type": "string",
                "description": "Optional explicit server name. Inferred from request if omitted.",
            },
            "command": {
                "type": "string",
                "description": "Optional explicit command to run the MCP server (overrides auto-detection).",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit command arguments.",
            },
            "env": {
                "type": "object",
                "description": "Optional environment variables for the server.",
            },
            "test": {
                "type": "boolean",
                "default": True,
                "description": "Whether to test the server before registering.",
            },
            "save": {
                "type": "boolean",
                "default": True,
                "description": "Whether to save the configuration to config.yaml.",
            },
            "config_path": {
                "type": "string",
                "default": "config.yaml",
                "description": "Path to the agent config file.",
            },
        },
        "required": ["request"],
    }

    def execute(
        self,
        request: str,
        server_name: str = "",
        command: str = "",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, Any]] = None,
        test: bool = True,
        save: bool = True,
        config_path: str = "config.yaml",
        **kwargs,
    ) -> ToolResult:
        try:
            cfg = McpAutoConfigurator()

            if command:
                name = server_name or "custom"
                server_config = {
                    "enabled": True,
                    "command": command,
                    "args": list(args or []),
                    "env": dict(env or {}),
                    "timeout": 60,
                    "init_timeout": 120,
                }
            else:
                name, server_config = cfg.infer_and_build(request)
                if not server_config:
                    known = ", ".join(cfg.KNOWN.keys())
                    return ToolResult(
                        success=False,
                        output="",
                        error=(
                            f"Could not infer MCP server from request. "
                            f"Known servers: {known}. "
                            f"You can also provide explicit 'command' and 'args'."
                        ),
                    )
                if server_name:
                    name = server_name

            # Test connection.
            tools: List[Dict[str, Any]] = []
            if test:
                ok, err, tools = cfg.test_config(name, server_config)
                if not ok:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"MCP test failed: {err}",
                        metadata={"config": server_config},
                    )

            # Persist configuration.
            if save:
                cfg.save_config(config_path, name, server_config)

            # Register at runtime.
            manager = get_global_mcp_manager()
            manager.add_server(name, server_config)
            register_mcp_tools(manager)
            registered = manager._registered_tool_names.get(name, [])

            tool_names = ", ".join(registered) if registered else "(none)"
            return ToolResult(
                success=True,
                output=(
                    f"MCP server '{name}' configured. "
                    f"Discovered {len(tools)} tool(s), registered {len(registered)}.\n"
                    f"Registered: {tool_names}"
                ),
                metadata={
                    "server": name,
                    "config": server_config,
                    "discovered": [t.get("name") for t in tools],
                    "registered": registered,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


# Auto-register on import.
TOOLS_REGISTRY.register(McpConfiguratorTool())
