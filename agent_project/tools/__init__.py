"""
Tool base classes, registry, and OpenAI-compatible tool schema export.
Also registers all built-in tools on import.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import json
from pydantic import BaseModel, Field


# =========== Core types ===========

class ToolCall(BaseModel):
    """Normalized tool call payload."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of executing a tool."""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========== Base tool ===========

class BaseTool(ABC):
    """All tools must inherit from this."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# =========== Registry ===========

class ToolRegistry:
    """Global tool registry."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._descriptions: List[Dict[str, Any]] = []

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        self._descriptions.append(tool.to_dict())
        return self

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if it existed."""
        if name not in self._tools:
            return False
        del self._tools[name]
        self._descriptions = [d for d in self._descriptions if d.get("name") != name]
        return True

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_prompt_description(self) -> str:
        lines: List[str] = ["Available tools:"]
        for desc in self._descriptions:
            lines.append(f"\n- {desc['name']}: {desc['description']}")
            props = desc["parameters"].get("properties", {})
            if props:
                lines.append("  Parameters:")
                required = set(desc["parameters"].get("required", []))
                for param, spec in props.items():
                    req = "(required)" if param in required else "(optional)"
                    lines.append(f"    {param}: {spec.get('description', '?')} {req}")
        return "\n".join(lines)

    def get_tools_dict(self) -> Dict[str, str]:
        return {t.name: t.description for t in self._tools.values()}

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert registered tools to OpenAI / Anthropic function-calling format."""
        tools = []
        for desc in self._descriptions:
            tools.append({
                "type": "function",
                "function": {
                    "name": desc["name"],
                    "description": desc["description"],
                    "parameters": {
                        "type": "object",
                        "properties": desc["parameters"].get("properties", {}),
                        "required": desc["parameters"].get("required", []),
                    },
                },
            })
        return tools

    # ---- Anthropic tool_use format helpers ----

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Return tools in Anthropic Messages API format (no `type: function` wrapper)."""
        result = []
        for desc in self._descriptions:
            result.append({
                "name": desc["name"],
                "description": desc["description"],
                "input_schema": {
                    "type": "object",
                    "properties": desc["parameters"].get("properties", {}),
                    "required": desc["parameters"].get("required", []),
                },
            })
        return result

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the raw tool description dict for a registered tool."""
        for desc in self._descriptions:
            if desc.get("name") == name:
                return desc
        return None

    def refresh(self) -> int:
        """Re-run dynamic tool discovery and register any new tools."""
        from .discovery import auto_register_tools
        return auto_register_tools(self)

    def enabled_tools(self, config: Any) -> List[str]:
        """Return the subset of registered tool names allowed by ToolConfig."""
        if config is None:
            return self.list_tools()
        enabled_set = set(config.tools.enabled_tools)
        if enabled_set:
            return [name for name in self.list_tools() if name in enabled_set]
        return self.list_tools()


# Global singleton
TOOLS_REGISTRY = ToolRegistry()

# Optional harness kernel reference set by the agent at init time.
_harness_kernel_ref: Dict[str, Any] = {"kernel": None}


def set_harness_kernel(kernel: Any) -> None:
    """Inject the active harness kernel so tools can perform policy checks."""
    _harness_kernel_ref["kernel"] = kernel


def get_harness_kernel() -> Optional[Any]:
    """Return the injected harness kernel, if any."""
    return _harness_kernel_ref.get("kernel")


def register_builtin_tools() -> None:
    """确保所有内置工具都已注册（幂等，可重复调用）。"""
    from .web_search import WebSearchTool
    from .calculator import CalculatorTool
    from .python_exec import PythonExecTool
    from .file_ops import FileOpsTool
    from .api_call import ApiCallTool
    from .playwright_browser import PlaywrightBrowserTool
    from .git_ops import GitTool
    from .database import DatabaseTool
    from .github_search import GitHubSearchTool
    from .telegram_bot import TelegramBotTool, create_telegram_bot
    from .bash_exec import BashExecTool
    from .grep_tool import GrepTool, GlobTool  # noqa: F401
    from .project_context import ProjectContextTool  # noqa: F401
    from .weather import WeatherTool  # noqa: F401

    _defaults = [
        WebSearchTool(),
        CalculatorTool(),
        PythonExecTool(),
        FileOpsTool(),
        ApiCallTool(),
        PlaywrightBrowserTool(),
        GitTool(),
        DatabaseTool(),
        BashExecTool(),
        GrepTool(),
        GlobTool(),
        ProjectContextTool(),
        WeatherTool(),
        GitHubSearchTool(),
    ]
    for t in _defaults:
        if t.name not in TOOLS_REGISTRY._tools:
            TOOLS_REGISTRY.register(t)

register_builtin_tools()

# =========== Import and auto-register all built-in tools ============

from .web_search import WebSearchTool  # noqa: E402
from .calculator import CalculatorTool  # noqa: E402
from .python_exec import PythonExecTool  # noqa: E402
from .file_ops import FileOpsTool  # noqa: E402
from .api_call import ApiCallTool  # noqa: E402
from .playwright_browser import PlaywrightBrowserTool  # noqa: E402
from .git_ops import GitTool  # noqa: E402
from .database import DatabaseTool  # noqa: E402
from .github_search import GitHubSearchTool  # noqa: E402
from .telegram_bot import TelegramBotTool, create_telegram_bot  # noqa: E402
from .bash_exec import BashExecTool  # noqa: E402
from .grep_tool import GrepTool, GlobTool  # noqa: E402,F401
from .weather import WeatherTool  # noqa: E402,F401
from .mcp_setup import McpConfiguratorTool  # noqa: E402,F401
from .discovery import auto_register_tools  # noqa: E402

# Discover and register any BaseTool subclasses not in the explicit list above.
try:
    auto_register_tools(TOOLS_REGISTRY)
except Exception as _discovery_err:
    import logging
    logging.getLogger(__name__).warning("Dynamic tool discovery failed: %s", _discovery_err)

__all__ = [
    "ToolCall",
    "ToolResult",
    "BaseTool",
    "ToolRegistry",
    "TOOLS_REGISTRY",
    "WebSearchTool",
    "CalculatorTool",
    "PythonExecTool",
    "FileOpsTool",
    "ApiCallTool",
    "PlaywrightBrowserTool",
    "GitTool",
    "DatabaseTool",
    "GitHubSearchTool",
    "TelegramBotTool",
    "create_telegram_bot",
    "BashExecTool",
    "ProcessManagerTool",
    "GrepTool",
    "GlobTool",
    "WeatherTool",
    ]
