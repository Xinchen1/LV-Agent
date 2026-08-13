"""Dynamic tool discovery for the Lv Super Agent tool registry.

Scans the tools directory for BaseTool subclasses and registers them
automatically, removing the need to edit manual import lists for every
new tool.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from . import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)


def _is_basetool_subclass(node: ast.AST) -> Optional[str]:
    """Return the class name if *node* is a non-abstract BaseTool subclass."""
    if not isinstance(node, ast.ClassDef):
        return None
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseTool":
            return node.name
        if (
            isinstance(base, ast.Attribute)
            and base.attr == "BaseTool"
        ):
            return node.name
    return None


def _module_defines_tools(module_path: Path) -> List[str]:
    """Return class names of BaseTool subclasses defined in a module."""
    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError:
        return []
    if "BaseTool" not in source:
        return []
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError:
        return []

    tools: List[str] = []
    for stmt in tree.body:
        name = _is_basetool_subclass(stmt)
        if name:
            tools.append(name)
    return tools


def discover_tool_modules(tools_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """Scan the tools directory and map module names to tool class names."""
    if tools_dir is None:
        tools_dir = Path(__file__).resolve().parent

    result: Dict[str, List[str]] = {}
    for path in sorted(tools_dir.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "discovery.py"}:
            continue
        class_names = _module_defines_tools(path)
        if class_names:
            module_name = f"agent_project.tools.{path.stem}"
            result[module_name] = class_names
    return result


def _is_instantiable(tool_cls: Type[BaseTool]) -> bool:
    """Return True if the class can be instantiated with no arguments."""
    try:
        sig = inspect.signature(tool_cls.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return False
        return True
    except Exception:
        return False


def auto_register_tools(registry: ToolRegistry, tools_dir: Optional[Path] = None) -> int:
    """Discover and register all BaseTool subclasses in the tools directory.

    Already-registered tool names are skipped to avoid duplicates with the
    explicit manual registration list.
    """
    mapping = discover_tool_modules(tools_dir)
    registered = 0
    for module_name, class_names in mapping.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", module_name, e)
            continue

        for class_name in class_names:
            tool_cls = getattr(module, class_name, None)
            if tool_cls is None or not issubclass(tool_cls, BaseTool):
                continue
            instance: Optional[BaseTool] = None
            try:
                if _is_instantiable(tool_cls):
                    instance = tool_cls()
            except Exception as e:
                logger.debug("Could not instantiate %s: %s", class_name, e)
                continue

            if instance is None:
                continue

            if instance.name in registry._tools:
                logger.debug("Tool %s already registered; skipping discovery", instance.name)
                continue

            registry.register(instance)
            registered += 1
            logger.info("Auto-registered tool: %s", instance.name)

    return registered
