"""Lv Agent terminal UI package.

Building blocks per the frontend design proposal:
- Theme: built-in light/dark/minimal themes + NO_COLOR support
- Renderer: ANSI-aware output helper
- Banner: startup portrait + brand text
- StatusBar: persistent status line with context bar (width-adaptive)
- ToolCard / MessageCard: tool & message card rendering
- CLIApp: simple line-mode CLI entry point
"""

from .themes import Theme, load_theme, BUILTIN_THEMES
from .renderer import Renderer
from .banner import render_banner, render_system_status
from .status_bar import StatusBar
from .cards import ToolCard, MessageCard
from .app import CLIApp

__all__ = [
    "Theme",
    "load_theme",
    "BUILTIN_THEMES",
    "Renderer",
    "render_banner",
    "render_system_status",
    "StatusBar",
    "ToolCard",
    "MessageCard",
    "CLIApp",
]
