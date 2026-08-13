"""Terminal/color helpers shared across the agent package.

This module is kept as a thin backward-compatibility layer. New code should
prefer ``agent_project.ui.Renderer`` and ``agent_project.ui.Theme`` for
structured output.

Respects the NO_COLOR convention (https://no-color.org/) and FORCE_COLOR override.
"""

from __future__ import annotations

import os
import sys

from agent_project.ui import load_theme, Renderer


def supports_color() -> bool:
    """Return True if the current stdout is expected to support ANSI color codes."""
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Legacy singleton renderer using the dark theme by default.
_theme = load_theme("dark")
_renderer = Renderer(_theme)


def set_theme(name: str = "dark") -> str:
    """Switch the singleton renderer to a named built-in theme (light/dark/minimal).

    Returns the active theme name. NO_COLOR continues to take precedence.
    """
    global _theme, _renderer
    _theme = load_theme(name)
    _renderer = Renderer(_theme)
    return _theme.name


def active_theme() -> str:
    """Return the currently active theme name."""
    return _theme.name


def style(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI *codes* only if color output is supported.

    Example: style("hello", "2") -> dim "hello" in TTY, plain "hello" otherwise.
    """
    return _renderer.style(text, *codes)


def token(text: str, name: str) -> str:
    """Style *text* using a named theme token (e.g. 'brand', 'muted', 'success').

    Unlike ``style`` which takes raw SGR codes, this resolves the active theme's
    token so switching themes (light/dark/minimal) visibly changes output.
    """
    return _renderer.themed(text, name)


def dim(text: str) -> str:
    """Dim style shortcut."""
    return _renderer.dim(text)
