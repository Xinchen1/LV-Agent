"""Theme system for Lv Agent CLI.

Themes map named tokens to ANSI SGR code strings. The renderer combines these
codes into escape sequences. When NO_COLOR is set (or the minimal theme is
chosen), color output is disabled while structure (box-drawing, spacing) is
preserved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Theme:
    """A terminal color/theme definition."""

    name: str
    tokens: Dict[str, str] = field(default_factory=dict)
    supports_color: bool = True

    def code(self, token: str) -> str:
        """Return the ANSI SGR code for a named token, or empty string."""
        if not self.supports_color:
            return ""
        return self.tokens.get(token, "")

    def style(self, text: str, token: str) -> str:
        """Wrap text in the ANSI sequence for a named token."""
        code = self.code(token)
        if not code:
            return text
        return f"\033[{code}m{text}\033[0m"


# Built-in themes. Only 3 semantic colors + neutrals (dim/bold).
# 1. PRIMARY (brand) - Main accent: headers, highlights, links, primary actions
# 2. SUCCESS - Positive states: success, info, confirmations, completed
# 3. ERROR - Negative states: errors, warnings, failures, critical issues
# Neutrals: dim (faint), bold, and implicit reset
BUILTIN_THEMES: Dict[str, Theme] = {
    "light": Theme(
        name="light",
        tokens={
            "primary": "38;5;30",           # teal #0d9488 - primary brand
            "success": "38;5;36",           # forest mint #059669
            "error": "38;5;167",            # coral #dc2626
            "dim": "2",                     # faint
            "bold": "1",                    # bold
        },
        supports_color=True,
    ),
    "dark": Theme(
        name="dark",
        tokens={
            "primary": "38;5;80",           # teal #2dd4bf - primary brand
            "success": "38;5;79",           # mint #34d399
            "error": "38;5;210",            # coral #f87171
            "dim": "2",                     # faint
            "bold": "1",                    # bold
        },
        supports_color=True,
    ),
    "minimal": Theme(
        name="minimal",
        tokens={},
        supports_color=False,
    ),
}


def _color_forced() -> bool:
    return os.getenv("FORCE_COLOR", "").lower() in ("1", "true", "yes")


def _color_disabled() -> bool:
    return os.getenv("NO_COLOR", "") != ""


def load_theme(
    name: str = "dark",
    custom_overrides: Optional[Dict[str, str]] = None,
) -> Theme:
    """Load a built-in theme and apply optional custom token overrides.

    Args:
        name: Theme name (light, dark, minimal) or a path to a custom theme
            (not implemented in Phase 1).
        custom_overrides: Map of token name -> ANSI SGR code to override.

    Returns:
        A Theme instance. NO_COLOR takes precedence and disables color.
    """
    if _color_disabled() and not _color_forced():
        return BUILTIN_THEMES["minimal"]

    base = BUILTIN_THEMES.get(name, BUILTIN_THEMES["dark"])
    tokens = dict(base.tokens)
    if custom_overrides:
        tokens.update(custom_overrides)

    supports_color = base.supports_color and not (name == "minimal")
    if _color_forced():
        supports_color = True

    return Theme(name=name, tokens=tokens, supports_color=supports_color)


def display_config_from(config: Optional[Any] = None) -> Dict[str, Any]:
    """Extract display settings from the agent config, or return defaults."""
    defaults = {
        "theme": "dark",
        "status_bar": {
            "enabled": True,
            "show_cost": False,
            "show_context_bar": True,
            "compact_threshold": 76,
            "minimal_threshold": 52,
        },
        "prompt": {
            "show_cwd": True,
            "show_branch": True,
            "show_mode_badge": True,
            "history_size": 1000,
        },
        "cards": {
            "collapsed_by_default": True,
            "max_visible_lines": 6,
        },
        "accessibility": {
            "no_color": False,
            "high_contrast": False,
            "reduced_motion": False,
        },
    }
    if config is None:
        return defaults

    raw = getattr(config, "display", None)
    if isinstance(raw, dict):
        # Shallow merge; nested dicts are replaced, which is acceptable for Phase 1.
        defaults.update(raw)
    return defaults
