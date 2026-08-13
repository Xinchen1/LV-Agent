"""Tool / message card rendering for Lv Agent CLI.

Implements the four-state tool card from the design proposal:
    pending ◌ (gray)  → executing ◐ (yellow)  → success ✓ (green) / error ✗ (red)
Long outputs are folded to ``max_visible_lines`` (default 6).
"""

from __future__ import annotations

import re
from typing import Optional

from .renderer import Renderer


class ToolCard:
    """Render a tool invocation + its result as a status card."""

    def __init__(
        self,
        renderer: Renderer,
        max_visible_lines: int = 6,
        folded: bool = True,
    ):
        self.r = renderer
        self.max_visible_lines = max_visible_lines
        self.folded = folded
        self._tool_name = ""

    # ------------------------------------------------------------------
    # State markers (design proposal)
    # ------------------------------------------------------------------
    MARKERS = {
        "pending": ("◌", "90"),
        "running": ("◐", "220"),
        "success": ("✓", "114"),
        "error": ("✗", "203"),
        "policy": ("◐", "33"),
    }

    @staticmethod
    def extract_tool_name(text: str) -> str:
        """Extract tool name from call text: bash_exec(...) / bash_exec: {...} / bash_exec {...}."""
        m = re.match(r"^\s*([a-zA-Z_][\w_]*)\s*[(:{=]", text)
        return m.group(1) if m else text.split("(", 1)[0].strip()[:20]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_call(self, text: str) -> str:
        """Render a tool invocation line (pending state)."""
        self._tool_name = self.extract_tool_name(text)
        mark, color = self.MARKERS["pending"]
        return f"  {self.r.style(mark, color)} {self.r.dim(text)}"

    def render_result(self, text: str, state: str = "success") -> str:
        """Render a tool result as a bordered card with a state marker."""
        is_error = state == "error"
        is_policy = state == "policy"
        mark, _ = self.MARKERS.get(state, self.MARKERS["success"])
        name = self._tool_name or "exec"
        lines = text.rstrip().split("\n")

        if is_policy:
            border_fg, name_fg, text_fg = "33", "33", "37"
        elif is_error:
            border_fg, name_fg, text_fg = "31", "31", "31"
        else:
            border_fg, name_fg, text_fg = "2", "34", "37"

        width = max(self.r.columns() - 8, 40)
        border = "─" * width
        out = [f" {self.r.style(border_fg)}╭─ {self.r.style(mark, border_fg)} {self.r.style(name, 'bold', name_fg)} {border}"]

        shown = lines[: self.max_visible_lines]
        more = len(lines) - len(shown)
        for ln in shown:
            if len(ln) > width:
                ln = ln[: width - 1] + "…"
            out.append(f" {self.r.style(text_fg)}│ {ln}")
        if more > 0:
            out.append(f" {self.r.style(text_fg)}│ …（其余 {more} 行已折叠）")

        tail = " · blocked" if is_policy else (" · error" if is_error else "")
        out.append(f" {self.r.style(border_fg)}╰─ {len(lines)} 行{tail}")
        return "\n".join(out)


class MessageCard:
    """Render a user/assistant message with a role prefix."""

    def __init__(self, renderer: Renderer):
        self.r = renderer

    def render(self, role: str, content: str) -> str:
        prefix = "You" if role == "user" else "Lv"
        color = "2" if role == "user" else "1;38;5;180"
        return f"{self.r.style(prefix, color)} {content}"
