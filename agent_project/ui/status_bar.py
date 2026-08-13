"""Status bar component for Lv Agent CLI.

Renders the persistent status line (directory · context bar · duration ·
commands) with three width-adaptive layouts per the frontend design proposal.

The context bar uses a thin 1/8-height block character and four soft color
thresholds: green(114) < 50%, yellow(220) < 80%, orange(216) < 95%, red(203).
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from .renderer import Renderer

# Soft color thresholds (design tokens)
_BAR_COLORS = (
    "38;5;114",   # green < 50%
    "38;5;220",   # yellow < 80%
    "38;5;216",   # orange < 95%
    "38;5;203",   # red >= 95%
)
_DIM = "38;5;240"


class StatusBar:
    """Compose the bottom status line for a terminal of any width."""

    def __init__(
        self,
        renderer: Renderer,
        max_context_tokens: int = 200_000,
        session_start: Optional[float] = None,
        commands_full: str = "/deep · /research · /model · /code · /help",
        commands_compact: str = "/model · /code · /help",
    ):
        self.r = renderer
        self.max_context_tokens = max_context_tokens
        self.session_start = session_start or time.time()
        self.commands_full = commands_full
        self.commands_compact = commands_compact

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, used_tokens: int = 0, width: int = 0) -> str:
        """Return the styled status line for the given width."""
        if width <= 0:
            try:
                width = shutil.get_terminal_size().columns
            except Exception:
                width = 80

        left = self._cwd()
        sep = self.r.style("│", "2", _DIM)

        # Context bar
        pct = min(100, int(used_tokens / self.max_context_tokens * 100)) if self.max_context_tokens else 0
        bar_color = _BAR_COLORS[3 if pct >= 95 else (2 if pct >= 80 else (1 if pct >= 50 else 0))]
        bar_width = 10
        filled = int(pct / 100 * bar_width)
        bar = "▁" * filled + " " * (bar_width - filled)
        token_str = f"{used_tokens // 1000}K/{self.max_context_tokens // 1000}K" if used_tokens >= 1000 else f"{used_tokens}/{self.max_context_tokens}"
        pct_str = self.r.style(f"{pct}%", "2", _DIM)

        mins = max(0, int((time.time() - self.session_start) / 60))
        duration = f"{mins}m"

        ctx_mid = f" {self.r.style(token_str, '2', _DIM)} {sep} {self.r.style(bar, bar_color)} {pct_str} {sep} {self.r.style(duration, '2', _DIM)} "

        # Commands by width
        if width >= 76:
            right = self.commands_full
        elif width >= 52:
            right = self.commands_compact
        else:
            right = ""

        # Layout: directory | context bar | commands
        mid_len = len(ctx_mid.strip())
        right_len = len(right)
        left_budget = max(6, width - mid_len - right_len - 2)
        if len(left) > left_budget:
            left = "…" + left[-(left_budget - 1):]
        pad = max(1, width - len(left) - mid_len - right_len)
        line = left + " " * pad + ctx_mid.strip() + " " + right
        return self.r.style(line[:width], "2", _DIM)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cwd() -> str:
        try:
            cwd = os.getcwd()
            home = os.path.expanduser("~")
            if cwd.startswith(home):
                cwd = "~" + cwd[len(home):]
            return cwd
        except Exception:
            return "."
