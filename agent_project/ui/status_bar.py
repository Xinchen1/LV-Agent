"""Status bar component for Lv Agent CLI.

Renders the persistent status line (directory · context bar · duration ·
commands) with three width-adaptive layouts per the frontend design proposal.

The context bar uses solid block characters and four soft color
thresholds: teal < 50%, amber < 80%, orange < 95%, red >= 95%.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from .renderer import Renderer

# Modern color thresholds (design tokens)
_BAR_COLORS = (
    "38;5;79",    # mint < 50%
    "38;5;215",   # soft amber < 80%
    "38;5;216",   # peach < 95%
    "38;5;210",   # coral >= 95%
)
_DIM = "38;5;238"
_MUTED = "38;5;245"


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
        sep = self.r.style("│", _DIM)

        # Context usage mini-bar
        pct = min(100, int(used_tokens / self.max_context_tokens * 100)) if self.max_context_tokens else 0
        bar_color = _BAR_COLORS[3 if pct >= 95 else (2 if pct >= 80 else (1 if pct >= 50 else 0))]
        bar_width = 8
        # Ensure at least one visible block when there is any usage, avoid blank gaps
        filled = max(1, int(pct / 100 * bar_width)) if pct > 0 else 0
        bar = self.r.style("█" * filled + "░" * (bar_width - filled), bar_color)
        token_str = f"{used_tokens // 1000}k/{self.max_context_tokens // 1000}k" if used_tokens >= 1000 else f"{used_tokens}/{self.max_context_tokens}"
        pct_str = self.r.style(f"{pct}%", _MUTED)

        mins = max(0, int((time.time() - self.session_start) / 60))
        duration = self.r.style(f"{mins}m", _MUTED)

        ctx_mid = f" {token_str} {sep} {bar} {pct_str} {sep} {duration} "

        # Commands by width
        if width >= 72:
            right = self.commands_full
        elif width >= 50:
            right = self.commands_compact
        else:
            right = ""

        # Layout: directory | context bar | commands
        mid_len = len(ctx_mid)
        right_len = len(right)
        left_budget = max(6, width - mid_len - right_len - 2)
        if len(left) > left_budget:
            left = "…" + left[-(left_budget - 1):]
        pad = max(1, width - len(left) - mid_len - right_len)
        line = left + " " * pad + ctx_mid + right
        return self.r.style(line[:width], _DIM)

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
