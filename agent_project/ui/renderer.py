"""ANSI rendering engine for Lv Agent CLI.

Renderer centralizes all terminal output. It respects the active Theme and
supports structured helpers (lines, panels, status rows) while remaining
compatible with the legacy terminal.style/terminal.dim API.
"""

from __future__ import annotations

import shutil
import sys
from typing import Optional, TextIO

from .themes import Theme


class Renderer:
    """Terminal output renderer with theme-aware ANSI styling."""

    def __init__(self, theme: Theme, out: Optional[TextIO] = None):
        self.theme = theme
        self.out = out or sys.stdout

    # ------------------------------------------------------------------
    # Low-level styling
    # ------------------------------------------------------------------
    def style(self, text: str, *codes: str) -> str:
        """Wrap text in ANSI codes if color is supported.

        Args accept raw SGR codes (e.g. "1", "38;5;188") or named theme
        tokens (e.g. "brand", "muted"). Named tokens are expanded first.
        """
        if not self.theme.supports_color or not codes:
            return text

        resolved: list[str] = []
        for code in codes:
            token_code = self.theme.code(code)
            resolved.append(token_code if token_code else code)

        return f"\033[{';'.join(resolved)}m{text}\033[0m"

    def themed(self, text: str, token: str, bold: bool = False) -> str:
        """Style text using a named theme token."""
        if bold:
            return self.style(text, "bold", token)
        return self.theme.style(text, token)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def dim(self, text: str) -> str:
        return self.style(text, "dim")

    def bold(self, text: str) -> str:
        return self.style(text, "bold")

    def brand(self, text: str) -> str:
        return self.themed(text, "primary")

    def brand_dim(self, text: str) -> str:
        return self.style(text, "dim", "primary")

    def muted(self, text: str) -> str:
        return self.style(text, "dim")

    def ink(self, text: str) -> str:
        return text

    def success(self, text: str) -> str:
        return self.themed(text, "success")

    def warning(self, text: str) -> str:
        return self.themed(text, "error")

    def error(self, text: str) -> str:
        return self.themed(text, "error")

    def accent(self, text: str) -> str:
        return self.themed(text, "primary")

    # ------------------------------------------------------------------
    # Output primitives
    # ------------------------------------------------------------------
    def write(self, text: str = "", end: str = "") -> None:
        """Write raw text to the configured output stream."""
        self.out.write(text + end)
        self.out.flush()

    def print(self, text: str = "", end: str = "\n") -> None:
        """Print a line (or partial line) to the configured output stream."""
        self.write(text, end=end)

    def line(self, char: str = "─", width: Optional[int] = None) -> None:
        """Print a full-width horizontal rule."""
        if width is None:
            try:
                width = shutil.get_terminal_size().columns
            except Exception:
                width = 80
        self.print(self.dim(char * width))

    def blank(self) -> None:
        """Print a blank line."""
        self.print()

    def columns(self) -> int:
        """Return current terminal width, defaulting to 80."""
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80

    # ------------------------------------------------------------------
    # Structured output helpers (used by higher-level components)
    # ------------------------------------------------------------------
    def label_value(
        self,
        label: str,
        value: str,
        label_token: str = "muted",
        value_token: str = "ink",
    ) -> str:
        """Return a 'label  value' pair styled with theme tokens."""
        return f"{self.themed(label, label_token)} {self.themed(value, value_token)}"

    def status_row(self, items: list[tuple[str, str]]) -> str:
        """Join label/value pairs with a muted separator."""
        sep = self.muted("·")
        parts = [self.label_value(l, v) for l, v in items]
        return f" {sep} ".join(parts)

    def banner_pair(self, left: str, right: str, gap: int = 3) -> None:
        """Print two blocks side by side, aligning line by line."""
        left_lines = left.splitlines()
        right_lines = right.splitlines()
        max_lines = max(len(left_lines), len(right_lines))
        left_lines += [""] * (max_lines - len(left_lines))
        right_lines += [""] * (max_lines - len(right_lines))
        for l_line, r_line in zip(left_lines, right_lines):
            self.print(f" {l_line}{' ' * gap}{r_line}")
