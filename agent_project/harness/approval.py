"""Interactive approval callback for the harness kernel.

TTY callers get a styled y/n prompt; non-TTY / CI callers fail closed so that
automated runs cannot accidentally approve destructive effects.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .effects import Effect

_console = Console()


def _truncate(s: str, n: int = 150) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def console_approval(effect: Effect, reason: str, timeout: Optional[float] = None) -> bool:
    """Ask the user whether to allow an effect (styled one-screen prompt)."""
    if not sys.stdin.isatty():
        return False

    tool = effect.tool_name
    args = effect.arguments if isinstance(effect.arguments, dict) else {}

    if args:
        try:
            compact = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            details = Text(_truncate(compact), style="bold", overflow="ellipsis", no_wrap=True)
        except Exception:
            details = Text(_truncate(str(args)), style="bold", overflow="ellipsis", no_wrap=True)
    else:
        details = Text("(no arguments)", style="dim", no_wrap=True)

    panel = Panel(
        details,
        title=" Harness approval ",
        title_align="left",
        subtitle=f" {_truncate(reason, 70)} ",
        subtitle_align="left",
        border_style="yellow",
        width=76,
        padding=(0, 1),
    )

    _console.print()
    _console.print(panel)
    _console.print(
        Text("   ") + Text("[y] allow", style="bold green")
        + Text("    ") + Text("[N] deny", style="dim")
        + Text("   ·   ") + Text(tool, style="italic cyan")
    )

    try:
        from ..stream_adapters import pause_active_spinner, resume_active_spinner
    except Exception:
        pause_active_spinner = resume_active_spinner = lambda: None

    pause_active_spinner()
    try:
        answer = input("   Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        resume_active_spinner()
        _console.print("   [red]denied[/red]")
        return False
    finally:
        resume_active_spinner()

    ok = answer in ("y", "yes")
    _console.print("   [green]allowed[/green]" if ok else "   [red]denied[/red]")
    return ok


def auto_deny(_effect: Effect, _reason: str) -> bool:
    """Headless / CI approval callback that always denies ASK decisions."""
    return False
