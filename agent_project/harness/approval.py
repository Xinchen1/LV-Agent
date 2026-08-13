"""Interactive approval callback for the harness kernel.

TTY callers get a y/n prompt; non-TTY / CI callers fail closed so that
automated runs cannot accidentally approve destructive effects.
"""

from __future__ import annotations

import sys
from typing import Optional

from .effects import Effect


def console_approval(effect: Effect, reason: str, timeout: Optional[float] = None) -> bool:
    """Ask the user whether to allow an effect.

    Args:
        effect: The effect being requested.
        reason: Human-readable policy reason.
        timeout: Optional seconds to wait for input (None = no timeout).

    Returns:
        True if the user explicitly approves, False otherwise.
    """
    if not sys.stdin.isatty():
        return False

    tool = effect.tool_name
    args = effect.arguments
    cmd = args.get("command") if isinstance(args, dict) else None

    print()
    print("=" * 60)
    print("Harness approval required")
    print("-" * 60)
    print(f"Tool: {tool}")
    if cmd:
        print(f"Command: {cmd}")
    if args:
        import json
        print(f"Arguments: {json.dumps(args, ensure_ascii=False, indent=2)}")
    print(f"Reason: {reason}")
    print("-" * 60)

    try:
        answer = input("Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer in ("y", "yes")


def auto_deny(_effect: Effect, _reason: str) -> bool:
    """Headless / CI approval callback that always denies ASK decisions."""
    return False
