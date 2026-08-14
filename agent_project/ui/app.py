"""CLI application entry point for Lv Agent (Phase 1).

This module replaces the ad-hoc logic in agent_project/__main__.py with a
small, testable CLIApp class that uses the new Renderer/Banner components.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .banner import render_banner, render_system_status
from .renderer import Renderer
from .themes import display_config_from, load_theme


class CLIApp:
    """Line-mode CLI frontend for Lv Agent."""

    def __init__(self, config: Any, args: Optional[argparse.Namespace] = None):
        self.config = config
        self.args = args or argparse.Namespace()
        self.display_cfg = display_config_from(config)
        self.theme = load_theme(
            self.display_cfg.get("theme", "dark"),
            custom_overrides=None,
        )
        self.renderer = Renderer(self.theme)
        self.agent: Optional[Any] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _model_name(self) -> str:
        """Return the currently configured model name."""
        backend = getattr(self.config, "backend", "unknown")
        section = getattr(self.config, backend, None)
        if isinstance(section, dict):
            return section.get("model", "unknown")
        return "unknown"

    def _backend_name(self) -> str:
        """Return the backend class/identifier name."""
        if self.agent is not None:
            return type(self.agent.backend).__name__
        return getattr(self.config, "backend", "unknown")

    def _tool_count(self) -> int:
        """Return the number of available tools."""
        try:
            from agent_project.tools import TOOLS_REGISTRY

            return len(list(TOOLS_REGISTRY.list_tools()))
        except Exception:
            return 0

    def _loop_count(self) -> int:
        """Return the configured max outer loop count."""
        return getattr(self.config, "max_outer_loops", 1)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def show_banner(self) -> None:
        """Render the startup banner and system status."""
        r = self.renderer
        cols = r.columns()
        r.line(width=cols)
        r.blank()

        minimal = not r.theme.supports_color or not sys.stdout.isatty()
        # 自适应头像宽度: 终端越宽, 头像越大, 像素点越多(更细腻)
        _pw = 34
        if cols >= 150:
            _pw = 52
        elif cols >= 120:
            _pw = 44
        banner = render_banner(r, portrait_width=_pw, show_minimal=minimal)
        r.print(banner)

        r.blank()
        status = render_system_status(
            r,
            backend=self._backend_name(),
            model=self._model_name(),
            tools=self._tool_count(),
            loops=self._loop_count(),
        )
        r.print(status)
        r.blank()
        r.line(width=cols)
        r.blank()

    def show_version(self) -> None:
        from agent_project import __version__

        self.renderer.print(f"Lv Super Agent v{__version__}")

    def list_tools(self) -> None:
        try:
            from agent_project.tools import TOOLS_REGISTRY

            self.renderer.print("\nAvailable tools:")
            for tool in TOOLS_REGISTRY.list_tools():
                self.renderer.print(f"  - {tool}")
        except Exception as e:
            self.renderer.print(self.renderer.error(f"Failed to list tools: {e}"))

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------
    def initialize_agent(self) -> bool:
        """Initialize the OpenMythosAgent."""
        try:
            from agent_project.agent import OpenMythosAgent

            self.agent = OpenMythosAgent(self.config)
            self.renderer.print(self.renderer.dim("  agent initialized"))
            return True
        except Exception as e:
            self.renderer.print(self.renderer.error(f"  agent init failed: {e}"))
            return False

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------
    def _handle_slash(self, text: str) -> bool:
        """Handle slash commands. Return True if the command was consumed."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self.renderer.print(self.renderer.dim("  goodbye!"))
            sys.exit(0)
        if cmd == "/help":
            self._show_help()
            return True
        if cmd == "/model":
            self.renderer.print(
                self.renderer.status_row([("backend", self._backend_name()), ("model", self._model_name())])
            )
            return True
        if cmd == "/tools":
            self.list_tools()
            return True
        if cmd == "/version":
            self.show_version()
            return True
        if cmd.startswith("/"):
            self.renderer.print(self.renderer.warning(f"  unknown command: {cmd}"))
            return True
        return False

    def _show_help(self) -> None:
        r = self.renderer
        r.print(r.brand(" Lv Agent commands"))
        commands = [
            ("/help", "显示此帮助"),
            ("/model", "显示当前后端与模型"),
            ("/tools", "列出可用工具"),
            ("/version", "显示版本"),
            ("/exit 或 /quit", "退出"),
        ]
        for cmd, desc in commands:
            r.print(f"  {r.accent(cmd):<18} {r.muted(desc)}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the interactive CLI loop."""
        r = self.renderer

        if not self.initialize_agent():
            sys.exit(1)

        r.print(r.muted("  进入交互模式，输入 '/help' 查看命令，'/exit' 退出"))

        while True:
            try:
                prompt = self._build_prompt()
                user_input = input(prompt).strip()
            except KeyboardInterrupt:
                r.print("\n" + r.dim("  Interrupted by user"))
                break
            except EOFError:
                r.print(r.dim("  goodbye!"))
                break

            if not user_input:
                continue

            if self._handle_slash(user_input):
                continue

            self._run_agent_turn(user_input)

    def _build_prompt(self) -> str:
        """Build the user input prompt."""
        r = self.renderer
        parts = [r.brand("Lv")]

        cfg = self.display_cfg.get("prompt", {})
        if cfg.get("show_cwd", True):
            try:
                cwd = os.getcwd()
                home = os.path.expanduser("~")
                if cwd.startswith(home):
                    cwd = "~" + cwd[len(home):]
            except Exception:
                cwd = "."
            parts.append(r.muted(cwd))

        if cfg.get("show_branch", True):
            branch = self._git_branch()
            if branch:
                parts.append(r.muted(branch))

        prompt_text = " → "
        return " ".join(parts) + r.themed(prompt_text, "muted")

    def _git_branch(self) -> str:
        """Return the current git branch, or empty string if not available."""
        try:
            from subprocess import run

            result = run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _run_agent_turn(self, user_input: str) -> None:
        """Run one agent turn and render the result."""
        r = self.renderer
        r.print(r.dim("  Thinking…"))

        try:
            result = self.agent.run(user_input)
            self._render_result(result)
        except KeyboardInterrupt:
            r.print("\n" + r.dim("  Turn interrupted"))
        except Exception as e:
            r.print(r.error(f"  Error: {e}"))

    def _render_result(self, result: Dict[str, Any]) -> None:
        """Render the agent result and metadata."""
        r = self.renderer
        ok = bool(result.get("success"))

        if ok:
            output = result.get("final_output") or result.get("final_answer") or "No output"
            r.print(f"{r.success('✓')} {output}")
        else:
            error = result.get("error") or result.get("failure") or "Unknown error"
            r.print(f"{r.error('✗')} {error}")

        # Metadata footer
        metadata = result.get("metadata", {})
        duration = metadata.get("duration_ms", 0) / 1000
        loops = result.get("outer_loops", 0)
        steps = result.get("thinking_steps", 0)
        tokens = result.get("live_tokens") or result.get("session_token_usage", {}).get("last_call_tokens", 0)
        tokens_display = f"{tokens // 1000}.{tokens % 1000 // 100}k" if tokens >= 1000 else str(tokens)

        status = r.success("ok") if ok else r.error("failed")
        meta_parts = [
            status,
            r.dim(f"{loops} loops"),
            r.dim(f"{steps} steps"),
            r.dim(f"{duration:.1f}s"),
            r.dim(f"{tokens_display} tokens"),
        ]
        r.print(" " + r.dim("· " + "─" * 38 + " ·"))
        r.print(" " + r.dim(" · ").join(meta_parts))


# ------------------------------------------------------------------------------
# Public entry point used by __main__.py
# ------------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lv Super Agent - Deep-thinking agent with recurrent architecture"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--loops",
        type=int,
        help="Override default thinking loops",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Override theme (light, dark, minimal)",
    )

    args = parser.parse_args(argv)

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from agent_project.config import load_config
    from agent_project.logging_module import setup_logging

    config = load_config(args.config)
    if args.loops:
        config.default_thinking_loops = args.loops
        config.max_thinking_loops = args.loops * 2

    setup_logging(config.logging)

    # Theme override from CLI
    if args.theme:
        if not hasattr(config, "display") or config.display is None:
            config.display = {}
        config.display["theme"] = args.theme

    app = CLIApp(config, args)

    if args.version:
        app.show_version()
        return 0

    if args.list_tools:
        app.list_tools()
        return 0

    app.show_banner()
    try:
        app.run()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    return 0
