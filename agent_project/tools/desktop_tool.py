"""Desktop control tool - notifications, opening targets, screenshots, clipboard and volume.

Uses native OS commands (macOS first, Linux/Windows fallbacks) with strict argv
execution (never a shell). Unsupported actions fail with a clear error instead of
crashing the agent loop.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import assistant_data_dir

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


def _run(argv: List[str], input_text: Optional[str] = None, timeout: int = 15):
    return subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class DesktopTool(BaseTool):
    name = "desktop"
    description = (
        "Control the desktop: show notifications/sounds, open URLs/files/apps, reveal files, "
        "take screenshots, read/write the clipboard, set volume and control music playback."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["notify", "open", "reveal", "screenshot", "clipboard_get",
                         "clipboard_set", "volume", "media", "capabilities"],
                "description": "Desktop action to perform.",
            },
            "title": {"type": "string", "description": "Notification title."},
            "message": {"type": "string", "description": "Notification message / clipboard text."},
            "sound": {"type": "string", "description": "Notification sound name."},
            "target": {"type": "string", "description": "URL, file path or app name to open."},
            "path": {"type": "string", "description": "Output path (screenshot) or reveal path."},
            "app": {"type": "boolean", "description": "Treat target as an application name."},
            "level": {"type": "integer", "description": "Volume level 0-100."},
            "mute": {"type": "boolean", "description": "Mute audio."},
            "media_action": {"type": "string", "enum": ["playpause", "next", "previous", "stop"],
                             "description": "Playback control."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "notify": self._notify,
                "open": self._open,
                "reveal": self._reveal,
                "screenshot": self._screenshot,
                "clipboard_get": self._clipboard_get,
                "clipboard_set": self._clipboard_set,
                "volume": self._volume,
                "media": self._media,
                "capabilities": self._capabilities,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Desktop command timed out")
        except FileNotFoundError as e:
            return ToolResult(success=False, output="", error=f"Required command not found: {e}")
        except Exception as e:
            logger.debug("desktop action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _notify(self, title: str = "Agent", message: str = "", sound: str = "",
                **kwargs) -> ToolResult:
        message = message or title
        if _SYSTEM == "Darwin":
            script = f'display notification "{_escape_applescript(message)}" with title "{_escape_applescript(title)}"'
            if sound:
                script += f' sound name "{_escape_applescript(sound)}"'
            proc = _run(["osascript", "-e", script])
        elif _SYSTEM == "Linux" and shutil.which("notify-send"):
            proc = _run(["notify-send", title, message])
        else:
            return ToolResult(success=False, output="",
                              error=f"Notifications unsupported on {_SYSTEM}")
        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=proc.stderr.strip() or "notify failed")
        return ToolResult(success=True, output=f"Notification sent: {title}")

    def _open(self, target: str = "", app: bool = False, **kwargs) -> ToolResult:
        if not target:
            return ToolResult(success=False, output="", error="target is required")
        if not app and "://" not in target and not target.startswith(("mailto:", "file:")):
            p = Path(target).expanduser()
            if p.exists() or "/" in target:
                target = str(p.resolve())
        if _SYSTEM == "Darwin":
            argv = ["open", "-a", target] if app else ["open", target]
        elif _SYSTEM == "Windows":
            os.startfile(target)  # type: ignore[attr-defined]
            return ToolResult(success=True, output=f"Opened: {target}")
        else:
            argv = ["xdg-open", target]
        proc = _run(argv)
        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=proc.stderr.strip() or "open failed")
        return ToolResult(success=True, output=f"Opened: {target}", metadata={"target": target})

    def _reveal(self, path: str = "", target: str = "", **kwargs) -> ToolResult:
        raw = path or target
        if not raw:
            return ToolResult(success=False, output="", error="path is required")
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {p}")
        if _SYSTEM == "Darwin":
            proc = _run(["open", "-R", str(p)])
        elif _SYSTEM == "Windows":
            proc = _run(["explorer", "/select,", str(p)])
        else:
            proc = _run(["xdg-open", str(p.parent)])
        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=proc.stderr.strip() or "reveal failed")
        return ToolResult(success=True, output=f"Revealed: {p}", metadata={"path": str(p)})

    def _screenshot(self, path: str = "", **kwargs) -> ToolResult:
        if _SYSTEM != "Darwin":
            return ToolResult(success=False, output="",
                              error=f"Screenshot unsupported on {_SYSTEM}")
        if path:
            out = Path(path).expanduser()
        else:
            folder = assistant_data_dir() / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / f"screenshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = _run(["screencapture", "-x", str(out)])
        if proc.returncode != 0 or not out.exists():
            return ToolResult(success=False, output="",
                              error=proc.stderr.strip() or "screenshot failed")
        return ToolResult(success=True, output=f"Screenshot saved: {out}",
                          metadata={"path": str(out), "bytes": out.stat().st_size})

    def _clipboard_get(self, **kwargs) -> ToolResult:
        if _SYSTEM == "Darwin":
            proc = _run(["pbpaste"])
        elif _SYSTEM == "Windows":
            proc = _run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])
        elif shutil.which("wl-paste"):
            proc = _run(["wl-paste", "--no-newline"])
        elif shutil.which("xclip"):
            proc = _run(["xclip", "-selection", "clipboard", "-o"])
        else:
            return ToolResult(success=False, output="", error="Clipboard unsupported")
        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=proc.stderr.strip() or "clipboard read failed")
        return ToolResult(success=True, output=proc.stdout, metadata={"characters": len(proc.stdout)})

    def _clipboard_set(self, message: str = "", text: str = "", **kwargs) -> ToolResult:
        value = message if message else text
        if value is None:
            return ToolResult(success=False, output="", error="message is required")
        if _SYSTEM == "Darwin":
            proc = _run(["pbcopy"], input_text=value)
        elif _SYSTEM == "Windows":
            proc = _run(["powershell", "-NoProfile", "-Command", "Set-Clipboard"], input_text=value)
        elif shutil.which("wl-copy"):
            proc = _run(["wl-copy"], input_text=value)
        elif shutil.which("xclip"):
            proc = _run(["xclip", "-selection", "clipboard"], input_text=value)
        else:
            return ToolResult(success=False, output="", error="Clipboard unsupported")
        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=proc.stderr.strip() or "clipboard write failed")
        return ToolResult(success=True, output=f"Copied {len(value)} characters to clipboard",
                          metadata={"characters": len(value)})

    def _volume(self, level: Optional[int] = None, mute: bool = False,
                **kwargs) -> ToolResult:
        if mute:
            if _SYSTEM == "Darwin":
                proc = _run(["osascript", "-e", "set volume output muted true"])
            elif shutil.which("pactl"):
                proc = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
            else:
                return ToolResult(success=False, output="", error="Mute unsupported")
            return ToolResult(success=proc.returncode == 0,
                              output="Muted" if proc.returncode == 0 else "",
                              error=None if proc.returncode == 0 else "mute failed")
        if level is None:
            return ToolResult(success=False, output="", error="level or mute is required")
        value = max(0, min(100, int(level)))
        if _SYSTEM == "Darwin":
            proc = _run(["osascript", "-e", f"set volume output volume {value}"])
        elif shutil.which("pactl"):
            proc = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{value}%"])
        else:
            return ToolResult(success=False, output="", error="Volume control unsupported")
        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=proc.stderr.strip() or "volume failed")
        return ToolResult(success=True, output=f"Volume set to {value}%", metadata={"level": value})

    def _media(self, media_action: str = "playpause", **kwargs) -> ToolResult:
        if _SYSTEM != "Darwin":
            return ToolResult(success=False, output="",
                              error=f"Media control unsupported on {_SYSTEM}")
        commands = {
            "playpause": 'tell application "Music" to playpause',
            "next": 'tell application "Music" to next track',
            "previous": 'tell application "Music" to previous track',
            "stop": 'tell application "Music" to stop',
        }
        script = commands.get(media_action)
        if not script:
            return ToolResult(success=False, output="", error=f"Unknown media_action: {media_action}")
        proc = _run(["osascript", "-e", script])
        if proc.returncode != 0:
            return ToolResult(success=False, output="",
                              error=proc.stderr.strip() or "media control failed (is Music running?)")
        return ToolResult(success=True, output=f"Media {media_action} sent",
                          metadata={"media_action": media_action})

    def _capabilities(self, **kwargs) -> ToolResult:
        data: Dict[str, Any] = {"platform": _SYSTEM}
        checks = {
            "notify": "osascript" if _SYSTEM == "Darwin" else "notify-send",
            "open": "open" if _SYSTEM == "Darwin" else "xdg-open",
            "screenshot": "screencapture" if _SYSTEM == "Darwin" else "",
            "clipboard": "pbpaste" if _SYSTEM == "Darwin" else ("wl-paste" if shutil.which("wl-paste") else "xclip"),
        }
        for name, binary in checks.items():
            data[name] = bool(binary) and (shutil.which(binary) is not None)
        lines = [f"platform: {_SYSTEM}"] + [f"{k}: {'yes' if v else 'no'}" for k, v in data.items() if k != "platform"]
        return ToolResult(success=True, output="\n".join(lines), metadata=data)


TOOLS_REGISTRY.register(DesktopTool())
