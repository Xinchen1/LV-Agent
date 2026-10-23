"""
Personal Profile Tool - 个人偏好/画像管理。
独立 SQLite 存储，无需 Agent 实例即可使用；供 MCP、快捷命令、PersonalAgent 复用。

存储键值对：timezone, language, home_address, work_address, currency, wake_time,
sleep_time, focus_minutes, briefing_sections, favorite_contacts 等。
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("personal_profile.db")
_LOCK = threading.RLock()

_KNOWN_KEYS = (
    "timezone", "language", "home_address", "work_address", "default_calendar",
    "currency", "wake_time", "sleep_time", "focus_minutes", "briefing_sections",
    "favorite_contacts",
)


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS personal_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _get(key: str, default: Any = None) -> Any:
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT value FROM personal_profile WHERE key = ?", (key,)).fetchone()
        conn.close()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]


def _set(key: str, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, default=str)
    now = datetime.now().isoformat()
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO personal_profile(key, value, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, payload, now),
        )
        conn.commit()
        conn.close()


class PersonalProfileTool(BaseTool):
    name = "personal_profile"
    description = (
        "Manage personal profile / preferences: timezone, language, home & work address, "
        "currency, wake/sleep time, default focus minutes, briefing sections, favorite contacts. "
        "Actions: get (single key), set (key+value), list (all)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set", "list"],
                "description": "get: read one key; set: write key+value; list: dump all preferences.",
            },
            "key": {
                "type": "string",
                "description": "Preference key (e.g. timezone, focus_minutes, home_address).",
            },
            "value": {
                "description": "Preference value to set (string/number/list). Required for 'set'.",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, key: Optional[str] = None, value: Any = None,
                **kwargs) -> ToolResult:
        try:
            if action == "get":
                if not key:
                    return ToolResult(success=False, output="", error="key is required for 'get'")
                v = _get(key)
                if v is None:
                    return ToolResult(success=True, output=f"未设置偏好：{key}", metadata={"key": key, "value": None})
                return ToolResult(success=True, output=f"{key} = {v}", metadata={"key": key, "value": v})
            elif action == "set":
                if not key:
                    return ToolResult(success=False, output="", error="key is required for 'set'")
                if value is None and "value" not in kwargs:
                    return ToolResult(success=False, output="", error="value is required for 'set'")
                if value is None:
                    value = kwargs.get("value")
                _set(key, value)
                return ToolResult(success=True, output=f"已设置 {key} = {value}", metadata={"key": key, "value": value})
            elif action == "list":
                with _LOCK:
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute("SELECT key, value FROM personal_profile ORDER BY key").fetchall()
                    conn.close()
                if not rows:
                    return ToolResult(success=True, output="暂无个人偏好设置。用 /profile <key> <value> 设置。", metadata={"profile": {}})
                lines = ["👤 个人偏好："]
                profile: Dict[str, Any] = {}
                for r in rows:
                    try:
                        v = json.loads(r["value"])
                    except Exception:
                        v = r["value"]
                    profile[r["key"]] = v
                    lines.append(f"  {r['key']}: {v}")
                return ToolResult(success=True, output="\n".join(lines), metadata={"profile": profile})
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


TOOLS_REGISTRY.register(PersonalProfileTool())
