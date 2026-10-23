"""Habit tracking tool - daily check-ins, streaks and completion stats."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path

logger = logging.getLogger(__name__)

_SCHEMA_READY: set = set()


def _connect() -> sqlite3.Connection:
    path = db_path("habits.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    key = str(path)
    if key not in _SCHEMA_READY:
        _init_db(conn)
        _SCHEMA_READY.add(key)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            cadence TEXT DEFAULT 'daily',
            target_per_period INTEGER DEFAULT 1,
            color TEXT,
            is_archived INTEGER DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS habit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            note TEXT,
            created_at TEXT,
            UNIQUE(habit_id, log_date)
        );
        CREATE INDEX IF NOT EXISTS idx_habit_logs_date ON habit_logs(habit_id, log_date);
        """
    )
    conn.commit()


def _parse_day(value: Optional[str]) -> str:
    if not value:
        return date.today().isoformat()
    v = value.strip().lower()
    if v in ("today", "now"):
        return date.today().isoformat()
    if v == "tomorrow":
        return (date.today() + timedelta(days=1)).isoformat()
    if v == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(value.strip()).date().isoformat()
    except ValueError:
        return date.today().isoformat()


def _streaks(days: List[str]) -> Dict[str, int]:
    """Return current and longest streak from a list of ISO dates."""
    if not days:
        return {"current": 0, "longest": 0}
    parsed = sorted({datetime.fromisoformat(d).date() for d in days})
    longest = 1
    run = 1
    for prev, cur in zip(parsed, parsed[1:]):
        if (cur - prev).days == 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    today = date.today()
    present = set(parsed)
    anchor = today if today in present else today - timedelta(days=1)
    current = 0
    cursor = anchor
    while cursor in present:
        current += 1
        cursor -= timedelta(days=1)
    return {"current": current, "longest": longest}


class HabitsTool(BaseTool):
    name = "habits"
    description = (
        "Track daily habits: create habits, check in/uncheck, list today's progress, "
        "and compute current/longest streaks and completion rates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "check", "uncheck", "list", "stats", "archive", "unarchive", "delete"],
                "description": "Action to perform.",
            },
            "habit_id": {"type": "integer", "description": "Habit ID."},
            "name": {"type": "string", "description": "Habit name (unique)."},
            "description": {"type": "string", "description": "Habit description."},
            "cadence": {"type": "string", "enum": ["daily", "weekly"], "description": "Cadence (default daily)."},
            "target_per_period": {"type": "integer", "description": "Times per period (default 1)."},
            "color": {"type": "string", "description": "Optional color label."},
            "date": {"type": "string", "description": "Check-in date (ISO/today/yesterday)."},
            "note": {"type": "string", "description": "Optional check-in note."},
            "days": {"type": "integer", "description": "Stats window in days (default 30)."},
            "include_archived": {"type": "boolean", "description": "Include archived habits."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "create": self._create,
                "check": self._check,
                "uncheck": self._uncheck,
                "list": self._list,
                "stats": self._stats,
                "archive": lambda **kw: self._set_archived(True, **kw),
                "unarchive": lambda **kw: self._set_archived(False, **kw),
                "delete": self._delete,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("habits action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _resolve(self, conn: sqlite3.Connection, habit_id: Optional[int],
                 name: Optional[str]) -> Optional[sqlite3.Row]:
        if habit_id is not None:
            return conn.execute("SELECT * FROM habits WHERE id = ?", (int(habit_id),)).fetchone()
        if name:
            return conn.execute(
                "SELECT * FROM habits WHERE lower(name) = lower(?)", (name.strip(),)
            ).fetchone()
        return None

    def _create(self, name: str = "", **kwargs) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, output="", error="name is required")
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            existing = self._resolve(conn, None, name)
            if existing:
                return ToolResult(success=False, output="", error=f"Habit '{name}' already exists")
            cur = conn.execute(
                """INSERT INTO habits (name, description, cadence, target_per_period, color, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    name.strip(),
                    kwargs.get("description", ""),
                    kwargs.get("cadence", "daily"),
                    max(1, int(kwargs.get("target_per_period") or 1)),
                    kwargs.get("color", ""),
                    now,
                ),
            )
            conn.commit()
            habit_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Habit '#{habit_id} {name.strip()}' created",
                          metadata={"habit_id": habit_id})

    def _check(self, habit_id: Optional[int] = None, name: Optional[str] = None,
               **kwargs) -> ToolResult:
        day = _parse_day(kwargs.get("date"))
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            habit = self._resolve(conn, habit_id, name)
            if habit is None:
                return ToolResult(success=False, output="", error="Habit not found")
            conn.execute(
                """INSERT OR REPLACE INTO habit_logs (habit_id, log_date, note, created_at)
                   VALUES (?, ?, ?, ?)""",
                (habit["id"], day, kwargs.get("note", ""), now),
            )
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Checked in '{habit['name']}' for {day}",
                          metadata={"habit_id": habit["id"], "date": day})

    def _uncheck(self, habit_id: Optional[int] = None, name: Optional[str] = None,
                 **kwargs) -> ToolResult:
        day = _parse_day(kwargs.get("date"))
        conn = _connect()
        try:
            habit = self._resolve(conn, habit_id, name)
            if habit is None:
                return ToolResult(success=False, output="", error="Habit not found")
            cur = conn.execute(
                "DELETE FROM habit_logs WHERE habit_id = ? AND log_date = ?", (habit["id"], day)
            )
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"No check-in for {day}")
        return ToolResult(success=True, output=f"Unchecked '{habit['name']}' for {day}",
                          metadata={"habit_id": habit["id"], "date": day})

    def _list(self, include_archived: bool = False, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            sql = "SELECT * FROM habits"
            if not include_archived:
                sql += " WHERE is_archived = 0"
            sql += " ORDER BY is_archived, id"
            habits = conn.execute(sql).fetchall()
            results = []
            for h in habits:
                logs = conn.execute(
                    "SELECT log_date FROM habit_logs WHERE habit_id = ? ORDER BY log_date",
                    (h["id"],),
                ).fetchall()
                days = [r["log_date"] for r in logs]
                results.append({
                    "id": h["id"],
                    "name": h["name"],
                    "cadence": h["cadence"],
                    "archived": bool(h["is_archived"]),
                    "done_today": date.today().isoformat() in set(days),
                    "total_checkins": len(days),
                    **_streaks(days),
                })
        finally:
            conn.close()
        if not results:
            return ToolResult(success=True, output="No habits yet", metadata={"habits": []})
        lines = [f"{len(results)} habit(s):"]
        for r in results:
            mark = "[x]" if r["done_today"] else "[ ]"
            arch = " (archived)" if r["archived"] else ""
            lines.append(
                f"  {mark} #{r['id']} {r['name']} - streak {r['current']}d, best {r['longest']}d, "
                f"total {r['total_checkins']}{arch}"
            )
        return ToolResult(success=True, output="\n".join(lines), metadata={"habits": results})

    def _stats(self, habit_id: Optional[int] = None, days: int = 30, **kwargs) -> ToolResult:
        days = max(1, int(days or 30))
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        conn = _connect()
        try:
            habit = self._resolve(conn, habit_id, kwargs.get("name"))
            if habit is None:
                return ToolResult(success=False, output="", error="Habit not found")
            logs = conn.execute(
                "SELECT log_date FROM habit_logs WHERE habit_id = ? AND log_date >= ? ORDER BY log_date",
                (habit["id"], since),
            ).fetchall()
            all_logs = conn.execute(
                "SELECT log_date FROM habit_logs WHERE habit_id = ? ORDER BY log_date",
                (habit["id"],),
            ).fetchall()
        finally:
            conn.close()
        all_days = [r["log_date"] for r in all_logs]
        window_days = [r["log_date"] for r in logs]
        streaks = _streaks(all_days)
        rate = len(window_days) / days * 100
        output = (
            f"Habit '{habit['name']}' stats:\n"
            f"  current streak: {streaks['current']} days\n"
            f"  longest streak: {streaks['longest']} days\n"
            f"  last {days} days: {len(window_days)}/{days} ({rate:.0f}%)\n"
            f"  total check-ins: {len(all_days)}"
        )
        return ToolResult(success=True, output=output, metadata={
            "habit_id": habit["id"], "current_streak": streaks["current"],
            "longest_streak": streaks["longest"], "completion_rate": round(rate, 1),
            "window_days": days, "total": len(all_days),
        })

    def _set_archived(self, archived: bool, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            habit = self._resolve(conn, kwargs.get("habit_id"), kwargs.get("name"))
            if habit is None:
                return ToolResult(success=False, output="", error="Habit not found")
            conn.execute("UPDATE habits SET is_archived = ? WHERE id = ?",
                         (1 if archived else 0, habit["id"]))
            conn.commit()
        finally:
            conn.close()
        state = "archived" if archived else "restored"
        return ToolResult(success=True, output=f"Habit '{habit['name']}' {state}",
                          metadata={"habit_id": habit["id"], "archived": archived})

    def _delete(self, habit_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            habit = self._resolve(conn, habit_id, kwargs.get("name"))
            if habit is None:
                return ToolResult(success=False, output="", error="Habit not found")
            conn.execute("DELETE FROM habit_logs WHERE habit_id = ?", (habit["id"],))
            conn.execute("DELETE FROM habits WHERE id = ?", (habit["id"],))
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Habit '{habit['name']}' deleted",
                          metadata={"habit_id": habit["id"]})


TOOLS_REGISTRY.register(HabitsTool())
