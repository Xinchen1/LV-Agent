"""
Calendar/Schedule Tool - Personal assistant scheduling using local SQLite storage.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path, is_isoish


DB_PATH = db_path("calendar.db")


def _init_db():
    """Initialize the calendar database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            location TEXT,
            attendees TEXT,
            recurrence TEXT,
            reminder_minutes INTEGER DEFAULT 15,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time)
    """)
    conn.commit()
    conn.close()


class CalendarTool(BaseTool):
    name = "calendar"
    description = (
        "Manage personal calendar events: create, read, update, delete, and list events. "
        "Supports recurring events, reminders, and conflict detection."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "update", "delete", "list", "search", "conflicts"],
                "description": "Action to perform on calendar events."
            },
            "event_id": {
                "type": "integer",
                "description": "Event ID for read/update/delete actions."
            },
            "title": {
                "type": "string",
                "description": "Event title (required for create)."
            },
            "date": {
                "type": "string",
                "description": "Event date for create: ISO date ('2026-10-12') or natural language ('明天', '下周五'). Optional when start_time is used."
            },
            "time": {
                "type": "string",
                "description": "Start time for create: 'HH:MM', 'YYYY-MM-DD HH:MM' or natural language ('下午三点'). Optional (default 09:00)."
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Event duration in minutes for create. Default 60."
            },
            "description": {
                "type": "string",
                "description": "Event description."
            },
            "start_time": {
                "type": "string",
                "description": "Start time in ISO format (e.g., '2026-10-12T10:00:00'). Required for create."
            },
            "end_time": {
                "type": "string",
                "description": "End time in ISO format (e.g., '2026-10-12T11:00:00'). Required for create."
            },
            "location": {
                "type": "string",
                "description": "Event location."
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee emails/names."
            },
            "recurrence": {
                "type": "string",
                "enum": ["none", "daily", "weekly", "monthly", "yearly"],
                "description": "Recurrence pattern. Default 'none'."
            },
            "reminder_minutes": {
                "type": "integer",
                "description": "Reminder minutes before event. Default 15."
            },
            "date_from": {
                "type": "string",
                "description": "Start date for list/search (ISO format)."
            },
            "date_to": {
                "type": "string",
                "description": "End date for list/search (ISO format)."
            },
            "query": {
                "type": "string",
                "description": "Search query for event titles/descriptions."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            # 语义别名: LLM/用户常给 create/update/delete 的同义词
            _ALIASES = {"add": "create", "new": "create", "remove": "delete",
                        "cancel": "delete", "get": "read", "find": "search", "events": "list"}
            action = _ALIASES.get(action, action)

            if action == "create":
                # 友好参数: date [+ time] / duration_minutes → start_time/end_time
                if not kwargs.get("start_time") and (kwargs.get("date") or kwargs.get("time")):
                    from .assist_data import resolve_when
                    expr = (kwargs.get("date") or "")
                    if kwargs.get("time"):
                        expr = f"{expr}{kwargs.get('time')}" if expr else kwargs.get("time")
                    s, e, _conf = resolve_when(expr, all_day_if_no_time=True) if expr else (None, None, 0)
                    if s is not None:
                        dur = int(kwargs.get("duration_minutes") or 60)
                        kwargs["start_time"] = s.isoformat()
                        kwargs["end_time"] = e.isoformat() if (e and e != s) else (s + timedelta(minutes=dur)).isoformat()
                if kwargs.get("start_time") and not kwargs.get("end_time"):
                    dur = int(kwargs.get("duration_minutes") or 60)
                    _s = datetime.fromisoformat(kwargs["start_time"].replace("T", " "))
                    kwargs["end_time"] = (_s + timedelta(minutes=dur)).isoformat()
                return self._create_event(**kwargs)
            elif action == "read":
                return self._read_event(**kwargs)
            elif action == "update":
                return self._update_event(**kwargs)
            elif action == "delete":
                return self._delete_event(**kwargs)
            elif action == "list":
                # "date" 单日 → date_from/date_to
                if kwargs.get("date"):
                    d = kwargs.pop("date")
                    kwargs.setdefault("date_from", d)
                    kwargs.setdefault("date_to", d)
                return self._list_events(**kwargs)
            elif action == "search":
                return self._search_events(**kwargs)
            elif action == "conflicts":
                return self._check_conflicts(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _create_event(self, title: str, start_time: str, end_time: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO events (title, description, start_time, end_time, location, attendees, recurrence, reminder_minutes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title,
            kwargs.get("description", ""),
            start_time,
            end_time,
            kwargs.get("location", ""),
            json.dumps(kwargs.get("attendees", [])),
            kwargs.get("recurrence", "none"),
            kwargs.get("reminder_minutes", 15),
            now,
            now
        ))
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Event created with ID: {event_id}",
            metadata={"event_id": event_id, "title": title, "start_time": start_time}
        )

    def _read_event(self, event_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Event {event_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"event": dict(row)}
        )

    def _update_event(self, event_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Event {event_id} not found")

        updates = []
        values = []
        for field in ["title", "description", "start_time", "end_time", "location", "recurrence", "reminder_minutes"]:
            if field in kwargs:
                updates.append(f"{field} = ?")
                values.append(kwargs[field])
        if "attendees" in kwargs:
            updates.append("attendees = ?")
            values.append(json.dumps(kwargs["attendees"]))

        if not updates:
            conn.close()
            return ToolResult(success=False, output="", error="No fields to update")

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(event_id)

        cursor.execute(f"UPDATE events SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Event {event_id} updated", metadata={"event_id": event_id})

    def _delete_event(self, event_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Event {event_id} not found")
        return ToolResult(success=True, output=f"Event {event_id} deleted", metadata={"event_id": event_id})

    def _list_events(self, date_from: str = None, date_to: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if date_from and is_isoish(date_from) and len(date_from) == 10:  # 纯日期
            date_from += "T00:00:00"
        if date_to and is_isoish(date_to) and len(date_to) == 10:
            date_to += "T23:59:59"

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if date_from:
            query += " AND start_time >= ?"
            params.append(date_from)
        if date_to:
            query += " AND start_time <= ?"
            params.append(date_to + "T23:59:59")

        query += " ORDER BY start_time ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No events found", metadata={"events": []})

        lines = [f"Found {len(rows)} event(s):"]
        for r in rows:
            start = r["start_time"][:16].replace("T", " ")
            end = r["end_time"][:16].replace("T", " ")
            loc = f" @ {r['location']}" if r["location"] else ""
            rec = f" [{r['recurrence']}]" if r["recurrence"] != "none" else ""
            lines.append(f"  #{r['id']}: {r['title']} | {start}~{end}{loc}{rec}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"events": rows, "count": len(rows)}
        )

    def _search_events(self, query: str, date_from: str = None, date_to: str = None) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM events WHERE (title LIKE ? OR description LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if date_from:
            sql += " AND start_time >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND start_time <= ?"
            params.append(date_to + "T23:59:59")

        sql += " ORDER BY start_time ASC"
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output=f"No events matching '{query}'", metadata={"events": []})

        lines = [f"Found {len(rows)} event(s) matching '{query}':"]
        for r in rows:
            start = r["start_time"][:16].replace("T", " ")
            lines.append(f"  #{r['id']}: {r['title']} | {start}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"events": rows, "count": len(rows)}
        )

    def _check_conflicts(self, start_time: str, end_time: str, exclude_id: int = None) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = """
            SELECT * FROM events
            WHERE start_time < ? AND end_time > ?
        """
        params = [end_time, start_time]

        if exclude_id:
            sql += " AND id != ?"
            params.append(exclude_id)

        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(
                success=True,
                output="No conflicts found",
                metadata={"conflicts": [], "has_conflicts": False}
            )

        lines = [f"Found {len(rows)} conflict(s):"]
        for r in rows:
            s = r["start_time"][:16].replace("T", " ")
            e = r["end_time"][:16].replace("T", " ")
            lines.append(f"  #{r['id']}: {r['title']} | {s}~{e}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"conflicts": rows, "has_conflicts": True}
        )


TOOLS_REGISTRY.register(CalendarTool())