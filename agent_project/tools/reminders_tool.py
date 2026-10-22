"""
Reminders/Alarms Tool - Personal assistant reminders with recurring patterns and notifications.
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("reminders.db")


def _init_db():
    """Initialize the reminders database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            message TEXT,
            trigger_time TEXT NOT NULL,
            recurrence TEXT DEFAULT 'none',
            recurrence_end TEXT,
            is_active INTEGER DEFAULT 1,
            is_completed INTEGER DEFAULT 0,
            completed_at TEXT,
            notification_methods TEXT DEFAULT '["popup"]',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_trigger_time ON reminders(trigger_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_active ON reminders(is_active, is_completed)")
    conn.commit()
    conn.close()


class RemindersTool(BaseTool):
    name = "reminders"
    description = (
        "Manage personal reminders and alarms: create, read, update, delete, list, and manage reminders. "
        "Supports one-time and recurring reminders (daily, weekly, monthly, yearly), "
        "with popup/sound/email notification methods."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "update", "delete", "list", "complete", "snooze", "upcoming"],
                "description": "Action to perform on reminders."
            },
            "reminder_id": {
                "type": "integer",
                "description": "Reminder ID for read/update/delete/complete/snooze actions."
            },
            "title": {
                "type": "string",
                "description": "Reminder title (required for create)."
            },
            "message": {
                "type": "string",
                "description": "Reminder message/description."
            },
            "trigger_time": {
                "type": "string",
                "description": "When to trigger in ISO format (e.g., '2026-10-12T10:00:00'). Required for create."
            },
            "recurrence": {
                "type": "string",
                "enum": ["none", "daily", "weekly", "monthly", "yearly", "weekdays", "weekends"],
                "description": "Recurrence pattern. Default 'none'."
            },
            "recurrence_end": {
                "type": "string",
                "description": "End date for recurrence in ISO format."
            },
            "notification_methods": {
                "type": "array",
                "items": {"type": "string", "enum": ["popup", "sound", "email", "webhook"]},
                "description": "Notification methods. Default ['popup']."
            },
            "snooze_minutes": {
                "type": "integer",
                "description": "Minutes to snooze. Default 10."
            },
            "hours_ahead": {
                "type": "integer",
                "description": "List reminders due within N hours. Default 24."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            _ALIASES = {"add": "create", "new": "create", "remove": "delete",
                        "get": "read", "find": "search", "update_time": "snooze"}
            action = _ALIASES.get(action, action)

            if action == "create":
                # 友好参数: message/title 互换; time/datetime → trigger_time
                if not kwargs.get("title"):
                    for k in ("message", "text", "content"):
                        if kwargs.get(k):
                            kwargs["title"] = kwargs[k]
                            break
                if not kwargs.get("trigger_time"):
                    for k in ("time", "datetime", "schedule", "when"):
                        if kwargs.get(k):
                            kwargs["trigger_time"] = kwargs[k]
                            break
                if kwargs.get("trigger_time") and not kwargs.get("message"):
                    kwargs["message"] = kwargs.get("title", "")
                # 灵活时间: "18:00" / "明晚九点" → ISO
                tt = kwargs.get("trigger_time")
                if tt:
                    from .assist_data import resolve_when
                    s, _e, _conf = resolve_when(tt)
                    if s is not None:
                        kwargs["trigger_time"] = s.isoformat()
                if not kwargs.get("trigger_time"):
                    return ToolResult(success=False, output="",
                                      error="create needs trigger_time ('HH:MM'/'明天下午三点'/'YYYY-MM-DD HH:MM')")
                return self._create_reminder(**kwargs)
            elif action == "read":
                return self._read_reminder(**kwargs)
            elif action == "update":
                return self._update_reminder(**kwargs)
            elif action == "delete":
                return self._delete_reminder(**kwargs)
            elif action == "list":
                return self._list_reminders(**kwargs)
            elif action == "complete":
                return self._complete_reminder(**kwargs)
            elif action == "snooze":
                return self._snooze_reminder(**kwargs)
            elif action == "upcoming":
                return self._upcoming_reminders(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _create_reminder(self, title: str, trigger_time: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO reminders (title, message, trigger_time, recurrence, recurrence_end, is_active, notification_methods, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title,
            kwargs.get("message", ""),
            trigger_time,
            kwargs.get("recurrence", "none"),
            kwargs.get("recurrence_end"),
            1,
            json.dumps(kwargs.get("notification_methods", ["popup"])),
            now,
            now
        ))
        reminder_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Reminder created with ID: {reminder_id}",
            metadata={"reminder_id": reminder_id, "title": title, "trigger_time": trigger_time}
        )

    def _read_reminder(self, reminder_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Reminder {reminder_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"reminder": dict(row)}
        )

    def _update_reminder(self, reminder_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Reminder {reminder_id} not found")

        updates = []
        values = []
        for field in ["title", "message", "trigger_time", "recurrence", "recurrence_end"]:
            if field in kwargs:
                updates.append(f"{field} = ?")
                values.append(kwargs[field])

        if "notification_methods" in kwargs:
            updates.append("notification_methods = ?")
            values.append(json.dumps(kwargs["notification_methods"]))

        if not updates:
            conn.close()
            return ToolResult(success=False, output="", error="No fields to update")

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(reminder_id)

        cursor.execute(f"UPDATE reminders SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Reminder {reminder_id} updated", metadata={"reminder_id": reminder_id})

    def _delete_reminder(self, reminder_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Reminder {reminder_id} not found")
        return ToolResult(success=True, output=f"Reminder {reminder_id} deleted", metadata={"reminder_id": reminder_id})

    def _list_reminders(self, active_only: bool = True, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM reminders WHERE 1=1"
        params = []

        if active_only:
            query += " AND is_active = 1 AND is_completed = 0"

        query += " ORDER BY trigger_time ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No reminders found", metadata={"reminders": []})

        lines = [f"Found {len(rows)} reminder(s):"]
        for r in rows:
            trigger = r["trigger_time"][:16].replace("T", " ")
            rec = f" 🔄{r['recurrence']}" if r["recurrence"] != "none" else ""
            status = "✅" if r["is_completed"] else ("🔔" if r["is_active"] else "🔕")
            lines.append(f"  {status} #{r['id']}: {r['title']} | {trigger}{rec}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"reminders": rows, "count": len(rows)}
        )

    def _complete_reminder(self, reminder_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT recurrence FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return ToolResult(success=False, output="", error=f"Reminder {reminder_id} not found")

        recurrence = row[0]

        if recurrence != "none":
            # For recurring reminders, calculate next occurrence
            next_trigger = self._calculate_next_trigger(reminder_id, recurrence)
            if next_trigger:
                cursor.execute("""
                    UPDATE reminders SET trigger_time = ?, is_completed = 0, updated_at = ? WHERE id = ?
                """, (next_trigger, datetime.now().isoformat(), reminder_id))
                conn.commit()
                conn.close()
                return ToolResult(
                    success=True,
                    output=f"Reminder {reminder_id} completed, next occurrence: {next_trigger}",
                    metadata={"reminder_id": reminder_id, "next_trigger": next_trigger}
                )

        # Non-recurring: mark completed
        cursor.execute("""
            UPDATE reminders SET is_completed = 1, completed_at = ?, updated_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), datetime.now().isoformat(), reminder_id))
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Reminder {reminder_id} marked as completed", metadata={"reminder_id": reminder_id})

    def _snooze_reminder(self, reminder_id: int, snooze_minutes: int = 10) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT trigger_time FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return ToolResult(success=False, output="", error=f"Reminder {reminder_id} not found")

        current_trigger = datetime.fromisoformat(row["trigger_time"])
        new_trigger = current_trigger + timedelta(minutes=snooze_minutes)

        cursor.execute("""
            UPDATE reminders SET trigger_time = ?, updated_at = ? WHERE id = ?
        """, (new_trigger.isoformat(), datetime.now().isoformat(), reminder_id))
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Reminder {reminder_id} snoozed to {new_trigger.strftime('%Y-%m-%d %H:%M')}",
            metadata={"reminder_id": reminder_id, "new_trigger": new_trigger.isoformat()}
        )

    def _upcoming_reminders(self, hours_ahead: int = 24) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        now = datetime.now()
        future = now + timedelta(hours=hours_ahead)

        cursor.execute("""
            SELECT * FROM reminders
            WHERE is_active = 1 AND is_completed = 0
            AND trigger_time >= ? AND trigger_time <= ?
            ORDER BY trigger_time ASC
        """, (now.isoformat(), future.isoformat()))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output=f"No upcoming reminders in next {hours_ahead} hours", metadata={"reminders": []})

        lines = [f"⏰ Upcoming reminders (next {hours_ahead}h):"]
        for r in rows:
            trigger = datetime.fromisoformat(r["trigger_time"])
            diff = trigger - now
            hours = int(diff.total_seconds() / 3600)
            minutes = int((diff.total_seconds() % 3600) / 60)
            rec = f" 🔄{r['recurrence']}" if r["recurrence"] != "none" else ""
            lines.append(f"  🔔 #{r['id']}: {r['title']} | in {hours}h {minutes}m ({trigger.strftime('%m-%d %H:%M')}){rec}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"reminders": rows, "count": len(rows)}
        )

    def _calculate_next_trigger(self, reminder_id: int, recurrence: str) -> Optional[str]:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT trigger_time, recurrence_end FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        current = datetime.fromisoformat(row["trigger_time"])
        end_date = datetime.fromisoformat(row["recurrence_end"]) if row["recurrence_end"] else None

        if recurrence == "daily":
            next_time = current + timedelta(days=1)
        elif recurrence == "weekly":
            next_time = current + timedelta(weeks=1)
        elif recurrence == "monthly":
            # Add one month (approximate)
            next_time = current + timedelta(days=30)
        elif recurrence == "yearly":
            next_time = current + timedelta(days=365)
        elif recurrence == "weekdays":
            next_time = current + timedelta(days=1)
            while next_time.weekday() >= 5:  # Skip weekends
                next_time += timedelta(days=1)
        elif recurrence == "weekends":
            next_time = current + timedelta(days=1)
            while next_time.weekday() < 5:
                next_time += timedelta(days=1)
        else:
            return None

        if end_date and next_time > end_date:
            return None

        return next_time.isoformat()


TOOLS_REGISTRY.register(RemindersTool())