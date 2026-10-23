"""Goals tool - track objectives, milestones and progress over time."""

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
    path = db_path("goals.db")
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
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT 'Personal',
            target_value REAL DEFAULT 1,
            current_value REAL DEFAULT 0,
            unit TEXT DEFAULT '',
            due_date TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS goal_milestones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            is_done INTEGER DEFAULT 0,
            due_date TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_goal_status ON goals(status);
        CREATE INDEX IF NOT EXISTS idx_milestone_goal ON goal_milestones(goal_id);
        """
    )
    conn.commit()


def _parse_day(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v == "today":
        return date.today().isoformat()
    if v == "tomorrow":
        return (date.today() + timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(value.strip()).date().isoformat()
    except ValueError:
        return None


class GoalsTool(BaseTool):
    name = "goals"
    description = (
        "Personal goal tracker: create goals with target values and deadlines, update progress, "
        "manage milestones, and review active/achieved/overdue goals."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "update", "progress", "add_milestone",
                         "complete_milestone", "review", "delete"],
                "description": "Action to perform.",
            },
            "goal_id": {"type": "integer", "description": "Goal ID."},
            "title": {"type": "string", "description": "Goal title."},
            "description": {"type": "string", "description": "Goal description."},
            "category": {"type": "string", "description": "Category, e.g. Health/Career."},
            "target_value": {"type": "number", "description": "Target numeric value."},
            "current_value": {"type": "number", "description": "Current numeric value."},
            "unit": {"type": "string", "description": "Unit for values, e.g. kg/books."},
            "due_date": {"type": "string", "description": "Due date (ISO/today)."},
            "status": {"type": "string", "enum": ["active", "achieved", "paused", "abandoned"],
                       "description": "Goal status."},
            "value": {"type": "number", "description": "Absolute value for progress."},
            "delta": {"type": "number", "description": "Increment for progress."},
            "milestone_id": {"type": "integer", "description": "Milestone ID."},
            "milestone_title": {"type": "string", "description": "Milestone title."},
            "include_done": {"type": "boolean", "description": "Include achieved goals in list."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "create": self._create,
                "list": self._list,
                "update": self._update,
                "progress": self._progress,
                "add_milestone": self._add_milestone,
                "complete_milestone": self._complete_milestone,
                "review": self._review,
                "delete": self._delete,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("goals action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _resolve(self, conn: sqlite3.Connection, goal_id: Optional[int],
                 title: Optional[str]) -> Optional[sqlite3.Row]:
        if goal_id is not None:
            return conn.execute("SELECT * FROM goals WHERE id = ?", (int(goal_id),)).fetchone()
        if title:
            return conn.execute(
                "SELECT * FROM goals WHERE lower(title) = lower(?) ORDER BY id DESC LIMIT 1",
                (title.strip(),),
            ).fetchone()
        return None

    def _create(self, title: str = "", **kwargs) -> ToolResult:
        if not title or not title.strip():
            return ToolResult(success=False, output="", error="title is required")
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            cur = conn.execute(
                """INSERT INTO goals
                   (title, description, category, target_value, current_value, unit, due_date, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    title.strip(),
                    kwargs.get("description", ""),
                    kwargs.get("category") or "Personal",
                    float(kwargs.get("target_value") or 1),
                    float(kwargs.get("current_value") or 0),
                    kwargs.get("unit", ""),
                    _parse_day(kwargs.get("due_date")),
                    now,
                    now,
                ),
            )
            conn.commit()
            goal_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Goal '#{goal_id} {title.strip()}' created",
                          metadata={"goal_id": goal_id})

    def _list(self, include_done: bool = True, category: Optional[str] = None,
              **kwargs) -> ToolResult:
        sql = "SELECT * FROM goals"
        clauses, params = [], []
        if not include_done:
            clauses.append("status = 'active'")
        if category:
            clauses.append("category = ?")
            params.append(category)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, due_date IS NULL, due_date, id"
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            milestones: Dict[int, List[Dict[str, Any]]] = {}
            for row in rows:
                ms = conn.execute(
                    "SELECT * FROM goal_milestones WHERE goal_id = ? ORDER BY is_done, due_date",
                    (row["id"],),
                ).fetchall()
                milestones[row["id"]] = [dict(m) for m in ms]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No goals yet", metadata={"goals": []})
        lines = [f"{len(rows)} goal(s):"]
        for r in rows:
            pct = _pct(r["current_value"], r["target_value"])
            due = f" due {r['due_date']}" if r["due_date"] else ""
            lines.append(f"  #{r['id']} [{r['status']}] {r['title']} - {pct:.0f}% "
                         f"({_num(r['current_value'])}/{_num(r['target_value'])}{r['unit']}){due}")
            for m in milestones.get(r["id"], []):
                mark = "[x]" if m["is_done"] else "[ ]"
                lines.append(f"      {mark} #{m['id']} {m['title']}")
        for row in rows:
            row["progress_percent"] = round(_pct(row["current_value"], row["target_value"]), 1)
            row["milestones"] = milestones.get(row["id"], [])
        return ToolResult(success=True, output="\n".join(lines), metadata={"goals": rows})

    def _update(self, goal_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            goal = self._resolve(conn, goal_id, kwargs.get("title"))
            if goal is None:
                return ToolResult(success=False, output="", error="Goal not found")
            fields, values = [], []
            for key in ("title", "description", "category", "target_value", "current_value", "unit", "status"):
                if key in kwargs and kwargs[key] is not None:
                    fields.append(f"{key} = ?")
                    values.append(kwargs[key])
            if "due_date" in kwargs:
                fields.append("due_date = ?")
                values.append(_parse_day(kwargs["due_date"]))
            if not fields:
                return ToolResult(success=False, output="", error="No fields to update")
            fields.append("updated_at = ?")
            values.append(datetime.now().isoformat(timespec="seconds"))
            values.append(goal["id"])
            conn.execute(f"UPDATE goals SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Goal '{goal['title']}' updated",
                          metadata={"goal_id": goal["id"]})

    def _progress(self, goal_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            goal = self._resolve(conn, goal_id, kwargs.get("title"))
            if goal is None:
                return ToolResult(success=False, output="", error="Goal not found")
            current = float(goal["current_value"] or 0)
            if kwargs.get("value") is not None:
                current = float(kwargs["value"])
            elif kwargs.get("delta") is not None:
                current += float(kwargs["delta"])
            else:
                return ToolResult(success=False, output="", error="value or delta is required")
            target = float(goal["target_value"] or 0)
            status = "achieved" if target > 0 and current >= target else goal["status"]
            conn.execute(
                "UPDATE goals SET current_value = ?, status = ?, updated_at = ? WHERE id = ?",
                (current, status, datetime.now().isoformat(timespec="seconds"), goal["id"]),
            )
            conn.commit()
        finally:
            conn.close()
        pct = _pct(current, target)
        return ToolResult(success=True, output=f"Goal '{goal['title']}' progress {pct:.0f}% "
                                            f"({_num(current)}/{_num(target)}{goal['unit']})",
                          metadata={"goal_id": goal["id"], "current_value": current,
                                    "progress_percent": round(pct, 1), "status": status})

    def _add_milestone(self, goal_id: Optional[int] = None, milestone_title: str = "",
                       **kwargs) -> ToolResult:
        if not milestone_title or not milestone_title.strip():
            return ToolResult(success=False, output="", error="milestone_title is required")
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            goal = self._resolve(conn, goal_id, kwargs.get("title"))
            if goal is None:
                return ToolResult(success=False, output="", error="Goal not found")
            cur = conn.execute(
                "INSERT INTO goal_milestones (goal_id, title, due_date, created_at) VALUES (?, ?, ?, ?)",
                (goal["id"], milestone_title.strip(), _parse_day(kwargs.get("due_date")), now),
            )
            conn.commit()
            mid = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Milestone '#{mid} {milestone_title.strip()}' added",
                          metadata={"milestone_id": mid, "goal_id": goal["id"]})

    def _complete_milestone(self, milestone_id: Optional[int] = None, **kwargs) -> ToolResult:
        if milestone_id is None:
            return ToolResult(success=False, output="", error="milestone_id is required")
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE goal_milestones SET is_done = 1 WHERE id = ?", (int(milestone_id),)
            )
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Milestone {milestone_id} not found")
        return ToolResult(success=True, output=f"Milestone #{milestone_id} completed",
                          metadata={"milestone_id": int(milestone_id)})

    def _review(self, **kwargs) -> ToolResult:
        today = date.today().isoformat()
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM goals").fetchall()]
        finally:
            conn.close()
        active = [r for r in rows if r["status"] == "active"]
        achieved = [r for r in rows if r["status"] == "achieved"]
        overdue = [r for r in active if r["due_date"] and r["due_date"] < today]
        lines = [
            "Goal review:",
            f"  total: {len(rows)}",
            f"  active: {len(active)}",
            f"  achieved: {len(achieved)}",
            f"  overdue: {len(overdue)}",
        ]
        for r in sorted(active, key=lambda x: x["due_date"] or "9999")[:5]:
            lines.append(f"  -> {r['title']}: {_pct(r['current_value'], r['target_value']):.0f}%"
                         + (f", due {r['due_date']}" if r["due_date"] else ""))
        return ToolResult(success=True, output="\n".join(lines), metadata={
            "total": len(rows), "active": len(active), "achieved": len(achieved),
            "overdue": [{"id": r["id"], "title": r["title"], "due_date": r["due_date"]} for r in overdue],
        })

    def _delete(self, goal_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            goal = self._resolve(conn, goal_id, kwargs.get("title"))
            if goal is None:
                return ToolResult(success=False, output="", error="Goal not found")
            conn.execute("DELETE FROM goal_milestones WHERE goal_id = ?", (goal["id"],))
            conn.execute("DELETE FROM goals WHERE id = ?", (goal["id"],))
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Goal '{goal['title']}' deleted",
                          metadata={"goal_id": goal["id"]})


def _pct(current: Any, target: Any) -> float:
    try:
        target = float(target or 0)
        current = float(current or 0)
    except (TypeError, ValueError):
        return 0.0
    if target <= 0:
        return 100.0 if current > 0 else 0.0
    return max(0.0, min(100.0, current / target * 100))


def _num(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


TOOLS_REGISTRY.register(GoalsTool())
