"""
Task/Todo Management Tool - Personal assistant task management with projects, priorities, due dates.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("tasks.db")


def _init_db():
    """Initialize the tasks database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            project TEXT,
            priority INTEGER DEFAULT 3,
            status TEXT DEFAULT 'pending',
            due_date TEXT,
            completed_at TEXT,
            tags TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            color TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date)")
    conn.commit()
    conn.close()


class TaskTool(BaseTool):
    name = "tasks"
    description = (
        "Manage personal tasks and todos: create, read, update, delete, list, and organize tasks. "
        "Supports projects, priorities (1=highest, 5=lowest), due dates, tags, and status tracking."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "update", "delete", "list", "complete", "reopen", "search", "stats"],
                "description": "Action to perform on tasks."
            },
            "task_id": {
                "type": "integer",
                "description": "Task ID for read/update/delete/complete/reopen actions."
            },
            "title": {
                "type": "string",
                "description": "Task title (required for create)."
            },
            "description": {
                "type": "string",
                "description": "Task description."
            },
            "project": {
                "type": "string",
                "description": "Project name to organize tasks."
            },
            "priority": {
                "type": "integer",
                "description": "Priority 1-5 (1=highest, 5=lowest). Default 3.",
                "minimum": 1,
                "maximum": 5
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"],
                "description": "Task status."
            },
            "due_date": {
                "type": "string",
                "description": "Due date in ISO format (e.g., '2026-10-15T23:59:59')."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags for categorization."
            },
            "project_filter": {
                "type": "string",
                "description": "Filter tasks by project name."
            },
            "status_filter": {
                "type": "string",
                "enum": ["pending", "in_progress", "blocked", "completed", "cancelled", "all"],
                "description": "Filter tasks by status. Default 'pending'."
            },
            "query": {
                "type": "string",
                "description": "Search query for task titles/descriptions."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tasks to return. Default 50."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "create":
                return self._create_task(**kwargs)
            elif action == "read":
                return self._read_task(**kwargs)
            elif action == "update":
                return self._update_task(**kwargs)
            elif action == "delete":
                return self._delete_task(**kwargs)
            elif action == "list":
                return self._list_tasks(**kwargs)
            elif action == "complete":
                return self._complete_task(**kwargs)
            elif action == "reopen":
                return self._reopen_task(**kwargs)
            elif action == "search":
                return self._search_tasks(**kwargs)
            elif action == "stats":
                return self._get_stats(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _create_task(self, title: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO tasks (title, description, project, priority, status, due_date, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title,
            kwargs.get("description", ""),
            kwargs.get("project", "Inbox"),
            kwargs.get("priority", 3),
            "pending",
            kwargs.get("due_date"),
            json.dumps(kwargs.get("tags", [])),
            now,
            now
        ))
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Task created with ID: {task_id}",
            metadata={"task_id": task_id, "title": title, "project": kwargs.get("project", "Inbox")}
        )

    def _read_task(self, task_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Task {task_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"task": dict(row)}
        )

    def _update_task(self, task_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Task {task_id} not found")

        updates = []
        values = []
        allowed_fields = ["title", "description", "project", "priority", "status", "due_date"]
        for field in allowed_fields:
            if field in kwargs:
                updates.append(f"{field} = ?")
                values.append(kwargs[field])

        if "tags" in kwargs:
            updates.append("tags = ?")
            values.append(json.dumps(kwargs["tags"]))

        if not updates:
            conn.close()
            return ToolResult(success=False, output="", error="No fields to update")

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(task_id)

        cursor.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Task {task_id} updated", metadata={"task_id": task_id})

    def _delete_task(self, task_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Task {task_id} not found")
        return ToolResult(success=True, output=f"Task {task_id} deleted", metadata={"task_id": task_id})

    def _complete_task(self, task_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tasks SET status = 'completed', completed_at = ?, updated_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), datetime.now().isoformat(), task_id))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if updated == 0:
            return ToolResult(success=False, output="", error=f"Task {task_id} not found")
        return ToolResult(success=True, output=f"Task {task_id} marked as completed", metadata={"task_id": task_id})

    def _reopen_task(self, task_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tasks SET status = 'pending', completed_at = NULL, updated_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), task_id))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if updated == 0:
            return ToolResult(success=False, output="", error=f"Task {task_id} not found")
        return ToolResult(success=True, output=f"Task {task_id} reopened", metadata={"task_id": task_id})

    def _list_tasks(self, project_filter: str = None, status_filter: str = "pending", limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if status_filter != "all":
            query += " AND status = ?"
            params.append(status_filter)

        if project_filter:
            query += " AND project = ?"
            params.append(project_filter)

        query += " ORDER BY priority ASC, due_date ASC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No tasks found", metadata={"tasks": []})

        lines = [f"Found {len(rows)} task(s):"]
        for r in rows:
            priority_marks = "!" * (6 - r["priority"])
            due = f" 📅{r['due_date'][:10]}" if r["due_date"] else ""
            proj = f" [{r['project']}]" if r["project"] else ""
            tags = f" #{' #'.join(json.loads(r['tags']))}" if r['tags'] else ""
            status_icon = {"pending": "○", "in_progress": "◐", "blocked": "⊘", "completed": "●", "cancelled": "✗"}.get(r["status"], "?")
            lines.append(f"  {status_icon} #{r['id']}: {r['title']}{proj}{due} {priority_marks}{tags}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"tasks": rows, "count": len(rows)}
        )

    def _search_tasks(self, query: str, project_filter: str = None, status_filter: str = "all") -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM tasks WHERE (title LIKE ? OR description LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if status_filter != "all":
            sql += " AND status = ?"
            params.append(status_filter)
        if project_filter:
            sql += " AND project = ?"
            params.append(project_filter)

        sql += " ORDER BY priority ASC, due_date ASC"
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output=f"No tasks matching '{query}'", metadata={"tasks": []})

        lines = [f"Found {len(rows)} task(s) matching '{query}':"]
        for r in rows:
            priority_marks = "!" * (6 - r["priority"])
            due = f" 📅{r['due_date'][:10]}" if r["due_date"] else ""
            lines.append(f"  #{r['id']}: {r['title']} [{r['project']}]{due} {priority_marks}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"tasks": rows, "count": len(rows)}
        )

    def _get_stats(self) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        status_counts = dict(cursor.fetchall())

        cursor.execute("SELECT project, COUNT(*) FROM tasks WHERE status != 'completed' GROUP BY project ORDER BY COUNT(*) DESC")
        project_counts = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM tasks WHERE due_date IS NOT NULL AND due_date < ? AND status NOT IN ('completed', 'cancelled')", (datetime.now().isoformat(),))
        overdue = cursor.fetchone()[0]

        conn.close()

        lines = ["📊 Task Statistics:"]
        lines.append(f"  Total: {sum(status_counts.values())}")
        for status, count in status_counts.items():
            lines.append(f"  {status}: {count}")
        lines.append(f"  Overdue: {overdue}")
        lines.append("")
        lines.append("  By Project (active):")
        for proj, count in project_counts.items():
            lines.append(f"    {proj}: {count}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"status_counts": status_counts, "project_counts": project_counts, "overdue": overdue}
        )


TOOLS_REGISTRY.register(TaskTool())