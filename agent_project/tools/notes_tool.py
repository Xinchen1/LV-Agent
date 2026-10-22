"""
Notes/Knowledge Management Tool - Personal assistant note-taking with tags, categories, and search.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("notes.db")


def _init_db():
    """Initialize the notes database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT,
            category TEXT,
            tags TEXT,
            is_pinned INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            color TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_pinned ON notes(is_pinned)")
    conn.commit()
    conn.close()


class NotesTool(BaseTool):
    name = "notes"
    description = (
        "Manage personal notes and knowledge base: create, read, update, delete, list, and search notes. "
        "Supports categories, tags, pinning, and full-text search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "update", "delete", "list", "search", "pin", "unpin", "categories"],
                "description": "Action to perform on notes."
            },
            "note_id": {
                "type": "integer",
                "description": "Note ID for read/update/delete/pin/unpin actions."
            },
            "title": {
                "type": "string",
                "description": "Note title (required for create)."
            },
            "content": {
                "type": "string",
                "description": "Note content (markdown supported)."
            },
            "category": {
                "type": "string",
                "description": "Category name for organization."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags for categorization."
            },
            "category_filter": {
                "type": "string",
                "description": "Filter notes by category."
            },
            "query": {
                "type": "string",
                "description": "Search query for note titles/content."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of notes to return. Default 50."
            },
            "pinned_only": {
                "type": "boolean",
                "description": "Only return pinned notes. Default false."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "create":
                return self._create_note(**kwargs)
            elif action == "read":
                return self._read_note(**kwargs)
            elif action == "update":
                return self._update_note(**kwargs)
            elif action == "delete":
                return self._delete_note(**kwargs)
            elif action == "list":
                return self._list_notes(**kwargs)
            elif action == "search":
                return self._search_notes(**kwargs)
            elif action == "pin":
                return self._pin_note(**kwargs)
            elif action == "unpin":
                return self._unpin_note(**kwargs)
            elif action == "categories":
                return self._list_categories(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _create_note(self, title: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO notes (title, content, category, tags, is_pinned, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            title,
            kwargs.get("content", ""),
            kwargs.get("category", "General"),
            json.dumps(kwargs.get("tags", [])),
            0,
            now,
            now
        ))
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Note created with ID: {note_id}",
            metadata={"note_id": note_id, "title": title, "category": kwargs.get("category", "General")}
        )

    def _read_note(self, note_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Note {note_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"note": dict(row)}
        )

    def _update_note(self, note_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Note {note_id} not found")

        updates = []
        values = []
        for field in ["title", "content", "category"]:
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
        values.append(note_id)

        cursor.execute(f"UPDATE notes SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Note {note_id} updated", metadata={"note_id": note_id})

    def _delete_note(self, note_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Note {note_id} not found")
        return ToolResult(success=True, output=f"Note {note_id} deleted", metadata={"note_id": note_id})

    def _list_notes(self, category_filter: str = None, limit: int = 50, pinned_only: bool = False) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM notes WHERE 1=1"
        params = []

        if pinned_only:
            query += " AND is_pinned = 1"

        if category_filter:
            query += " AND category = ?"
            params.append(category_filter)

        query += " ORDER BY is_pinned DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No notes found", metadata={"notes": []})

        lines = [f"Found {len(rows)} note(s):"]
        for r in rows:
            pin = "📌 " if r["is_pinned"] else ""
            cat = f" [{r['category']}]" if r["category"] else ""
            tags = f" #{' #'.join(json.loads(r['tags']))}" if r['tags'] else ""
            preview = r["content"][:80].replace("\n", " ") + ("..." if len(r["content"]) > 80 else "")
            lines.append(f"  {pin}#{r['id']}: {r['title']}{cat}{tags} - {preview}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"notes": rows, "count": len(rows)}
        )

    def _search_notes(self, query: str, category_filter: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM notes WHERE (title LIKE ? OR content LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if category_filter:
            sql += " AND category = ?"
            params.append(category_filter)

        sql += " ORDER BY is_pinned DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output=f"No notes matching '{query}'", metadata={"notes": []})

        lines = [f"Found {len(rows)} note(s) matching '{query}':"]
        for r in rows:
            pin = "📌 " if r["is_pinned"] else ""
            cat = f" [{r['category']}]" if r["category"] else ""
            preview = r["content"][:80].replace("\n", " ") + ("..." if len(r["content"]) > 80 else "")
            lines.append(f"  {pin}#{r['id']}: {r['title']}{cat} - {preview}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"notes": rows, "count": len(rows)}
        )

    def _pin_note(self, note_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE notes SET is_pinned = 1, updated_at = ? WHERE id = ?", (datetime.now().isoformat(), note_id))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if updated == 0:
            return ToolResult(success=False, output="", error=f"Note {note_id} not found")
        return ToolResult(success=True, output=f"Note {note_id} pinned", metadata={"note_id": note_id})

    def _unpin_note(self, note_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE notes SET is_pinned = 0, updated_at = ? WHERE id = ?", (datetime.now().isoformat(), note_id))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if updated == 0:
            return ToolResult(success=False, output="", error=f"Note {note_id} not found")
        return ToolResult(success=True, output=f"Note {note_id} unpinned", metadata={"note_id": note_id})

    def _list_categories(self) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM categories ORDER BY name")
        cats = [dict(row) for row in cursor.fetchall()]

        # Also get note counts per category
        cursor.execute("SELECT category, COUNT(*) as cnt FROM notes GROUP BY category")
        counts = {row["category"]: row["cnt"] for row in cursor.fetchall() if row["category"]}
        conn.close()

        if not cats and not counts:
            return ToolResult(success=True, output="No categories found", metadata={"categories": []})

        lines = ["📁 Categories:"]
        for c in cats:
            cnt = counts.get(c["name"], 0)
            lines.append(f"  {c['name']}: {cnt} notes")
        for cat, cnt in counts.items():
            if not any(c["name"] == cat for c in cats):
                lines.append(f"  {cat}: {cnt} notes (auto-created)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"categories": cats, "counts": counts}
        )


TOOLS_REGISTRY.register(NotesTool())