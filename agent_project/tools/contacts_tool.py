"""
Contacts Management Tool - Personal assistant contact management with groups, tags, and search.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("contacts.db")


def _init_db():
    """Initialize the contacts database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            company TEXT,
            position TEXT,
            address TEXT,
            birthday TEXT,
            notes TEXT,
            groups TEXT,
            tags TEXT,
            is_favorite INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            color TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contacts_name ON contacts(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email)")
    conn.commit()
    conn.close()


class ContactsTool(BaseTool):
    name = "contacts"
    description = (
        "Manage personal contacts: create, read, update, delete, list, and search contacts. "
        "Supports groups, tags, favorites, multiple phone/email, company, position, birthday, and notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "update", "delete", "list", "search", "favorite", "unfavorite", "groups", "birthdays"],
                "description": "Action to perform on contacts."
            },
            "contact_id": {
                "type": "integer",
                "description": "Contact ID for read/update/delete/favorite/unfavorite actions."
            },
            "name": {
                "type": "string",
                "description": "Contact name (required for create)."
            },
            "email": {
                "type": "string",
                "description": "Contact email address."
            },
            "phone": {
                "type": "string",
                "description": "Contact phone number."
            },
            "company": {
                "type": "string",
                "description": "Company/organization name."
            },
            "position": {
                "type": "string",
                "description": "Job title/position."
            },
            "address": {
                "type": "string",
                "description": "Physical address."
            },
            "birthday": {
                "type": "string",
                "description": "Birthday in YYYY-MM-DD format."
            },
            "notes": {
                "type": "string",
                "description": "Additional notes."
            },
            "groups": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of group names."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags."
            },
            "group_filter": {
                "type": "string",
                "description": "Filter contacts by group name."
            },
            "query": {
                "type": "string",
                "description": "Search query for name/email/company/notes."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of contacts to return. Default 50."
            },
            "upcoming_days": {
                "type": "integer",
                "description": "Days ahead to check for birthdays. Default 30."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            _ALIASES = {"add": "create", "new": "create", "remove": "delete",
                        "get": "read", "find": "search"}
            action = _ALIASES.get(action, action)
            if action == "create":
                return self._create_contact(**kwargs)
            elif action == "read":
                return self._read_contact(**kwargs)
            elif action == "update":
                return self._update_contact(**kwargs)
            elif action == "delete":
                return self._delete_contact(**kwargs)
            elif action == "list":
                return self._list_contacts(**kwargs)
            elif action == "search":
                return self._search_contacts(**kwargs)
            elif action == "favorite":
                return self._favorite_contact(**kwargs)
            elif action == "unfavorite":
                return self._unfavorite_contact(**kwargs)
            elif action == "groups":
                return self._list_groups(**kwargs)
            elif action == "birthdays":
                return self._upcoming_birthdays(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _create_contact(self, name: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO contacts (name, email, phone, company, position, address, birthday, notes, groups, tags, is_favorite, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            kwargs.get("email"),
            kwargs.get("phone"),
            kwargs.get("company"),
            kwargs.get("position"),
            kwargs.get("address"),
            kwargs.get("birthday"),
            kwargs.get("notes"),
            json.dumps(kwargs.get("groups", [])),
            json.dumps(kwargs.get("tags", [])),
            0,
            now,
            now
        ))
        contact_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Contact created with ID: {contact_id}",
            metadata={"contact_id": contact_id, "name": name}
        )

    def _read_contact(self, contact_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Contact {contact_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"contact": dict(row)}
        )

    def _update_contact(self, contact_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Contact {contact_id} not found")

        updates = []
        values = []
        allowed_fields = ["name", "email", "phone", "company", "position", "address", "birthday", "notes"]
        for field in allowed_fields:
            if field in kwargs:
                updates.append(f"{field} = ?")
                values.append(kwargs[field])

        for field in ["groups", "tags"]:
            if field in kwargs:
                updates.append(f"{field} = ?")
                values.append(json.dumps(kwargs[field]))

        if not updates:
            conn.close()
            return ToolResult(success=False, output="", error="No fields to update")

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(contact_id)

        cursor.execute(f"UPDATE contacts SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Contact {contact_id} updated", metadata={"contact_id": contact_id})

    def _delete_contact(self, contact_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Contact {contact_id} not found")
        return ToolResult(success=True, output=f"Contact {contact_id} deleted", metadata={"contact_id": contact_id})

    def _list_contacts(self, group_filter: str = None, limit: int = 50, favorites_only: bool = False) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM contacts WHERE 1=1"
        params = []

        if favorites_only:
            query += " AND is_favorite = 1"

        if group_filter:
            query += " AND groups LIKE ?"
            params.append(f"%{group_filter}%")

        query += " ORDER BY is_favorite DESC, name ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No contacts found", metadata={"contacts": []})

        lines = [f"Found {len(rows)} contact(s):"]
        for r in rows:
            fav = "⭐ " if r["is_favorite"] else ""
            company = f" @ {r['company']}" if r["company"] else ""
            phone = f" 📞{r['phone']}" if r["phone"] else ""
            email = f" 📧{r['email']}" if r["email"] else ""
            groups = f" #{' #'.join(json.loads(r['groups']))}" if r['groups'] else ""
            lines.append(f"  {fav}#{r['id']}: {r['name']}{company}{phone}{email}{groups}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"contacts": rows, "count": len(rows)}
        )

    def _search_contacts(self, query: str, group_filter: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = """
            SELECT * FROM contacts
            WHERE name LIKE ? OR email LIKE ? OR company LIKE ? OR notes LIKE ? OR phone LIKE ?
        """
        params = [f"%{query}%"] * 5

        if group_filter:
            sql += " AND groups LIKE ?"
            params.append(f"%{group_filter}%")

        sql += " ORDER BY is_favorite DESC, name ASC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output=f"No contacts matching '{query}'", metadata={"contacts": []})

        lines = [f"Found {len(rows)} contact(s) matching '{query}':"]
        for r in rows:
            fav = "⭐ " if r["is_favorite"] else ""
            company = f" @ {r['company']}" if r["company"] else ""
            lines.append(f"  {fav}#{r['id']}: {r['name']}{company}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"contacts": rows, "count": len(rows)}
        )

    def _favorite_contact(self, contact_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE contacts SET is_favorite = 1, updated_at = ? WHERE id = ?", (datetime.now().isoformat(), contact_id))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if updated == 0:
            return ToolResult(success=False, output="", error=f"Contact {contact_id} not found")
        return ToolResult(success=True, output=f"Contact {contact_id} added to favorites", metadata={"contact_id": contact_id})

    def _unfavorite_contact(self, contact_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE contacts SET is_favorite = 0, updated_at = ? WHERE id = ?", (datetime.now().isoformat(), contact_id))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if updated == 0:
            return ToolResult(success=False, output="", error=f"Contact {contact_id} not found")
        return ToolResult(success=True, output=f"Contact {contact_id} removed from favorites", metadata={"contact_id": contact_id})

    def _list_groups(self) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM groups ORDER BY name")
        groups = [dict(row) for row in cursor.fetchall()]

        # Get contact counts per group
        cursor.execute("SELECT groups FROM contacts")
        all_contacts = cursor.fetchall()
        conn.close()

        group_counts = {}
        for row in all_contacts:
            try:
                gs = json.loads(row["groups"] or "[]")
                for g in gs:
                    group_counts[g] = group_counts.get(g, 0) + 1
            except (json.JSONDecodeError, TypeError) as e:
                # Silent: malformed groups JSON in DB, skip this row
                import logging
                logging.getLogger(__name__).debug("Failed to parse groups JSON: %s", e)

        if not groups and not group_counts:
            return ToolResult(success=True, output="No groups found", metadata={"groups": []})

        lines = ["👥 Contact Groups:"]
        for g in groups:
            cnt = group_counts.get(g["name"], 0)
            lines.append(f"  {g['name']}: {cnt} contacts")
        for grp, cnt in group_counts.items():
            if not any(g["name"] == grp for g in groups):
                lines.append(f"  {grp}: {cnt} contacts (auto-created)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"groups": groups, "counts": group_counts}
        )

    def _upcoming_birthdays(self, upcoming_days: int = 30) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        today = datetime.now()
        this_year = today.year
        future = today + timedelta(days=upcoming_days)
        future_year = future.year

        cursor.execute("SELECT * FROM contacts WHERE birthday IS NOT NULL AND birthday != ''")
        all_contacts = [dict(row) for row in cursor.fetchall()]
        conn.close()

        upcoming = []
        for c in all_contacts:
            try:
                bday = datetime.strptime(c["birthday"], "%Y-%m-%d")
                # Check this year's birthday
                bday_this_year = bday.replace(year=this_year)
                if today <= bday_this_year <= future:
                    upcoming.append((c, bday_this_year, (bday_this_year - today).days))
                # Check next year's birthday if within range
                elif this_year != future_year:
                    bday_next_year = bday.replace(year=this_year + 1)
                    if today <= bday_next_year <= future:
                        upcoming.append((c, bday_next_year, (bday_next_year - today).days))
            except (ValueError, TypeError) as e:
                # Silent: invalid birthday format in DB, skip this contact
                import logging
                logging.getLogger(__name__).debug("Failed to parse birthday: %s", e)

        upcoming.sort(key=lambda x: x[2])

        if not upcoming:
            return ToolResult(success=True, output=f"No upcoming birthdays in next {upcoming_days} days", metadata={"birthdays": []})

        lines = [f"🎂 Upcoming Birthdays (next {upcoming_days} days):"]
        for contact, bday, days in upcoming:
            age = bday.year - datetime.strptime(contact["birthday"], "%Y-%m-%d").year
            lines.append(f"  🎂 {contact['name']} - {bday.strftime('%m-%d')} (in {days} days, turning {age})")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"birthdays": [{"contact": c, "date": d.isoformat(), "days": days} for c, d, days in upcoming]}
        )


TOOLS_REGISTRY.register(ContactsTool())