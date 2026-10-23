"""Bookmarks / read-later tool - save links with tags, folders and archive state."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path

logger = logging.getLogger(__name__)

_SCHEMA_READY: set = set()


def _connect() -> sqlite3.Connection:
    path = db_path("bookmarks.db")
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
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            title TEXT,
            description TEXT,
            tags TEXT,
            folder TEXT DEFAULT 'Inbox',
            is_archived INTEGER DEFAULT 0,
            read_at TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_bookmarks_url ON bookmarks(url);
        CREATE INDEX IF NOT EXISTS idx_bookmarks_folder ON bookmarks(folder);
        """
    )
    conn.commit()


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url and not url.startswith("mailto:"):
        url = "https://" + url
    return url


def _tags(row: Dict[str, Any]) -> List[str]:
    raw = row.get("tags")
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else [str(value)]
    except (ValueError, TypeError):
        return []


class BookmarksTool(BaseTool):
    name = "bookmarks"
    description = (
        "Read-later bookmark manager: add URLs with title/tags/folder, search and list "
        "saved links, mark read, archive, and show tag statistics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "get", "read", "archive", "unarchive", "delete", "tags", "stats"],
                "description": "Action to perform.",
            },
            "bookmark_id": {"type": "integer", "description": "Bookmark ID."},
            "url": {"type": "string", "description": "URL to save (scheme optional)."},
            "title": {"type": "string", "description": "Title."},
            "description": {"type": "string", "description": "Short description/notes."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags."},
            "folder": {"type": "string", "description": "Folder name (default Inbox)."},
            "query": {"type": "string", "description": "Search keyword."},
            "include_archived": {"type": "boolean", "description": "Include archived bookmarks."},
            "limit": {"type": "integer", "description": "Max rows (default 20)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "add": self._add,
                "list": self._list,
                "search": self._search,
                "get": self._get,
                "read": self._mark_read,
                "archive": lambda **kw: self._set_archived(True, **kw),
                "unarchive": lambda **kw: self._set_archived(False, **kw),
                "delete": self._delete,
                "tags": self._tags,
                "stats": self._stats,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("bookmarks action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _row(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["tags"] = _tags(data)
        return data

    def _find(self, conn: sqlite3.Connection, bookmark_id: Optional[int],
              url: Optional[str] = None) -> Optional[sqlite3.Row]:
        if bookmark_id is not None:
            return conn.execute("SELECT * FROM bookmarks WHERE id = ?", (int(bookmark_id),)).fetchone()
        if url:
            return conn.execute("SELECT * FROM bookmarks WHERE url = ?",
                                (_normalize_url(url),)).fetchone()
        return None

    def _add(self, url: str = "", **kwargs) -> ToolResult:
        url = _normalize_url(url)
        if not url:
            return ToolResult(success=False, output="", error="url is required")
        if not urlparse(url).scheme:
            return ToolResult(success=False, output="", error="invalid url")
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            existing = self._find(conn, None, url)
            if existing:
                return ToolResult(
                    success=False, output="",
                    error=f"Bookmark already exists: #{existing['id']} {existing['url']}",
                )
            cur = conn.execute(
                """INSERT INTO bookmarks (url, title, description, tags, folder, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    url,
                    kwargs.get("title", ""),
                    kwargs.get("description", ""),
                    json.dumps(kwargs.get("tags") or [], ensure_ascii=False),
                    kwargs.get("folder") or "Inbox",
                    now,
                    now,
                ),
            )
            conn.commit()
            bookmark_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Bookmark #{bookmark_id} saved: {url}",
                          metadata={"bookmark_id": bookmark_id, "url": url})

    def _list(self, include_archived: bool = False, folder: Optional[str] = None,
              limit: int = 20, **kwargs) -> ToolResult:
        sql = "SELECT * FROM bookmarks"
        clauses, params = [], []
        if not include_archived:
            clauses.append("is_archived = 0")
        if folder:
            clauses.append("folder = ?")
            params.append(folder)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY is_archived, created_at DESC LIMIT ?"
        params.append(max(1, int(limit or 20)))
        conn = _connect()
        try:
            rows = [self._row(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No bookmarks found", metadata={"bookmarks": []})
        lines = [f"Found {len(rows)} bookmark(s):"]
        for r in rows:
            tags = f" #{' #'.join(r['tags'])}" if r["tags"] else ""
            read = " (read)" if r["read_at"] else ""
            title = r["title"] or r["url"]
            lines.append(f"  #{r['id']} [{r['folder']}] {title}{tags}{read}\n      {r['url']}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"bookmarks": rows})

    def _search(self, query: str = "", limit: int = 20, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, output="", error="query is required")
        like = f"%{query}%"
        conn = _connect()
        try:
            rows = [
                self._row(r)
                for r in conn.execute(
                    """SELECT * FROM bookmarks
                       WHERE url LIKE ? OR title LIKE ? OR description LIKE ? OR tags LIKE ? OR folder LIKE ?
                       ORDER BY is_archived, created_at DESC LIMIT ?""",
                    (like, like, like, like, like, max(1, int(limit or 20))),
                ).fetchall()
            ]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output=f"No bookmarks matching '{query}'",
                              metadata={"bookmarks": []})
        lines = [f"Found {len(rows)} matching bookmark(s):"]
        for r in rows:
            lines.append(f"  #{r['id']} {r['title'] or r['url']} - {r['url']}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"bookmarks": rows})

    def _get(self, bookmark_id: int = None, **kwargs) -> ToolResult:
        if bookmark_id is None:
            return ToolResult(success=False, output="", error="bookmark_id is required")
        conn = _connect()
        try:
            row = self._find(conn, bookmark_id)
        finally:
            conn.close()
        if row is None:
            return ToolResult(success=False, output="", error=f"Bookmark {bookmark_id} not found")
        data = self._row(row)
        return ToolResult(success=True, output=json.dumps(data, ensure_ascii=False, indent=2),
                          metadata={"bookmark": data})

    def _mark_read(self, bookmark_id: int = None, **kwargs) -> ToolResult:
        if bookmark_id is None:
            return ToolResult(success=False, output="", error="bookmark_id is required")
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE bookmarks SET read_at = ?, updated_at = ? WHERE id = ?",
                (now, now, int(bookmark_id)),
            )
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Bookmark {bookmark_id} not found")
        return ToolResult(success=True, output=f"Bookmark #{bookmark_id} marked read",
                          metadata={"bookmark_id": int(bookmark_id), "read_at": now})

    def _set_archived(self, archived: bool, **kwargs) -> ToolResult:
        bookmark_id = kwargs.get("bookmark_id")
        if bookmark_id is None:
            return ToolResult(success=False, output="", error="bookmark_id is required")
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE bookmarks SET is_archived = ?, updated_at = ? WHERE id = ?",
                (1 if archived else 0, datetime.now().isoformat(timespec="seconds"), int(bookmark_id)),
            )
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Bookmark {bookmark_id} not found")
        state = "archived" if archived else "restored"
        return ToolResult(success=True, output=f"Bookmark #{bookmark_id} {state}",
                          metadata={"bookmark_id": int(bookmark_id), "archived": archived})

    def _delete(self, bookmark_id: int = None, **kwargs) -> ToolResult:
        if bookmark_id is None:
            return ToolResult(success=False, output="", error="bookmark_id is required")
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM bookmarks WHERE id = ?", (int(bookmark_id),))
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Bookmark {bookmark_id} not found")
        return ToolResult(success=True, output=f"Bookmark #{bookmark_id} deleted",
                          metadata={"bookmark_id": int(bookmark_id)})

    def _tags(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            rows = conn.execute("SELECT tags FROM bookmarks WHERE is_archived = 0").fetchall()
        finally:
            conn.close()
        counts: Dict[str, int] = {}
        for r in rows:
            data = {"tags": r["tags"]}
            for tag in _tags(data):
                counts[tag] = counts.get(tag, 0) + 1
        if not counts:
            return ToolResult(success=True, output="No tags yet", metadata={"tags": {}})
        ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        lines = ["Tags:"] + [f"  #{tag} ({n})" for tag, n in ordered.items()]
        return ToolResult(success=True, output="\n".join(lines), metadata={"tags": ordered})

    def _stats(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            total = conn.execute("SELECT COUNT(*) AS c FROM bookmarks").fetchone()["c"]
            unread = conn.execute(
                "SELECT COUNT(*) AS c FROM bookmarks WHERE is_archived = 0 AND read_at IS NULL"
            ).fetchone()["c"]
            archived = conn.execute(
                "SELECT COUNT(*) AS c FROM bookmarks WHERE is_archived = 1"
            ).fetchone()["c"]
            folders = conn.execute(
                "SELECT folder, COUNT(*) AS c FROM bookmarks WHERE is_archived = 0 "
                "GROUP BY folder ORDER BY c DESC"
            ).fetchall()
        finally:
            conn.close()
        lines = [
            "Bookmark stats:",
            f"  total: {total}",
            f"  unread: {unread}",
            f"  archived: {archived}",
        ]
        if folders:
            lines.append("  folders: " + ", ".join(f"{r['folder']}->{r['c']}" for r in folders))
        return ToolResult(success=True, output="\n".join(lines), metadata={
            "total": total, "unread": unread, "archived": archived,
            "folders": {r["folder"]: r["c"] for r in folders},
        })


TOOLS_REGISTRY.register(BookmarksTool())
