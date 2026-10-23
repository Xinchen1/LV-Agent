"""Journal / diary tool - daily entries, mood tracking, gratitude and search."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path

logger = logging.getLogger(__name__)

_SCHEMA_READY: set = set()

_MOOD_SCORES = {
    "great": 5, "excellent": 5, "happy": 5, "很好": 5, "开心": 5,
    "good": 4, "fine": 4, "不错": 4, "挺好": 4,
    "neutral": 3, "ok": 3, "一般": 3, "还行": 3,
    "bad": 2, "low": 2, "down": 2, "难受": 2, "不好": 2,
    "awful": 1, "terrible": 1, "糟糕": 1, "很差": 1,
}


def _connect() -> sqlite3.Connection:
    path = db_path("journal.db")
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
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            mood TEXT,
            mood_score INTEGER,
            gratitude TEXT,
            tags TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_journal_date ON journal(entry_date);
        """
    )
    conn.commit()


def _parse_day(value: Optional[str]) -> str:
    if not value:
        return date.today().isoformat()
    v = value.strip().lower()
    if v == "today":
        return date.today().isoformat()
    if v == "tomorrow":
        return (date.today() + timedelta(days=1)).isoformat()
    if v == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(value.strip()).date().isoformat()
    except ValueError:
        return date.today().isoformat()


def _mood_score(mood: Optional[str], score: Optional[int]) -> Optional[int]:
    if score is not None:
        try:
            return max(1, min(5, int(score)))
        except (TypeError, ValueError):
            return None
    if not mood:
        return None
    return _MOOD_SCORES.get(str(mood).strip().lower())


def _row_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("tags",):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (ValueError, TypeError):
                d[key] = []
        else:
            d[key] = []
    return d


class JournalTool(BaseTool):
    name = "journal"
    description = (
        "Personal journal and diary: write entries with mood and gratitude, list/search past "
        "entries, and review mood statistics over time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "get", "search", "update", "delete", "mood_stats"],
                "description": "Action to perform.",
            },
            "entry_id": {"type": "integer", "description": "Journal entry ID."},
            "entry_date": {"type": "string", "description": "Entry date (ISO, today, yesterday)."},
            "title": {"type": "string", "description": "Optional short title."},
            "content": {"type": "string", "description": "Entry body (required for create)."},
            "mood": {"type": "string", "description": "Mood word, e.g. great/good/neutral/bad/awful."},
            "mood_score": {"type": "integer", "description": "Mood score from 1 (worst) to 5 (best)."},
            "gratitude": {"type": "string", "description": "What you are grateful for today."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags."},
            "query": {"type": "string", "description": "Search keyword."},
            "date_from": {"type": "string", "description": "Start date filter (inclusive)."},
            "date_to": {"type": "string", "description": "End date filter (inclusive)."},
            "days": {"type": "integer", "description": "Window size for mood_stats (default 30)."},
            "limit": {"type": "integer", "description": "Max rows to return (default 20)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "create": self._create,
                "list": self._list,
                "get": self._get,
                "search": self._search,
                "update": self._update,
                "delete": self._delete,
                "mood_stats": self._mood_stats,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("journal action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _create(self, content: str = "", **kwargs) -> ToolResult:
        if not content or not content.strip():
            return ToolResult(success=False, output="", error="content is required")
        now = datetime.now().isoformat(timespec="seconds")
        entry_date = _parse_day(kwargs.get("entry_date"))
        tags = kwargs.get("tags") or []
        mood = kwargs.get("mood")
        score = _mood_score(mood, kwargs.get("mood_score"))
        conn = _connect()
        try:
            cur = conn.execute(
                """INSERT INTO journal
                   (entry_date, title, content, mood, mood_score, gratitude, tags, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_date,
                    kwargs.get("title", ""),
                    content.strip(),
                    mood,
                    score,
                    kwargs.get("gratitude", ""),
                    json.dumps(tags, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()
            entry_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(
            success=True,
            output=f"Journal entry #{entry_id} saved for {entry_date}",
            metadata={"entry_id": entry_id, "entry_date": entry_date},
        )

    def _list(self, limit: int = 20, date_from: Optional[str] = None,
              date_to: Optional[str] = None, **kwargs) -> ToolResult:
        sql = "SELECT * FROM journal"
        params: List[Any] = []
        clauses = []
        if date_from:
            clauses.append("entry_date >= ?")
            params.append(_parse_day(date_from))
        if date_to:
            clauses.append("entry_date <= ?")
            params.append(_parse_day(date_to))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY entry_date DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit or 20)))
        conn = _connect()
        try:
            rows = [_row_dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No journal entries found", metadata={"entries": []})
        lines = [f"Found {len(rows)} journal entr{'y' if len(rows) == 1 else 'ies'}:"]
        for r in rows:
            mood = f" mood={r['mood']}({r['mood_score']})" if r.get("mood_score") else ""
            preview = (r["content"] or "")[:100].replace("\n", " ")
            lines.append(f"  #{r['id']} {r['entry_date']}{mood} - {r.get('title') or preview}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"entries": rows})

    def _get(self, entry_id: Optional[int] = None, **kwargs) -> ToolResult:
        if entry_id is None:
            return ToolResult(success=False, output="", error="entry_id is required")
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM journal WHERE id = ?", (int(entry_id),)).fetchone()
        finally:
            conn.close()
        if row is None:
            return ToolResult(success=False, output="", error=f"Journal entry {entry_id} not found")
        data = _row_dict(row)
        return ToolResult(
            success=True,
            output=json.dumps(data, ensure_ascii=False, indent=2),
            metadata={"entry": data},
        )

    def _search(self, query: str = "", limit: int = 20, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, output="", error="query is required")
        like = f"%{query}%"
        conn = _connect()
        try:
            rows = [
                _row_dict(r)
                for r in conn.execute(
                    """SELECT * FROM journal
                       WHERE content LIKE ? OR title LIKE ? OR gratitude LIKE ? OR tags LIKE ?
                       ORDER BY entry_date DESC LIMIT ?""",
                    (like, like, like, like, max(1, int(limit or 20))),
                ).fetchall()
            ]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output=f"No entries matching '{query}'", metadata={"entries": []})
        lines = [f"Found {len(rows)} matching entr{'y' if len(rows) == 1 else 'ies'}:"]
        for r in rows:
            preview = (r["content"] or "")[:120].replace("\n", " ")
            lines.append(f"  #{r['id']} {r['entry_date']} - {preview}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"entries": rows})

    def _update(self, entry_id: Optional[int] = None, **kwargs) -> ToolResult:
        if entry_id is None:
            return ToolResult(success=False, output="", error="entry_id is required")
        fields = []
        values: List[Any] = []
        for key in ("title", "content", "mood", "gratitude", "entry_date"):
            if key in kwargs and kwargs[key] is not None:
                if key == "entry_date":
                    fields.append("entry_date = ?")
                    values.append(_parse_day(kwargs[key]))
                else:
                    fields.append(f"{key} = ?")
                    values.append(kwargs[key])
        if "mood_score" in kwargs or "mood" in kwargs:
            fields.append("mood_score = ?")
            values.append(_mood_score(kwargs.get("mood"), kwargs.get("mood_score")))
        if "tags" in kwargs and kwargs["tags"] is not None:
            fields.append("tags = ?")
            values.append(json.dumps(kwargs["tags"], ensure_ascii=False))
        if not fields:
            return ToolResult(success=False, output="", error="No fields to update")
        fields.append("updated_at = ?")
        values.append(datetime.now().isoformat(timespec="seconds"))
        values.append(int(entry_id))
        conn = _connect()
        try:
            cur = conn.execute(f"UPDATE journal SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Journal entry {entry_id} not found")
        return ToolResult(success=True, output=f"Journal entry #{entry_id} updated",
                          metadata={"entry_id": int(entry_id)})

    def _delete(self, entry_id: Optional[int] = None, **kwargs) -> ToolResult:
        if entry_id is None:
            return ToolResult(success=False, output="", error="entry_id is required")
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM journal WHERE id = ?", (int(entry_id),))
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Journal entry {entry_id} not found")
        return ToolResult(success=True, output=f"Journal entry #{entry_id} deleted",
                          metadata={"entry_id": int(entry_id)})

    def _mood_stats(self, days: int = 30, **kwargs) -> ToolResult:
        days = max(1, int(days or 30))
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT entry_date, mood, mood_score FROM journal WHERE entry_date >= ? ORDER BY entry_date",
                (since,),
            ).fetchall()
        finally:
            conn.close()
        scores = [(r["entry_date"], r["mood_score"]) for r in rows if r["mood_score"] is not None]
        if not scores:
            return ToolResult(success=True, output=f"No mood data in the last {days} days",
                              metadata={"days": days, "entries": 0})
        values = [s for _, s in scores]
        avg = sum(values) / len(values)
        distribution: Dict[str, int] = {}
        for _, s in scores:
            distribution[str(s)] = distribution.get(str(s), 0) + 1
        best = max(scores, key=lambda x: x[1])
        worst = min(scores, key=lambda x: x[1])
        lines = [
            f"Mood stats (last {days} days):",
            f"  entries with mood: {len(values)}",
            f"  average score: {avg:.2f} / 5",
            f"  best day: {best[0]} ({best[1]})",
            f"  worst day: {worst[0]} ({worst[1]})",
            "  distribution: " + ", ".join(f"{k}->{v}" for k, v in sorted(distribution.items())),
        ]
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"days": days, "entries": len(values), "average": round(avg, 2),
                      "distribution": distribution, "best": best, "worst": worst},
        )


TOOLS_REGISTRY.register(JournalTool())
