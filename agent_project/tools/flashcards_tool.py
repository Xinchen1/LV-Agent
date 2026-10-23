"""Flashcards tool - spaced repetition review with a lightweight SM-2 scheduler."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path

logger = logging.getLogger(__name__)

_SCHEMA_READY: set = set()

_RATINGS = {
    "again": 0, "forgot": 0, "1": 0,
    "hard": 1, "2": 1,
    "good": 2, "ok": 2, "3": 2,
    "easy": 3, "4": 3,
}


def _connect() -> sqlite3.Connection:
    path = db_path("flashcards.db")
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
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deck TEXT DEFAULT 'Default',
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            ease REAL DEFAULT 2.5,
            interval_days INTEGER DEFAULT 0,
            repetitions INTEGER DEFAULT 0,
            due_date TEXT NOT NULL,
            created_at TEXT,
            last_reviewed TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cards_due ON cards(due_date);
        CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck);
        """
    )
    conn.commit()


class FlashcardsTool(BaseTool):
    name = "flashcards"
    description = (
        "Spaced-repetition flashcards: add cards to decks, list due cards, review with "
        "again/hard/good/easy ratings, and show deck statistics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "due", "review", "stats", "delete", "decks"],
                "description": "Action to perform.",
            },
            "card_id": {"type": "integer", "description": "Card ID."},
            "deck": {"type": "string", "description": "Deck name (default Default)."},
            "front": {"type": "string", "description": "Question / front side."},
            "back": {"type": "string", "description": "Answer / back side."},
            "rating": {"type": "string", "enum": list(_RATINGS), "description": "Review rating."},
            "limit": {"type": "integer", "description": "Max cards (default 20)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "add": self._add,
                "list": self._list,
                "due": self._due,
                "review": self._review,
                "stats": self._stats,
                "delete": self._delete,
                "decks": self._decks,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("flashcards action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _add(self, front: str = "", back: str = "", **kwargs) -> ToolResult:
        if not front or not front.strip():
            return ToolResult(success=False, output="", error="front is required")
        if not back or not back.strip():
            return ToolResult(success=False, output="", error="back is required")
        today = date.today().isoformat()
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            cur = conn.execute(
                """INSERT INTO cards (deck, front, back, due_date, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (kwargs.get("deck") or "Default", front.strip(), back.strip(), today, now),
            )
            conn.commit()
            card_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Card #{card_id} added to deck "
                                            f"'{(kwargs.get('deck') or 'Default')}'",
                          metadata={"card_id": card_id})

    def _list(self, deck: Optional[str] = None, limit: int = 20, **kwargs) -> ToolResult:
        sql = "SELECT * FROM cards"
        params: List[Any] = []
        if deck:
            sql += " WHERE deck = ?"
            params.append(deck)
        sql += " ORDER BY due_date, id LIMIT ?"
        params.append(max(1, int(limit or 20)))
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No cards found", metadata={"cards": []})
        lines = [f"{len(rows)} card(s):"]
        for r in rows:
            lines.append(f"  #{r['id']} [{r['deck']}] {r['front'][:60]} -> {r['back'][:60]} "
                         f"(due {r['due_date']}, ease {r['ease']:.2f})")
        return ToolResult(success=True, output="\n".join(lines), metadata={"cards": rows})

    def _due(self, deck: Optional[str] = None, limit: int = 20, **kwargs) -> ToolResult:
        today = date.today().isoformat()
        sql = "SELECT * FROM cards WHERE due_date <= ?"
        params: List[Any] = [today]
        if deck:
            sql += " AND deck = ?"
            params.append(deck)
        sql += " ORDER BY due_date, id LIMIT ?"
        params.append(max(1, int(limit or 20)))
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No cards due today", metadata={"due": []})
        lines = [f"{len(rows)} card(s) due:"]
        for r in rows:
            lines.append(f"  #{r['id']} [{r['deck']}] {r['front']}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"due": rows})

    def _review(self, card_id: Optional[int] = None, rating: str = "good",
                **kwargs) -> ToolResult:
        if card_id is None:
            return ToolResult(success=False, output="", error="card_id is required")
        rating_key = str(rating).strip().lower()
        if rating_key not in _RATINGS:
            return ToolResult(success=False, output="", error=f"rating must be one of {sorted(_RATINGS)}")
        level = _RATINGS[rating_key]
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM cards WHERE id = ?", (int(card_id),)).fetchone()
            if row is None:
                return ToolResult(success=False, output="", error=f"Card {card_id} not found")
            ease = float(row["ease"] or 2.5)
            interval = int(row["interval_days"] or 0)
            reps = int(row["repetitions"] or 0)
            if level == 0:
                reps, interval = 0, 0
                ease = max(1.3, ease - 0.2)
            elif level == 1:
                interval = max(1, round(interval * 1.2)) if interval else 1
                ease = max(1.3, ease - 0.15)
                reps += 1
            elif level == 2:
                if reps == 0:
                    interval = 1
                elif reps == 1:
                    interval = 6
                else:
                    interval = max(1, round(interval * ease))
                reps += 1
            else:
                if reps == 0:
                    interval = 4
                else:
                    interval = max(1, round((interval or 1) * ease * 1.3))
                ease = min(3.0, ease + 0.15)
                reps += 1
            due = (date.today() + timedelta(days=interval)).isoformat()
            conn.execute(
                """UPDATE cards SET ease = ?, interval_days = ?, repetitions = ?, due_date = ?,
                   last_reviewed = ? WHERE id = ?""",
                (round(ease, 2), interval, reps, due,
                 datetime.now().isoformat(timespec="seconds"), int(card_id)),
            )
            conn.commit()
        finally:
            conn.close()
        return ToolResult(
            success=True,
            output=f"Card #{card_id} reviewed ({rating_key}); next due {due} (interval {interval}d)",
            metadata={"card_id": int(card_id), "rating": rating_key, "ease": round(ease, 2),
                      "interval_days": interval, "repetitions": reps, "due_date": due},
        )

    def _stats(self, **kwargs) -> ToolResult:
        today = date.today().isoformat()
        conn = _connect()
        try:
            total = conn.execute("SELECT COUNT(*) AS c FROM cards").fetchone()["c"]
            due = conn.execute("SELECT COUNT(*) AS c FROM cards WHERE due_date <= ?",
                               (today,)).fetchone()["c"]
            decks = conn.execute(
                "SELECT deck, COUNT(*) AS c, SUM(CASE WHEN due_date <= ? THEN 1 ELSE 0 END) AS d "
                "FROM cards GROUP BY deck ORDER BY c DESC",
                (today,),
            ).fetchall()
        finally:
            conn.close()
        lines = ["Flashcard stats:", f"  total: {total}", f"  due today: {due}"]
        for r in decks:
            lines.append(f"  [{r['deck']}] {r['c']} cards, {r['d']} due")
        return ToolResult(success=True, output="\n".join(lines), metadata={
            "total": total, "due": due,
            "decks": {r["deck"]: {"cards": r["c"], "due": r["d"]} for r in decks},
        })

    def _decks(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            rows = conn.execute("SELECT deck, COUNT(*) AS c FROM cards GROUP BY deck ORDER BY deck").fetchall()
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No decks yet", metadata={"decks": {}})
        data = {r["deck"]: r["c"] for r in rows}
        return ToolResult(success=True, output="\n".join(f"  {k}: {v}" for k, v in data.items()),
                          metadata={"decks": data})

    def _delete(self, card_id: Optional[int] = None, **kwargs) -> ToolResult:
        if card_id is None:
            return ToolResult(success=False, output="", error="card_id is required")
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM cards WHERE id = ?", (int(card_id),))
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Card {card_id} not found")
        return ToolResult(success=True, output=f"Card #{card_id} deleted",
                          metadata={"card_id": int(card_id)})


TOOLS_REGISTRY.register(FlashcardsTool())
