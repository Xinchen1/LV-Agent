"""SQLite session memory with FTS5 keyword search fallback.

Stores per-turn conversation history and supports retrieval by recent
history or keyword search. Works with stdlib sqlite3; FTS5 is optional
and degrades to LIKE queries when unavailable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    session_id: str
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SQLiteSessionStore:
    """Persistent session store backed by SQLite."""

    def __init__(self, db_path: str = "./data/sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts5_available = self._check_fts5()
        self._ensure_tables()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _check_fts5(self) -> bool:
        try:
            with self._connection() as conn:
                conn.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='sqlite_fts5_test'").fetchall()
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sqlite_fts5_test USING fts5(text)")
                conn.execute("DROP TABLE IF EXISTS sqlite_fts5_test")
                return True
        except Exception:
            return False

    def _ensure_tables(self) -> None:
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, created_at)"
                )
                if self._fts5_available:
                    try:
                        conn.execute(
                            "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(content, content_rowid=turns)"
                        )
                    except Exception as e:
                        logger.warning(f"FTS5 table creation failed: {e}")
                        self._fts5_available = False
        except Exception as e:
            logger.warning(f"SQLite session store initialization failed: {e}")

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO turns (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        role,
                        content,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        datetime.now().isoformat(),
                    ),
                )
                row_id = cursor.lastrowid
                if self._fts5_available:
                    try:
                        conn.execute(
                            "INSERT INTO turns_fts (rowid, content) VALUES (?, ?)",
                            (row_id, content),
                        )
                    except Exception:
                        pass
            return True
        except Exception as e:
            logger.warning(f"Could not add turn to session store: {e}")
            return False

    def recent(self, session_id: str = "", n: int = 10) -> List[TurnRecord]:
        try:
            with self._connection() as conn:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                        (session_id, n),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM turns ORDER BY created_at DESC LIMIT ?",
                        (n,),
                    ).fetchall()
                return [self._row_to_record(row) for row in reversed(rows)]
        except Exception as e:
            logger.warning(f"Could not retrieve recent turns: {e}")
            return []

    def search(self, query: str, session_id: str = "", k: int = 10) -> List[TurnRecord]:
        try:
            with self._connection() as conn:
                if self._fts5_available:
                    if session_id:
                        rows = conn.execute(
                            """
                            SELECT t.* FROM turns t
                            JOIN turns_fts f ON t.id = f.rowid
                            WHERE turns_fts MATCH ? AND t.session_id = ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (query, session_id, k),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT t.* FROM turns t
                            JOIN turns_fts f ON t.id = f.rowid
                            WHERE turns_fts MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (query, k),
                        ).fetchall()
                else:
                    # 分词: 任一关键词命中即可(避免整句 LIKE 匹配失败)
                    import re as _re
                    terms = [
                        t for t in _re.split(r"[\s,，。.;;：:!?！？、]+", query)
                        if len(t.strip()) >= 2
                    ]
                    if not terms:
                        terms = [query.strip()]
                    conds = " OR ".join(["content LIKE ?"] * len(terms))
                    params = [f"%{t}%" for t in terms]
                    if session_id:
                        rows = conn.execute(
                            f"SELECT * FROM turns WHERE ({conds}) AND session_id = ? ORDER BY created_at DESC LIMIT ?",
                            params + [session_id, k],
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            f"SELECT * FROM turns WHERE {conds} ORDER BY created_at DESC LIMIT ?",
                            params + [k],
                        ).fetchall()
                return [self._row_to_record(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not search session store: {e}")
            return []

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TurnRecord:
        return TurnRecord(
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
        )


def create_sqlite_session_store(db_path: str = "./data/sessions.db") -> SQLiteSessionStore:
    return SQLiteSessionStore(db_path=db_path)
