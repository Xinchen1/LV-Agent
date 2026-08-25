"""
Database Tool - SQLite database operations
Read-only mode by default, can enable writes
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

from . import BaseTool, ToolResult, TOOLS_REGISTRY


class DatabaseTool(BaseTool):
    """
    Database operations tool (primarily SQLite)
    Supports queries, schema inspection, and safe operations
    """

    name = "database"
    description = "Execute SQL queries on SQLite databases. Read-only by default, can enable writes with 'allow_write' flag."

    parameters = {
        "type": "object",
        "properties": {
            "db_path": {
                "type": "string",
                "description": "Path to SQLite database file"
            },
            "query": {
                "type": "string",
                "description": "SQL query to execute"
            },
            "params": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Parameters for query (for parameterized queries)"
            },
            "allow_write": {
                "type": "boolean",
                "description": "Allow INSERT/UPDATE/DELETE operations (default: false)",
                "default": False
            },
            "fetch_all": {
                "type": "boolean",
                "description": "Fetch all results vs single row",
                "default": True
            }
        },
        "required": ["db_path", "query"]
    }

    # Blocked operations for safety
    BLOCKED_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'PRAGMA', 'ATTACH', 'DETACH']

    def __init__(self, allowed_dbs: List[str] = None):
        self.allowed_dbs = [Path(d).resolve() for d in allowed_dbs] if allowed_dbs else []
        self.logger = None

    def execute(
        self,
        db_path: str,
        query: str,
        params: Optional[List[str]] = None,
        allow_write: bool = False,
        fetch_all: bool = True
    ) -> ToolResult:
        """Execute SQL query"""

        try:
            path = Path(db_path).resolve()

            # Check if DB is allowed
            if self.allowed_dbs and not any(
                path.is_relative_to(allowed) for allowed in self.allowed_dbs if path.is_relative_to(allowed) or path == allowed
            ):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Database path not in allowed list: {path}"
                )

            if not path.exists():
                return ToolResult(success=False, output="", error=f"Database file not found: {path}")

            # Safety check for write operations
            if not allow_write:
                blocked = [kw for kw in self.BLOCKED_KEYWORDS if kw in query.upper().split()]
                if blocked:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Write/modify operations blocked. Use allow_write=true if needed. Blocked: {blocked}"
                    )

            # Execute query
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()

            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Fetch results for SELECT
                if query.strip().upper().startswith(('SELECT', 'PRAGMA', '.schema')):
                    if fetch_all:
                        rows = cursor.fetchall()
                        results = [dict(row) for row in rows]
                    else:
                        row = cursor.fetchone()
                        results = [dict(row)] if row else []
                else:
                    results = [{"rows_affected": cursor.rowcount}]

                conn.commit()

                return ToolResult(
                    success=True,
                    output=json.dumps(results, indent=2, default=str),
                    metadata={
                        "row_count": len(results),
                        "columns": list(results[0].keys()) if results else []
                    }
                )

            finally:
                cursor.close()
                conn.close()

        except sqlite3.Error as e:
            return ToolResult(success=False, output="", error=f"SQLite error: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Database error: {str(e)}")

    def get_schema(self, db_path: str) -> ToolResult:
        """Get database schema"""
        query = """
        SELECT name, type FROM sqlite_master
        WHERE type IN ('table', 'view', 'index')
        ORDER BY type, name;
        """
        return self.execute(db_path, query, fetch_all=True)


TOOLS_REGISTRY.register(DatabaseTool())
