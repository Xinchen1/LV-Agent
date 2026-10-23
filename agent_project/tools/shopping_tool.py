"""Shopping list tool - groceries and household items with quantities and estimates."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path

logger = logging.getLogger(__name__)

_SCHEMA_READY: set = set()

_PRIORITY_LABELS = {1: "urgent", 2: "normal", 3: "low"}


def _connect() -> sqlite3.Connection:
    path = db_path("shopping.db")
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
        CREATE TABLE IF NOT EXISTS shopping_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            quantity REAL DEFAULT 1,
            unit TEXT DEFAULT '',
            category TEXT DEFAULT 'Other',
            price REAL,
            store TEXT DEFAULT '',
            note TEXT DEFAULT '',
            priority INTEGER DEFAULT 2,
            is_checked INTEGER DEFAULT 0,
            created_at TEXT,
            checked_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_shopping_checked ON shopping_items(is_checked);
        CREATE INDEX IF NOT EXISTS idx_shopping_category ON shopping_items(category);
        """
    )
    conn.commit()


class ShoppingTool(BaseTool):
    name = "shopping"
    description = (
        "Shopping and grocery list: add items with quantity/unit/category/price, check off, "
        "update, clear completed items, group by category and estimate total cost."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "check", "uncheck", "update", "delete", "clear_checked", "categories", "estimate"],
                "description": "Action to perform.",
            },
            "item_id": {"type": "integer", "description": "Item ID."},
            "name": {"type": "string", "description": "Item name."},
            "quantity": {"type": "number", "description": "Quantity (default 1)."},
            "unit": {"type": "string", "description": "Unit, e.g. kg/bottle/pack."},
            "category": {"type": "string", "description": "Category, e.g. Produce/Dairy/Household."},
            "price": {"type": "number", "description": "Unit price."},
            "store": {"type": "string", "description": "Where to buy."},
            "note": {"type": "string", "description": "Note."},
            "priority": {"type": "integer", "description": "1=urgent, 2=normal, 3=low."},
            "include_checked": {"type": "boolean", "description": "Include checked items."},
            "category_filter": {"type": "string", "description": "Filter list by category."},
            "limit": {"type": "integer", "description": "Max rows (default 100)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "add": self._add,
                "list": self._list,
                "check": lambda **kw: self._set_checked(True, **kw),
                "uncheck": lambda **kw: self._set_checked(False, **kw),
                "update": self._update,
                "delete": self._delete,
                "clear_checked": self._clear_checked,
                "categories": self._categories,
                "estimate": self._estimate,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("shopping action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _find(self, conn: sqlite3.Connection, item_id: Optional[int],
              name: Optional[str]) -> Optional[sqlite3.Row]:
        if item_id is not None:
            return conn.execute("SELECT * FROM shopping_items WHERE id = ?", (int(item_id),)).fetchone()
        if name:
            return conn.execute(
                "SELECT * FROM shopping_items WHERE lower(name) = lower(?) AND is_checked = 0 "
                "ORDER BY id DESC LIMIT 1",
                (name.strip(),),
            ).fetchone()
        return None

    def _add(self, name: str = "", **kwargs) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, output="", error="name is required")
        now = datetime.now().isoformat(timespec="seconds")
        quantity = kwargs.get("quantity")
        conn = _connect()
        try:
            cur = conn.execute(
                """INSERT INTO shopping_items
                   (name, quantity, unit, category, price, store, note, priority, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name.strip(),
                    float(quantity) if quantity is not None else 1.0,
                    kwargs.get("unit", ""),
                    kwargs.get("category") or "Other",
                    kwargs.get("price"),
                    kwargs.get("store", ""),
                    kwargs.get("note", ""),
                    int(kwargs.get("priority") or 2),
                    now,
                ),
            )
            conn.commit()
            item_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Added '{name.strip()}' to shopping list (#{item_id})",
                          metadata={"item_id": item_id})

    def _list(self, include_checked: bool = False, category_filter: Optional[str] = None,
              limit: int = 100, **kwargs) -> ToolResult:
        sql = "SELECT * FROM shopping_items"
        clauses, params = [], []
        if not include_checked:
            clauses.append("is_checked = 0")
        if category_filter:
            clauses.append("category = ?")
            params.append(category_filter)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY is_checked, priority, category, name LIMIT ?"
        params.append(max(1, int(limit or 100)))
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="Shopping list is empty", metadata={"items": []})
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["category"], []).append(row)
        lines = [f"Shopping list ({len(rows)} item(s)):"]
        for category, items in grouped.items():
            lines.append(f"  [{category}]")
            for item in items:
                mark = "[x]" if item["is_checked"] else "[ ]"
                qty = f"{item['quantity']:g}{(' ' + item['unit']) if item['unit'] else ''}"
                price = f" @{item['price']:.2f}" if item["price"] is not None else ""
                note = f" ({item['note']})" if item["note"] else ""
                lines.append(f"    {mark} #{item['id']} {item['name']} x{qty}{price}{note}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"items": rows})

    def _set_checked(self, checked: bool, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            item = self._find(conn, kwargs.get("item_id"), kwargs.get("name"))
            if item is None:
                return ToolResult(success=False, output="", error="Item not found")
            conn.execute(
                "UPDATE shopping_items SET is_checked = ?, checked_at = ? WHERE id = ?",
                (1 if checked else 0,
                 datetime.now().isoformat(timespec="seconds") if checked else None,
                 item["id"]),
            )
            conn.commit()
        finally:
            conn.close()
        state = "checked" if checked else "unchecked"
        return ToolResult(success=True, output=f"'{item['name']}' {state}",
                          metadata={"item_id": item["id"], "checked": checked})

    def _update(self, item_id: Optional[int] = None, **kwargs) -> ToolResult:
        if item_id is None:
            return ToolResult(success=False, output="", error="item_id is required")
        fields, values = [], []
        for key in ("name", "quantity", "unit", "category", "price", "store", "note", "priority"):
            if key in kwargs and kwargs[key] is not None:
                fields.append(f"{key} = ?")
                values.append(kwargs[key])
        if not fields:
            return ToolResult(success=False, output="", error="No fields to update")
        values.append(int(item_id))
        conn = _connect()
        try:
            cur = conn.execute(f"UPDATE shopping_items SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
            changed = cur.rowcount
        finally:
            conn.close()
        if not changed:
            return ToolResult(success=False, output="", error=f"Item {item_id} not found")
        return ToolResult(success=True, output=f"Item #{item_id} updated",
                          metadata={"item_id": int(item_id)})

    def _delete(self, item_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            item = self._find(conn, item_id, kwargs.get("name"))
            if item is None:
                return ToolResult(success=False, output="", error="Item not found")
            conn.execute("DELETE FROM shopping_items WHERE id = ?", (item["id"],))
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Deleted '{item['name']}'",
                          metadata={"item_id": item["id"]})

    def _clear_checked(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM shopping_items WHERE is_checked = 1")
            conn.commit()
            removed = cur.rowcount
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Cleared {removed} checked item(s)",
                          metadata={"removed": removed})

    def _categories(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT category, COUNT(*) AS c FROM shopping_items WHERE is_checked = 0 "
                "GROUP BY category ORDER BY c DESC"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No pending items", metadata={"categories": {}})
        data = {r["category"]: r["c"] for r in rows}
        lines = ["Pending by category:"] + [f"  {k}: {v}" for k, v in data.items()]
        return ToolResult(success=True, output="\n".join(lines), metadata={"categories": data})

    def _estimate(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT name, quantity, price FROM shopping_items "
                "WHERE is_checked = 0 AND price IS NOT NULL"
            ).fetchall()
            missing = conn.execute(
                "SELECT COUNT(*) AS c FROM shopping_items WHERE is_checked = 0 AND price IS NULL"
            ).fetchone()["c"]
        finally:
            conn.close()
        total = sum((r["quantity"] or 0) * (r["price"] or 0) for r in rows)
        lines = [
            f"Estimated total: {total:.2f}",
            f"  priced items: {len(rows)}",
            f"  items without price: {missing}",
        ]
        return ToolResult(success=True, output="\n".join(lines),
                          metadata={"total": round(total, 2), "priced": len(rows), "missing_price": missing})


TOOLS_REGISTRY.register(ShoppingTool())
