"""Subscription tracker - recurring bills, renewal reminders and monthly totals."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path

logger = logging.getLogger(__name__)

_SCHEMA_READY: set = set()

_CYCLES = ("weekly", "monthly", "quarterly", "yearly")
_PER_MONTH = {"weekly": 52 / 12, "monthly": 1.0, "quarterly": 1 / 3, "yearly": 1 / 12}
_NEXT_DELTA = {"weekly": 7, "monthly": 30, "quarterly": 91, "yearly": 365}


def _connect() -> sqlite3.Connection:
    path = db_path("subscriptions.db")
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
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT DEFAULT 'CNY',
            cycle TEXT DEFAULT 'monthly',
            category TEXT DEFAULT 'Other',
            next_due TEXT,
            payment_method TEXT DEFAULT '',
            note TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_subs_active ON subscriptions(is_active);
        """
    )
    conn.commit()


def _parse_day(value: Optional[str], default: Optional[date] = None) -> Optional[str]:
    if not value:
        return default.isoformat() if default else None
    v = value.strip().lower()
    if v == "today":
        return date.today().isoformat()
    if v == "tomorrow":
        return (date.today() + timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(value.strip()).date().isoformat()
    except ValueError:
        return default.isoformat() if default else None


def _roll_forward(day: str, cycle: str) -> str:
    """Advance a past due date to the next occurrence on/after today."""
    try:
        current = datetime.fromisoformat(day).date()
    except ValueError:
        return day
    delta = _NEXT_DELTA.get(cycle, 30)
    today = date.today()
    guard = 0
    while current < today and guard < 500:
        current += timedelta(days=delta)
        guard += 1
    return current.isoformat()


class SubscriptionsTool(BaseTool):
    name = "subscriptions"
    description = (
        "Track recurring subscriptions and bills: add/update/delete, list upcoming renewals, "
        "pause/resume, and compute normalized monthly totals per currency."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "update", "delete", "upcoming", "total", "pause", "resume"],
                "description": "Action to perform.",
            },
            "subscription_id": {"type": "integer", "description": "Subscription ID."},
            "name": {"type": "string", "description": "Subscription name."},
            "amount": {"type": "number", "description": "Amount per cycle."},
            "currency": {"type": "string", "description": "Currency code (default CNY)."},
            "cycle": {"type": "string", "enum": list(_CYCLES), "description": "Billing cycle."},
            "category": {"type": "string", "description": "Category."},
            "next_due": {"type": "string", "description": "Next due date (ISO/today/tomorrow)."},
            "payment_method": {"type": "string", "description": "Payment method."},
            "note": {"type": "string", "description": "Note."},
            "days": {"type": "integer", "description": "Upcoming window in days (default 30)."},
            "include_inactive": {"type": "boolean", "description": "Include paused subscriptions."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "add": self._add,
                "list": self._list,
                "update": self._update,
                "delete": self._delete,
                "upcoming": self._upcoming,
                "total": self._total,
                "pause": lambda **kw: self._set_active(False, **kw),
                "resume": lambda **kw: self._set_active(True, **kw),
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("subscriptions action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _resolve(self, conn: sqlite3.Connection, sub_id: Optional[int],
                 name: Optional[str]) -> Optional[sqlite3.Row]:
        if sub_id is not None:
            return conn.execute("SELECT * FROM subscriptions WHERE id = ?", (int(sub_id),)).fetchone()
        if name:
            return conn.execute(
                "SELECT * FROM subscriptions WHERE lower(name) = lower(?) ORDER BY id DESC LIMIT 1",
                (name.strip(),),
            ).fetchone()
        return None

    def _add(self, name: str = "", amount: Optional[float] = None, **kwargs) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, output="", error="name is required")
        if amount is None:
            return ToolResult(success=False, output="", error="amount is required")
        cycle = kwargs.get("cycle") or "monthly"
        if cycle not in _CYCLES:
            return ToolResult(success=False, output="", error=f"cycle must be one of {_CYCLES}")
        now = datetime.now().isoformat(timespec="seconds")
        conn = _connect()
        try:
            cur = conn.execute(
                """INSERT INTO subscriptions
                   (name, amount, currency, cycle, category, next_due, payment_method, note, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name.strip(), float(amount), kwargs.get("currency") or "CNY", cycle,
                    kwargs.get("category") or "Other", _parse_day(kwargs.get("next_due")),
                    kwargs.get("payment_method", ""), kwargs.get("note", ""), now, now,
                ),
            )
            conn.commit()
            sub_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Subscription '{name.strip()}' added (#{sub_id})",
                          metadata={"subscription_id": sub_id})

    def _list(self, include_inactive: bool = False, **kwargs) -> ToolResult:
        sql = "SELECT * FROM subscriptions"
        if not include_inactive:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY is_active DESC, name"
        conn = _connect()
        try:
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output="No subscriptions tracked",
                              metadata={"subscriptions": []})
        lines = [f"{len(rows)} subscription(s):"]
        for r in rows:
            due = _roll_forward(r["next_due"], r["cycle"]) if r["next_due"] else "-"
            state = "" if r["is_active"] else " (paused)"
            lines.append(
                f"  #{r['id']} {r['name']} - {r['amount']:.2f} {r['currency']}/{r['cycle']}, "
                f"next {due}{state}"
            )
        return ToolResult(success=True, output="\n".join(lines), metadata={"subscriptions": rows})

    def _update(self, subscription_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            sub = self._resolve(conn, subscription_id, kwargs.get("name"))
            if sub is None:
                return ToolResult(success=False, output="", error="Subscription not found")
            fields, values = [], []
            for key in ("name", "amount", "currency", "cycle", "category", "payment_method", "note"):
                if key in kwargs and kwargs[key] is not None:
                    fields.append(f"{key} = ?")
                    values.append(kwargs[key])
            if "next_due" in kwargs and kwargs["next_due"] is not None:
                fields.append("next_due = ?")
                values.append(_parse_day(kwargs["next_due"]))
            if not fields:
                return ToolResult(success=False, output="", error="No fields to update")
            fields.append("updated_at = ?")
            values.append(datetime.now().isoformat(timespec="seconds"))
            values.append(sub["id"])
            conn.execute(f"UPDATE subscriptions SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Subscription '{sub['name']}' updated",
                          metadata={"subscription_id": sub["id"]})

    def _delete(self, subscription_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            sub = self._resolve(conn, subscription_id, kwargs.get("name"))
            if sub is None:
                return ToolResult(success=False, output="", error="Subscription not found")
            conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub["id"],))
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Subscription '{sub['name']}' deleted",
                          metadata={"subscription_id": sub["id"]})

    def _set_active(self, active: bool, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            sub = self._resolve(conn, kwargs.get("subscription_id"), kwargs.get("name"))
            if sub is None:
                return ToolResult(success=False, output="", error="Subscription not found")
            conn.execute(
                "UPDATE subscriptions SET is_active = ?, updated_at = ? WHERE id = ?",
                (1 if active else 0, datetime.now().isoformat(timespec="seconds"), sub["id"]),
            )
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Subscription '{sub['name']}' {'resumed' if active else 'paused'}",
                          metadata={"subscription_id": sub["id"], "is_active": active})

    def _upcoming(self, days: int = 30, **kwargs) -> ToolResult:
        days = max(0, int(days or 30))
        horizon = date.today() + timedelta(days=days)
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM subscriptions WHERE is_active = 1 AND next_due IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        items = []
        for r in rows:
            due_str = _roll_forward(r["next_due"], r["cycle"])
            try:
                due = datetime.fromisoformat(due_str).date()
            except ValueError as e:
                logger.debug("subscription #%s invalid due date %r: %s", r["id"], due_str, e)
                continue
            if due <= horizon:
                items.append({
                    "id": r["id"], "name": r["name"], "amount": r["amount"],
                    "currency": r["currency"], "cycle": r["cycle"],
                    "next_due": due_str, "days_until": (due - date.today()).days,
                })
        items.sort(key=lambda x: x["next_due"])
        if not items:
            return ToolResult(success=True, output=f"No renewals in the next {days} days",
                              metadata={"upcoming": []})
        lines = [f"Upcoming renewals ({days}d):"]
        for item in items:
            when = "today" if item["days_until"] == 0 else f"in {item['days_until']}d"
            lines.append(f"  #{item['id']} {item['name']} {item['amount']:.2f} {item['currency']} "
                         f"due {item['next_due']} ({when})")
        return ToolResult(success=True, output="\n".join(lines), metadata={"upcoming": items})

    def _total(self, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            rows = conn.execute("SELECT * FROM subscriptions WHERE is_active = 1").fetchall()
        finally:
            conn.close()
        totals: Dict[str, float] = {}
        for r in rows:
            per_month = r["amount"] * _PER_MONTH.get(r["cycle"], 1.0)
            totals[r["currency"]] = totals.get(r["currency"], 0.0) + per_month
        if not totals:
            return ToolResult(success=True, output="No active subscriptions", metadata={"totals": {}})
        lines = ["Normalized monthly total:"]
        for currency, value in sorted(totals.items()):
            lines.append(f"  {currency}: {value:.2f}/month ({value * 12:.2f}/year)")
        return ToolResult(success=True, output="\n".join(lines),
                          metadata={"totals": {k: round(v, 2) for k, v in totals.items()}})


TOOLS_REGISTRY.register(SubscriptionsTool())
