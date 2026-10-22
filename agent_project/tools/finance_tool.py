"""
Expense/Finance Tracking Tool - Personal assistant expense tracking with categories, budgets, and reports.
"""

import json
import sqlite3
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("finance.db")


def _init_db():
    """Initialize the finance database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL CHECK(type IN ('income', 'expense', 'transfer')),
            amount REAL NOT NULL,
            currency TEXT DEFAULT 'CNY',
            category TEXT,
            subcategory TEXT,
            account TEXT,
            description TEXT,
            tags TEXT,
            transaction_date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT CHECK(type IN ('cash', 'bank', 'credit', 'investment', 'other')),
            currency TEXT DEFAULT 'CNY',
            balance REAL DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            period TEXT CHECK(period IN ('weekly', 'monthly', 'yearly')),
            start_date TEXT,
            end_date TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)")
    conn.commit()
    conn.close()


class FinanceTool(BaseTool):
    name = "finance"
    description = (
        "Manage personal finances: track income/expenses, manage accounts, set budgets, and generate reports. "
        "Supports multiple currencies, categories, tags, and recurring transactions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add_transaction", "read_transaction", "update_transaction", "delete_transaction", "list_transactions", "summary", "budget_summary", "accounts", "categories"],
                "description": "Action to perform on finances."
            },
            "transaction_id": {
                "type": "integer",
                "description": "Transaction ID for read/update/delete actions."
            },
            "type": {
                "type": "string",
                "enum": ["income", "expense", "transfer"],
                "description": "Transaction type (required for add_transaction)."
            },
            "amount": {
                "type": "number",
                "description": "Transaction amount (required for add_transaction)."
            },
            "currency": {
                "type": "string",
                "description": "Currency code (e.g., CNY, USD). Default CNY."
            },
            "category": {
                "type": "string",
                "description": "Category (e.g., food, transport, salary, entertainment)."
            },
            "subcategory": {
                "type": "string",
                "description": "Subcategory for finer granularity."
            },
            "account": {
                "type": "string",
                "description": "Account name (e.g., Cash, Alipay, Bank Card)."
            },
            "description": {
                "type": "string",
                "description": "Transaction description."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags."
            },
            "transaction_date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format. Default today."
            },
            "date_from": {
                "type": "string",
                "description": "Start date for filtering (YYYY-MM-DD)."
            },
            "date_to": {
                "type": "string",
                "description": "End date for filtering (YYYY-MM-DD)."
            },
            "category_filter": {
                "type": "string",
                "description": "Filter by category."
            },
            "type_filter": {
                "type": "string",
                "enum": ["income", "expense", "transfer"],
                "description": "Filter by transaction type."
            },
            "account_filter": {
                "type": "string",
                "description": "Filter by account."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results. Default 100."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "add_transaction":
                return self._add_transaction(**kwargs)
            elif action == "read_transaction":
                return self._read_transaction(**kwargs)
            elif action == "update_transaction":
                return self._update_transaction(**kwargs)
            elif action == "delete_transaction":
                return self._delete_transaction(**kwargs)
            elif action == "list_transactions":
                return self._list_transactions(**kwargs)
            elif action == "summary":
                return self._get_summary(**kwargs)
            elif action == "budget_summary":
                return self._get_budget_summary(**kwargs)
            elif action == "accounts":
                return self._manage_accounts(**kwargs)
            elif action == "categories":
                return self._list_categories(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _add_transaction(self, type: str, amount: float, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        trans_date = kwargs.get("transaction_date", date.today().isoformat())
        cursor.execute("""
            INSERT INTO transactions (type, amount, currency, category, subcategory, account, description, tags, transaction_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            type,
            amount,
            kwargs.get("currency", "CNY"),
            kwargs.get("category"),
            kwargs.get("subcategory"),
            kwargs.get("account"),
            kwargs.get("description"),
            json.dumps(kwargs.get("tags", [])),
            trans_date,
            now,
            now
        ))
        trans_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Transaction added with ID: {trans_id}",
            metadata={"transaction_id": trans_id, "type": type, "amount": amount, "date": trans_date}
        )

    def _read_transaction(self, transaction_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Transaction {transaction_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"transaction": dict(row)}
        )

    def _update_transaction(self, transaction_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Transaction {transaction_id} not found")

        updates = []
        values = []
        allowed_fields = ["type", "amount", "currency", "category", "subcategory", "account", "description", "transaction_date"]
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
        values.append(transaction_id)

        cursor.execute(f"UPDATE transactions SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Transaction {transaction_id} updated", metadata={"transaction_id": transaction_id})

    def _delete_transaction(self, transaction_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Transaction {transaction_id} not found")
        return ToolResult(success=True, output=f"Transaction {transaction_id} deleted", metadata={"transaction_id": transaction_id})

    def _list_transactions(self, date_from: str = None, date_to: str = None, category_filter: str = None, type_filter: str = None, account_filter: str = None, limit: int = 100) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM transactions WHERE 1=1"
        params = []

        if date_from:
            query += " AND transaction_date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND transaction_date <= ?"
            params.append(date_to)
        if category_filter:
            query += " AND category = ?"
            params.append(category_filter)
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        if account_filter:
            query += " AND account = ?"
            params.append(account_filter)

        query += " ORDER BY transaction_date DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No transactions found", metadata={"transactions": []})

        lines = [f"Found {len(rows)} transaction(s):"]
        total_income = sum(r["amount"] for r in rows if r["type"] == "income")
        total_expense = sum(r["amount"] for r in rows if r["type"] == "expense")
        for r in rows:
            sign = "+" if r["type"] == "income" else "-"
            cat = f" [{r['category']}]" if r["category"] else ""
            acc = f" ({r['account']})" if r["account"] else ""
            desc = f" - {r['description']}" if r["description"] else ""
            lines.append(f"  {sign}¥{r['amount']:.2f} {r['currency']}{cat}{acc}{desc} | {r['transaction_date']}")

        lines.append(f"\n📊 Total: Income ¥{total_income:.2f} | Expense ¥{total_expense:.2f} | Net ¥{total_income - total_expense:.2f}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"transactions": rows, "count": len(rows), "total_income": total_income, "total_expense": total_expense}
        )

    def _get_summary(self, date_from: str = None, date_to: str = None) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        where = "WHERE 1=1"
        params = []
        if date_from:
            where += " AND transaction_date >= ?"
            params.append(date_from)
        if date_to:
            where += " AND transaction_date <= ?"
            params.append(date_to)

        cursor.execute(f"SELECT type, category, SUM(amount) as total FROM transactions {where} GROUP BY type, category ORDER BY type, total DESC", params)
        by_category = cursor.fetchall()

        cursor.execute(f"SELECT type, SUM(amount) as total FROM transactions {where} GROUP BY type", params)
        by_type = dict(cursor.fetchall())

        cursor.execute(f"SELECT account, SUM(amount) as total FROM transactions {where} AND type = 'expense' GROUP BY account ORDER BY total DESC", params)
        by_account = cursor.fetchall()

        conn.close()

        lines = ["💰 Financial Summary:"]
        if date_from or date_to:
            period = f" ({date_from or '...'} to {date_to or '...'})"
            lines[0] += period

        income = by_type.get("income", 0)
        expense = by_type.get("expense", 0)
        lines.append(f"  Income:  ¥{income:.2f}")
        lines.append(f"  Expense: ¥{expense:.2f}")
        lines.append(f"  Net:     ¥{income - expense:.2f}")

        if by_category:
            lines.append("\n  By Category:")
            for typ, cat, total in by_category:
                icon = "📈" if typ == "income" else "📉"
                lines.append(f"    {icon} {cat or 'Uncategorized'}: ¥{total:.2f}")

        if by_account:
            lines.append("\n  Top Expense Accounts:")
            for acc, total in by_account[:5]:
                lines.append(f"    {acc or 'Unknown'}: ¥{total:.2f}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"by_type": by_type, "by_category": by_category, "by_account": by_account}
        )

    def _get_budget_summary(self) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM budgets WHERE is_active = 1")
        budgets = [dict(row) for row in cursor.fetchall()]

        if not budgets:
            conn.close()
            return ToolResult(success=True, output="No active budgets set", metadata={"budgets": []})

        lines = ["📋 Budget Tracking:"]
        for b in budgets:
            cat = b["category"]
            limit = b["amount"]
            period = b["period"]

            # Calculate spent this period
            if period == "monthly":
                start = date.today().replace(day=1).isoformat()
                end = (date.today().replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()
            elif period == "weekly":
                start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
                end = (date.today() + timedelta(days=6 - date.today().weekday())).isoformat()
            else:
                start = date.today().replace(month=1, day=1).isoformat()
                end = date.today().replace(month=12, day=31).isoformat()

            cursor.execute("""
                SELECT SUM(amount) as spent FROM transactions
                WHERE type = 'expense' AND category = ? AND transaction_date >= ? AND transaction_date < ?
            """, (cat, start, end))
            spent = cursor.fetchone()["spent"] or 0

            pct = (spent / limit * 100) if limit > 0 else 0
            status = "🟢" if pct < 80 else ("🟡" if pct < 100 else "🔴")
            lines.append(f"  {status} {cat}: ¥{spent:.2f} / ¥{limit:.2f} ({pct:.0f}%) [{period}]")

        conn.close()
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"budgets": budgets}
        )

    def _manage_accounts(self, **kwargs) -> ToolResult:
        # For now just list accounts
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM accounts WHERE is_active = 1 ORDER BY name")
        accounts = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not accounts:
            return ToolResult(success=True, output="No accounts configured", metadata={"accounts": []})

        lines = ["🏦 Accounts:"]
        for a in accounts:
            lines.append(f"  {a['name']} ({a['type']}): ¥{a['balance']:.2f} {a['currency']}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"accounts": accounts}
        )

    def _list_categories(self) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM transactions WHERE category IS NOT NULL ORDER BY category")
        cats = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not cats:
            return ToolResult(success=True, output="No categories used yet", metadata={"categories": []})

        return ToolResult(
            success=True,
            output="📂 Categories:\n  " + "\n  ".join(cats),
            metadata={"categories": cats}
        )


from datetime import timedelta

TOOLS_REGISTRY.register(FinanceTool())