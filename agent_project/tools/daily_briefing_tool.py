"""
Daily Briefing Tool - Unified view of calendar, tasks, reminders, finance, and health for a given day.
Cross-tool aggregation for morning briefing and planning.
"""

import json
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY


class DailyBriefingTool(BaseTool):
    name = "daily_briefing"
    description = (
        "Generate a unified daily briefing aggregating calendar events, tasks, reminders, "
        "finance transactions, and health data for a specific date. "
        "Perfect for morning planning and end-of-day review."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["generate", "morning", "evening", "weekly"],
                "description": "Briefing type: 'generate' for custom date, 'morning' for today AM, 'evening' for today PM, 'weekly' for week overview."
            },
            "target_date": {
                "type": "string",
                "description": "Date for briefing in YYYY-MM-DD format. Default: today."
            },
            "include_sections": {
                "type": "array",
                "items": {"type": "string", "enum": ["calendar", "tasks", "reminders", "finance", "health", "weather"]},
                "description": "Sections to include. Default: all."
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "text", "json"],
                "description": "Output format. Default: markdown."
            }
        },
        "required": ["action"]
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "generate":
                return self._generate_briefing(**kwargs)
            elif action == "morning":
                return self._morning_briefing(**kwargs)
            elif action == "evening":
                return self._evening_briefing(**kwargs)
            elif action == "weekly":
                return self._weekly_briefing(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _get_date(self, target_date: str = None) -> date:
        if target_date:
            return datetime.strptime(target_date, "%Y-%m-%d").date()
        return date.today()

    def _generate_briefing(self, target_date: str = None, include_sections: List[str] = None,
                           format: str = "markdown", action: str = "morning") -> ToolResult:
        d = self._get_date(target_date)
        sections = include_sections or ["calendar", "tasks", "reminders", "finance", "health", "weather"]
        
        data = {}
        if "calendar" in sections:
            data["calendar"] = self._get_calendar(d)
        if "tasks" in sections:
            data["tasks"] = self._get_tasks(d)
        if "reminders" in sections:
            data["reminders"] = self._get_reminders(d)
        if "finance" in sections:
            data["finance"] = self._get_finance(d)
        if "health" in sections:
            data["health"] = self._get_health(d)
        if "weather" in sections:
            data["weather"] = self._get_weather(d)

        if format == "json":
            return ToolResult(success=True, output=json.dumps(data, ensure_ascii=False, indent=2), metadata=data)

        output = self._format_briefing(d, data, is_morning=(action != "evening"))
        return ToolResult(success=True, output=output, metadata=data)

    def _morning_briefing(self, **kwargs) -> ToolResult:
        kwargs.pop("action", None)
        return self._generate_briefing(**kwargs)

    def _evening_briefing(self, target_date: str = None, **kwargs) -> ToolResult:
        d = self._get_date(target_date)
        data = {
            "calendar": self._get_calendar(d, completed_only=False),
            "tasks": self._get_tasks(d, show_completed=True),
            "reminders": self._get_reminders(d),
            "finance": self._get_finance(d),
            "health": self._get_health(d, summary=True),
        }
        output = self._format_evening_review(d, data)
        return ToolResult(success=True, output=output, metadata=data)

    def _weekly_briefing(self, target_date: str = None, **kwargs) -> ToolResult:
        d = self._get_date(target_date)
        week_start = d - timedelta(days=d.weekday())
        week_end = week_start + timedelta(days=6)
        
        output = [f"📅 Weekly Briefing ({week_start} ~ {week_end})", "=" * 40]
        
        for i in range(7):
            day = week_start + timedelta(days=i)
            day_data = {
                "calendar": self._get_calendar(day),
                "tasks": self._get_tasks(day),
                "reminders": self._get_reminders(day),
            }
            has_items = any(v for v in day_data.values())
            if has_items:
                day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day.weekday()]
                output.append(f"\n{day_name} {day.strftime('%m/%d')}:")
                for cal in day_data["calendar"][:3]:
                    output.append(f"  📅 {cal['start_time'][:5]} {cal['title']}")
                for task in day_data["tasks"][:3]:
                    output.append(f"  ☐ {task['title']}")
                for rem in day_data["reminders"][:2]:
                    output.append(f"  🔔 {rem['title']}")
        
        # Weekly finance summary
        fin = self._get_finance_week(week_start, week_end)
        output.append(f"\n💰 Weekly Finance: Income ¥{fin['income']:.0f} | Expense ¥{fin['expense']:.0f} | Net ¥{fin['net']:.0f}")
        
        # Weekly health summary
        hlth = self._get_health_week(week_start, week_end)
        output.append(f"🏃 Weekly Health: {hlth['workouts']} workouts, {hlth['avg_sleep']:.1f}h avg sleep")
        
        return ToolResult(success=True, output="\n".join(output), metadata={"week_start": week_start.isoformat()})

    # Data fetchers
    def _get_calendar(self, d: date, completed_only: bool = False) -> List[Dict]:
        from agent_project.tools.calendar_tool import DB_PATH as CAL_DB
        conn = sqlite3.connect(CAL_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        day_start = d.isoformat()
        day_end = (d + timedelta(days=1)).isoformat()
        cursor.execute("""
            SELECT * FROM events 
            WHERE start_time >= ? AND start_time < ?
            ORDER BY start_time
        """, (day_start, day_end))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def _get_tasks(self, d: date, show_completed: bool = False) -> List[Dict]:
        from agent_project.tools.task_tool import DB_PATH as TASK_DB
        conn = sqlite3.connect(TASK_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        day_str = d.isoformat()
        query = "SELECT * FROM tasks WHERE (due_date LIKE ? OR due_date IS NULL) AND 1=1"
        params = [f"{day_str}%"]
        if not show_completed:
            query += " AND status != 'completed'"
        query += " ORDER BY priority ASC, due_date ASC"
        cursor.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def _get_reminders(self, d: date) -> List[Dict]:
        from agent_project.tools.reminders_tool import DB_PATH as REM_DB
        conn = sqlite3.connect(REM_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        day_start = d.isoformat()
        day_end = (d + timedelta(days=1)).isoformat()
        cursor.execute("""
            SELECT * FROM reminders 
            WHERE is_active = 1 AND trigger_time >= ? AND trigger_time < ?
            ORDER BY trigger_time
        """, (day_start, day_end))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def _get_finance(self, d: date) -> Dict:
        from agent_project.tools.finance_tool import DB_PATH as FIN_DB
        conn = sqlite3.connect(FIN_DB)
        cursor = conn.cursor()
        day_str = d.isoformat()
        cursor.execute("SELECT type, SUM(amount) as total FROM transactions WHERE transaction_date = ? GROUP BY type", (day_str,))
        by_type = dict(cursor.fetchall())
        cursor.execute("SELECT category, SUM(amount) as total FROM transactions WHERE transaction_date = ? AND type = 'expense' GROUP BY category", (day_str,))
        by_cat = dict(cursor.fetchall())
        conn.close()
        return {
            "income": by_type.get("income", 0),
            "expense": by_type.get("expense", 0),
            "net": by_type.get("income", 0) - by_type.get("expense", 0),
            "by_category": by_cat
        }

    def _get_finance_week(self, start: date, end: date) -> Dict:
        from agent_project.tools.finance_tool import DB_PATH as FIN_DB
        conn = sqlite3.connect(FIN_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT type, SUM(amount) as total FROM transactions 
            WHERE transaction_date >= ? AND transaction_date <= ? GROUP BY type
        """, (start.isoformat(), end.isoformat()))
        by_type = dict(cursor.fetchall())
        conn.close()
        return {
            "income": by_type.get("income", 0),
            "expense": by_type.get("expense", 0),
            "net": by_type.get("income", 0) - by_type.get("expense", 0)
        }

    def _get_health(self, d: date, summary: bool = False) -> Dict:
        from agent_project.tools.health_tool import DB_PATH as HLTH_DB
        conn = sqlite3.connect(HLTH_DB)
        cursor = conn.cursor()
        day_str = d.isoformat()
        
        cursor.execute("SELECT COUNT(*) as cnt, SUM(duration_minutes) as min, SUM(calories_burned) as cal FROM workouts WHERE workout_date = ?", (day_str,))
        w = cursor.fetchone()
        
        cursor.execute("SELECT SUM(calories) as cal, SUM(protein_g) as prot, SUM(water_ml) as water FROM nutrition WHERE nutrition_date = ?", (day_str,))
        n = cursor.fetchone()
        
        cursor.execute("SELECT AVG(duration_hours) as avg_dur, AVG(quality) as avg_qual FROM sleep WHERE sleep_date = ?", (day_str,))
        s = cursor.fetchone()
        
        conn.close()
        
        return {
            "workouts": w[0] or 0,
            "total_minutes": w[1] or 0,
            "calories_burned": w[2] or 0,
            "calories_consumed": n[0] or 0,
            "protein": n[1] or 0,
            "water": n[2] or 0,
            "sleep_hours": s[0] or 0,
            "sleep_quality": s[1] or 0
        }

    def _get_health_week(self, start: date, end: date) -> Dict:
        from agent_project.tools.health_tool import DB_PATH as HLTH_DB
        conn = sqlite3.connect(HLTH_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt, AVG(duration_hours) as avg_sleep FROM workouts WHERE workout_date BETWEEN ? AND ?", (start.isoformat(), end.isoformat()))
        w = cursor.fetchone()
        cursor.execute("SELECT AVG(duration_hours) as avg_dur FROM sleep WHERE sleep_date BETWEEN ? AND ?", (start.isoformat(), end.isoformat()))
        s = cursor.fetchone()
        conn.close()
        return {
            "workouts": w[0] or 0,
            "avg_sleep": s[0] or 0
        }

    def _get_weather(self, d: date) -> Dict:
        # Placeholder - would call weather tool
        return {"status": "available", "note": "Use weather tool for details"}

    def _format_briefing(self, d: date, data: Dict, is_morning: bool = True) -> str:
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()]
        header = f"🌅 Morning Briefing" if is_morning else f"🌙 Evening Review"
        lines = [f"{header} — {day_name}, {d.strftime('%Y-%m-%d')}", "=" * 45]

        # Calendar
        cal = data.get("calendar", [])
        if cal:
            lines.append(f"\n📅 Calendar ({len(cal)} events):")
            for c in cal:
                start = c["start_time"][11:16]
                end = c["end_time"][11:16]
                loc = f" @ {c['location']}" if c["location"] else ""
                lines.append(f"  {start}~{end} {c['title']}{loc}")
        else:
            lines.append("\n📅 Calendar: No events scheduled")

        # Tasks
        tasks = data.get("tasks", [])
        pending = [t for t in tasks if t["status"] != "completed"]
        if pending:
            lines.append(f"\n☑️ Tasks ({len(pending)} pending):")
            for t in pending[:5]:
                pri = "!" * (6 - t["priority"])
                due = f" 📅{t['due_date'][:10]}" if t["due_date"] else ""
                proj = f" [{t['project']}]" if t["project"] else ""
                lines.append(f"  {pri} {t['title']}{proj}{due}")
            if len(pending) > 5:
                lines.append(f"  ... and {len(pending) - 5} more")
        else:
            lines.append("\n☑️ Tasks: All caught up!")

        # Reminders
        rem = data.get("reminders", [])
        if rem:
            lines.append(f"\n🔔 Reminders ({len(rem)}):")
            for r in rem[:5]:
                t = r["trigger_time"][11:16]
                lines.append(f"  {t} {r['title']}")
        elif is_morning:
            lines.append("\n🔔 Reminders: None for today")

        # Finance
        fin = data.get("finance", {})
        if fin.get("expense", 0) > 0 or fin.get("income", 0) > 0:
            lines.append(f"\n💰 Today: Income ¥{fin.get('income', 0):.0f} | Expense ¥{fin.get('expense', 0):.0f} | Net ¥{fin.get('net', 0):.0f}")
            if fin.get("by_category"):
                top_cat = max(fin["by_category"].items(), key=lambda x: x[1])
                lines.append(f"  Top expense: {top_cat[0]} ¥{top_cat[1]:.0f}")

        # Health
        hlth = data.get("health", {})
        if any(hlth.values()):
            lines.append(f"\n🏃 Health: {hlth.get('workouts', 0)} workouts ({hlth.get('total_minutes', 0)} min)")
            lines.append(f"  Calories: in {hlth.get('calories_consumed', 0)} / out {hlth.get('calories_burned', 0)}")
            lines.append(f"  Sleep: {hlth.get('sleep_hours', 0):.1f}h (quality {hlth.get('sleep_quality', 0):.0f}/10)")
            lines.append(f"  Water: {hlth.get('water', 0)}ml | Protein: {hlth.get('protein', 0):.0f}g")

        if is_morning:
            lines.append("\n" + "─" * 45)
            lines.append("💡 Have a productive day!")
        
        return "\n".join(lines)

    def _format_evening_review(self, d: date, data: Dict) -> str:
        lines = [f"🌙 Evening Review — {d.strftime('%Y-%m-%d')}", "=" * 45]

        # Completed tasks
        tasks = data.get("tasks", [])
        completed = [t for t in tasks if t["status"] == "completed"]
        pending = [t for t in tasks if t["status"] != "completed"]
        lines.append(f"\n✅ Completed ({len(completed)}):")
        for t in completed:
            lines.append(f"  ✓ {t['title']}")
        if pending:
            lines.append(f"\n⏳ Carried over ({len(pending)}):")
            for t in pending[:5]:
                lines.append(f"  → {t['title']}")

        # Finance
        fin = data.get("finance", {})
        lines.append(f"\n💰 Finance: Income ¥{fin.get('income', 0):.0f} | Expense ¥{fin.get('expense', 0):.0f}")

        # Health
        hlth = data.get("health", {})
        lines.append(f"\n🏃 Health: {hlth.get('workouts', 0)} workouts, {hlth.get('sleep_hours', 0):.1f}h sleep")
        lines.append(f"  Net calories: {hlth.get('calories_consumed', 0) - hlth.get('calories_burned', 0):+.0f}")

        lines.append("\n" + "─" * 45)
        lines.append("🌙 Good night! Rest well for tomorrow.")
        
        return "\n".join(lines)


TOOLS_REGISTRY.register(DailyBriefingTool())