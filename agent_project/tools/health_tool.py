"""
Health/Fitness Tracking Tool - Personal assistant health tracking with workouts, nutrition, sleep, and vitals.
"""

import json
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("health.db")


def _init_db():
    """Initialize the health database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_type TEXT NOT NULL,
            duration_minutes INTEGER,
            distance_km REAL,
            calories_burned INTEGER,
            intensity TEXT CHECK(intensity IN ('low', 'moderate', 'high')),
            notes TEXT,
            workout_date TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nutrition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_type TEXT CHECK(meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
            food_items TEXT,
            calories INTEGER,
            protein_g REAL,
            carbs_g REAL,
            fat_g REAL,
            fiber_g REAL,
            sugar_g REAL,
            water_ml INTEGER,
            nutrition_date TEXT NOT NULL,
            meal_time TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sleep (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep_date TEXT NOT NULL,
            bedtime TEXT,
            wake_time TEXT,
            duration_hours REAL,
            quality INTEGER CHECK(quality BETWEEN 1 AND 10),
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            measured_at TEXT NOT NULL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_type TEXT NOT NULL,
            target_value REAL NOT NULL,
            unit TEXT,
            period TEXT CHECK(period IN ('daily', 'weekly', 'monthly')),
            start_date TEXT,
            end_date TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(workout_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nutrition_date ON nutrition(nutrition_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep(sleep_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vitals_metric ON vitals(metric)")
    conn.commit()
    conn.close()


class HealthTool(BaseTool):
    name = "health"
    description = (
        "Track health and fitness: log workouts, nutrition, sleep, vitals (weight, heart rate, etc.), "
        "and set health goals. Provides summaries and progress tracking."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["log_workout", "list_workouts", "log_nutrition", "list_nutrition", "log_sleep", "list_sleep", "log_vitals", "list_vitals", "set_goal", "list_goals", "summary", "progress"],
                "description": "Action to perform on health data."
            },
            # Workout fields
            "activity_type": {
                "type": "string",
                "description": "Activity type (running, cycling, swimming, strength, yoga, walking, etc.)."
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Workout duration in minutes."
            },
            "distance_km": {
                "type": "number",
                "description": "Distance in kilometers."
            },
            "calories_burned": {
                "type": "integer",
                "description": "Calories burned."
            },
            "intensity": {
                "type": "string",
                "enum": ["low", "moderate", "high"],
                "description": "Workout intensity."
            },
            "workout_date": {
                "type": "string",
                "description": "Workout date (YYYY-MM-DD). Default today."
            },
            "start_time": {
                "type": "string",
                "description": "Start time (HH:MM)."
            },
            "end_time": {
                "type": "string",
                "description": "End time (HH:MM)."
            },
            "workout_notes": {
                "type": "string",
                "description": "Workout notes."
            },
            # Nutrition fields
            "meal_type": {
                "type": "string",
                "enum": ["breakfast", "lunch", "dinner", "snack"],
                "description": "Meal type."
            },
            "food_items": {
                "type": "string",
                "description": "Food items description."
            },
            "calories": {
                "type": "integer",
                "description": "Calories consumed."
            },
            "protein_g": {
                "type": "number",
                "description": "Protein in grams."
            },
            "carbs_g": {
                "type": "number",
                "description": "Carbohydrates in grams."
            },
            "fat_g": {
                "type": "number",
                "description": "Fat in grams."
            },
            "fiber_g": {
                "type": "number",
                "description": "Fiber in grams."
            },
            "sugar_g": {
                "type": "number",
                "description": "Sugar in grams."
            },
            "water_ml": {
                "type": "integer",
                "description": "Water intake in ml."
            },
            "nutrition_date": {
                "type": "string",
                "description": "Nutrition date (YYYY-MM-DD). Default today."
            },
            "meal_time": {
                "type": "string",
                "description": "Meal time (HH:MM)."
            },
            "nutrition_notes": {
                "type": "string",
                "description": "Nutrition notes."
            },
            # Sleep fields
            "sleep_date": {
                "type": "string",
                "description": "Sleep date (YYYY-MM-DD). Default today."
            },
            "bedtime": {
                "type": "string",
                "description": "Bedtime (HH:MM)."
            },
            "wake_time": {
                "type": "string",
                "description": "Wake time (HH:MM)."
            },
            "duration_hours": {
                "type": "number",
                "description": "Sleep duration in hours."
            },
            "quality": {
                "type": "integer",
                "description": "Sleep quality 1-10.",
                "minimum": 1,
                "maximum": 10
            },
            "sleep_notes": {
                "type": "string",
                "description": "Sleep notes."
            },
            # Vitals fields
            "metric": {
                "type": "string",
                "description": "Vital metric (weight, heart_rate, blood_pressure_sys, blood_pressure_dia, body_fat, steps, etc.)."
            },
            "value": {
                "type": "number",
                "description": "Metric value."
            },
            "unit": {
                "type": "string",
                "description": "Unit (kg, bpm, mmHg, %, steps, etc.)."
            },
            "measured_at": {
                "type": "string",
                "description": "Measurement date/time (YYYY-MM-DD or YYYY-MM-DDTHH:MM). Default now."
            },
            "vitals_notes": {
                "type": "string",
                "description": "Vitals notes."
            },
            # Goals fields
            "goal_type": {
                "type": "string",
                "description": "Goal type (workouts_per_week, daily_steps, daily_calories, daily_water, sleep_hours, weight, etc.)."
            },
            "target_value": {
                "type": "number",
                "description": "Target value for the goal."
            },
            "goal_unit": {
                "type": "string",
                "description": "Goal unit."
            },
            "period": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly"],
                "description": "Goal period."
            },
            "start_date": {
                "type": "string",
                "description": "Goal start date (YYYY-MM-DD). Default today."
            },
            "end_date": {
                "type": "string",
                "description": "Goal end date (YYYY-MM-DD)."
            },
            # Common fields
            "date_from": {
                "type": "string",
                "description": "Start date for summaries (YYYY-MM-DD)."
            },
            "date_to": {
                "type": "string",
                "description": "End date for summaries (YYYY-MM-DD)."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results. Default 50."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "log_workout":
                return self._log_workout(**kwargs)
            elif action == "list_workouts":
                return self._list_workouts(**kwargs)
            elif action == "log_nutrition":
                return self._log_nutrition(**kwargs)
            elif action == "list_nutrition":
                return self._list_nutrition(**kwargs)
            elif action == "log_sleep":
                return self._log_sleep(**kwargs)
            elif action == "list_sleep":
                return self._list_sleep(**kwargs)
            elif action == "log_vitals":
                return self._log_vitals(**kwargs)
            elif action == "list_vitals":
                return self._list_vitals(**kwargs)
            elif action == "set_goal":
                return self._set_goal(**kwargs)
            elif action == "list_goals":
                return self._list_goals(**kwargs)
            elif action == "summary":
                return self._get_summary(**kwargs)
            elif action == "progress":
                return self._get_progress(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # Workout methods
    def _log_workout(self, activity_type: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        workout_date = kwargs.get("workout_date", date.today().isoformat())
        cursor.execute("""
            INSERT INTO workouts (activity_type, duration_minutes, distance_km, calories_burned, intensity, notes, workout_date, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            activity_type,
            kwargs.get("duration_minutes"),
            kwargs.get("distance_km"),
            kwargs.get("calories_burned"),
            kwargs.get("intensity"),
            kwargs.get("workout_notes"),
            workout_date,
            kwargs.get("start_time"),
            kwargs.get("end_time")
        ))
        workout_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Workout logged: {activity_type} ({kwargs.get('duration_minutes', 0)} min)",
            metadata={"workout_id": workout_id, "activity_type": activity_type, "date": workout_date}
        )

    def _list_workouts(self, date_from: str = None, date_to: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM workouts WHERE 1=1"
        params = []
        if date_from:
            query += " AND workout_date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND workout_date <= ?"
            params.append(date_to)
        query += " ORDER BY workout_date DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No workouts logged", metadata={"workouts": []})

        lines = [f"🏃 Found {len(rows)} workout(s):"]
        total_min = sum(r["duration_minutes"] or 0 for r in rows)
        total_cal = sum(r["calories_burned"] or 0 for r in rows)
        total_dist = sum(r["distance_km"] or 0 for r in rows)
        for r in rows:
            dist = f" {r['distance_km']:.1f}km" if r["distance_km"] else ""
            cal = f" {r['calories_burned']}cal" if r["calories_burned"] else ""
            inten = f" [{r['intensity']}]" if r["intensity"] else ""
            lines.append(f"  • {r['workout_date']}: {r['activity_type']} {r['duration_minutes']}min{dist}{cal}{inten}")
        lines.append(f"\n  Total: {total_min} min, {total_cal} cal, {total_dist:.1f} km")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"workouts": rows, "count": len(rows), "total_minutes": total_min, "total_calories": total_cal}
        )

    # Nutrition methods
    def _log_nutrition(self, meal_type: str, food_items: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        nutrition_date = kwargs.get("nutrition_date", date.today().isoformat())
        cursor.execute("""
            INSERT INTO nutrition (meal_type, food_items, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, water_ml, nutrition_date, meal_time, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            meal_type, food_items,
            kwargs.get("calories"),
            kwargs.get("protein_g"),
            kwargs.get("carbs_g"),
            kwargs.get("fat_g"),
            kwargs.get("fiber_g"),
            kwargs.get("sugar_g"),
            kwargs.get("water_ml"),
            nutrition_date,
            kwargs.get("meal_time"),
            kwargs.get("nutrition_notes")
        ))
        nutrition_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Meal logged: {meal_type} - {food_items}",
            metadata={"nutrition_id": nutrition_id, "meal_type": meal_type, "date": nutrition_date}
        )

    def _list_nutrition(self, date_from: str = None, date_to: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM nutrition WHERE 1=1"
        params = []
        if date_from:
            query += " AND nutrition_date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND nutrition_date <= ?"
            params.append(date_to)
        query += " ORDER BY nutrition_date DESC, meal_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No nutrition entries", metadata={"nutrition": []})

        lines = [f"🍽️ Found {len(rows)} meal(s):"]
        total_cal = sum(r["calories"] or 0 for r in rows)
        total_protein = sum(r["protein_g"] or 0 for r in rows)
        total_water = sum(r["water_ml"] or 0 for r in rows)
        for r in rows:
            cal = f" {r['calories']}cal" if r["calories"] else ""
            prot = f" {r['protein_g']}g protein" if r["protein_g"] else ""
            lines.append(f"  • {r['nutrition_date']} {r['meal_type']}: {r['food_items']}{cal}{prot}")
        lines.append(f"\n  Total: {total_cal} cal, {total_protein:.1f}g protein, {total_water}ml water")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"nutrition": rows, "count": len(rows), "total_calories": total_cal, "total_protein": total_protein}
        )

    # Sleep methods
    def _log_sleep(self, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        sleep_date = kwargs.get("sleep_date", date.today().isoformat())

        # Auto-calculate duration if bedtime and wake_time provided
        duration = kwargs.get("duration_hours")
        if not duration and kwargs.get("bedtime") and kwargs.get("wake_time"):
            try:
                bed = datetime.strptime(f"{sleep_date} {kwargs['bedtime']}", "%Y-%m-%d %H:%M")
                wake = datetime.strptime(f"{sleep_date} {kwargs['wake_time']}", "%Y-%m-%d %H:%M")
                if wake < bed:
                    wake += timedelta(days=1)
                duration = (wake - bed).total_seconds() / 3600
            except (ValueError, KeyError) as e:
                # Silent: invalid time format, skip auto-calculation
                import logging
                logging.getLogger(__name__).debug("Failed to auto-calculate sleep duration: %s", e)

        cursor.execute("""
            INSERT INTO sleep (sleep_date, bedtime, wake_time, duration_hours, quality, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sleep_date,
            kwargs.get("bedtime"),
            kwargs.get("wake_time"),
            duration,
            kwargs.get("quality"),
            kwargs.get("sleep_notes")
        ))
        sleep_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Sleep logged: {sleep_date} ({duration:.1f}h)" if duration else f"Sleep logged: {sleep_date}",
            metadata={"sleep_id": sleep_id, "date": sleep_date, "duration": duration}
        )

    def _list_sleep(self, date_from: str = None, date_to: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM sleep WHERE 1=1"
        params = []
        if date_from:
            query += " AND sleep_date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND sleep_date <= ?"
            params.append(date_to)
        query += " ORDER BY sleep_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No sleep entries", metadata={"sleep": []})

        lines = [f"😴 Found {len(rows)} sleep record(s):"]
        total_hours = sum(r["duration_hours"] or 0 for r in rows)
        avg_quality = sum(r["quality"] or 0 for r in rows) / len([r for r in rows if r["quality"]]) if rows else 0
        for r in rows:
            dur = f" {r['duration_hours']:.1f}h" if r["duration_hours"] else ""
            qual = f" ⭐{r['quality']}/10" if r["quality"] else ""
            bed = f" {r['bedtime']}" if r["bedtime"] else ""
            wake = f" {r['wake_time']}" if r["wake_time"] else ""
            lines.append(f"  • {r['sleep_date']}{dur}{qual}{bed}~{wake}")
        lines.append(f"\n  Avg: {total_hours/len(rows):.1f}h/night, Quality: {avg_quality:.1f}/10")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"sleep": rows, "count": len(rows), "avg_duration": total_hours/len(rows), "avg_quality": avg_quality}
        )

    # Vitals methods
    def _log_vitals(self, metric: str, value: float, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        measured_at = kwargs.get("measured_at", datetime.now().isoformat())
        cursor.execute("""
            INSERT INTO vitals (metric, value, unit, measured_at, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            metric, value,
            kwargs.get("unit"),
            measured_at,
            kwargs.get("vitals_notes")
        ))
        vital_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Vital logged: {metric} = {value} {kwargs.get('unit', '')}",
            metadata={"vital_id": vital_id, "metric": metric, "value": value}
        )

    def _list_vitals(self, metric: str = None, date_from: str = None, date_to: str = None, limit: int = 50) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM vitals WHERE 1=1"
        params = []
        if metric:
            query += " AND metric = ?"
            params.append(metric)
        if date_from:
            query += " AND measured_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND measured_at <= ?"
            params.append(date_to)
        query += " ORDER BY measured_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No vitals recorded", metadata={"vitals": []})

        lines = [f"💓 Found {len(rows)} vital record(s):"]
        for r in rows:
            measured = r["measured_at"][:16].replace("T", " ")
            lines.append(f"  • {measured}: {r['metric']} = {r['value']} {r['unit'] or ''}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"vitals": rows, "count": len(rows)}
        )

    # Goals methods
    def _set_goal(self, goal_type: str, target_value: float, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        start = kwargs.get("start_date", date.today().isoformat())
        cursor.execute("""
            INSERT INTO goals (goal_type, target_value, unit, period, start_date, end_date, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            goal_type, target_value,
            kwargs.get("goal_unit"),
            kwargs.get("period", "daily"),
            start,
            kwargs.get("end_date"),
            1
        ))
        goal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Goal set: {goal_type} = {target_value} {kwargs.get('goal_unit', '')} ({kwargs.get('period', 'daily')})",
            metadata={"goal_id": goal_id, "goal_type": goal_type, "target": target_value}
        )

    def _list_goals(self, active_only: bool = True, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM goals WHERE 1=1"
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY created_at DESC"

        cursor.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No goals set", metadata={"goals": []})

        lines = [f"🎯 Found {len(rows)} goal(s):"]
        for r in rows:
            status = "✅" if r["is_active"] else "⏸️"
            lines.append(f"  {status} {r['goal_type']}: {r['target_value']} {r['unit'] or ''} / {r['period']} (since {r['start_date']})")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"goals": rows, "count": len(rows)}
        )

    # Summary & Progress
    def _get_summary(self, date_from: str = None, date_to: str = None, **kwargs) -> ToolResult:
        if not date_from:
            date_from = (date.today() - timedelta(days=7)).isoformat()
        if not date_to:
            date_to = date.today().isoformat()

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Workouts this week
        cursor.execute("""
            SELECT COUNT(*) as count, SUM(duration_minutes) as total_min, SUM(calories_burned) as total_cal
            FROM workouts WHERE workout_date BETWEEN ? AND ?
        """, (date_from, date_to))
        w = cursor.fetchone()

        # Nutrition this week
        cursor.execute("""
            SELECT SUM(calories) as total_cal, SUM(protein_g) as total_prot, SUM(water_ml) as total_water
            FROM nutrition WHERE nutrition_date BETWEEN ? AND ?
        """, (date_from, date_to))
        n = cursor.fetchone()

        # Sleep this week
        cursor.execute("""
            SELECT AVG(duration_hours) as avg_dur, AVG(quality) as avg_qual
            FROM sleep WHERE sleep_date BETWEEN ? AND ?
        """, (date_from, date_to))
        s = cursor.fetchone()

        # Latest vitals
        cursor.execute("""
            SELECT metric, value, unit, measured_at FROM vitals
            WHERE measured_at >= ? ORDER BY measured_at DESC
        """, (date_from,))
        vitals = cursor.fetchall()

        conn.close()

        lines = [f"📊 Health Summary ({date_from} to {date_to}):"]
        lines.append(f"\n  🏃 Workouts: {w['count'] or 0} sessions, {w['total_min'] or 0} min, {w['total_cal'] or 0} cal")
        lines.append(f"  🍽️ Nutrition: {n['total_cal'] or 0} cal, {n['total_prot'] or 0:.1f}g protein, {n['total_water'] or 0}ml water")
        lines.append(f"  😴 Sleep: {s['avg_dur'] or 0:.1f}h avg, Quality {s['avg_qual'] or 0:.1f}/10")

        if vitals:
            lines.append("\n  💓 Latest Vitals:")
            for v in vitals[:5]:
                lines.append(f"    {v[0]}: {v[1]} {v[2]} ({v[3][:10]})")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"workouts": w, "nutrition": n, "sleep": s, "vitals": vitals}
        )

    def _get_progress(self, date_from: str = None, date_to: str = None, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM goals WHERE is_active = 1")
        goals = [dict(row) for row in cursor.fetchall()]

        if not goals:
            conn.close()
            return ToolResult(success=True, output="No active goals to track", metadata={"goals": []})

        lines = ["📈 Goal Progress:"]
        for g in goals:
            gtype = g["goal_type"]
            target = g["target_value"]
            period = g["period"]

            # Calculate current progress based on goal type
            if period == "daily":
                check_date = date.today().isoformat()
            elif period == "weekly":
                check_date = (date.today() - timedelta(days=date.today().weekday())).isoformat()
            else:
                check_date = date.today().replace(day=1).isoformat()

            current = 0
            if gtype == "workouts_per_week":
                cursor.execute("SELECT COUNT(*) FROM workouts WHERE workout_date >= ?", (check_date,))
                current = cursor.fetchone()[0]
            elif gtype == "daily_steps":
                cursor.execute("SELECT SUM(value) FROM vitals WHERE metric = 'steps' AND measured_at >= ?", (check_date,))
                current = cursor.fetchone()[0] or 0
            elif gtype == "daily_calories":
                cursor.execute("SELECT SUM(calories) FROM nutrition WHERE nutrition_date >= ?", (check_date,))
                current = cursor.fetchone()[0] or 0
            elif gtype == "daily_water":
                cursor.execute("SELECT SUM(water_ml) FROM nutrition WHERE nutrition_date >= ?", (check_date,))
                current = cursor.fetchone()[0] or 0
            elif gtype == "sleep_hours":
                cursor.execute("SELECT AVG(duration_hours) FROM sleep WHERE sleep_date >= ?", (check_date,))
                current = cursor.fetchone()[0] or 0
            elif gtype == "weight":
                cursor.execute("SELECT value FROM vitals WHERE metric = 'weight' ORDER BY measured_at DESC LIMIT 1")
                current = cursor.fetchone()[0] or 0

            pct = (current / target * 100) if target > 0 else 0
            status = "🟢" if pct >= 100 else ("🟡" if pct >= 80 else "🔴")
            lines.append(f"  {status} {gtype}: {current:.1f}/{target} {g['unit'] or ''} ({pct:.0f}%)")

        conn.close()
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"goals": goals}
        )


TOOLS_REGISTRY.register(HealthTool())