"""Meals tool - recipes, weekly meal planning and generated shopping lists."""

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

_MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def _connect() -> sqlite3.Connection:
    path = db_path("meals.db")
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
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            ingredients TEXT,
            steps TEXT,
            servings INTEGER DEFAULT 2,
            tags TEXT,
            prep_minutes INTEGER,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS meal_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            recipe_id INTEGER,
            note TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_meal_plans_date ON meal_plans(plan_date);
        """
    )
    conn.commit()


def _parse_day(value: Optional[str], default: Optional[date] = None) -> str:
    if not value:
        return (default or date.today()).isoformat()
    v = value.strip().lower()
    if v == "today":
        return date.today().isoformat()
    if v == "tomorrow":
        return (date.today() + timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(value.strip()).date().isoformat()
    except ValueError:
        return (default or date.today()).isoformat()


def _json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return [v.strip() for v in str(value).split(",") if v.strip()]


class MealsTool(BaseTool):
    name = "meals"
    description = (
        "Recipes and meal planning: save recipes with ingredients, plan meals by date, "
        "show a week plan, and generate an aggregated shopping list."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add_recipe", "list_recipes", "search_recipes", "get_recipe",
                         "delete_recipe", "plan", "plan_week", "shopping_list"],
                "description": "Action to perform.",
            },
            "recipe_id": {"type": "integer", "description": "Recipe ID."},
            "name": {"type": "string", "description": "Recipe name."},
            "ingredients": {
                "type": "array",
                "description": "Ingredient list, e.g. [{\"name\":\"egg\",\"quantity\":2,\"unit\":\"pcs\"}].",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                },
            },
            "steps": {"type": "string", "description": "Cooking steps."},
            "servings": {"type": "integer", "description": "Servings (default 2)."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags."},
            "prep_minutes": {"type": "integer", "description": "Preparation minutes."},
            "query": {"type": "string", "description": "Search keyword."},
            "plan_date": {"type": "string", "description": "Plan date (ISO/today/tomorrow)."},
            "meal_type": {"type": "string", "enum": list(_MEAL_TYPES), "description": "Meal type."},
            "note": {"type": "string", "description": "Plan note when no recipe."},
            "week_start": {"type": "string", "description": "Week start date (default Monday)."},
            "limit": {"type": "integer", "description": "Max rows (default 20)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "add_recipe": self._add_recipe,
                "list_recipes": self._list_recipes,
                "search_recipes": self._search_recipes,
                "get_recipe": self._get_recipe,
                "delete_recipe": self._delete_recipe,
                "plan": self._plan,
                "plan_week": self._plan_week,
                "shopping_list": self._shopping_list,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("meals action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _recipe_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["ingredients"] = _json_list(data.get("ingredients"))
        data["tags"] = _json_list(data.get("tags"))
        return data

    def _resolve_recipe(self, conn: sqlite3.Connection, recipe_id: Optional[int],
                        name: Optional[str]) -> Optional[sqlite3.Row]:
        if recipe_id is not None:
            return conn.execute("SELECT * FROM recipes WHERE id = ?", (int(recipe_id),)).fetchone()
        if name:
            return conn.execute(
                "SELECT * FROM recipes WHERE lower(name) = lower(?)", (name.strip(),)
            ).fetchone()
        return None

    def _add_recipe(self, name: str = "", **kwargs) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, output="", error="name is required")
        ingredients = kwargs.get("ingredients") or []
        if not isinstance(ingredients, list):
            ingredients = _json_list(ingredients)
        conn = _connect()
        try:
            if self._resolve_recipe(conn, None, name):
                return ToolResult(success=False, output="",
                                  error=f"Recipe '{name}' already exists")
            cur = conn.execute(
                """INSERT INTO recipes (name, ingredients, steps, servings, tags, prep_minutes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    name.strip(),
                    json.dumps(ingredients, ensure_ascii=False),
                    kwargs.get("steps", ""),
                    max(1, int(kwargs.get("servings") or 2)),
                    json.dumps(kwargs.get("tags") or [], ensure_ascii=False),
                    kwargs.get("prep_minutes"),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
            recipe_id = cur.lastrowid
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Recipe '#{recipe_id} {name.strip()}' saved",
                          metadata={"recipe_id": recipe_id})

    def _list_recipes(self, limit: int = 20, tag: Optional[str] = None,
                      **kwargs) -> ToolResult:
        conn = _connect()
        try:
            rows = [self._recipe_dict(r) for r in conn.execute(
                "SELECT * FROM recipes ORDER BY name LIMIT ?", (max(1, int(limit or 20)),)
            ).fetchall()]
        finally:
            conn.close()
        if tag:
            rows = [r for r in rows if tag in r["tags"]]
        if not rows:
            return ToolResult(success=True, output="No recipes saved", metadata={"recipes": []})
        lines = [f"{len(rows)} recipe(s):"]
        for r in rows:
            tags = f" #{' #'.join(r['tags'])}" if r["tags"] else ""
            lines.append(f"  #{r['id']} {r['name']} ({r['servings']} servings, "
                         f"{len(r['ingredients'])} ingredients){tags}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"recipes": rows})

    def _search_recipes(self, query: str = "", limit: int = 20, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, output="", error="query is required")
        like = f"%{query}%"
        conn = _connect()
        try:
            rows = [self._recipe_dict(r) for r in conn.execute(
                """SELECT * FROM recipes WHERE name LIKE ? OR ingredients LIKE ? OR steps LIKE ?
                   OR tags LIKE ? ORDER BY name LIMIT ?""",
                (like, like, like, like, max(1, int(limit or 20))),
            ).fetchall()]
        finally:
            conn.close()
        if not rows:
            return ToolResult(success=True, output=f"No recipes matching '{query}'",
                              metadata={"recipes": []})
        return ToolResult(success=True,
                          output="\n".join(f"  #{r['id']} {r['name']}" for r in rows),
                          metadata={"recipes": rows})

    def _get_recipe(self, **kwargs) -> ToolResult:
        recipe_id = kwargs.get("recipe_id")
        name = kwargs.get("name")
        if recipe_id is None and not name:
            return ToolResult(success=False, output="", error="recipe_id or name is required")
        conn = _connect()
        try:
            row = self._resolve_recipe(conn, recipe_id, name)
        finally:
            conn.close()
        if row is None:
            return ToolResult(success=False, output="", error="Recipe not found")
        data = self._recipe_dict(row)
        lines = [f"#{data['id']} {data['name']} ({data['servings']} servings)",
                 "Ingredients:"]
        for ing in data["ingredients"]:
            if isinstance(ing, dict):
                qty = ing.get("quantity", "")
                unit = ing.get("unit", "")
                lines.append(f"  - {qty}{(' ' + unit) if unit else ''} {ing.get('name', '')}".strip())
            else:
                lines.append(f"  - {ing}")
        if data["steps"]:
            lines.extend(["Steps:", data["steps"]])
        return ToolResult(success=True, output="\n".join(lines), metadata={"recipe": data})

    def _delete_recipe(self, recipe_id: Optional[int] = None, **kwargs) -> ToolResult:
        conn = _connect()
        try:
            row = self._resolve_recipe(conn, recipe_id, kwargs.get("name"))
            if row is None:
                return ToolResult(success=False, output="", error="Recipe not found")
            conn.execute("DELETE FROM recipes WHERE id = ?", (row["id"],))
            conn.execute("UPDATE meal_plans SET recipe_id = NULL, note = note WHERE recipe_id = ?",
                         (row["id"],))
            conn.commit()
        finally:
            conn.close()
        return ToolResult(success=True, output=f"Recipe '{row['name']}' deleted",
                          metadata={"recipe_id": row["id"]})

    def _plan(self, meal_type: str = "dinner", **kwargs) -> ToolResult:
        if meal_type not in _MEAL_TYPES:
            return ToolResult(success=False, output="", error=f"meal_type must be one of {_MEAL_TYPES}")
        plan_date = _parse_day(kwargs.get("plan_date"))
        recipe_id = kwargs.get("recipe_id")
        note = kwargs.get("note", "")
        conn = _connect()
        try:
            if recipe_id is not None:
                recipe = self._resolve_recipe(conn, recipe_id, kwargs.get("name"))
                if recipe is None:
                    return ToolResult(success=False, output="", error="Recipe not found")
                recipe_id = recipe["id"]
            elif kwargs.get("name"):
                recipe = self._resolve_recipe(conn, None, kwargs.get("name"))
                if recipe is None:
                    note = note or str(kwargs.get("name"))
                else:
                    recipe_id = recipe["id"]
            existing = conn.execute(
                "SELECT id FROM meal_plans WHERE plan_date = ? AND meal_type = ?",
                (plan_date, meal_type),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE meal_plans SET recipe_id = ?, note = ? WHERE id = ?",
                    (recipe_id, note, existing["id"]),
                )
                plan_id = existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO meal_plans (plan_date, meal_type, recipe_id, note, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (plan_date, meal_type, recipe_id, note,
                     datetime.now().isoformat(timespec="seconds")),
                )
                plan_id = cur.lastrowid
            conn.commit()
            label = conn.execute("SELECT name FROM recipes WHERE id = ?",
                                 (recipe_id,)).fetchone() if recipe_id else None
        finally:
            conn.close()
        shown = label["name"] if label else (note or "meal")
        return ToolResult(success=True, output=f"Planned {meal_type} on {plan_date}: {shown}",
                          metadata={"plan_id": plan_id, "plan_date": plan_date,
                                    "meal_type": meal_type, "recipe_id": recipe_id})

    def _week_rows(self, week_start: Optional[str]) -> List[Dict[str, Any]]:
        start = date.today()
        if week_start:
            start = datetime.fromisoformat(_parse_day(week_start)).date()
        else:
            start = start - timedelta(days=start.weekday())
        end = start + timedelta(days=6)
        conn = _connect()
        try:
            rows = conn.execute(
                """SELECT p.*, r.name AS recipe_name FROM meal_plans p
                   LEFT JOIN recipes r ON r.id = p.recipe_id
                   WHERE p.plan_date BETWEEN ? AND ? ORDER BY p.plan_date, p.meal_type""",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows], start, end

    def _plan_week(self, week_start: Optional[str] = None, **kwargs) -> ToolResult:
        rows, start, end = self._week_rows(week_start)
        if not rows:
            return ToolResult(success=True, output=f"No meals planned for {start}..{end}",
                              metadata={"week_start": start.isoformat(), "week_end": end.isoformat(),
                                        "plans": []})
        lines = [f"Meal plan {start} .. {end}:"]
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_day.setdefault(row["plan_date"], []).append(row)
        for day, plans in by_day.items():
            parts = [f"{p['meal_type']}={p['recipe_name'] or p['note'] or '-'}" for p in plans]
            lines.append(f"  {day}: " + ", ".join(parts))
        return ToolResult(success=True, output="\n".join(lines), metadata={
            "week_start": start.isoformat(), "week_end": end.isoformat(), "plans": rows,
        })

    def _shopping_list(self, week_start: Optional[str] = None, **kwargs) -> ToolResult:
        rows, start, end = self._week_rows(week_start)
        if not rows:
            return ToolResult(success=True, output="No planned meals to aggregate",
                              metadata={"items": []})
        conn = _connect()
        try:
            recipe_ids = [r["recipe_id"] for r in rows if r["recipe_id"]]
            recipes = {}
            for rid in set(recipe_ids):
                row = conn.execute("SELECT * FROM recipes WHERE id = ?", (rid,)).fetchone()
                if row:
                    recipes[rid] = self._recipe_dict(row)
        finally:
            conn.close()
        aggregated: Dict[tuple, float] = {}
        for row in rows:
            recipe = recipes.get(row["recipe_id"])
            if not recipe:
                continue
            for ing in recipe["ingredients"]:
                if isinstance(ing, str):
                    ing = {"name": ing, "quantity": 1, "unit": ""}
                name = str(ing.get("name", "")).strip()
                if not name:
                    continue
                unit = str(ing.get("unit", "")).strip()
                try:
                    quantity = float(ing.get("quantity", 1))
                except (TypeError, ValueError):
                    quantity = 1.0
                key = (name.lower(), unit)
                aggregated[key] = aggregated.get(key, 0.0) + quantity
        if not aggregated:
            return ToolResult(success=True, output="No ingredients found in planned recipes",
                              metadata={"items": []})
        items = [
            {"name": name, "unit": unit, "quantity": round(qty, 2)}
            for (name, unit), qty in sorted(aggregated.items())
        ]
        lines = [f"Shopping list for {start} .. {end}:"]
        for item in items:
            lines.append(f"  - {item['name']}: {item['quantity']:g}"
                         f"{(' ' + item['unit']) if item['unit'] else ''}")
        return ToolResult(success=True, output="\n".join(lines), metadata={"items": items})


TOOLS_REGISTRY.register(MealsTool())
