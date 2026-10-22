"""
Travel Planning Tool - Personal assistant trip planning with itineraries, bookings, and packing lists.
"""

import json
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .assist_data import db_path


DB_PATH = db_path("travel.db")


def _init_db():
    """Initialize the travel database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            destination TEXT,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'planning',
            budget REAL,
            currency TEXT DEFAULT 'CNY',
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS itinerary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id INTEGER NOT NULL,
            day INTEGER NOT NULL,
            date TEXT NOT NULL,
            title TEXT,
            description TEXT,
            location TEXT,
            start_time TEXT,
            end_time TEXT,
            transport TEXT,
            accommodation TEXT,
            cost REAL,
            currency TEXT DEFAULT 'CNY',
            confirmation TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS packing_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            category TEXT,
            quantity INTEGER DEFAULT 1,
            is_packed INTEGER DEFAULT 0,
            is_essential INTEGER DEFAULT 0,
            notes TEXT,
            FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id INTEGER NOT NULL,
            type TEXT CHECK(type IN ('flight', 'hotel', 'train', 'rental', 'activity', 'restaurant', 'other')),
            provider TEXT,
            reference TEXT,
            details TEXT,
            cost REAL,
            currency TEXT DEFAULT 'CNY',
            booking_date TEXT,
            status TEXT DEFAULT 'confirmed',
            FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_itinerary_trip ON itinerary(trip_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_packing_trip ON packing_list(trip_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bookings_trip ON bookings(trip_id)")
    conn.commit()
    conn.close()


class TravelTool(BaseTool):
    name = "travel"
    description = (
        "Plan and manage trips: create trips, build day-by-day itineraries, manage packing lists, "
        "track bookings (flights, hotels, etc.), and monitor budgets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_trip", "read_trip", "update_trip", "delete_trip", "list_trips",
                         "add_itinerary", "list_itinerary", "add_packing", "list_packing", "toggle_packing",
                         "add_booking", "list_bookings", "trip_summary"],
                "description": "Action to perform on travel plans."
            },
            "trip_id": {
                "type": "integer",
                "description": "Trip ID for trip-specific actions."
            },
            "name": {
                "type": "string",
                "description": "Trip name (required for create_trip)."
            },
            "destination": {
                "type": "string",
                "description": "Destination city/country."
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format (required for create_trip)."
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format (required for create_trip)."
            },
            "status": {
                "type": "string",
                "enum": ["planning", "booked", "in_progress", "completed", "cancelled"],
                "description": "Trip status."
            },
            "budget": {
                "type": "number",
                "description": "Total budget for the trip."
            },
            "currency": {
                "type": "string",
                "description": "Currency code. Default CNY."
            },
            "notes": {
                "type": "string",
                "description": "Trip notes."
            },
            # Itinerary fields
            "day": {
                "type": "integer",
                "description": "Day number (1-based) for itinerary items."
            },
            "itinerary_date": {
                "type": "string",
                "description": "Date for itinerary item (YYYY-MM-DD)."
            },
            "itinerary_title": {
                "type": "string",
                "description": "Itinerary item title."
            },
            "itinerary_description": {
                "type": "string",
                "description": "Itinerary item description."
            },
            "location": {
                "type": "string",
                "description": "Location for itinerary item."
            },
            "start_time": {
                "type": "string",
                "description": "Start time in HH:MM format."
            },
            "end_time": {
                "type": "string",
                "description": "End time in HH:MM format."
            },
            "transport": {
                "type": "string",
                "description": "Transport method."
            },
            "accommodation": {
                "type": "string",
                "description": "Accommodation details."
            },
            "cost": {
                "type": "number",
                "description": "Cost for this item."
            },
            "confirmation": {
                "type": "string",
                "description": "Confirmation number/details."
            },
            # Packing fields
            "item": {
                "type": "string",
                "description": "Packing item name."
            },
            "category": {
                "type": "string",
                "description": "Item category (clothing, electronics, documents, etc.)."
            },
            "quantity": {
                "type": "integer",
                "description": "Quantity needed. Default 1."
            },
            "is_essential": {
                "type": "boolean",
                "description": "Whether item is essential. Default false."
            },
            # Booking fields
            "booking_type": {
                "type": "string",
                "enum": ["flight", "hotel", "train", "rental", "activity", "restaurant", "other"],
                "description": "Booking type."
            },
            "provider": {
                "type": "string",
                "description": "Booking provider (airline, hotel name, etc.)."
            },
            "reference": {
                "type": "string",
                "description": "Booking reference/confirmation number."
            },
            "details": {
                "type": "string",
                "description": "Booking details."
            },
            "booking_status": {
                "type": "string",
                "enum": ["confirmed", "pending", "cancelled"],
                "description": "Booking status."
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        _init_db()

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "create_trip":
                return self._create_trip(**kwargs)
            elif action == "read_trip":
                return self._read_trip(**kwargs)
            elif action == "update_trip":
                return self._update_trip(**kwargs)
            elif action == "delete_trip":
                return self._delete_trip(**kwargs)
            elif action == "list_trips":
                return self._list_trips(**kwargs)
            elif action == "add_itinerary":
                return self._add_itinerary(**kwargs)
            elif action == "list_itinerary":
                return self._list_itinerary(**kwargs)
            elif action == "add_packing":
                return self._add_packing(**kwargs)
            elif action == "list_packing":
                return self._list_packing(**kwargs)
            elif action == "toggle_packing":
                return self._toggle_packing(**kwargs)
            elif action == "add_booking":
                return self._add_booking(**kwargs)
            elif action == "list_bookings":
                return self._list_bookings(**kwargs)
            elif action == "trip_summary":
                return self._trip_summary(**kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # Trip management
    def _create_trip(self, name: str, start_date: str, end_date: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO trips (name, destination, start_date, end_date, status, budget, currency, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            kwargs.get("destination"),
            start_date,
            end_date,
            kwargs.get("status", "planning"),
            kwargs.get("budget"),
            kwargs.get("currency", "CNY"),
            kwargs.get("notes"),
            now,
            now
        ))
        trip_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Trip created with ID: {trip_id}",
            metadata={"trip_id": trip_id, "name": name, "start_date": start_date, "end_date": end_date}
        )

    def _read_trip(self, trip_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trips WHERE id = ?", (trip_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")
        return ToolResult(
            success=True,
            output=json.dumps(dict(row), indent=2, ensure_ascii=False),
            metadata={"trip": dict(row)}
        )

    def _update_trip(self, trip_id: int, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trips WHERE id = ?", (trip_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")

        updates = []
        values = []
        for field in ["name", "destination", "start_date", "end_date", "status", "budget", "currency", "notes"]:
            if field in kwargs:
                updates.append(f"{field} = ?")
                values.append(kwargs[field])

        if not updates:
            conn.close()
            return ToolResult(success=False, output="", error="No fields to update")

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(trip_id)

        cursor.execute(f"UPDATE trips SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ToolResult(success=True, output=f"Trip {trip_id} updated", metadata={"trip_id": trip_id})

    def _delete_trip(self, trip_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trips WHERE id = ?", (trip_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")
        return ToolResult(success=True, output=f"Trip {trip_id} deleted", metadata={"trip_id": trip_id})

    def _list_trips(self) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trips ORDER BY start_date DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No trips planned", metadata={"trips": []})

        lines = [f"🗺️ Found {len(rows)} trip(s):"]
        for r in rows:
            status_icon = {"planning": "📝", "booked": "✅", "in_progress": "🚀", "completed": "✨", "cancelled": "❌"}.get(r["status"], "?")
            budget = f" | Budget: ¥{r['budget']:.2f}" if r["budget"] else ""
            lines.append(f"  {status_icon} #{r['id']}: {r['name']} → {r['destination']} | {r['start_date']}~{r['end_date']}{budget}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"trips": rows, "count": len(rows)}
        )

    # Itinerary management
    def _add_itinerary(self, trip_id: int, day: int = None, itinerary_date: str = None, **kwargs) -> ToolResult:
        try:
            day = int(day) if day is not None else 1
        except (TypeError, ValueError):
            day = 1
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, start_date FROM trips WHERE id = ?", (trip_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")
        # itinerary_date 未给时按 trip 出发日 + (day-1) 推导
        if not itinerary_date:
            try:
                start = datetime.strptime(row[1][:10], "%Y-%m-%d")
                itinerary_date = (start + timedelta(days=day - 1)).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                itinerary_date = kwargs.get("trip_date")

        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO itinerary (trip_id, day, date, title, description, location, start_time, end_time, transport, accommodation, cost, currency, confirmation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trip_id, day, itinerary_date,
            kwargs.get("itinerary_title"),
            kwargs.get("itinerary_description"),
            kwargs.get("location"),
            kwargs.get("start_time"),
            kwargs.get("end_time"),
            kwargs.get("transport"),
            kwargs.get("accommodation"),
            kwargs.get("cost"),
            kwargs.get("currency", "CNY"),
            kwargs.get("confirmation"),
            now
        ))
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Itinerary item added for Day {day}",
            metadata={"item_id": item_id, "trip_id": trip_id, "day": day}
        )

    def _list_itinerary(self, trip_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM itinerary WHERE trip_id = ? ORDER BY day, start_time", (trip_id,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No itinerary items yet", metadata={"itinerary": []})

        lines = [f"📅 Itinerary for Trip #{trip_id}:"]
        current_day = None
        for r in rows:
            if r["day"] != current_day:
                current_day = r["day"]
                lines.append(f"\n  📍 Day {current_day} ({r['date']}):")
            time_str = f" {r['start_time']}~{r['end_time']}" if r["start_time"] and r["end_time"] else ""
            loc = f" @ {r['location']}" if r["location"] else ""
            cost = f" ¥{r['cost']:.2f}" if r["cost"] else ""
            lines.append(f"    • {r['title']}{time_str}{loc}{cost}")
            if r["description"]:
                lines.append(f"      {r['description']}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"itinerary": rows, "count": len(rows)}
        )

    # Packing list
    def _add_packing(self, trip_id: int, item: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM trips WHERE id = ?", (trip_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")

        cursor.execute("""
            INSERT INTO packing_list (trip_id, item, category, quantity, is_packed, is_essential, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trip_id, item,
            kwargs.get("category"),
            kwargs.get("quantity", 1),
            0,
            kwargs.get("is_essential", False),
            kwargs.get("notes")
        ))
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Packing item added: {item}",
            metadata={"item_id": item_id, "trip_id": trip_id}
        )

    def _list_packing(self, trip_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM packing_list WHERE trip_id = ? ORDER BY category, item", (trip_id,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="Packing list is empty", metadata={"packing": []})

        lines = [f"🎒 Packing List for Trip #{trip_id}:"]
        current_cat = None
        packed_count = 0
        for r in rows:
            if r["category"] != current_cat:
                current_cat = r["category"] or "Other"
                lines.append(f"\n  📦 {current_cat}:")
            status = "✅" if r["is_packed"] else "☐"
            essential = " ⭐" if r["is_essential"] else ""
            qty = f" x{r['quantity']}" if r["quantity"] > 1 else ""
            if r["is_packed"]:
                packed_count += 1
            lines.append(f"    {status} {r['item']}{qty}{essential}")
        total = len(rows)
        lines.append(f"\n  Progress: {packed_count}/{total} packed")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"packing": rows, "packed": packed_count, "total": total}
        )

    def _toggle_packing(self, trip_id: int, item: str) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, is_packed FROM packing_list WHERE trip_id = ? AND item = ?", (trip_id, item))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return ToolResult(success=False, output="", error=f"Item '{item}' not found in trip {trip_id}")

        new_status = 0 if row["is_packed"] else 1
        cursor.execute("UPDATE packing_list SET is_packed = ? WHERE id = ?", (new_status, row["id"]))
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Item '{item}' marked as {'packed' if new_status else 'unpacked'}",
            metadata={"item": item, "is_packed": bool(new_status)}
        )

    # Bookings
    def _add_booking(self, trip_id: int, booking_type: str, provider: str, **kwargs) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM trips WHERE id = ?", (trip_id,))
        if not cursor.fetchone():
            conn.close()
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")

        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO bookings (trip_id, type, provider, reference, details, cost, currency, booking_date, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trip_id, booking_type, provider,
            kwargs.get("reference"),
            kwargs.get("details"),
            kwargs.get("cost"),
            kwargs.get("currency", "CNY"),
            date.today().isoformat(),
            kwargs.get("booking_status", "confirmed")
        ))
        booking_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return ToolResult(
            success=True,
            output=f"Booking added: {booking_type} with {provider}",
            metadata={"booking_id": booking_id, "trip_id": trip_id}
        )

    def _list_bookings(self, trip_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bookings WHERE trip_id = ? ORDER BY booking_date", (trip_id,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not rows:
            return ToolResult(success=True, output="No bookings yet", metadata={"bookings": []})

        lines = [f"🎫 Bookings for Trip #{trip_id}:"]
        type_icons = {"flight": "✈️", "hotel": "🏨", "train": "🚂", "rental": "🚗", "activity": "🎯", "restaurant": "🍽️", "other": "📋"}
        for r in rows:
            icon = type_icons.get(r["type"], "📋")
            status = "✅" if r["status"] == "confirmed" else ("⏳" if r["status"] == "pending" else "❌")
            cost = f" ¥{r['cost']:.2f}" if r["cost"] else ""
            ref = f" #{r['reference']}" if r["reference"] else ""
            lines.append(f"  {icon} {status} {r['type'].title()}: {r['provider']}{ref}{cost}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"bookings": rows, "count": len(rows)}
        )

    # Trip summary
    def _trip_summary(self, trip_id: int) -> ToolResult:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trips WHERE id = ?", (trip_id,))
        trip = cursor.fetchone()
        if not trip:
            conn.close()
            return ToolResult(success=False, output="", error=f"Trip {trip_id} not found")

        cursor.execute("SELECT * FROM itinerary WHERE trip_id = ? ORDER BY day", (trip_id,))
        itinerary = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM packing_list WHERE trip_id = ?", (trip_id,))
        packing = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM bookings WHERE trip_id = ?", (trip_id,))
        bookings = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT SUM(cost) as total FROM bookings WHERE trip_id = ?", (trip_id,))
        total_booked = cursor.fetchone()["total"] or 0

        conn.close()

        lines = [f"🗺️ Trip Summary: {trip['name']}"]
        lines.append(f"  📍 Destination: {trip['destination']}")
        lines.append(f"  📅 Dates: {trip['start_date']} ~ {trip['end_date']} ({(datetime.fromisoformat(trip['end_date']) - datetime.fromisoformat(trip['start_date'])).days + 1} days)")
        lines.append(f"  📊 Status: {trip['status']}")
        if trip["budget"]:
            lines.append(f"  💰 Budget: ¥{trip['budget']:.2f} {trip['currency']} | Booked: ¥{total_booked:.2f} | Remaining: ¥{trip['budget'] - total_booked:.2f}")

        lines.append(f"\n  📅 Itinerary: {len(itinerary)} items")
        lines.append(f"  🎒 Packing: {len(packing)} items ({sum(1 for p in packing if p['is_packed'])}/{len(packing)} packed)")
        lines.append(f"  🎫 Bookings: {len(bookings)}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"trip": dict(trip), "itinerary_count": len(itinerary), "packing_count": len(packing), "bookings_count": len(bookings), "total_booked": total_booked}
        )


TOOLS_REGISTRY.register(TravelTool())