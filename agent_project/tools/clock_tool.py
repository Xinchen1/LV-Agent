"""Clock tool - current time, timezone conversion, date arithmetic and countdowns (offline)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import BaseTool, ToolResult, TOOLS_REGISTRY

logger = logging.getLogger(__name__)

_ALIASES = {
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "china": "Asia/Shanghai",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo",
    "london": "Europe/London", "uk": "Europe/London",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin",
    "nyc": "America/New_York", "new_york": "America/New_York", "est": "America/New_York",
    "la": "America/Los_Angeles", "los_angeles": "America/Los_Angeles", "pst": "America/Los_Angeles",
    "sydney": "Australia/Sydney", "singapore": "Asia/Singapore", "dubai": "Asia/Dubai",
    "utc": "UTC", "gmt": "UTC",
}


def _zone(name: Optional[str]) -> Optional[ZoneInfo]:
    if not name:
        return None
    key = name.strip().replace(" ", "_").lower()
    target = _ALIASES.get(key, name.strip())
    try:
        return ZoneInfo(target)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown timezone: {name}")


def _parse_dt(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw or raw.lower() == "now":
        return datetime.now().astimezone()
    if raw.lower() == "today":
        return datetime.combine(date.today(), datetime.min.time()).astimezone()
    if raw.lower() == "tomorrow":
        return datetime.combine(date.today() + timedelta(days=1), datetime.min.time()).astimezone()
    if raw.lower() == "yesterday":
        return datetime.combine(date.today() - timedelta(days=1), datetime.min.time()).astimezone()
    cleaned = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError as e:
                logger.debug("clock parse format %s failed: %s", fmt, e)
                continue
    raise ValueError(f"Cannot parse datetime: {value}")


def _human_delta(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return sign + " ".join(parts)


class ClockTool(BaseTool):
    name = "clock"
    description = (
        "Time and date utilities: current time in any timezone, timezone conversion, "
        "date arithmetic, difference between dates, countdowns and business-day counts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["now", "convert", "diff", "add", "countdown", "workdays"],
                "description": "Action to perform.",
            },
            "timezone": {"type": "string", "description": "Timezone for now (e.g. Asia/Shanghai)."},
            "value": {"type": "string", "description": "Datetime (ISO or 'now'/'today')."},
            "start": {"type": "string", "description": "Start datetime."},
            "end": {"type": "string", "description": "End datetime."},
            "from_timezone": {"type": "string", "description": "Source timezone."},
            "to_timezone": {"type": "string", "description": "Target timezone."},
            "days": {"type": "integer", "description": "Days to add."},
            "weeks": {"type": "integer", "description": "Weeks to add."},
            "hours": {"type": "integer", "description": "Hours to add."},
            "minutes": {"type": "integer", "description": "Minutes to add."},
            "target": {"type": "string", "description": "Target datetime for countdown."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "now": self._now,
                "convert": self._convert,
                "diff": self._diff,
                "add": self._add,
                "countdown": self._countdown,
                "workdays": self._workdays,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("clock action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _now(self, timezone: Optional[str] = None, **kwargs) -> ToolResult:
        zone = _zone(timezone)
        now = datetime.now(zone) if zone else datetime.now().astimezone()
        data = {
            "iso": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": str(now.tzinfo),
            "unix": int(now.timestamp()),
            "iso_week": now.isocalendar().week,
        }
        lines = [
            f"{data['iso']} ({data['timezone']})",
            f"weekday: {data['weekday']}, ISO week: {data['iso_week']}",
            f"unix: {data['unix']}",
        ]
        return ToolResult(success=True, output="\n".join(lines), metadata=data)

    def _convert(self, value: str = "", from_timezone: Optional[str] = None,
                 to_timezone: str = "", **kwargs) -> ToolResult:
        if not to_timezone:
            return ToolResult(success=False, output="", error="to_timezone is required")
        dt = _parse_dt(value)
        src = _zone(from_timezone)
        if src is not None:
            dt = dt.replace(tzinfo=src) if dt.tzinfo is None else dt.astimezone(src)
        elif dt.tzinfo is None:
            dt = dt.astimezone()
        converted = dt.astimezone(_zone(to_timezone))
        return ToolResult(
            success=True,
            output=f"{converted.isoformat(timespec='seconds')} ({converted.strftime('%A')})",
            metadata={"iso": converted.isoformat(timespec="seconds"),
                      "source": dt.isoformat(timespec="seconds"),
                      "timezone": str(converted.tzinfo)},
        )

    def _diff(self, start: str = "", end: str = "", **kwargs) -> ToolResult:
        if not start or not end:
            return ToolResult(success=False, output="", error="start and end are required")
        a = _parse_dt(start)
        b = _parse_dt(end)
        if (a.tzinfo is None) != (b.tzinfo is None):
            a = a.astimezone()
            b = b.astimezone()
        delta = b - a
        total = int(delta.total_seconds())
        data = {
            "human": _human_delta(delta),
            "days": delta.days,
            "total_hours": round(total / 3600, 2),
            "total_minutes": round(total / 60, 2),
            "total_seconds": total,
        }
        return ToolResult(success=True,
                          output=f"{data['human']} ({data['total_hours']}h)",
                          metadata=data)

    def _add(self, value: str = "", days: int = 0, weeks: int = 0,
             hours: int = 0, minutes: int = 0, **kwargs) -> ToolResult:
        dt = _parse_dt(value or "now")
        delta = timedelta(days=int(days or 0) + int(weeks or 0) * 7,
                          hours=int(hours or 0), minutes=int(minutes or 0))
        result = dt + delta
        return ToolResult(success=True,
                          output=f"{result.isoformat(timespec='seconds')} ({result.strftime('%A')})",
                          metadata={"iso": result.isoformat(timespec="seconds"),
                                    "weekday": result.strftime("%A")})

    def _countdown(self, target: str = "", **kwargs) -> ToolResult:
        if not target:
            return ToolResult(success=False, output="", error="target is required")
        now = datetime.now().astimezone()
        due = _parse_dt(target)
        if due.tzinfo is None:
            due = due.astimezone()
        delta = due - now
        human = _human_delta(delta)
        verb = "until" if delta.total_seconds() >= 0 else "since"
        return ToolResult(
            success=True,
            output=f"{human} {verb} {due.date().isoformat()}",
            metadata={"human": human, "seconds": int(delta.total_seconds()),
                      "overdue": delta.total_seconds() < 0},
        )

    def _workdays(self, start: str = "", end: str = "", **kwargs) -> ToolResult:
        if not start or not end:
            return ToolResult(success=False, output="", error="start and end are required")
        a = _parse_dt(start).date()
        b = _parse_dt(end).date()
        sign = 1 if b >= a else -1
        current, stop = (a, b) if sign > 0 else (b, a)
        count = 0
        while current <= stop:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return ToolResult(success=True, output=f"{count * sign} business day(s)",
                          metadata={"business_days": count * sign,
                                    "calendar_days": (b - a).days})


TOOLS_REGISTRY.register(ClockTool())
