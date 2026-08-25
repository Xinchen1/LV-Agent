"""
Weather tool - fetch current weather and forecast via Open-Meteo (no API key needed).
"""
from __future__ import annotations

import json
import ssl
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional

from . import BaseTool, ToolResult

# Prefer requests/certifi if available for better SSL handling on macOS.
try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl.create_default_context()


# WMO Weather interpretation codes (simplified)
_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _wmo_description(code: int) -> str:
    return _WMO_CODES.get(code, f"weather code {code}")


def _http_get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    if _HAS_REQUESTS:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "OpenMythosAgent/1.0"})
        resp.raise_for_status()
        return resp.json()

    req = urllib.request.Request(url, headers={"User-Agent": "OpenMythosAgent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except ssl.SSLCertVerificationError:
        # Fallback: retry without certificate verification
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _geocode(location: str) -> Optional[Dict[str, Any]]:
    query = urllib.parse.urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{query}"
    data = _http_get_json(url)
    results = data.get("results") or []
    return results[0] if results else None


def _fetch_weather(lat: float, lon: float, days: int = 3) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
        "timezone": "auto",
        "forecast_days": max(1, min(7, days)),
    }
    query = urllib.parse.urlencode(params)
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    return _http_get_json(url)


class WeatherTool(BaseTool):
    name = "weather"
    description = (
        "Get current weather and a short forecast for a location. "
        "Examples: location='Beijing', location='Shanghai', location='New York'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or place name in English or Chinese.",
            },
            "days": {
                "type": "integer",
                "description": "Number of forecast days (1-7). Default 3.",
                "default": 3,
            },
        },
        "required": ["location"],
    }

    def execute(self, location: str, days: int = 3, **kwargs) -> ToolResult:
        try:
            place = _geocode(location)
            if not place:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Could not find location: {location}",
                )

            lat = place.get("latitude")
            lon = place.get("longitude")
            name = place.get("name", location)
            country = place.get("country", "")
            admin = place.get("admin1", "")
            display_name = ", ".join(p for p in [name, admin, country] if p)

            data = _fetch_weather(lat, lon, days)
            current = data.get("current", {})
            daily = data.get("daily", {})

            lines = [f"Weather for {display_name}", ""]
            lines.append("Current:")
            lines.append(
                f"  Temperature: {current.get('temperature_2m', '?')}°C"
            )
            lines.append(
                f"  Condition: {_wmo_description(current.get('weather_code', -1))}"
            )
            lines.append(
                f"  Humidity: {current.get('relative_humidity_2m', '?')}%"
            )
            lines.append(
                f"  Wind: {current.get('wind_speed_10m', '?')} km/h"
            )

            if daily and daily.get("time"):
                lines.append("")
                lines.append("Forecast:")
                for i, date in enumerate(daily["time"]):
                    max_t = daily.get("temperature_2m_max", [])
                    min_t = daily.get("temperature_2m_min", [])
                    code = daily.get("weather_code", [])
                    lines.append(
                        f"  {date}: {min_t[i] if i < len(min_t) else '?'}°C ~ "
                        f"{max_t[i] if i < len(max_t) else '?'}°C, "
                        f"{_wmo_description(code[i] if i < len(code) else -1)}"
                    )

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"location": display_name, "latitude": lat, "longitude": lon},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
