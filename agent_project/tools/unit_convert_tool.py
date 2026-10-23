"""Unit conversion tool - offline length/mass/temperature/volume/area/speed/data/time/energy."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from . import BaseTool, ToolResult, TOOLS_REGISTRY

logger = logging.getLogger(__name__)

# factor = value in base unit
_UNITS: Dict[str, Dict[str, float]] = {
    "length": {
        "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "um": 1e-6, "nm": 1e-9,
        "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254, "nmi": 1852.0,
    },
    "mass": {
        "kg": 1.0, "g": 0.001, "mg": 1e-6, "t": 1000.0, "lb": 0.45359237,
        "oz": 0.028349523125, "st": 6.35029318,
    },
    "volume": {
        "l": 1.0, "ml": 0.001, "m3": 1000.0, "gal": 3.785411784, "qt": 0.946352946,
        "pt": 0.473176473, "cup": 0.2365882365, "floz": 0.0295735295625,
        "tbsp": 0.01478676478125, "tsp": 0.00492892159375,
    },
    "area": {
        "m2": 1.0, "km2": 1e6, "cm2": 1e-4, "ha": 10000.0, "acre": 4046.8564224,
        "ft2": 0.09290304, "in2": 0.00064516,
    },
    "speed": {
        "mps": 1.0, "kph": 1 / 3.6, "mph": 0.44704, "knot": 0.514444, "fps": 0.3048,
    },
    "data": {
        "b": 1.0, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "pb": 1e15,
        "kib": 1024.0, "mib": 1024.0 ** 2, "gib": 1024.0 ** 3, "tib": 1024.0 ** 4,
    },
    "time": {
        "s": 1.0, "ms": 0.001, "us": 1e-6, "min": 60.0, "h": 3600.0,
        "day": 86400.0, "week": 604800.0, "year": 31557600.0,
    },
    "pressure": {
        "pa": 1.0, "kpa": 1000.0, "mpa": 1e6, "bar": 1e5, "psi": 6894.757293168,
        "atm": 101325.0, "mmhg": 133.322387415,
    },
    "energy": {
        "j": 1.0, "kj": 1000.0, "cal": 4.184, "kcal": 4184.0, "wh": 3600.0,
        "kwh": 3.6e6, "btu": 1055.05585262,
    },
}

_ALIASES = {
    "meter": "m", "meters": "m", "metre": "m", "kilometer": "km", "kilometers": "km",
    "mile": "mi", "miles": "mi", "yard": "yd", "yards": "yd", "foot": "ft", "feet": "ft",
    "inch": "in", "inches": "in", "公斤": "kg", "千克": "kg", "克": "g", "斤": "lb",
    "磅": "lb", "米": "m", "公里": "km", "千米": "km", "厘米": "cm", "毫米": "mm",
    "升": "l", "毫升": "ml", "加仑": "gal", "小时": "h", "分钟": "min", "秒": "s",
    "天": "day", "周": "week", "年": "year", "卡": "cal", "千卡": "kcal", "大卡": "kcal",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k", "摄氏度": "c", "华氏度": "f",
    "c": "c", "f": "f", "k": "k",
}


def _norm(unit: str) -> str:
    key = (unit or "").strip().lower().replace(" ", "")
    return _ALIASES.get(key, key)


def _convert_temperature(value: float, src: str, dst: str) -> float:
    if src == dst:
        return value
    celsius = value
    if src == "f":
        celsius = (value - 32) * 5 / 9
    elif src == "k":
        celsius = value - 273.15
    if dst == "f":
        return celsius * 9 / 5 + 32
    if dst == "k":
        return celsius + 273.15
    return celsius


def _find_category(unit: str) -> Optional[Tuple[str, str]]:
    unit = _norm(unit)
    if unit in ("c", "f", "k"):
        return "temperature", unit
    for category, units in _UNITS.items():
        if unit in units:
            return category, unit
    return None


class UnitConvertTool(BaseTool):
    name = "unit_convert"
    description = (
        "Offline unit converter for length, mass, temperature, volume, area, speed, "
        "data size, time, pressure and energy. Accepts common English/Chinese aliases."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["convert", "list"],
                "description": "Convert a value or list supported units.",
            },
            "value": {"type": "number", "description": "Value to convert."},
            "from_unit": {"type": "string", "description": "Source unit."},
            "to_unit": {"type": "string", "description": "Target unit."},
            "category": {"type": "string", "description": "Category for list (optional)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "list":
                return self._list(**kwargs)
            if action == "convert":
                return self._convert(**kwargs)
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            logger.debug("unit_convert %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _list(self, category: Optional[str] = None, **kwargs) -> ToolResult:
        if category:
            units = _UNITS.get(category.lower())
            if units is None:
                return ToolResult(success=False, output="", error=f"Unknown category: {category}")
            return ToolResult(success=True, output=f"{category}: " + ", ".join(sorted(units)),
                              metadata={"category": category, "units": sorted(units)})
        lines = ["temperature: c, f, k"]
        data = {"temperature": ["c", "f", "k"]}
        for name, units in _UNITS.items():
            lines.append(f"{name}: " + ", ".join(sorted(units)))
            data[name] = sorted(units)
        return ToolResult(success=True, output="\n".join(lines), metadata=data)

    def _convert(self, value: Optional[float] = None, from_unit: str = "",
                 to_unit: str = "", **kwargs) -> ToolResult:
        if value is None:
            return ToolResult(success=False, output="", error="value is required")
        if not from_unit or not to_unit:
            return ToolResult(success=False, output="", error="from_unit and to_unit are required")
        src_category = _find_category(from_unit)
        dst_category = _find_category(to_unit)
        if src_category is None:
            return ToolResult(success=False, output="", error=f"Unknown unit: {from_unit}")
        if dst_category is None:
            return ToolResult(success=False, output="", error=f"Unknown unit: {to_unit}")
        src_cat, src = src_category
        dst_cat, dst = dst_category
        if src_cat != dst_cat and not ("temperature" in (src_cat, dst_cat)):
            return ToolResult(success=False, output="",
                              error=f"Incompatible units: {from_unit} ({src_cat}) vs {to_unit} ({dst_cat})")
        if src_cat == "temperature":
            result = _convert_temperature(float(value), src, dst)
        else:
            factor_src = _UNITS[src_cat][src]
            factor_dst = _UNITS[src_cat][dst]
            result = float(value) * factor_src / factor_dst
        result = round(result, 6)
        output = f"{value:g} {from_unit} = {result:g} {to_unit}"
        return ToolResult(success=True, output=output, metadata={
            "value": value, "from": from_unit, "to": to_unit, "result": result,
            "category": src_cat,
        })


TOOLS_REGISTRY.register(UnitConvertTool())
