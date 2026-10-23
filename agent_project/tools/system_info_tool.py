"""System information tool - disk, memory, CPU, battery, processes and network (offline)."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


def _run(argv: List[str], timeout: int = 10) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.stdout if proc.returncode == 0 else ""


def _bytes_human(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def _memory() -> Optional[Dict[str, Any]]:
    if _SYSTEM == "Darwin":
        total_raw = _run(["sysctl", "-n", "hw.memsize"]).strip()
        vm = _run(["vm_stat"])
        if not total_raw or not vm:
            return None
        page_size = 4096
        for line in vm.splitlines():
            if "page size of" in line:
                try:
                    page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
                except (IndexError, ValueError):
                    page_size = 4096
                break
        values: Dict[str, int] = {}
        for line in vm.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            number = raw.strip().rstrip(".")
            if number.isdigit():
                values[key.strip()] = int(number)
        free_pages = values.get("Pages free", 0) + values.get("Pages inactive", 0)
        total = int(total_raw)
        free = free_pages * page_size
        used = max(0, total - free)
        return {"total": total, "used": used, "free": free}
    if _SYSTEM == "Linux":
        info: Dict[str, int] = {}
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    if rest.strip().endswith("kB"):
                        try:
                            info[key] = int(rest.split()[0]) * 1024
                        except (ValueError, IndexError) as e:
                            logger.debug("meminfo parse %s failed: %s", key, e)
                            continue
        except OSError as e:
            logger.debug("read /proc/meminfo failed: %s", e)
            return None
        total = info.get("MemTotal", 0)
        free = info.get("MemAvailable", info.get("MemFree", 0))
        if not total:
            return None
        return {"total": total, "used": total - free, "free": free}
    return None


def _battery() -> Optional[Dict[str, Any]]:
    if _SYSTEM == "Darwin":
        out = _run(["pmset", "-g", "batt"])
        for line in out.splitlines():
            if "%" not in line:
                continue
            percent = None
            for token in line.replace(";", " ").split():
                if token.endswith("%"):
                    try:
                        percent = int(token.rstrip("%"))
                    except ValueError:
                        percent = None
            if percent is not None:
                state = "charging" if "AC Power" in line or "charging" in line else "battery"
                return {"percent": percent, "state": state, "raw": line.strip()}
    if _SYSTEM == "Linux":
        base = "/sys/class/power_supply"
        try:
            for name in os.listdir(base):
                if not name.upper().startswith("BAT"):
                    continue
                with open(os.path.join(base, name, "capacity"), encoding="utf-8") as fh:
                    percent = int(fh.read().strip())
                status = ""
                status_path = os.path.join(base, name, "status")
                if os.path.exists(status_path):
                    with open(status_path, encoding="utf-8") as fh:
                        status = fh.read().strip()
                return {"percent": percent, "state": status.lower() or "unknown"}
        except (OSError, ValueError) as e:
            logger.debug("read battery failed: %s", e)
            return None
    return None


class SystemInfoTool(BaseTool):
    name = "system_info"
    description = (
        "Inspect the local machine: OS summary, disk usage, memory, CPU/load, battery, "
        "top processes and network interfaces. Read-only and offline."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["summary", "disk", "memory", "cpu", "battery", "processes", "network"],
                "description": "Information to read.",
            },
            "path": {"type": "string", "description": "Disk path (default /)."},
            "limit": {"type": "integer", "description": "Max processes (default 10)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            handler = {
                "summary": self._summary,
                "disk": self._disk,
                "memory": self._memory_action,
                "cpu": self._cpu,
                "battery": self._battery_action,
                "processes": self._processes,
                "network": self._network,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(**kwargs)
        except Exception as e:
            logger.debug("system_info action %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _disk_data(self, path: str = "/") -> Dict[str, Any]:
        target = path or "/"
        usage = shutil.disk_usage(target)
        return {
            "path": target,
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent_used": round(usage.used / usage.total * 100, 1) if usage.total else 0,
        }

    def _disk(self, path: str = "/", **kwargs) -> ToolResult:
        data = self._disk_data(path)
        output = (f"{data['path']}: {_bytes_human(data['used'])} used / "
                  f"{_bytes_human(data['total'])} total ({data['percent_used']}%), "
                  f"{_bytes_human(data['free'])} free")
        readable = {k: (_bytes_human(v) if k in ("total", "used", "free") else v)
                    for k, v in data.items()}
        return ToolResult(success=True, output=output, metadata=readable)

    def _memory_action(self, **kwargs) -> ToolResult:
        data = _memory()
        if not data:
            return ToolResult(success=False, output="", error="Memory info unavailable")
        percent = round(data["used"] / data["total"] * 100, 1) if data["total"] else 0
        output = (f"Memory: {_bytes_human(data['used'])} used / {_bytes_human(data['total'])} "
                  f"({percent}%), {_bytes_human(data['free'])} available")
        return ToolResult(success=True, output=output, metadata={
            "total": _bytes_human(data["total"]), "used": _bytes_human(data["used"]),
            "free": _bytes_human(data["free"]), "percent_used": percent,
        })

    def _cpu(self, **kwargs) -> ToolResult:
        load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        data = {
            "processor": platform.processor() or platform.machine(),
            "cores": os.cpu_count(),
            "load_1m": round(load[0], 2),
            "load_5m": round(load[1], 2),
            "load_15m": round(load[2], 2),
            "python": sys.version.split()[0],
        }
        lines = [
            f"CPU: {data['processor']} ({data['cores']} logical cores)",
            f"load: {data['load_1m']} / {data['load_5m']} / {data['load_15m']} (1/5/15m)",
            f"python: {data['python']}",
        ]
        return ToolResult(success=True, output="\n".join(lines), metadata=data)

    def _battery_action(self, **kwargs) -> ToolResult:
        data = _battery()
        if not data:
            return ToolResult(success=False, output="", error="Battery info unavailable")
        return ToolResult(success=True,
                          output=f"Battery: {data['percent']}% ({data['state']})",
                          metadata=data)

    def _processes(self, limit: int = 10, **kwargs) -> ToolResult:
        limit = max(1, int(limit or 10))
        if _SYSTEM == "Darwin":
            out = _run(["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"])
        else:
            out = _run(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"])
        rows: List[Dict[str, Any]] = []
        for line in out.splitlines()[1:limit + 1]:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            try:
                rows.append({
                    "pid": int(parts[0]),
                    "cpu_percent": float(parts[1]),
                    "mem_percent": float(parts[2]),
                    "command": parts[3].strip(),
                })
            except ValueError as e:
                logger.debug("ps parse failed for %r: %s", line, e)
                continue
        if not rows:
            return ToolResult(success=False, output="", error="Process list unavailable")
        lines = ["Top processes by CPU:"] + [
            f"  {r['pid']:>7} cpu {r['cpu_percent']:>5.1f}% mem {r['mem_percent']:>4.1f}% {r['command']}"
            for r in rows
        ]
        return ToolResult(success=True, output="\n".join(lines), metadata={"processes": rows})

    def _network(self, **kwargs) -> ToolResult:
        host = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(host)
        except OSError:
            local_ip = ""
        interfaces: List[str] = []
        for name in ("en0", "en1", "eth0", "wlan0", "Wi-Fi"):
            if _SYSTEM != "Darwin" and name.startswith("en"):
                continue
            out = _run(["ipconfig", "getifaddr", name]) if _SYSTEM == "Darwin" else ""
            if out.strip():
                interfaces.append(f"{name}: {out.strip()}")
        if not interfaces:
            try:
                addresses = socket.getaddrinfo(host, None)
                seen = set()
                for addr in addresses:
                    ip = addr[4][0]
                    if ":" in ip or ip in seen or ip.startswith("127."):
                        continue
                    seen.add(ip)
                    interfaces.append(f"{host}: {ip}")
            except OSError as e:
                logger.debug("getaddrinfo failed: %s", e)
        data = {"hostname": host, "local_ip": local_ip, "interfaces": interfaces}
        lines = [f"hostname: {host}", f"local ip: {local_ip or '-'}"]
        lines.extend(f"  {item}" for item in interfaces)
        return ToolResult(success=True, output="\n".join(lines), metadata=data)

    def _summary(self, **kwargs) -> ToolResult:
        disk = self._disk_data("/")
        mem = _memory()
        battery = _battery()
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        data: Dict[str, Any] = {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cores": os.cpu_count(),
            "load_1m": round(load, 2),
            "disk_percent": disk["percent_used"],
            "disk_free": _bytes_human(disk["free"]),
            "memory_percent": round(mem["used"] / mem["total"] * 100, 1) if mem else None,
            "battery_percent": battery["percent"] if battery else None,
        }
        lines = [
            f"OS: {data['os']} ({data['machine']})",
            f"Python: {data['python']}, cores: {data['cores']}, load(1m): {data['load_1m']}",
            f"Disk: {data['disk_percent']}% used, {data['disk_free']} free",
            f"Memory: {data['memory_percent']}% used" if data["memory_percent"] is not None
            else "Memory: n/a",
            f"Battery: {data['battery_percent']}%" if data["battery_percent"] is not None
            else "Battery: n/a",
        ]
        return ToolResult(success=True, output="\n".join(lines), metadata=data)


TOOLS_REGISTRY.register(SystemInfoTool())
