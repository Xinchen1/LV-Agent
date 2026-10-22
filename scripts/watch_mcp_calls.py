#!/usr/bin/env python3
"""盯 MCP 端点被调用的实时哨兵。

轮询 /healthz 的 calls 计数器, 一旦对面真的调了工具就把快照追加进日志,
这样面板那边跑起来之后不必守着终端问「调了没」。

    python scripts/watch_mcp_calls.py                 # 常驻监听(默认 5s 一次)
    python scripts/watch_mcp_calls.py --once          # 只看当前快照
    python scripts/watch_mcp_calls.py --interval 2    # 加密轮询
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "mcp_calls.log"
DEFAULT_HEALTH = "http://127.0.0.1:8788/healthz"


def fetch(health_url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(health_url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def append(record: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(record + "\n")


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch LV Agent MCP endpoint usage.")
    parser.add_argument("--health", default=DEFAULT_HEALTH)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="打印一次快照后退出")
    args = parser.parse_args()

    seen = -1
    errors = 0
    while True:
        try:
            data = fetch(args.health)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            errors += 1
            if args.once:
                print(f"[{stamp()}] 端点不可达: {exc}", flush=True)
                return 1
            if errors == 1 or errors % 20 == 0:
                line = f"[{stamp()}] 端点不可达(连续 {errors} 次): {exc}"
                print(line, flush=True)
                append(line)
            time.sleep(args.interval)
            continue

        errors = 0
        calls = int(data.get("calls", 0))
        err_count = int(data.get("errors", 0))
        snapshot = (
            f"[{stamp()}] calls={calls} errors={err_count} "
            f"last_tool={data.get('last_tool')} last_call_at={data.get('last_call_at')} "
            f"(端点 tools={data.get('tools')} auth={data.get('auth')})"
        )

        if args.once:
            print(snapshot, flush=True)
            return 0
        if calls != seen:
            seen = calls
            print(snapshot, flush=True)
            append(snapshot)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())