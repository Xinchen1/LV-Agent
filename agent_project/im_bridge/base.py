"""Shared runtime for IM platform bridges."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def emit(event: str, **data: Any) -> None:
    payload = {"event": event, **data}
    print("@@IM@@" + json.dumps(payload, ensure_ascii=False), flush=True)


def log(text: str) -> None:
    print(f"[im] {text}", file=sys.stderr, flush=True)


def split_text(text: str, limit: int = 3500) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return [c for c in chunks if c]


def _load_config():
    from agent_project.config import load_config

    env_cfg = os.getenv("LV_AGENT_CONFIG")
    if env_cfg and Path(env_cfg).exists():
        return load_config(env_cfg)
    root = Path(__file__).resolve().parents[2]
    for name in ("config.yaml", "config.json"):
        p = root / name
        if p.exists():
            return load_config(str(p))
    return load_config()


class AgentWorker:
    """Single agent instance behind a queue so platform callbacks never race."""

    def __init__(self, agent: Any = None) -> None:
        self._agent = agent
        self._config = None
        self._queue: "queue.Queue[Optional[tuple[str, queue.Queue]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="im-agent", daemon=True)
        self._thread.start()

    def _ensure_agent(self):
        if self._agent is None:
            from agent_project.agent import OpenMythosAgent

            if self._config is None:
                self._config = _load_config()
            log("initializing agent…")
            self._agent = OpenMythosAgent(self._config)
            log("agent ready")
        return self._agent

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            text, out = item
            try:
                agent = self._ensure_agent()
                result = agent.run(text, code_mode=False)
                answer = ""
                if isinstance(result, dict):
                    answer = str(
                        result.get("final_answer")
                        or result.get("answer")
                        or result.get("response")
                        or ""
                    )
                elif result is not None:
                    answer = str(result)
                if not answer.strip():
                    answer = "（Agent 没有返回文本结果）"
                try:
                    from agent_project.response_filter import strip_tool_tags

                    answer = strip_tool_tags(answer)
                except Exception:
                    pass
                out.put(("ok", answer))
            except Exception as exc:  # noqa: BLE001
                log(f"agent error: {exc}")
                out.put(("error", f"Agent 执行失败：{exc}"))
            finally:
                self._queue.task_done()

    def ask(self, text: str, timeout: float = 900.0) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        out: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)
        self._queue.put((text, out))
        try:
            status, answer = out.get(timeout=timeout)
        except queue.Empty:
            return "任务超时，请稍后重试。"
        return answer if status == "ok" else answer

    def stop(self) -> None:
        self._queue.put(None)


def require(module: str, package: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"缺少依赖 {package}，请在项目 venv 中执行：pip install {package}"
        )
