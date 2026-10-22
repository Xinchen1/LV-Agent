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


class ThinkingProgress:
    """思考过程收集器: status/tool_call/reasoning 事件 -> 可读文本.

    send/update 为平台适配钩子, 可都传 None 变纯收集器(final-summary 渠道用):
      send(body) -> message_id        : 首条占位消息, 返回消息 ID 失败返回 ""
      update(message_id, body)        : 节流后的编辑刷新
    worker 线程内回调 on_event, 平台钩子需自管线程安全(asyncio 渠道用 run_coroutine_threadsafe).
    """

    MAX_OPS = 24
    MAX_REASON = 1400

    def __init__(self, send=None, update=None, *, interval: float = 2.0) -> None:
        self._send = send
        self._update = update
        self._interval = interval
        self.message_id: Optional[str] = None
        self.ops: list[str] = []
        self._reasoning = ""
        self._last_flush = 0.0
        self._last_body = ""
        self._lock = threading.Lock()
        self._done = False

    def on_event(self, kind: str, text: str) -> None:
        """worker 线程回调: 聚合事件并节流刷新."""
        with self._lock:
            if self._done:
                return
            snippet = (text or "").strip().splitlines()
            if not snippet:
                return
            first = snippet[0][:160]
            if kind == "tool_call":
                self.ops.append(f"🔧 {first}")
            elif kind == "status":
                if first == "thinking":
                    return
                self.ops.append(f"· {first}")
            elif kind == "reasoning":
                self._reasoning = (self._reasoning + text)[-self.MAX_REASON :]
            else:
                return
            if len(self.ops) > self.MAX_OPS:
                self.ops = self.ops[-self.MAX_OPS :]
            self._flush(force=self.message_id is None)

    def finish(self, label: str = "💭 思考过程 ✅") -> tuple[bool, Optional[str], str]:
        """任务结束: (编辑平台标记完成) 并冻结收集. 返回 (是否完成标记, 消息ID, 最终body).
        收集器形态(无 update 钩子)只冻结, 不做任何发送."""
        with self._lock:
            self._done = True
            marked = False
            if self._update and self.message_id:
                marked = True
                body = self.render().replace("💭 思考中", label)
                if body != self._last_body:
                    self._update(self.message_id, body)
                    self._last_body = body
                return marked, self.message_id, self._last_body or body
            return marked, self.message_id, self._last_body

    def is_empty(self) -> bool:
        with self._lock:
            return not self.ops and not self._reasoning.strip()

    def summary(self, max_ops: int = 8, max_reason: int = 500) -> str:
        """最终回复附带的紧凑版思考过程(不支持消息编辑的渠道把这段拼到答案前)."""
        with self._lock:
            if not self.ops and not self._reasoning.strip():
                return ""
            ops = ["🔧" if op.startswith("🔧") else "·" for op in self.ops]
            ops = "\n".join(self.ops[-max_ops:])
            reason = self._reasoning.strip()
            if len(reason) > max_reason:
                reason = "…" + reason[-max_reason:]
            parts = ["💭 思考过程:"]
            if ops:
                parts.append(ops)
            if reason:
                parts.append(reason)
            return "\n\n".join(parts)

    def _flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_flush < self._interval:
            return
        self._last_flush = now
        body = self.render()
        if body == self._last_body:
            return
        if not self._send:
            self._last_body = body
            return
        if self.message_id is None:
            self.message_id = self._send(body) or ""
            if self.message_id:
                self._last_body = body
            return
        if self._update and self.message_id:
            self._update(self.message_id, body)
            self._last_body = body

    def render(self) -> str:
        parts = ["💭 思考中"]
        if self.ops:
            parts.append("步骤:\n" + "\n".join(self.ops))
        if self._reasoning.strip():
            parts.append("思考:\n" + self._reasoning.strip())
        body = "\n\n".join(parts)
        while len(body.encode("utf-8")) > 3600:
            if len(self.ops) > 10:
                self.ops = self.ops[-10:]
            elif len(self._reasoning) > 800:
                self._reasoning = self._reasoning[-800:]
            else:
                break
            parts = (["💭 思考中"]
                     + (["步骤:\n" + "\n".join(self.ops)] if self.ops else [])
                     + (["思考:\n" + self._reasoning.strip()] if self._reasoning.strip() else []))
            body = "\n\n".join(parts)
        return body


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
            text, out, on_event = (item + (None,))[:3]
            try:
                agent = self._ensure_agent()

                def _stream_cb(kind: str, text: str, _on_event=on_event):
                    # IM 渠道保险丝: 过滤噪声与 [TOOL: 前缀内容, 只透传可读过程.
                    # content 不透传(答案完成时整条回复), 避免半成品刷屏.
                    if not _on_event or not text:
                        return
                    if kind not in ("status", "tool_call", "reasoning"):
                        return
                    cleaned = text.strip()
                    if not cleaned or cleaned.startswith("[TOOL:"):
                        return
                    try:
                        _on_event(kind, cleaned)
                    except Exception as exc:  # noqa: BLE001
                        import logging

                        logging.getLogger(__name__).debug(f"im progress callback error: {exc}")

                result = agent.run(
                    text,
                    code_mode=False,
                    stream_callback=_stream_cb if on_event else None,
                )
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
                except Exception as strip_err:
                    # strip_tool_tags 不可用时保留原文，属刻意 fallback
                    log(f"strip_tool_tags skipped: {strip_err}")
                out.put(("ok", answer))
            except Exception as exc:  # noqa: BLE001
                log(f"agent error: {exc}")
                out.put(("error", f"Agent 执行失败：{exc}"))
            finally:
                self._queue.task_done()

    def ask(self, text: str, timeout: float = 900.0) -> str:
        return self._submit(text, None, timeout)

    def ask_stream(self, text: str, on_event, timeout: float = 900.0) -> str:
        """与 ask 相同, 但 on_event(kind, text) 实时收到思考过程(status/tool_call/reasoning)."""
        return self._submit(text, on_event, timeout)

    def _submit(self, text: str, on_event, timeout: float) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        out: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)
        self._queue.put((text, out, on_event))
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
