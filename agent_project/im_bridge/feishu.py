"""Feishu / Lark bot over the official long-connection (WebSocket) transport."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Set

from .base import AgentWorker, emit, log, require, split_text

_MENTION_RE = re.compile(r"<at\b[^>]*>.*?</at>", re.S | re.I)


def register_app() -> tuple[str, str]:
    """Create a Feishu app by QR scan and return (app_id, app_secret)."""
    require("lark_oapi", "lark-oapi")
    import lark_oapi as lark
    import pyqrcode

    qr_dir = Path(os.getenv("IM_QR_DIR") or (Path.cwd() / "data" / "im" / "feishu"))
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr_path = qr_dir / "feishu-qr.png"

    def _write_qr(url: str, expire_in: Any = None) -> None:
        tmp = qr_path.with_suffix(".tmp")
        pyqrcode.create(url, error="M").png(str(tmp), scale=6)
        tmp.replace(qr_path)
        emit(
            "qr",
            platform="feishu",
            path=str(qr_path),
            url=url,
            expire_in=expire_in,
            ts=int(time.time() * 1000),
        )

    def _on_qr(info: dict[str, Any]) -> None:
        url = str(info.get("url") or "")
        if not url:
            return
        try:
            _write_qr(url, info.get("expire_in"))
        except Exception as exc:  # noqa: BLE001
            log(f"feishu qr render failed: {exc}")

    def _on_status(info: dict[str, Any]) -> None:
        state = str(info.get("status") or "waiting")
        emit("status", platform="feishu", state=state)

    emit("status", platform="feishu", state="waiting_scan")
    log("requesting Feishu app registration QR…")
    result = lark.register_app(
        on_qr_code=_on_qr,
        on_status_change=_on_status,
        addons={
            "scopes": {"tenant": ["im:message", "im:message:send_as_bot"]},
            "events": {"items": {"tenant": ["im.message.receive_v1"]}},
        },
        create_only=True,
    )
    app_id = str(result.get("client_id") or result.get("app_id") or "").strip()
    app_secret = str(result.get("client_secret") or result.get("app_secret") or "").strip()
    if not app_id or not app_secret:
        raise RuntimeError("飞书扫码创建成功，但未返回 App ID / App Secret")
    try:
        qr_path.unlink()
    except Exception:
        pass
    emit("credentials", platform="feishu", app_id=app_id, app_secret=app_secret)
    log(f"feishu app created: {app_id}")
    return app_id, app_secret


def _text_from_content(content: str) -> str:
    try:
        obj = json.loads(content or "{}")
    except Exception:
        return ""
    text = obj.get("text") or ""
    return _MENTION_RE.sub("", text).strip()


class FeishuBridge:
    def __init__(
        self,
        worker: AgentWorker,
        app_id: str | None = None,
        app_secret: str | None = None,
    ) -> None:
        require("lark_oapi", "lark-oapi")
        import lark_oapi as lark

        self.lark = lark
        self.worker = worker
        self.app_id = (app_id or os.getenv("FEISHU_APP_ID") or "").strip()
        self.app_secret = (app_secret or os.getenv("FEISHU_APP_SECRET") or "").strip()
        if not self.app_id or not self.app_secret:
            raise RuntimeError("请先填写飞书 App ID 和 App Secret")
        allowed = (os.getenv("FEISHU_ALLOWED_CHAT_IDS") or "").strip()
        self.allowed: Set[str] = {x.strip() for x in allowed.split(",") if x.strip()}
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self.client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .build()
        )

    def _duplicate(self, message_id: str) -> bool:
        now = time.time()
        with self._lock:
            if message_id in self._seen:
                return True
            self._seen[message_id] = now
            if len(self._seen) > 500:
                for key, ts in list(self._seen.items()):
                    if now - ts > 600:
                        self._seen.pop(key, None)
        return False

    def _reply(self, chat_id: str, text: str) -> None:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        for chunk in split_text(text):
            body = (
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(json.dumps({"text": chunk}, ensure_ascii=False))
                .build()
            )
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(body)
                .build()
            )
            resp = self.client.im.v1.message.create(req)
            if not resp.success():
                log(f"feishu reply failed: code={resp.code} msg={resp.msg}")
                return

    def _send_message(self, chat_id: str, text: str) -> str:
        """发送文本消息并返回 message_id(失败返回 '')."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        body = (
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(json.dumps({"text": text[:3800]}, ensure_ascii=False))
            .build()
        )
        req = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(body)
            .build()
        )
        resp = self.client.im.v1.message.create(req)
        if not resp.success():
            log(f"feishu progress create failed: code={resp.code} msg={resp.msg}")
            return ""
        return str(getattr(resp.data, "message_id", "") or "")

    def _update_message(self, message_id: str, text: str) -> None:
        """编辑既有文本消息(思考过程实时刷新用), 失败静默. 消息初次发出后仍会保留占位文本."""
        if not message_id:
            return
        from lark_oapi.api.im.v1 import UpdateMessageRequest, UpdateMessageRequestBody

        try:
            body = (
                UpdateMessageRequestBody.builder()
                .msg_type("text")
                .content(json.dumps({"text": text[:3800]}, ensure_ascii=False))
                .build()
            )
            req = (
                UpdateMessageRequest.builder()
                .message_id(message_id)
                .request_body(body)
                .build()
            )
            resp = self.client.im.v1.message.update(req)
            if not resp.success():
                log(f"feishu progress update failed: code={resp.code} msg={resp.msg}")
        except Exception as exc:  # noqa: BLE001
            log(f"feishu update error: {exc}")

    class _ThinkingProgress:
        """把 agent 思考过程(status/tool_call/reasoning)聚合成一条飞书消息并节流刷新."""

        MAX_OPS = 24
        MAX_REASON = 1400
        MIN_INTERVAL = 2.0

        def __init__(self, bot, chat_id: str) -> None:
            self._bot = bot
            self._chat_id = chat_id
            self._message_id = ""
            self._ops: list[str] = []
            self._reasoning = ""
            self._last_flush = 0.0
            self._last_body = ""
            self._lock = threading.Lock()
            self._done = False

        def on_event(self, kind: str, text: str) -> None:
            with self._lock:
                if self._done:
                    return
                snippet = text.splitlines()[0][:160] if text else ""
                if not snippet:
                    return
                if kind == "tool_call":
                    self._ops.append(f"🔧 {snippet}")
                elif kind == "status":
                    if snippet == "thinking":
                        return
                    self._ops.append(f"· {snippet}")
                elif kind == "reasoning":
                    self._reasoning = (self._reasoning + text)[-self.MAX_REASON :]
                else:
                    return
                if len(self._ops) > self.MAX_OPS:
                    self._ops = self._ops[-self.MAX_OPS :]
                self._flush(force=not self._message_id)

        def finish(self) -> None:
            with self._lock:
                self._done = True
                if not self._message_id:
                    return
                body = self._render().replace("💭 思考中", "💭 思考过程 ✅")
                if body != self._last_body:
                    self._bot._update_message(self._message_id, body)

        def _flush(self, force: bool = False) -> None:
            now = time.time()
            if not force and now - self._last_flush < self.MIN_INTERVAL:
                return
            self._last_flush = now
            body = self._render()
            if body == self._last_body:
                return
            if not self._message_id:
                self._message_id = self._bot._send_message(self._chat_id, body)
                self._last_body = body
                return
            self._bot._update_message(self._message_id, body)
            self._last_body = body

        def _render(self) -> str:
            parts = ["💭 思考中"]
            if self._ops:
                parts.append("步骤:\n" + "\n".join(self._ops))
            if self._reasoning.strip():
                parts.append("思考:\n" + self._reasoning.strip())
            body = "\n\n".join(parts)
            # 飞书单条文本 ~4000 字节, 超限精简
            while len(body.encode("utf-8")) > 3600:
                if self._ops:
                    self._ops = self._ops[-10:]
                    parts = parts[:2]
                else:
                    self._reasoning = self._reasoning[-800:]
                body = "\n\n".join(["💭 思考中"] + (["步骤:\n" + "\n".join(self._ops)] if self._ops else []) + (["思考:\n" + self._reasoning.strip()] if self._reasoning.strip() else []))
                if len(self._ops) <= 10 and len(self._reasoning) <= 800:
                    break
            return body

    def _on_message(self, data: Any) -> None:
        try:
            event = data.event
            message = event.message
            sender = event.sender
            if getattr(sender, "sender_type", "") == "app":
                return
            if self._duplicate(getattr(message, "message_id", "")):
                return
            if self.allowed and getattr(message, "chat_id", "") not in self.allowed:
                return
            if getattr(message, "message_type", "") != "text":
                return
            text = _text_from_content(getattr(message, "content", ""))
            if not text:
                return
            sender_id = getattr(sender, "sender_id", None)
            open_id = getattr(sender_id, "open_id", "") if sender_id else ""
            emit("message", platform="feishu", from_user=open_id, text=text[:200])
            progress = self._ThinkingProgress(self, getattr(message, "chat_id", ""))
            answer = self.worker.ask_stream(text, progress.on_event)
            progress.finish()
            self._reply(getattr(message, "chat_id", ""), answer)
        except Exception as exc:  # noqa: BLE001
            log(f"feishu handler error: {exc}")

    def run(self) -> None:
        lark = self.lark
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )
        client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )
        emit("status", platform="feishu", state="connecting")
        log("connecting to Feishu long connection…")
        client.start()
