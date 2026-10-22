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
            answer = self.worker.ask(text)
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
