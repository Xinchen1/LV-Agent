"""DingTalk bot over the official Stream (WebSocket) transport."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Set

from .base import AgentWorker, ThinkingProgress, emit, log, require, split_text

_DEFAULT_BASE = "https://oapi.dingtalk.com"
_DEFAULT_SOURCE = "DING_DWS_CLAW"


def _post(client: Any, base: str, path: str, payload: dict[str, str]) -> dict[str, Any]:
    resp = client.post(f"{base}{path}", json=payload)
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("钉钉扫码接口返回格式异常")
    errcode = int(data.get("errcode") or 0)
    if resp.status_code >= 400 or errcode:
        raise RuntimeError(f"{data.get('errmsg') or '钉钉扫码接口失败'} (errcode={errcode})")
    return data


def register_app() -> tuple[str, str]:
    """Create a DingTalk app by QR scan and return (client_id, client_secret)."""
    require("httpx", "httpx")
    import httpx
    import pyqrcode

    base = (os.getenv("DINGTALK_REGISTRATION_BASE_URL") or _DEFAULT_BASE).rstrip("/")
    source = os.getenv("DINGTALK_REGISTRATION_SOURCE") or _DEFAULT_SOURCE
    qr_dir = Path(os.getenv("IM_QR_DIR") or (Path.cwd() / "data" / "im" / "dingtalk"))
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr_path = qr_dir / "dingtalk-qr.png"

    emit("status", platform="dingtalk", state="waiting_scan")
    log("requesting DingTalk app registration QR…")
    with httpx.Client(timeout=20, follow_redirects=True, trust_env=True) as client:
        init = _post(client, base, "/app/registration/init", {"source": source})
        nonce = str(init.get("nonce") or "").strip()
        if not nonce:
            raise RuntimeError("钉钉扫码初始化失败：缺少 nonce")
        begin = _post(client, base, "/app/registration/begin", {"nonce": nonce})
        device_code = str(begin.get("device_code") or "").strip()
        url = str(begin.get("verification_uri_complete") or "").strip()
        if not device_code or not url:
            raise RuntimeError("钉钉扫码初始化失败：缺少 device_code 或二维码地址")
        expires_in = max(int(begin.get("expires_in") or 7200), 60)
        interval = max(int(begin.get("interval") or 3), 1)

        tmp = qr_path.with_suffix(".tmp")
        pyqrcode.create(url, error="M").png(str(tmp), scale=6)
        tmp.replace(qr_path)
        emit(
            "qr",
            platform="dingtalk",
            path=str(qr_path),
            url=url,
            expire_in=expires_in,
            ts=int(time.time() * 1000),
        )

        deadline = time.time() + expires_in
        while time.time() < deadline:
            poll = _post(client, base, "/app/registration/poll", {"device_code": device_code})
            status = str(poll.get("status") or "").upper()
            if status in {"WAITING", "CREATING"}:
                emit("status", platform="dingtalk", state=status.lower())
                time.sleep(interval)
                continue
            if status == "SUCCESS":
                client_id = str(poll.get("client_id") or "").strip()
                client_secret = str(poll.get("client_secret") or "").strip()
                if not client_id or not client_secret:
                    raise RuntimeError("钉钉扫码成功，但未返回 Client ID / Client Secret")
                try:
                    qr_path.unlink()
                except Exception:
                    pass
                emit(
                    "credentials",
                    platform="dingtalk",
                    app_id=client_id,
                    app_secret=client_secret,
                )
                log(f"dingtalk app created: {client_id}")
                return client_id, client_secret
            if status == "EXPIRED":
                raise RuntimeError("钉钉二维码已过期，请重新扫码")
            if status == "FAIL":
                raise RuntimeError(str(poll.get("fail_reason") or "钉钉扫码创建失败"))
            raise RuntimeError(f"钉钉扫码返回未知状态：{status or 'UNKNOWN'}")
    raise RuntimeError("钉钉扫码超时，请重试")


class DingTalkBridge:
    def __init__(
        self,
        worker: AgentWorker,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        require("dingtalk_stream", "dingtalk-stream")
        import dingtalk_stream

        self.sdk = dingtalk_stream
        self.worker = worker
        self.client_id = (client_id or os.getenv("DINGTALK_CLIENT_ID") or "").strip()
        self.client_secret = (client_secret or os.getenv("DINGTALK_CLIENT_SECRET") or "").strip()
        if not self.client_id or not self.client_secret:
            raise RuntimeError("请先填写钉钉 Client ID (AppKey) 和 Client Secret (AppSecret)")
        allowed = (os.getenv("DINGTALK_ALLOWED_USER_IDS") or "").strip()
        self.allowed: Set[str] = {x.strip() for x in allowed.split(",") if x.strip()}
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

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

    def _make_handler(self):
        sdk = self.sdk
        bridge = self

        class Handler(sdk.ChatbotHandler):
            async def process(self, callback: Any):
                try:
                    incoming = sdk.ChatbotMessage.from_dict(callback.data)
                    user_id = getattr(incoming, "sender_staff_id", "") or ""
                    if bridge.allowed and user_id not in bridge.allowed:
                        return sdk.AckMessage.STATUS_OK, "OK"
                    if bridge._duplicate(getattr(incoming, "message_id", "")):
                        return sdk.AckMessage.STATUS_OK, "OK"
                    text = ""
                    text_obj = getattr(incoming, "text", None)
                    if text_obj is not None:
                        text = getattr(text_obj, "content", "") or ""
                    text = text.strip()
                    if not text:
                        return sdk.AckMessage.STATUS_OK, "OK"
                    emit("message", platform="dingtalk", from_user=user_id, text=text[:200])
                    # 思考过程: 收集后在最终回复前折叠一份(钉钉机器人消息不支持编辑刷新)
                    collector = ThinkingProgress()
                    answer = bridge.worker.ask_stream(text, collector.on_event)
                    thinking = collector.summary()
                    if thinking:
                        answer = thinking + "\n\n———\n\n" + answer
                    for chunk in split_text(answer):
                        self.reply_text(chunk, incoming)
                except Exception as exc:  # noqa: BLE001
                    log(f"dingtalk handler error: {exc}")
                return sdk.AckMessage.STATUS_OK, "OK"

        return Handler()

    def run(self) -> None:
        sdk = self.sdk
        credential = sdk.Credential(self.client_id, self.client_secret)
        client = sdk.DingTalkStreamClient(credential)
        client.register_callback_handler(sdk.ChatbotMessage.TOPIC, self._make_handler())
        emit("status", platform="dingtalk", state="connecting")
        log("connecting to DingTalk Stream…")
        client.start_forever()
