"""Personal WeChat bridge using the unofficial web protocol (QR login)."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Set

from .base import AgentWorker, ThinkingProgress, emit, log, require, split_text


class WeChatBridge:
    def __init__(self, worker: AgentWorker) -> None:
        require("itchat", "itchat")
        import itchat
        from itchat.content import TEXT

        self.itchat = itchat
        self.TEXT = TEXT
        self.worker = worker
        self.qr_dir = Path(os.getenv("IM_QR_DIR") or (Path.cwd() / "data" / "im"))
        self.qr_dir.mkdir(parents=True, exist_ok=True)
        self.qr_path = self.qr_dir / "wechat-qr.png"
        self.status_path = self.qr_dir / "wechat-login.json"
        allowed = (os.getenv("WECHAT_ALLOWED_USERS") or "").strip()
        self.allowed: Set[str] = {x.strip() for x in allowed.split(",") if x.strip()}
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._nick = ""

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

    def _write_qr(self, png: bytes) -> None:
        if not png:
            return
        tmp = self.qr_path.with_suffix(".tmp")
        tmp.write_bytes(png)
        tmp.replace(self.qr_path)
        emit("qr", platform="wechat", path=str(self.qr_path), ts=int(time.time() * 1000))

    def _on_qr(self, uuid=None, status=None, qrcode=None, **_: Any) -> None:
        if qrcode:
            self._write_qr(qrcode)
        if status == "201":
            emit("status", platform="wechat", state="scanned")

    def _on_login(self) -> None:
        try:
            self._nick = getattr(self.itchat.instance.storageClass, "nickName", "") or ""
        except Exception:
            self._nick = ""
        emit("status", platform="wechat", state="online", account=self._nick)
        log(f"wechat logged in as {self._nick or 'unknown'}")
        try:
            if self.qr_path.exists():
                self.qr_path.unlink()
        except Exception:
            pass

    def _on_exit(self) -> None:
        emit("status", platform="wechat", state="offline")

    def _on_text(self, msg: Any) -> None:
        try:
            text = (msg.get("Text") or "").strip()
            if not text:
                return
            user = msg.get("FromUserName") or ""
            is_group = user.startswith("@@")
            if is_group and not getattr(msg, "isAt", False):
                return
            if self.allowed and user not in self.allowed:
                return
            if self._duplicate(str(msg.get("MsgId") or "")):
                return
            if is_group:
                text = text.split("\u2005", 1)[-1].strip()
            emit("message", platform="wechat", from_user=user, text=text[:200])
            # 思考过程: 收集后折叠进最终回复(网页版消息不可编辑)
            collector = ThinkingProgress()
            answer = self.worker.ask_stream(text, collector.on_event)
            thinking = collector.summary()
            if thinking:
                answer = thinking + "\n\n———\n\n" + answer
            for chunk in split_text(answer):
                msg.user.send(chunk)
        except Exception as exc:  # noqa: BLE001
            log(f"wechat handler error: {exc}")

    def run(self) -> None:
        self.itchat.msg_register(self.TEXT, isFriendChat=True, isGroupChat=True)(self._on_text)
        emit("status", platform="wechat", state="waiting_qr")
        log("requesting WeChat QR code…")
        try:
            self.itchat.auto_login(
                hotReload=True,
                enableCmdQR=False,
                picDir=str(self.qr_path),
                statusStorageDir=str(self.status_path),
                qrCallback=self._on_qr,
                loginCallback=self._on_login,
                exitCallback=self._on_exit,
            )
            self.itchat.run()
        except SystemExit:
            raise RuntimeError("无法连接微信网页版登录服务，请检查网络后重试")
