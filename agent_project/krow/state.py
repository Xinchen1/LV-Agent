"""防重复消费：持久化的「已送达 / 待送达 / 读取游标」三张表.

平台侧 ``GET /agent/v1/inject`` 是读即消费（服务端推进 last_read、改 wakeups 行为 injected），
但 LV-Agent 侧仍需要自己的一本账，原因有三：

1. **两条取数路径可能看到同一条消息**：REST ``inject`` 与 MCP ``hall_read`` 都能读到同一条大厅发言；
   ``team_inbox`` 不标已读，每次轮询都会再返回同一封信；
2. **取到 ≠ 送到**：取数成功后、拼进 prompt 之前进程若崩掉，服务端已视为消费过，这条就永远丢了——
   所以先落盘 ``pending``、送达后再挪进 ``seen``（对模型至少一次、按键恰好一次）；
3. **空闲轮询线程与执行中注入**是两个消费者，谁拿到都要互相看得见对方的账。

文件格式（JSON，一个文件）::

    {"version": 1,
     "seen": {"hall:7360": 1761290000.1, ...},      # 去重键 → 送达时间；超过 max_seen 条按时间淘汰最老的
     "pending": [ {InjectItem.to_dict()}, ... ],     # 已取到、还没送进模型
     "cursors": {"hall:<channel_id>": 7360, ...},    # 各取数源自己的读取游标（MCP 路径用）
     "mentions": {"7965": {"state": "answered", "mail_id": "…", "channel_id": "…", "post_message_id": 7970,
                           "attempts": 1, "updated_at": 1761290000.1}, ...},   # 大厅点名闭环账（v3 守候循环）
     "owner_replies": {"<mail_id>": {"state": "answered", "updated_at": …}, ...}}  # 主人直聊回信账（v3）

``mentions`` 以 ``hall_message_id`` 为键：``read`` = 已真读（服务端已翻 injected）还没回帖；``answered`` = 已带 reply_to_id 回帖；
``failed`` = 回帖失败且不再自动重试；``unverified`` = 上次进程在回帖途中死掉、又认不出自己的发言，宁可不重发也不刷屏。
重启后未读信再次出现在 ``unread_only`` 里，先查这本账：``answered`` 的只补读信不再回。

写盘用「临时文件 + os.replace」原子替换；``path=None`` 时只在内存里记（测试 / 一次性用）。
线程安全：所有公开方法持同一把 RLock。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from .items import InjectItem

logger = logging.getLogger("krow.state")

__all__ = ["DedupStore", "DEFAULT_STATE_PATH"]

#: 缺省落盘位置（与 ``super_agent._version_beacon`` 的 ``~/.lv_agent`` 同一目录）。
DEFAULT_STATE_PATH = os.path.join("~", ".lv_agent", "krow_state.json")
_VERSION = 1
#: 点名账 / 主人回信账各自最多保留多少条（按 updated_at 淘汰最老的）。
MAX_RECORDS = 2000


class DedupStore:
    def __init__(self, path: Optional[str] = DEFAULT_STATE_PATH, *, max_seen: int = 5000, max_records: int = MAX_RECORDS) -> None:
        self.path = os.path.expanduser(path) if path else None
        self.max_seen = max(100, int(max_seen))
        self.max_records = max(50, int(max_records))
        self._lock = threading.RLock()
        self._seen: Dict[str, float] = {}
        self._pending: List[InjectItem] = []
        self._cursors: Dict[str, Any] = {}
        # v3 守候循环的两本账: 大厅点名(hall_message_id → 记录) / 主人直聊回信(mail_id → 记录)
        self._mentions: Dict[str, Dict[str, Any]] = {}
        self._owner_replies: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ── 持久化 ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as e:  # noqa: BLE001 — 坏文件不拦启动：按空账起步，但留一行日志
            logger.warning("krow state unreadable (%s); starting empty: %s", self.path, e)
            return
        if not isinstance(raw, dict):
            return
        seen = raw.get("seen")
        if isinstance(seen, dict):
            self._seen = {str(k): float(v) for k, v in seen.items() if isinstance(v, (int, float))}
        pending = raw.get("pending")
        if isinstance(pending, list):
            self._pending = [InjectItem.from_dict(p) for p in pending if isinstance(p, dict) and p.get("key")]
        cursors = raw.get("cursors")
        if isinstance(cursors, dict):
            self._cursors = dict(cursors)
        for attr, key in (("_mentions", "mentions"), ("_owner_replies", "owner_replies")):
            table = raw.get(key)
            if isinstance(table, dict):
                setattr(self, attr, {str(k): dict(v) for k, v in table.items() if isinstance(v, dict)})

    def _save(self) -> None:
        if not self.path:
            return
        data = {
            "version": _VERSION,
            "seen": self._seen,
            "pending": [p.to_dict() for p in self._pending],
            "cursors": self._cursors,
            "mentions": self._mentions,
            "owner_replies": self._owner_replies,
        }
        directory = os.path.dirname(self.path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".krow_state.", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError as e:
                        logger.debug("krow state tmp cleanup failed (%s): %s", tmp, e)
        except Exception as e:  # noqa: BLE001 — 写不下只影响跨重启的去重，不影响本进程
            logger.warning("krow state save failed (%s): %s", self.path, e)

    # ── 去重 ───────────────────────────────────────────────────────────

    def is_known(self, key: str) -> bool:
        with self._lock:
            return key in self._seen or any(p.key == key for p in self._pending)

    def filter_new(self, items: Iterable[InjectItem]) -> List[InjectItem]:
        """去掉已送达 / 已在待送队列 / 本批重复的，保序。"""
        out: List[InjectItem] = []
        with self._lock:
            batch: set = set()
            for it in items:
                if not it.key or it.key in batch:
                    continue
                if it.key in self._seen or any(p.key == it.key for p in self._pending):
                    continue
                batch.add(it.key)
                out.append(it)
        return out

    def add_pending(self, items: Iterable[InjectItem]) -> List[InjectItem]:
        """新项先落盘进待送队列（取到即记账）。返回真正新增的项。"""
        with self._lock:
            new = self.filter_new(items)
            if new:
                self._pending.extend(new)
                self._save()
            return new

    def take_pending(self, limit: Optional[int] = None) -> List[InjectItem]:
        """看一眼待送队列（不出队——送达后调 :meth:`mark_delivered` 才出队）。"""
        with self._lock:
            items = list(self._pending)
            return items[:limit] if limit else items

    def mark_delivered(self, keys: Iterable[str]) -> None:
        now = time.time()
        with self._lock:
            ks = {k for k in keys if k}
            if not ks:
                return
            for k in ks:
                self._seen[k] = now
            self._pending = [p for p in self._pending if p.key not in ks]
            self._trim_seen()
            self._save()

    def _trim_seen(self) -> None:
        if len(self._seen) <= self.max_seen:
            return
        ordered = sorted(self._seen.items(), key=lambda kv: kv[1])
        for k, _ in ordered[: len(self._seen) - self.max_seen]:
            self._seen.pop(k, None)

    # ── 游标 ───────────────────────────────────────────────────────────

    def cursor(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self._cursors.get(name, default)

    def set_cursor(self, name: str, value: Any) -> None:
        with self._lock:
            if self._cursors.get(name) == value:
                return
            self._cursors[name] = value
            self._save()

    # ── 大厅点名闭环账 / 主人直聊回信账（v3 守候循环） ──────────────────────

    @staticmethod
    def _key(value: Any) -> str:
        return str(value).strip()

    def mention(self, hall_message_id: Any) -> Optional[Dict[str, Any]]:
        """某条大厅消息的闭环记录（没有返回 None）。"""
        with self._lock:
            rec = self._mentions.get(self._key(hall_message_id))
            return dict(rec) if rec else None

    def set_mention(self, hall_message_id: Any, **fields: Any) -> Dict[str, Any]:
        """合并写入（``state`` / ``mail_id`` / ``channel_id`` / ``post_message_id`` / ``attempts`` / ``error`` …），自动盖 ``updated_at``。"""
        with self._lock:
            key = self._key(hall_message_id)
            rec = dict(self._mentions.get(key) or {})
            rec.update({k: v for k, v in fields.items() if v is not None})
            rec["updated_at"] = time.time()
            self._mentions[key] = rec
            self._trim_records(self._mentions)
            self._save()
            return dict(rec)

    def bump_mention_attempts(self, hall_message_id: Any) -> int:
        """回帖前先 +1 并落盘：进程若死在回帖途中，重启后能看出「发过没发完」。"""
        with self._lock:
            key = self._key(hall_message_id)
            rec = dict(self._mentions.get(key) or {})
            rec["attempts"] = int(rec.get("attempts") or 0) + 1
            rec["updated_at"] = time.time()
            self._mentions[key] = rec
            self._save()
            return rec["attempts"]

    def mentions(self, state: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._mentions.items() if state is None or v.get("state") == state}

    def owner_reply(self, mail_id: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._owner_replies.get(self._key(mail_id))
            return dict(rec) if rec else None

    def set_owner_reply(self, mail_id: Any, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            key = self._key(mail_id)
            rec = dict(self._owner_replies.get(key) or {})
            rec.update({k: v for k, v in fields.items() if v is not None})
            rec["updated_at"] = time.time()
            self._owner_replies[key] = rec
            self._trim_records(self._owner_replies)
            self._save()
            return dict(rec)

    def _trim_records(self, table: Dict[str, Dict[str, Any]]) -> None:
        if len(table) <= self.max_records:
            return
        ordered = sorted(table.items(), key=lambda kv: float(kv[1].get("updated_at") or 0.0))
        for k, _ in ordered[: len(table) - self.max_records]:
            table.pop(k, None)

    # ── 观测 ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            by_state: Dict[str, int] = {}
            for rec in self._mentions.values():
                st = str(rec.get("state") or "?")
                by_state[st] = by_state.get(st, 0) + 1
            return {
                "path": self.path,
                "seen": len(self._seen),
                "pending": len(self._pending),
                "cursors": dict(self._cursors),
                "mentions": by_state,
                "owner_replies": len(self._owner_replies),
            }
