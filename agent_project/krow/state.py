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
     "cursors": {"hall:<channel_id>": 7360, ...}}    # 各取数源自己的读取游标（MCP 路径用）

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


class DedupStore:
    def __init__(self, path: Optional[str] = DEFAULT_STATE_PATH, *, max_seen: int = 5000) -> None:
        self.path = os.path.expanduser(path) if path else None
        self.max_seen = max(100, int(max_seen))
        self._lock = threading.RLock()
        self._seen: Dict[str, float] = {}
        self._pending: List[InjectItem] = []
        self._cursors: Dict[str, Any] = {}
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

    def _save(self) -> None:
        if not self.path:
            return
        data = {
            "version": _VERSION,
            "seen": self._seen,
            "pending": [p.to_dict() for p in self._pending],
            "cursors": self._cursors,
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

    # ── 观测 ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "path": self.path,
                "seen": len(self._seen),
                "pending": len(self._pending),
                "cursors": dict(self._cursors),
            }
