"""KrowBridge —— LV-Agent 侧的「原生注入 + 轮询 + 防重复消费」统一入口.

一个进程一个 bridge，三个消费者共用它、共用一把锁、共用一本账（:class:`DedupStore`）：

* **执行中注入**：:class:`~agent_project.execution_engine.ExecutionEngine` 在每次工具调用结束、下一步思考之前调
  :meth:`interjection_text`（等价于 kit 的 PostToolUse 钩子）；模型给出最终答案时调 :meth:`owner_only_text`
  （等价于 kit 的 ``--event stop`` 轮尾续跑：有主人插话就不收口、接着干）。
* **空闲轮询**：:meth:`start_idle_poller` 起一个后台线程，组员没在跑任务时定期 :meth:`poll`，有点名就交给
  ``on_items``（REPL 把它变成一次唤醒回合；守护模式直接跑）。
* **守护模式**：:mod:`agent_project.krow.wake` 的 ``KrowDaemon`` 直接 ``poll → 跑任务 → mark_delivered``。

取数源按配置与环境自动装配：有 ``KW_API_ENDPOINT`` + ``KW_WORKER_TOKEN`` 就挂 :class:`RestInjectSource`
（平台统一口子，读即消费）；LV-Agent 的 MCP 配置里有 krow 服务器就挂 :class:`McpHallSource`（轮询
``hall_channels`` / ``hall_read`` / ``team_inbox``）。两条都在时两条都拉，重复由账本去掉。

配置（``config.yaml`` 顶层 ``krow:`` 段，全部可省）::

    krow:
      enabled: true
      mcp_server: krow-platform     # mcp.servers 里哪一项是 krow 桥；省略则按工具名 hall_read 自动找
      self_names: ["lv-agent"]      # 我的显示名，用于 @ 判定；省略则按频道 mention_count 整体放行
      rest: auto                    # auto | on | off：是否用 GET /agent/v1/inject（需要 KW_API_ENDPOINT / KW_WORKER_TOKEN）
      mode: mentions                # mentions（只送点到我的）| all（有未读就送）
      inbox: true                   # 一并轮询站内信未读
      poll_interval_s: 20           # 空闲轮询间隔
      idle_wake: true               # REPL 空闲时自动把点名变成一次回合
      state_path: ~/.lv_agent/krow_state.json
      max_hall_per_step: 5
      max_owner_per_step: 3
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .items import InjectItem, contract_text, make_nonce, render_items
from .sources import McpHallSource, RestInjectSource, resolve_rest_env
from .state import DEFAULT_STATE_PATH, DedupStore

logger = logging.getLogger("krow.bridge")

__all__ = ["KrowBridge", "KrowConfig", "build_bridge", "find_krow_server"]

_SERVER_NAME_CANDIDATES = ("krow-platform", "krow_platform", "krow")
_PROBE_TOOL = "hall_read"


class KrowConfig:
    """``config.krow`` 字典的只读视图（给缺省值、做类型归一）。"""

    def __init__(self, raw: Optional[Mapping[str, Any]]) -> None:
        d = dict(raw or {})
        self.enabled = bool(d.get("enabled", False))
        self.mcp_server = str(d.get("mcp_server") or "").strip() or None
        names = d.get("self_names") or []
        if isinstance(names, str):
            names = [names]
        self.self_names = [str(n).strip() for n in names if str(n).strip()]
        self.rest = str(d.get("rest") or "auto").strip().lower()
        if self.rest not in ("auto", "on", "off"):
            self.rest = "auto"
        self.mode = str(d.get("mode") or "mentions").strip().lower()
        self.inbox = bool(d.get("inbox", True))
        self.poll_interval_s = max(5.0, float(d.get("poll_interval_s", 20) or 20))
        self.idle_wake = bool(d.get("idle_wake", True))
        self.state_path = str(d.get("state_path") or DEFAULT_STATE_PATH)
        self.max_hall_per_step = max(1, min(10, int(d.get("max_hall_per_step", 5) or 5)))
        self.max_owner_per_step = max(1, min(3, int(d.get("max_owner_per_step", 3) or 3)))
        self.ack = bool(d.get("ack", True))


def find_krow_server(mcp_manager: Any, preferred: Optional[str] = None) -> Optional[str]:
    """在 ``McpManager`` 里找 krow 桥：先按名字，再按「哪台服务器有 ``hall_read`` 工具」。找不到 / 还没就绪返回 None。"""
    if mcp_manager is None:
        return None
    servers = getattr(mcp_manager, "servers", None) or {}
    bridge_cfgs = getattr(mcp_manager, "_bridge_configs", None) or {}
    known = list(servers.keys()) + [k for k in bridge_cfgs.keys() if k not in servers]
    for name in ([preferred] if preferred else []) + list(_SERVER_NAME_CANDIDATES):
        if name and name in known:
            return name
    # 按工具名探测（纯 Python stdio 连接：conn.tools 在 _ready 后可用）
    for name, conn in servers.items():
        ready = getattr(conn, "_ready", None)
        if ready is not None and hasattr(ready, "is_set") and not ready.is_set():
            continue
        tools = getattr(conn, "tools", None) or []
        if any(isinstance(t, dict) and t.get("name") == _PROBE_TOOL for t in tools):
            return name
    return None


class KrowBridge:
    def __init__(
        self,
        cfg: KrowConfig,
        *,
        store: Optional[DedupStore] = None,
        sources: Optional[List[Any]] = None,
        nonce: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.store = store or DedupStore(cfg.state_path)
        self.sources: List[Any] = list(sources or [])
        self.nonce = nonce or make_nonce()
        self._lock = threading.RLock()
        self._busy = threading.Event()
        self._clock = clock
        self._stop = threading.Event()
        self._poller: Optional[threading.Thread] = None
        self.last_poll_at: float = 0.0
        self.last_new: int = 0
        self.polls: int = 0
        self.delivered: int = 0

    # ── 状态 ───────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.sources)

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def set_busy(self, flag: bool) -> None:
        if flag:
            self._busy.set()
        else:
            self._busy.clear()

    def contract_text(self) -> str:
        return contract_text(self.nonce)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sources": [getattr(s, "name", type(s).__name__) for s in self.sources],
            "source_errors": {getattr(s, "name", type(s).__name__): getattr(s, "last_error", "") for s in self.sources},
            "source_calls": {getattr(s, "name", type(s).__name__): getattr(s, "calls", 0) for s in self.sources},
            "polls": self.polls,
            "delivered": self.delivered,
            "last_poll_at": self.last_poll_at,
            "last_new": self.last_new,
            "busy": self.busy,
            "poller_alive": bool(self._poller and self._poller.is_alive()),
            "store": self.store.stats(),
        }

    # ── 取数 ───────────────────────────────────────────────────────────

    def poll(self, *, hall: bool = True, leader: bool = True, owner: bool = True, sources: Optional[Iterable[Any]] = None) -> List[InjectItem]:
        """拉一次所有取数源 → 去重 → 新项落盘进待送队列 → 返回**全部**待送项（含更早没送出去的）。

        调用方送达后必须 :meth:`mark_delivered`，否则下次还会拿到同一批（这是故意的：取到 ≠ 送到）。
        """
        with self._lock:
            self.polls += 1
            self.last_poll_at = self._clock()
            fetched: List[InjectItem] = []
            for src in (sources if sources is not None else self.sources):
                try:
                    fetched.extend(src.fetch(hall=hall, leader=leader, owner=owner))
                except Exception as e:  # noqa: BLE001 — 单源失败不拖累其它源
                    logger.warning("krow source %s failed: %s", getattr(src, "name", src), e)
            new = self.store.add_pending(fetched)
            self.last_new = len(new)
            pending = self.store.take_pending()
            if not hall:
                pending = [p for p in pending if p.kind != "hall"]
            if not leader:
                pending = [p for p in pending if p.kind != "leader"]
            if not owner:
                pending = [p for p in pending if p.kind not in ("owner", "mail")]
            return pending

    def mark_delivered(self, items: Iterable[InjectItem]) -> None:
        keys = [it.key for it in items]
        if keys:
            self.store.mark_delivered(keys)
            self.delivered += len(keys)

    # ── 执行中注入（ExecutionEngine 调） ──────────────────────────────────

    def interjection_text(self, ctx: Any = None) -> Optional[str]:
        """工具调用边界：有话就返回成品文本（已套抬头、已夹长度），并把这批记为已送达。无话 → None。"""
        if not self.enabled:
            return None
        items = self.poll()
        if not items:
            return None
        text = render_items(items, self.nonce)
        self.mark_delivered(items)
        return text or None

    def owner_only_text(self, ctx: Any = None) -> Optional[str]:
        """轮尾：只拉主人插话（REST 源；``hall=0`` 且不传 ``max_leader``——大厅是数据，不值得为它续跑）。"""
        if not self.enabled:
            return None
        rest = [s for s in self.sources if getattr(s, "name", "") == "rest"]
        if not rest:
            return None
        items = [p for p in self.poll(hall=False, leader=False, owner=True, sources=rest) if p.kind == "owner"]
        if not items:
            return None
        text = render_items(items, self.nonce)
        self.mark_delivered(items)
        return text or None

    # ── 空闲轮询线程 ─────────────────────────────────────────────────────

    def start_idle_poller(
        self,
        on_items: Callable[[List[InjectItem]], None],
        *,
        is_idle: Callable[[], bool] = lambda: True,
        interval_s: Optional[float] = None,
        name: str = "krow-idle-poller",
    ) -> Optional[threading.Thread]:
        """后台线程：``interval_s`` 一拍；只在 ``is_idle()`` 为真且 bridge 不忙时拉；有待送项就交 ``on_items``。

        ``on_items`` 拿到的是待送项（没出队）：由它决定怎么送、送完 :meth:`mark_delivered`。重复起线程是 no-op。
        """
        if not self.enabled:
            return None
        if self._poller and self._poller.is_alive():
            return self._poller
        interval = float(interval_s or self.cfg.poll_interval_s)
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    if self.busy or not is_idle():
                        continue
                    items = self.poll()
                    if items:
                        on_items(items)
                except Exception as e:  # noqa: BLE001 — 轮询线程绝不因一次失败退出
                    logger.warning("krow idle poller: %s", e)

        self._poller = threading.Thread(target=_loop, daemon=True, name=name)
        self._poller.start()
        return self._poller

    def stop(self) -> None:
        self._stop.set()


# ── 装配 ──────────────────────────────────────────────────────────────────


def _mcp_call_factory(mcp_manager: Any, preferred: Optional[str]) -> Callable[[str, Dict[str, Any]], str]:
    """返回 ``call_tool(tool, args)``：每次调用时再解析服务器名（MCP 服务器是后台懒启动的）。"""
    state: Dict[str, Optional[str]] = {"server": None}

    def _call(tool: str, args: Dict[str, Any]) -> str:
        if not state["server"]:
            state["server"] = find_krow_server(mcp_manager, preferred)
        if not state["server"]:
            raise RuntimeError("krow MCP server not found/ready (set krow.mcp_server or wait for MCP startup)")
        return mcp_manager.call_tool(state["server"], tool, args)

    return _call


def build_bridge(
    config: Any,
    *,
    mcp_manager: Any = None,
    environ: Optional[Mapping[str, str]] = None,
    store: Optional[DedupStore] = None,
) -> Optional[KrowBridge]:
    """按 ``config.krow`` 与环境装配 bridge；未启用或一个取数源都装不出来返回 None（调用方当作「没接 krow」）。"""
    raw = getattr(config, "krow", None)
    if raw is None and isinstance(config, Mapping):
        raw = config.get("krow")
    cfg = KrowConfig(raw if isinstance(raw, Mapping) else None)
    if not cfg.enabled:
        return None
    env = environ if environ is not None else os.environ
    st = store or DedupStore(cfg.state_path)
    sources: List[Any] = []

    if cfg.rest != "off":
        rest_env = resolve_rest_env(env)
        if rest_env:
            sources.append(
                RestInjectSource(
                    rest_env["api"],
                    rest_env["token"],
                    binding=rest_env["binding"],
                    max_owner=cfg.max_owner_per_step,
                    max_hall=cfg.max_hall_per_step,
                )
            )
        elif cfg.rest == "on":
            logger.warning("krow.rest=on but KW_API_ENDPOINT / KW_WORKER_TOKEN missing; REST source skipped")

    mcp_cfg = getattr(config, "mcp", None)
    mcp_enabled = bool(isinstance(mcp_cfg, Mapping) and mcp_cfg.get("enabled"))
    if mcp_manager is not None and (mcp_enabled or getattr(mcp_manager, "servers", None)):
        sources.append(
            McpHallSource(
                _mcp_call_factory(mcp_manager, cfg.mcp_server),
                self_names=cfg.self_names,
                mode=cfg.mode,
                inbox=cfg.inbox,
                ack=cfg.ack,
                cursor_get=st.cursor,
                cursor_set=st.set_cursor,
            )
        )

    if not sources:
        logger.warning("krow.enabled=true but no source available (no KW_* env, no krow MCP server); bridge disabled")
        return None
    bridge = KrowBridge(cfg, store=st, sources=sources)
    env_nonce = (env.get("KW_INJECT_NONCE") or "").strip() if hasattr(env, "get") else ""
    if env_nonce:
        bridge.nonce = env_nonce
    logger.info("krow bridge ready: sources=%s", [getattr(s, "name", "?") for s in sources])
    return bridge
