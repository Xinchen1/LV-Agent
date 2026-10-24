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
      channels: auto                # REST 逐频道轮询: auto | off | [channel_id(UUID), ...]; 非 UUID 项忽略、不中断其余频道
      channel_scan_interval_s: 10   # 频道扫描节流
      bad_channel_backoff_s: 300    # 显式 channel_id 回 400/404 后该频道隔离多久再试; 其余频道不受影响
      # ── v3 守候循环(收件箱长轮询 → 大厅点名闭环 / 主人直聊) ──
      mail_loop: true               # 常驻 krow_inbox(wait=20, unread_only=true); 关掉就退回 v1 的"每拍 team_inbox(wait=0)"
      mail_wait_s: 20               # 长轮询秒数, 夹到 0..25
      mail_backoff_max_s: 60        # 收件箱出错时指数退避上限(2s 起)
      mention_context: 20           # 点名时另取多少条频道上下文给模型(0 = 不取)
      reply_max_chars: 1800         # 回帖 / 回信正文上限
      post_retries: 2               # 回帖 / 回信瞬时错误重试次数(4xx 不重试)
      owner_reply: true             # 主人直聊(from_kind=owner)跑完回合后用 krow_send_message(to_role=owner, in_reply_to) 回信
      tools: {}                     # 工具注册名覆盖, 如 {team_inbox: krow_inbox}; 缺省按服务器工具列表探测(krow_* 优先)

``status()`` 里 REST 相关字段：``rest_channels`` / ``rest_invalid_channels`` / ``rest_bad_channels`` / ``rest_legacy_mode`` /
``rest_409_seen`` + ``rest_409_note``（409 只表示已见 / 未见：有 409 ⇒ 令牌非 mcp_pull 为单向推断，未见 409 不等于 mcp_pull 已确认，
不据此断言平台侧 H1 已回答）；``mail_loop``（守候循环：拍数 / 已处理点名与主人信 / 连续错误 / 点名账各状态条数）；``tools``（逻辑名 → 注册名）。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .items import InjectItem, contract_text, make_nonce, render_items
from .mailloop import KrowMailLoop, clamp_wait
from .sources import McpHallSource, RestInjectSource, resolve_rest_env
from .state import DEFAULT_STATE_PATH, DedupStore
from .tools import KrowTools

logger = logging.getLogger("krow.bridge")

__all__ = ["KrowBridge", "KrowConfig", "build_bridge", "find_krow_server", "list_server_tools"]

_SERVER_NAME_CANDIDATES = ("krow-platform", "krow_platform", "krow", "agent-hub", "agent_hub", "krow-agent-hub")
#: 探测"哪台 MCP 服务器是 krow 桥": 有这些工具之一就是(agent-hub 叫 krow_*, 平台 kit 桥叫 hall_* / team_*)
_PROBE_TOOLS = ("krow_hall_read", "krow_inbox", "hall_read", "team_inbox")


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
        # REST 路径的项目 / 小组频道逐个轮询(D11): "auto"(按 hall/channels 计数挑) | "off" | [channel_id, ...]
        ch = d.get("channels", "auto")
        if isinstance(ch, (list, tuple)):
            self.channels: Any = [str(c).strip() for c in ch if str(c).strip()] or "off"
        else:
            self.channels = str(ch or "auto").strip().lower() or "auto"
            if self.channels not in ("auto", "off"):
                self.channels = [str(ch).strip()]
        self.channel_scan_interval_s = max(0.0, float(d.get("channel_scan_interval_s", 10) or 0))
        # 显式 channel_id 回 400/404 等频道级错误后该频道隔离多久(秒); 其余频道不受影响(v2 坏频道隔离)
        self.bad_channel_backoff_s = max(0.0, float(d.get("bad_channel_backoff_s", 300) or 0))
        # v3 守候循环: 收件箱长轮询 → 大厅点名闭环 / 主人直聊
        self.mail_loop = bool(d.get("mail_loop", True))
        self.mail_wait_s = clamp_wait(d.get("mail_wait_s", 20))
        self.mail_backoff_max_s = max(2.0, float(d.get("mail_backoff_max_s", 60) or 60))
        self.mention_context = max(0, min(100, int(d.get("mention_context", 20) or 0)))
        self.reply_max_chars = max(80, int(d.get("reply_max_chars", 1800) or 1800))
        self.post_retries = max(0, int(d.get("post_retries", 2) or 0))
        self.owner_reply = bool(d.get("owner_reply", True))
        tools = d.get("tools")
        self.tools: Dict[str, str] = {str(k): str(v) for k, v in tools.items() if k and v} if isinstance(tools, Mapping) else {}


def list_server_tools(mcp_manager: Any, server: str) -> List[str]:
    """某台 MCP 服务器当前的工具名（纯 Python 连接读 ``conn.tools``；Rust 桥问 ``list_tools``）。拿不到 / 还没就绪返回空。"""
    if mcp_manager is None or not server:
        return []
    servers = getattr(mcp_manager, "servers", None) or {}
    conn = servers.get(server)
    if conn is not None:
        ready = getattr(conn, "_ready", None)
        if ready is not None and hasattr(ready, "is_set") and not ready.is_set():
            return []
        return [str(t.get("name")) for t in (getattr(conn, "tools", None) or []) if isinstance(t, dict) and t.get("name")]
    bridge = getattr(mcp_manager, "_bridge", None)
    if bridge is not None and server in (getattr(mcp_manager, "_bridge_configs", None) or {}):
        try:
            resp = bridge.list_tools(server, timeout=10)
            tools = (resp.get("metadata") or {}).get("tools", []) if isinstance(resp, dict) and resp.get("success") else []
            return [str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")]
        except Exception as e:  # noqa: BLE001
            logger.debug("krow: rust bridge list_tools(%s) failed: %s", server, e)
    return []


def find_krow_server(mcp_manager: Any, preferred: Optional[str] = None) -> Optional[str]:
    """在 ``McpManager`` 里找 krow 桥：先按名字，再按「哪台服务器有 krow 工具（``krow_hall_read`` / ``hall_read`` …）」。找不到 / 还没就绪返回 None。"""
    if mcp_manager is None:
        return None
    servers = getattr(mcp_manager, "servers", None) or {}
    bridge_cfgs = getattr(mcp_manager, "_bridge_configs", None) or {}
    known = list(servers.keys()) + [k for k in bridge_cfgs.keys() if k not in servers]
    for name in ([preferred] if preferred else []) + list(_SERVER_NAME_CANDIDATES):
        if name and name in known:
            return name
    # 按工具名探测（纯 Python stdio 连接：conn.tools 在 _ready 后可用；Rust 桥问 list_tools）
    for name in known:
        if any(t in _PROBE_TOOLS for t in list_server_tools(mcp_manager, name)):
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
        # v3: 收件箱守候循环(长轮询 → 大厅点名闭环 / 主人直聊); build_bridge 装配, 宿主再 set_run_task 接上模型回合
        self.mail_loop: Optional[KrowMailLoop] = None
        self.tools: Optional[KrowTools] = None

    # ── 状态 ───────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and (self.sources or self.mail_loop is not None))

    def attach_mail_loop(self, loop: KrowMailLoop) -> None:
        loop.nonce = loop.nonce or self.nonce
        self.mail_loop = loop

    def set_run_task(self, run_task: Optional[Callable[[str], Any]]) -> None:
        """宿主把「跑一回合」的函数接进来（守护模式 / REPL 都要：点名作答、主人直聊回信都靠它）。"""
        if self.mail_loop is not None:
            self.mail_loop.run_task = run_task

    def process_mail_queue(self) -> Dict[str, Any]:
        """REPL 主线程：处理空闲线程排进来的信（大厅点名读→答→回帖、主人直聊回信、其余进待送队列）。"""
        if self.mail_loop is None:
            return {"owner": 0, "mentions": 0, "other": 0, "new": 0}
        self.set_busy(True)
        try:
            return self.mail_loop.process_queued()
        finally:
            self.set_busy(False)

    def mail_queued(self) -> int:
        return self.mail_loop.queued() if self.mail_loop is not None else 0

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
        rest = next((s for s in self.sources if getattr(s, "name", "") == "rest"), None)
        rest_status: Dict[str, Any] = rest.status() if rest is not None and hasattr(rest, "status") else {}
        return {
            "enabled": self.enabled,
            "sources": [getattr(s, "name", type(s).__name__) for s in self.sources],
            "source_errors": {getattr(s, "name", type(s).__name__): getattr(s, "last_error", "") for s in self.sources},
            "source_calls": {getattr(s, "name", type(s).__name__): getattr(s, "calls", 0) for s in self.sources},
            # REST 路径: 频道轮询策略 / 坏频道隔离 / 旧接口判定 / 409 门禁
            "rest_channels": rest_status.get("channels"),
            "rest_invalid_channels": rest_status.get("invalid_channels", []),  # 固定列表里不是 UUID、被忽略的项
            "rest_bad_channels": rest_status.get("bad_channels", {}),  # 显式拉取回 400/404 等被隔离的频道(含 retry_at)
            "rest_legacy_mode": rest_status.get("legacy_mode"),  # None 未判定 / True 老口子 / False 统一口子(只由缺省调用判定)
            # 409 只说明"已见 / 未见": 有 409 ⇒ 令牌非 mcp_pull(单向); 未见 409 不等于 mcp_pull 已确认. 不据此断言 H1 已回答.
            "rest_409_seen": bool(rest_status.get("saw_409", False)),
            "rest_409_note": rest_status.get("note_409") if rest is not None else "无 REST 源：没发过请求，谈不上已见 / 未见 409；H1 未由此回答",
            "polls": self.polls,
            "delivered": self.delivered,
            "last_poll_at": self.last_poll_at,
            "last_new": self.last_new,
            "busy": self.busy,
            "poller_alive": bool(self._poller and self._poller.is_alive()),
            "store": self.store.stats(),
            # v3 守候循环 / 工具名解析
            "mail_loop": self.mail_loop.status() if self.mail_loop is not None else None,
            "tools": self.tools.resolved() if self.tools is not None else {},
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
        """后台线程：只在 ``is_idle()`` 为真且 bridge 不忙时拉；有待送项就交 ``on_items``。

        没有守候循环时 ``interval_s`` 一拍 ``poll()``（v1）。有守候循环时每拍先 ``mail_loop.wait_once(handle=False)`` 长轮询收件箱
        （长轮询本身就是等待；入队不处理——agent 回合要在主线程跑），再 ``poll()`` 其它源；队列里有东西或有待送项就叫 ``on_items``，
        队列没被主线程取走前不再打收件箱（未读信会原样再回来）。
        ``on_items`` 拿到的是待送项（没出队）：由它决定怎么送、送完 :meth:`mark_delivered`。重复起线程是 no-op。
        """
        if not self.enabled:
            return None
        if self._poller and self._poller.is_alive():
            return self._poller
        interval = float(interval_s or self.cfg.poll_interval_s)
        self._stop.clear()
        loop = self.mail_loop
        if loop is not None:
            loop.reset_stop()

        def _loop_v1() -> None:
            while not self._stop.wait(interval):
                try:
                    if self.busy or not is_idle():
                        continue
                    items = self.poll()
                    if items:
                        on_items(items)
                except Exception as e:  # noqa: BLE001 — 轮询线程绝不因一次失败退出
                    logger.warning("krow idle poller: %s", e)

        def _loop_v3() -> None:
            assert loop is not None
            while not self._stop.is_set():
                try:
                    if self.busy or not is_idle():
                        self._stop.wait(1.0)
                        continue
                    if loop.queued() > 0:
                        # 主线程还没取走上一批: 不再打收件箱(未读信会原样再回来), 只按间隔看别的源
                        on_items(self.poll())
                        self._stop.wait(min(interval, 5.0))
                        continue
                    summary = loop.wait_once(wait=min(loop.wait_s, int(interval)), handle=False)
                    items = self.poll()
                    if items or summary.get("queued"):
                        on_items(items)
                    delay = float(summary.get("delay") or 0.0)
                    if delay > 0:
                        self._stop.wait(delay)
                except Exception as e:  # noqa: BLE001
                    logger.warning("krow idle poller: %s", e)
                    self._stop.wait(2.0)

        self._poller = threading.Thread(target=_loop_v3 if loop is not None else _loop_v1, daemon=True, name=name)
        self._poller.start()
        return self._poller

    def stop(self) -> None:
        self._stop.set()
        if self.mail_loop is not None:
            self.mail_loop.stop()


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

    _call.state = state  # type: ignore[attr-defined] — list_tools 共用解析结果
    return _call


def _mcp_tools(mcp_manager: Any, cfg: KrowConfig) -> KrowTools:
    """装 :class:`KrowTools`：原始调用绑到 McpManager，工具列表按解析出的服务器懒取（没就绪返回空 → 先用 krow_* 候选、失败再换）。"""
    call = _mcp_call_factory(mcp_manager, cfg.mcp_server)

    def _list() -> List[str]:
        server = call.state.get("server") or find_krow_server(mcp_manager, cfg.mcp_server)  # type: ignore[attr-defined]
        return list_server_tools(mcp_manager, server) if server else []

    return KrowTools(call, list_tools=_list, overrides=cfg.tools)


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
                    channels=cfg.channels,
                    mode=cfg.mode,
                    channel_scan_interval=cfg.channel_scan_interval_s,
                    bad_channel_backoff=cfg.bad_channel_backoff_s,
                )
            )
        elif cfg.rest == "on":
            logger.warning("krow.rest=on but KW_API_ENDPOINT / KW_WORKER_TOKEN missing; REST source skipped")

    mcp_cfg = getattr(config, "mcp", None)
    mcp_enabled = bool(isinstance(mcp_cfg, Mapping) and mcp_cfg.get("enabled"))
    tools: Optional[KrowTools] = None
    mail_loop: Optional[KrowMailLoop] = None
    if mcp_manager is not None and (mcp_enabled or getattr(mcp_manager, "servers", None) or getattr(mcp_manager, "_bridge_configs", None)):
        tools = _mcp_tools(mcp_manager, cfg)
        # 守候循环接管收件箱(长轮询 + 分诊 + 点名闭环); 开着时 MCP 大厅源不再每拍 team_inbox(wait=0), 免得同一封信两头抢
        use_loop = bool(cfg.mail_loop)
        sources.append(
            McpHallSource(
                tools.call,  # 逻辑名 = v1 的叫法, 由 KrowTools 映射到服务器上的注册名(krow_* 优先)
                self_names=cfg.self_names,
                mode=cfg.mode,
                inbox=(cfg.inbox and not use_loop),
                ack=cfg.ack,
                cursor_get=st.cursor,
                cursor_set=st.set_cursor,
            )
        )
        if use_loop and cfg.inbox:
            mail_loop = KrowMailLoop(
                tools,
                st,
                wait_s=cfg.mail_wait_s,
                backoff_max_s=cfg.mail_backoff_max_s,
                context_limit=cfg.mention_context,
                reply_max_chars=cfg.reply_max_chars,
                post_retries=cfg.post_retries,
                owner_reply=cfg.owner_reply,
                self_names=cfg.self_names,
            )

    if not sources and mail_loop is None:
        logger.warning("krow.enabled=true but no source available (no KW_* env, no krow MCP server); bridge disabled")
        return None
    bridge = KrowBridge(cfg, store=st, sources=sources)
    env_nonce = (env.get("KW_INJECT_NONCE") or "").strip() if hasattr(env, "get") else ""
    if env_nonce:
        bridge.nonce = env_nonce
    bridge.tools = tools
    if mail_loop is not None:
        bridge.attach_mail_loop(mail_loop)
    logger.info("krow bridge ready: sources=%s mail_loop=%s", [getattr(s, "name", "?") for s in sources], mail_loop is not None)
    return bridge
