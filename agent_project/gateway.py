"""多端网关抽象(对标 Hermes gateway/session).

现状三端各自直调 agent, 事件形状各异:
- web/server.py      : agent.run(task, stream_callback) + (kind, token) pump -> WS JSON
- tools/telegram_bot : agent_callback(tg_msg, ctx) -> response 文本
- super_agent.py     : agent.run(prompt, code_mode=False) 直接同步调用

本模块只做抽象, 不合并老入口:
- ChannelMessage: 统一入站消息(渠道/用户/会话/文本/附件)
- ChannelEvent  : 统一出站事件(kind, token) —— 与现有 stream_callback 同形
- Channel(ABC)  : handle_inbound() 默认实现 = 先排队(FIFO)再 agent.run + 事件扇出
                  agent 忙时消息排队等待而非失败, 队列满则快速拒绝并给出可读原因
- Gateway       : 渠道注册表 + dispatch, 按 chat_id 做最小会话路由键
- LocalChannel  : 内存样板实现(单测/冒烟/脚本用), 证明抽象可用
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ChannelMessage:
    """统一入站消息."""

    text: str
    channel: str = "local"
    user_id: str = ""
    chat_id: str = ""
    message_id: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        """最小会话路由键: 渠道 + 会话(无会话则按用户)."""
        return f"{self.channel}:{self.chat_id or self.user_id or 'default'}"


@dataclass
class ChannelEvent:
    """统一出站事件, 与现有 stream_callback(kind, token) 同形."""

    kind: str
    token: str


class Channel(ABC):
    """渠道抽象: 入站调度默认实现, 出站投递由各端实现.

    排队语义: 同一渠道同一时刻只跑一个 agent.run; agent 忙时后续消息按
    先来先服务(FIFO)排队等待, 排到即执行, 不丢不撞车.
    """

    name = "base"
    max_queue: int = 64  # 等待中的消息上限, 超出快速拒绝不无脑堆积

    def __init__(self, agent: Any = None):
        self.agent = agent
        self.inbound = 0
        self.errors = 0
        self.queued_total = 0
        self.rejected = 0
        self._cv = threading.Condition()
        self._active = False
        self._active_thread: Optional[int] = None
        self._waiters = 0

    def attach(self, agent: Any) -> None:
        self.agent = agent

    @abstractmethod
    def sink(self, event: ChannelEvent, message: ChannelMessage) -> None:
        """投递单个事件(各端实现: WS 发送 / reply_text / 终端打印)."""

    def queue_depth(self) -> int:
        """当前排队等待中的消息数(不含正在执行的那条)."""
        with self._cv:
            return self._waiters

    def _take_ticket(self) -> bool:
        """排队等令牌; 返回 False 表示快速拒绝(同线程重入或队列已满).

        FIFO 说明: 等待方在同一 Condition 上等唤醒, 先到先等(近似先到先跑);
        渠道级串行的关键是"不并发、不丢", 严格逐条顺序由到达顺序近似保证.
        """
        ident = threading.get_ident()
        with self._cv:
            # 同线程重入(run 流程内部触发 handle_inbound): 回队即死锁, 快速拒绝
            if self._active and self._active_thread == ident:
                return False
            if self._active and self._waiters >= self.max_queue:
                return False
            if self._active:
                self._waiters += 1
                while self._active:
                    self._cv.wait()
                self._waiters -= 1
            self._active = True
            self._active_thread = ident
        return True

    def _release_ticket(self) -> None:
        """交出令牌并唤醒下一个等待者."""
        with self._cv:
            if self._active_thread == threading.get_ident():
                self._active = False
                self._active_thread = None
                self._cv.notify_all()

    @staticmethod
    def _busy_result(reason: str) -> Dict[str, Any]:
        return {"success": False, "final_answer": f"消息排队失败: {reason}",
                "actions": [], "metadata": {"queued": False, "reason": reason}}

    def handle_inbound(self, message: ChannelMessage,
                       extra_callbacks: Optional[List[Callable[[ChannelEvent], None]]] = None) -> Dict[str, Any]:
        """入站入口: 先排队(FIFO), 轮到再调度; 忙则等待而非失败."""
        self.queued_total += 1
        if not self._take_ticket():
            self.rejected += 1
            if self._active and self._active_thread == threading.get_ident():
                return self._busy_result("同线程正在处理中(禁止重入)")
            return self._busy_result(f"队列已满({self.max_queue}), 稍后再试")
        try:
            return self.run_agent(message, extra_callbacks=extra_callbacks)
        finally:
            self._release_ticket()

    def run_agent(self, message: ChannelMessage,
                  extra_callbacks: Optional[List[Callable[[ChannelEvent], None]]] = None) -> Dict[str, Any]:
        """默认调度: agent.run + 事件扇出. 各端一般无需重写."""
        if self.agent is None:
            raise RuntimeError(f"channel {self.name}: agent not attached")
        self.inbound += 1
        sinks = list(extra_callbacks or [])

        def stream_cb(kind: str, token: str) -> None:
            ev = ChannelEvent(kind=kind, token=token or "")
            try:
                self.sink(ev, message)
            except Exception:
                self.errors += 1
            for cb in sinks:
                try:
                    cb(ev)
                except Exception:
                    self.errors += 1  # 旁路回调失败只计数, 不掀主流程

        try:
            result = self.agent.run(message.text, stream_callback=stream_cb)
        except Exception as e:  # noqa: BLE001 — 渠道层兜底, 转错误结果不断连
            self.errors += 1
            result = {"success": False, "final_answer": f"处理失败: {type(e).__name__}",
                      "actions": [], "metadata": {"error": str(e)[:300]}}
        if not isinstance(result, dict):
            result = {"success": True, "final_answer": str(result),
                      "actions": [], "metadata": {}}
        return result

    def stats(self) -> Dict[str, Any]:
        return {"channel": self.name, "inbound": self.inbound, "errors": self.errors,
                "queued_total": self.queued_total, "queue_depth": self.queue_depth(),
                "rejected": self.rejected}


class LocalChannel(Channel):
    """内存样板渠道: 事件记入 outbox, 供单测/冒烟/脚本验证抽象."""

    name = "local"

    def __init__(self, agent: Any = None):
        super().__init__(agent)
        self.outbox: List[ChannelEvent] = []

    def sink(self, event: ChannelEvent, message: ChannelMessage) -> None:
        self.outbox.append(event)

    def last_answer_events(self) -> List[str]:
        return [e.token for e in self.outbox if e.kind == "content"]


class Gateway:
    """渠道注册表 + 按会话键 dispatch."""

    def __init__(self):
        self.channels: Dict[str, Channel] = {}
        self.sessions_seen: Dict[str, int] = {}

    def register(self, channel: Channel) -> None:
        self.channels[channel.name] = channel

    def dispatch(self, message: ChannelMessage,
                 extra_callbacks: Optional[List[Callable[[ChannelEvent], None]]] = None) -> Dict[str, Any]:
        ch = self.channels.get(message.channel)
        if ch is None:
            raise KeyError(f"unknown channel: {message.channel}")
        self.sessions_seen[message.session_key] = self.sessions_seen.get(message.session_key, 0) + 1
        # 走 handle_inbound 获得排队语义(agent 忙则 FIFO 等待); 直连 run_agent 无排队
        return ch.handle_inbound(message, extra_callbacks=extra_callbacks)

    def stats(self) -> Dict[str, Any]:
        return {"channels": sorted(self.channels),
                "sessions": dict(self.sessions_seen),
                **{name: ch.stats() for name, ch in self.channels.items()}}
