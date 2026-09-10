"""多端网关抽象(对标 Hermes gateway/session).

现状三端各自直调 agent, 事件形状各异:
- web/server.py      : agent.run(task, stream_callback) + (kind, token) pump -> WS JSON
- tools/telegram_bot : agent_callback(tg_msg, ctx) -> response 文本
- super_agent.py     : agent.run(prompt, code_mode=False) 直接同步调用

本模块只做抽象, 不合并老入口(行为不变):
- ChannelMessage: 统一入站消息(渠道/用户/会话/文本/附件)
- ChannelEvent  : 统一出站事件(kind, token) —— 与现有 stream_callback 同形
- Channel(ABC)  : handle_inbound() 默认实现 = agent.run + 事件扇出,
                  各端只需实现投递(sink), 不必重写调度
- Gateway       : 渠道注册表 + dispatch, 按 chat_id 做最小会话路由键
- LocalChannel  : 内存样板实现(单测/冒烟/脚本用), 证明抽象可用
"""

from __future__ import annotations

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
    """渠道抽象: 入站调度默认实现, 出站投递由各端实现."""

    name: str = "base"

    def __init__(self, agent: Any = None):
        self.agent = agent
        self.inbound = 0
        self.errors = 0

    def attach(self, agent: Any) -> None:
        self.agent = agent

    @abstractmethod
    def sink(self, event: ChannelEvent, message: ChannelMessage) -> None:
        """投递单个事件(各端实现: WS 发送 / reply_text / 终端打印)."""

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

    def handle_inbound(self, message: ChannelMessage) -> Dict[str, Any]:
        """入站入口(默认 = 调度; 需要鉴权/排队的端可重写前置)."""
        return self.run_agent(message)

    def stats(self) -> Dict[str, Any]:
        return {"channel": self.name, "inbound": self.inbound, "errors": self.errors}


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
        return ch.run_agent(message, extra_callbacks=extra_callbacks)

    def stats(self) -> Dict[str, Any]:
        return {"channels": sorted(self.channels),
                "sessions": dict(self.sessions_seen),
                **{name: ch.stats() for name, ch in self.channels.items()}}
