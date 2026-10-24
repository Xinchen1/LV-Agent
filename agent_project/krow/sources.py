"""取数源：REST 统一注入口子（首选）与 MCP 工具轮询（回退 / 补充）.

两条路径都把平台消息归一成 :class:`~agent_project.krow.items.InjectItem`，去重交给
:class:`~agent_project.krow.state.DedupStore`，拼文案交给 :func:`~agent_project.krow.items.render_items`。

**RestInjectSource** —— 与平台 kit 钩子同一口子、同一参数、同一回落：
``GET {KW_API_ENDPOINT}/agent/v1/inject?max_owner=3&max_hall=5&max_leader=1[&hall=0]``，
Bearer = ``KW_WORKER_TOKEN``（``KROW_TASK_SLOT=conversation`` 时用 ``KW_WORKER_TOKEN_CONVERSATION``），
可选头 ``X-Krow-Job-Binding``；老服务端 404 → ``GET /agent/v1/hall/inject?max=5``。服务端读即消费。
连不上 / 超时 / 5xx 进 30 秒冷却，窗口内不出网（滚动发布时只付一刀 3 秒）。
**只读环境变量名、不打印值**；token 不进日志、不进异常文本。

**McpHallSource** —— 只依赖一个 ``call_tool(tool_name, arguments) -> str`` 回调（由 bridge 绑到 LV-Agent 已配置的
krow MCP 服务器上），轮询 ``hall_channels`` → 有 ``mention_count``（或 ``unread_count``）的频道 → ``hall_read(unread_only)``
→ 解析每行 ``#<id> [<显示名>] <MM-DD HH:MM> <正文>`` → 按 ``@你`` / ``@<我的名字>`` 判点名 → 落账后 ``hall_ack``；
``team_inbox(unread_only=true, wait=0)`` 补站内信（不标已读——读全文是模型自己的决定）。
这条路径不依赖 ``inject`` 口子对 ``suppressed(not_wakeable)`` 行是否放行——那是外部工作台能不能被原生注入叫醒的
关键未知项，需要 captain 用自己的席位实测；无论放不放行，MCP 轮询都能把点名捞上来。
"""
from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .items import (
    InjectItem,
    extract_channel_id,
    hall_key,
    leader_key,
    mail_key,
    owner_key,
)

logger = logging.getLogger("krow.sources")

__all__ = [
    "HttpGet",
    "McpHallSource",
    "RestInjectSource",
    "parse_hall_read",
    "parse_inject_hall_text",
    "resolve_rest_env",
]

INJECT_PATH = "/agent/v1/inject"
HALL_INJECT_PATH = "/agent/v1/hall/inject"
MAX_OWNER_PER_STEP = 3
MAX_HALL_PER_STEP = 5
MAX_LEADER_PER_STEP = 1
TIMEOUT_SECS = 3.0
COOLDOWN_SECS = 30.0
JOB_BINDING_HEADER = "X-Krow-Job-Binding"

#: ``(url, headers, timeout) -> (status_code, body_text)``；非 2xx 直接抛 ``urllib.error.HTTPError``。
HttpGet = Callable[[str, Mapping[str, str], float], Tuple[int, str]]

_HALL_LINE_RE = re.compile(r"^#(?P<id>\d+)\s+(?P<rest>.*)$")
_HALL_READ_LINE_RE = re.compile(
    r"^#(?P<id>\d+) \[(?P<display>[^\]]*)\] (?P<time>\d{2}-\d{2} \d{2}:\d{2}|--:--|\S+) (?P<body>.*)$"
)
_MENTION_ME_MARK = "@你"


# ── 环境 ──────────────────────────────────────────────────────────────────


def resolve_rest_env(environ: Mapping[str, str]) -> Dict[str, str]:
    """算出 REST 口子要用的 ``{"api", "token", "binding", "nonce"}``；缺 api 或 token 返回空 dict。

    空串按缺失处理。``KW_INJECT_HOOK=0`` 整个关掉（与 kit 同名开关，方便和 kit 共存时一键静默）。
    只在这里读 token 值，调用方拿到的 dict 不得打印、不得写日志。
    """
    env = {k: v for k, v in environ.items() if isinstance(v, str) and v}
    if (env.get("KW_INJECT_HOOK") or "").strip() == "0":
        return {}
    api = (env.get("KW_API_ENDPOINT") or "").strip().rstrip("/")
    slot = (env.get("KROW_TASK_SLOT") or "").strip().lower()
    token_name = "KW_WORKER_TOKEN_CONVERSATION" if slot == "conversation" else "KW_WORKER_TOKEN"
    token = (env.get(token_name) or "").strip()
    if not api or not token:
        return {}
    return {
        "api": api,
        "token": token,
        "binding": (env.get("KROW_JOB_BINDING") or "").strip(),
        "nonce": (env.get("KW_INJECT_NONCE") or "").strip(),
    }


def _default_http_get(url: str, headers: Mapping[str, str], timeout: float) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status or 200), resp.read().decode("utf-8", "replace")


def _is_server_down(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError))


# ── 解析 ──────────────────────────────────────────────────────────────────


def parse_inject_hall_text(text: str, *, source: str = "rest") -> List[InjectItem]:
    """服务端 ``hall.text`` 是多行成品（``inject.rs::render_line``：每条 ``#<id> <显示名>：<正文>``，长正文可能跨行）。

    按 ``#<id>`` 起新项，其余行并入上一项；整段没有 ``#<id>`` 就当一条（键按正文散列）。
    """
    items: List[InjectItem] = []
    cur: Optional[InjectItem] = None
    for line in (text or "").splitlines():
        m = _HALL_LINE_RE.match(line.strip())
        if m:
            mid = int(m.group("id"))
            body = line.strip()
            cur = InjectItem(
                key=hall_key(mid),
                kind="hall",
                text=body,
                source=source,
                message_id=mid,
                channel_id=extract_channel_id(body),
                mentions_me=_MENTION_ME_MARK in body,
            )
            items.append(cur)
        elif cur is not None and line.strip():
            cur.text += "\n" + line.rstrip()
            if cur.channel_id is None:
                cur.channel_id = extract_channel_id(line)
            if _MENTION_ME_MARK in line:
                cur.mentions_me = True
    if not items and (text or "").strip():
        body = text.strip()
        items.append(
            InjectItem(
                key=hall_key(None, body),
                kind="hall",
                text=body,
                source=source,
                channel_id=extract_channel_id(body),
                mentions_me=_MENTION_ME_MARK in body,
            )
        )
    return items


def _legacy_hall_items(data: Mapping[str, Any]) -> List[InjectItem]:
    """老口子 ``/agent/v1/hall/inject`` 的结构化 ``items[]`` → 注入项（与 kit ``hall_inject.to_interjection`` 同一写法）。"""
    out: List[InjectItem] = []
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        return out
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        display = str(raw.get("display") or "").strip() or "？"
        body = str(raw.get("body") or "").strip()
        atts = raw.get("attachments")
        has_atts = isinstance(atts, list) and any(isinstance(a, dict) for a in atts)
        if not body and not has_atts:
            continue
        mid = raw.get("id")
        text = body or "（无正文）"
        mentions_me = bool(raw.get("mentions_me"))
        if mentions_me:
            text = _MENTION_ME_MARK + " " + text
        reply = raw.get("reply_to")
        if isinstance(reply, dict) and reply.get("id") is not None:
            text += f"（回复 #{reply['id']}）"
        if has_atts:
            names = [str(a.get("name") or a.get("filename") or a.get("file_id") or "附件") for a in atts if isinstance(a, dict)]
            text += "".join(f"（附件：{n}，用 hall_read(message_id={mid}) 取）" for n in names)
        head = f"#{mid} " if isinstance(mid, int) and mid > 0 else ""
        out.append(
            InjectItem(
                key=hall_key(mid, text),
                kind="hall",
                text=f"{head}{display}：{text}",
                source="rest",
                message_id=mid if isinstance(mid, int) else None,
                channel_id=str(raw.get("channel_id")) if raw.get("channel_id") else extract_channel_id(text),
                mentions_me=mentions_me,
            )
        )
    return out


def parse_hall_read(text: str, *, channel_id: Optional[str] = None, source: str = "mcp") -> List[InjectItem]:
    """``hall_read`` 的渲染文本（``hall_tools.render_messages``）→ 注入项。

    每条 ``#<id> [<显示名>] <MM-DD HH:MM> <正文>``；附件 / 接龙项另起缩进行，并入上一条。表头、「没有新消息」等非 ``#`` 行忽略。
    """
    items: List[InjectItem] = []
    cur: Optional[InjectItem] = None
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        m = _HALL_READ_LINE_RE.match(line)
        if m:
            mid = int(m.group("id"))
            display = m.group("display").strip() or "？"
            body = m.group("body").strip()
            cur = InjectItem(
                key=hall_key(mid),
                kind="hall",
                text=f"#{mid} {display}：{body}",
                source=source,
                message_id=mid,
                channel_id=channel_id or extract_channel_id(body),
                mentions_me=_MENTION_ME_MARK in body,
                created_at=m.group("time"),
                meta={"display": display, "body": body},
            )
            items.append(cur)
        elif cur is not None and line.startswith((" ", "\t")) and line.strip():
            cur.text += "\n" + line
    return items


def _mentions_any(body: str, names: Iterable[str]) -> bool:
    low = (body or "").lower()
    if _MENTION_ME_MARK in body:
        return True
    for n in names:
        n = (n or "").strip().lower()
        if n and f"@{n}" in low:
            return True
    return False


def _json_or_empty(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text) if text and text.strip() else {}
    except Exception:  # noqa: BLE001 — 桥把异常转成了可读文本，不是 JSON
        return {}
    return data if isinstance(data, dict) else {}


# ── REST ──────────────────────────────────────────────────────────────────


class RestInjectSource:
    """平台统一注入口子（读即消费）。``fetch(hall=..., leader=...)`` 返回本次拿到的注入项。"""

    name = "rest"

    def __init__(
        self,
        api: str,
        token: str,
        *,
        binding: str = "",
        timeout: float = TIMEOUT_SECS,
        cooldown: float = COOLDOWN_SECS,
        max_owner: int = MAX_OWNER_PER_STEP,
        max_hall: int = MAX_HALL_PER_STEP,
        max_leader: int = MAX_LEADER_PER_STEP,
        http_get: Optional[HttpGet] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.api = api.rstrip("/")
        self._token = token
        self.binding = binding
        self.timeout = timeout
        self.cooldown = cooldown
        self.max_owner = max_owner
        self.max_hall = max_hall
        self.max_leader = max_leader
        self._http_get = http_get or _default_http_get
        self._clock = clock
        self._down_until = 0.0
        self.last_error: str = ""
        self.calls = 0

    # 不要把 token 带进 repr / 日志。
    def __repr__(self) -> str:  # pragma: no cover - 可读性
        return f"RestInjectSource(api={self.api!r})"

    def cooling_down(self) -> bool:
        return self._clock() < self._down_until

    def _get(self, path: str, params: Dict[str, str]) -> Dict[str, Any]:
        url = f"{self.api}{path}?{urllib.parse.urlencode(params)}"
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if self.binding:
            headers[JOB_BINDING_HEADER] = self.binding
        self.calls += 1
        _status, body = self._http_get(url, headers, self.timeout)
        return _json_or_empty(body)

    def fetch(self, *, hall: bool = True, leader: bool = True, owner: bool = True) -> List[InjectItem]:
        if self.cooling_down():
            return []
        params = {"max_owner": str(self.max_owner), "max_hall": str(self.max_hall)}
        if leader:
            params["max_leader"] = str(self.max_leader)
        if not hall:
            params["hall"] = "0"
        try:
            try:
                data = self._get(INJECT_PATH, params)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                if not hall:
                    return []
                legacy = self._get(HALL_INJECT_PATH, {"max": str(self.max_hall)})
                self.last_error = ""
                return _legacy_hall_items(legacy)
        except Exception as e:  # noqa: BLE001 — 分类后冷却；调用方 fail-open
            self.last_error = f"{type(e).__name__}: {getattr(e, 'code', '') or ''}".strip(": ")
            if _is_server_down(e):
                self._down_until = self._clock() + self.cooldown
            logger.warning("krow rest inject failed (%s); %s", self.last_error, "cooling down" if self.cooling_down() else "no cooldown")
            return []
        self.last_error = ""
        return self._parse_unified(data, want_owner=owner, want_leader=leader, want_hall=hall)

    @staticmethod
    def _parse_unified(data: Mapping[str, Any], *, want_owner: bool, want_leader: bool, want_hall: bool) -> List[InjectItem]:
        out: List[InjectItem] = []
        owner = data.get("owner")
        if want_owner and isinstance(owner, dict):
            text = owner.get("text")
            if isinstance(text, str) and text.strip():
                out.append(InjectItem(key=owner_key(text.strip()), kind="owner", text=text.strip(), source="rest", mentions_me=True))
        leader = data.get("leader")
        if want_leader and isinstance(leader, dict) and isinstance(leader.get("items"), list):
            for it in leader["items"]:
                if not isinstance(it, dict):
                    continue
                body = str(it.get("body") or "").strip()
                if not body:
                    continue
                out.append(
                    InjectItem(
                        key=leader_key(it.get("addendum_id"), body),
                        kind="leader",
                        text=body,
                        source="rest",
                        mentions_me=True,
                        meta={
                            "author_name": it.get("author_name"),
                            "addendum_kind": it.get("addendum_kind"),
                            "task_id": it.get("task_id"),
                            "addendum_id": it.get("addendum_id"),
                        },
                    )
                )
        hall = data.get("hall")
        if want_hall and isinstance(hall, dict):
            text = hall.get("text")
            if isinstance(text, str) and text.strip():
                out.extend(parse_inject_hall_text(text))
        return out


# ── MCP ───────────────────────────────────────────────────────────────────


class McpHallSource:
    """经 LV-Agent 已配置的 krow MCP 服务器轮询大厅（与站内信）。

    ``call_tool(tool_name, arguments) -> str``：bridge 绑定到 ``McpManager.call_tool(server, tool, args)``。
    ``self_names``：我的显示名（``lv-agent`` / ``lv-agent#默认小组`` …），用于按 ``@名`` 判点名；空时按
    ``mention_count>0`` 的频道整体放行（服务端已经判定有点名，只是我们不知道自己叫什么）。
    ``mode``：``mentions``（只送点到我的）/ ``all``（有未读就送）。
    """

    name = "mcp"

    def __init__(
        self,
        call_tool: Callable[[str, Dict[str, Any]], str],
        *,
        self_names: Iterable[str] = (),
        mode: str = "mentions",
        inbox: bool = True,
        read_limit: int = 30,
        ack: bool = True,
        cursor_get: Optional[Callable[[str], Any]] = None,
        cursor_set: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        self._call = call_tool
        self.self_names = [n for n in (self_names or ()) if n]
        self.mode = mode if mode in ("mentions", "all") else "mentions"
        self.inbox = inbox
        self.read_limit = max(1, min(100, int(read_limit)))
        self.ack = ack
        self._cursor_get = cursor_get or (lambda _n: None)
        self._cursor_set = cursor_set or (lambda _n, _v: None)
        self.last_error: str = ""
        self.calls = 0

    def _tool(self, name: str, args: Dict[str, Any]) -> str:
        self.calls += 1
        return self._call(name, args) or ""

    def fetch(self, *, hall: bool = True, leader: bool = True, owner: bool = True) -> List[InjectItem]:
        out: List[InjectItem] = []
        if hall:
            try:
                out.extend(self._fetch_hall())
            except Exception as e:  # noqa: BLE001 — 轮询失败只少听一拍
                self.last_error = f"hall: {type(e).__name__}: {e}"[:200]
                logger.warning("krow mcp hall poll failed: %s", self.last_error)
        if self.inbox and owner:
            try:
                out.extend(self._fetch_inbox())
            except Exception as e:  # noqa: BLE001
                self.last_error = f"inbox: {type(e).__name__}: {e}"[:200]
                logger.warning("krow mcp inbox poll failed: %s", self.last_error)
        return out

    def _channels(self) -> List[Dict[str, Any]]:
        data = _json_or_empty(self._tool("hall_channels", {}))
        chans = data.get("channels")
        return [c for c in chans if isinstance(c, dict)] if isinstance(chans, list) else []

    def _fetch_hall(self) -> List[InjectItem]:
        items: List[InjectItem] = []
        for ch in self._channels():
            cid = str(ch.get("id") or "").strip()
            mention_n = int(ch.get("mention_count") or 0)
            unread_n = int(ch.get("unread_count") or 0)
            if self.mode == "mentions" and mention_n <= 0:
                continue
            if self.mode == "all" and unread_n <= 0 and mention_n <= 0:
                continue
            args: Dict[str, Any] = {"unread_only": True, "limit": self.read_limit}
            is_general = str(ch.get("kind") or "").lower() == "general" or str(ch.get("name") or "").strip() in ("general", "# 大厅", "大厅")
            if cid and not is_general:
                args["channel_id"] = cid
            since = self._cursor_get(f"hall:{cid or 'general'}")
            if isinstance(since, int) and since > 0:
                args["since_id"] = since
            text = self._tool("hall_read", args)
            parsed = parse_hall_read(text, channel_id=(cid if not is_general else None))
            if not parsed:
                continue
            max_id = max(p.message_id or 0 for p in parsed)
            if self.mode == "mentions":
                if self.self_names:
                    chosen = [p for p in parsed if _mentions_any(p.meta.get("body", p.text), self.self_names)]
                else:
                    chosen = parsed  # 服务端说有点名、我们不知道自己叫什么：整频道放行，由模型判断
                for p in chosen:
                    p.mentions_me = True
            else:
                chosen = parsed
                for p in chosen:
                    p.mentions_me = p.mentions_me or _mentions_any(p.meta.get("body", p.text), self.self_names)
            for p in chosen:
                p.meta.pop("body", None)
                if p.channel_id and "channel_id=" not in p.text:
                    p.text += f"（频道 channel_id={p.channel_id}）"
            items.extend(chosen)
            # 游标先走、再 ack：ack 失败也不会重复送（去重账在 bridge），下次 unread_only 仍能补读。
            if max_id > 0:
                self._cursor_set(f"hall:{cid or 'general'}", max_id)
                if self.ack:
                    ack_args: Dict[str, Any] = {"last_read_id": max_id}
                    if cid and not is_general:
                        ack_args["channel_id"] = cid
                    try:
                        self._tool("hall_ack", ack_args)
                    except Exception as e:  # noqa: BLE001 — ack 只影响未读计数
                        logger.debug("krow hall_ack failed: %s", e)
        return items

    def _fetch_inbox(self) -> List[InjectItem]:
        data = _json_or_empty(self._tool("team_inbox", {"unread_only": True, "wait": 0}))
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return []
        out: List[InjectItem] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id") or "").strip()
            if not mid:
                continue
            sender = str(m.get("from_worker_name") or m.get("from_address") or (m.get("ref") or {}).get("from_address") or "？")
            subject = str(m.get("subject") or "").strip() or "（无主题）"
            body = " ".join(str(m.get("body") or "").split())
            labels = m.get("labels") if isinstance(m.get("labels"), list) else []
            sys_labels = [str(l) for l in labels if str(l).startswith("sys:")]
            atts = m.get("attachments") if isinstance(m.get("attachments"), list) else []
            tail = f"（附件 {len(atts)} 个）" if atts else ""
            lab = f" [{' '.join(sys_labels)}]" if sys_labels else ""
            text = f"信 {mid} 来自 {sender}{lab}：{subject} — {body[:600]}{'…' if len(body) > 600 else ''}{tail}"
            out.append(
                InjectItem(
                    key=mail_key(mid),
                    kind="mail",
                    text=text,
                    source="mcp",
                    mentions_me=True,
                    created_at=str(m.get("created_at") or ""),
                    meta={"thread_id": m.get("thread_id"), "labels": sys_labels},
                )
            )
        return out
