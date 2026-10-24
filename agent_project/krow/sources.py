"""取数源：REST 统一注入口子（首选）与 MCP 工具轮询（回退 / 补充）.

两条路径都把平台消息归一成 :class:`~agent_project.krow.items.InjectItem`，去重交给
:class:`~agent_project.krow.state.DedupStore`，拼文案交给 :func:`~agent_project.krow.items.render_items`。

**RestInjectSource** —— 与平台 kit 钩子同一口子、同一参数、同一回落：
``GET {KW_API_ENDPOINT}/agent/v1/inject?max_owner=3&max_hall=5&max_leader=1[&hall=0]``，
Bearer = ``KW_WORKER_TOKEN``（``KROW_TASK_SLOT=conversation`` 时用 ``KW_WORKER_TOKEN_CONVERSATION``），
可选头 ``X-Krow-Job-Binding``；老服务端 404 → ``GET /agent/v1/hall/inject?max=5``。服务端读即消费。
连不上 / 超时 / 5xx 进 30 秒冷却，窗口内不出网（滚动发布时只付一刀 3 秒）。
**只读环境变量名、不打印值**；token 不进日志、不进异常文本。

外部工作台特有的一步（D11，李靖按 krowworker main 读码的结论，服务端行为以 captain 实测为准）：
外部工作台的点名在服务端是 ``suppressed(not_wakeable)`` 行；缺省 general 的那次调用只会把**别的频道的 pending 行**捎带进来
（``PENDING_SUMMONS_ELSEWHERE_SQL``），suppressed 行捎不到；但按 ``channel_id`` 读时间线的主路径（``INJECT_WHERE``）
不看唤醒行状态，读到即把行翻成 injected。所以这里在缺省调用之后，再对每个有 ``mention_count``（``all`` 模式下有未读）的
项目 / 小组频道各打一次 ``…/inject?…&channel_id=<id>``（老口子同样带 ``channel_id``）；频道列表来自
``GET /agent/v1/hall/channels``，扫描按 :data:`CHANNEL_SCAN_SECS` 节流。

坏频道隔离（v2；李靖按 main ``hall/inject.rs:460-473`` / ``hall/agent_api.rs:109-135`` 读码：显式 ``channel_id`` 走窄口径，
频道不存在 / 读者不是成员 → **404**「频道不存在或不可见」、不回落 general；``channel_id`` 按 serde ``Uuid`` 反序列化，
非 UUID → **400**）：

* 显式 ``channel_id`` 调用的 400 / 403 / 404 / 410 等**频道级**错误只隔离该频道（记进 :attr:`RestInjectSource.bad_channels`、
  :data:`BAD_CHANNEL_BACKOFF_SECS` 内不再打、到期重试一次、首次隔离记一条日志），循环**继续**轮询后面的频道；
  ``last_error`` 不被坏频道污染。401 / 409 / 429 / 5xx / 连不上 是**源级**错误：按原规则记错 / 冷却，本轮停止扫描，
  已拿到的项照样交出去（fail-open）。
* 旧接口（``/agent/v1/hall/inject``）兼容回落**只由缺省调用判定、每轮一次**：缺省调用统一口子 404 → ``legacy_mode=True``，本轮后续
  频道调用直接走老口子（不逐频道再探）；显式频道调用的 404 **不是**旧服务端信号，永远不触发回落。判定不跨轮粘住：下一轮缺省调用
  重新探统一口子（一次瞬时 404 不会让只在统一口子上才有的主人插话 / 任务单段丢 N 分钟）。
* ``krow.channels`` 固定列表入口按 UUID 校验：非法项记进 :attr:`RestInjectSource.invalid_channels`、不发请求，其余有效频道照常轮询。

409（``requires an active job binding``）：只记「**已见 409**」（:attr:`RestInjectSource.saw_409`）并冷却。按 ``worker_token.rs:486-496``
门禁只在「执行态 && placement_kind != mcp_pull && 无 job binding」时 409，所以 **有 409 ⇒ 不是 mcp_pull** 成立（单向）；
**未见 409 ⇏ mcp_pull**（conversation 作用域令牌同样放行）。这里不据此断言平台侧 H1 已回答，``placement_kind`` 的枚举值仍要
captain 另给（:data:`NOTE_409_SEEN` / :data:`NOTE_409_NOT_SEEN` 是 ``/krow`` 状态里的原话）。

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
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .items import (
    InjectItem,
    extract_channel_id,
    hall_key,
    leader_key,
    mail_key,
    owner_key,
)
from .tools import is_unknown_tool_error as _is_unknown_tool

logger = logging.getLogger("krow.sources")

__all__ = [
    "HttpGet",
    "McpHallSource",
    "NOTE_409_NOT_SEEN",
    "NOTE_409_SEEN",
    "RestInjectSource",
    "canonical_channel_id",
    "parse_hall_read",
    "parse_inject_hall_text",
    "resolve_rest_env",
]

INJECT_PATH = "/agent/v1/inject"
HALL_INJECT_PATH = "/agent/v1/hall/inject"
HALL_CHANNELS_PATH = "/agent/v1/hall/channels"
MAX_OWNER_PER_STEP = 3
MAX_HALL_PER_STEP = 5
MAX_LEADER_PER_STEP = 1
TIMEOUT_SECS = 3.0
COOLDOWN_SECS = 30.0
#: 项目 / 小组频道逐个轮询的最小间隔(秒): 工具边界每次都拉, 但频道扫描最多这么频繁.
CHANNEL_SCAN_SECS = 10.0
#: 显式 channel_id 回 400/404 等频道级错误后, 该频道隔离多久再试一次(秒). 其余频道不受影响.
BAD_CHANNEL_BACKOFF_SECS = 300.0
JOB_BINDING_HEADER = "X-Krow-Job-Binding"

#: 显式 channel_id 调用上, 这些状态码是"这个频道有问题"(不存在 / 不可见 / id 不是 UUID / 已删), 只隔离该频道.
#: 401(令牌) / 409(门禁) / 429(限流) / 5xx(服务端) 不在其中: 它们是整条 REST 源的问题, 按源级规则记错 / 冷却.
CHANNEL_SCOPED_HTTP_CODES = frozenset({400, 403, 404, 410, 422})

#: ``/krow`` 状态里 409 语义的原话: 409 只说明"已见 / 未见", 推断是单向的, 不替平台侧 H1 作答.
NOTE_409_SEEN = (
    "已见 409（requires an active job binding）：无 job binding 的执行态请求被门禁拦下 ⇒ 这枚令牌不是 mcp_pull（单向推断成立）；"
    "H1 的 placement_kind 枚举值仍需 captain 另给，本状态不替它作答"
)
NOTE_409_NOT_SEEN = (
    "未见 409：不等于 mcp_pull 已确认（conversation 作用域令牌同样放行，也可能尚未发过请求）；H1 未由此回答"
)

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


def canonical_channel_id(raw: Any) -> Optional[str]:
    """``krow.channels`` 固定列表里的一项 → 规范的带连字符小写 UUID；不是 UUID（频道名、空串、``general``…）返回 None。

    服务端两口子的 ``channel_id`` 都按 serde ``Uuid`` 反序列化（李靖核 ``handlers/inject.rs:132-149`` / ``hall/inject.rs:106-112``）：
    传名字会 400。这里在入口就挡掉，非法项不发请求、不中断其余有效频道。
    """
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        return str(uuid.UUID(s))
    except (ValueError, AttributeError, TypeError):
        return None


def _is_channel_scoped_error(exc: BaseException) -> bool:
    """显式 ``channel_id`` 调用抛出的错误是不是"只这个频道有问题"(见 :data:`CHANNEL_SCOPED_HTTP_CODES`)。"""
    return isinstance(exc, urllib.error.HTTPError) and int(getattr(exc, "code", 0) or 0) in CHANNEL_SCOPED_HTTP_CODES


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
        channels: Any = "auto",
        mode: str = "mentions",
        channel_scan_interval: float = CHANNEL_SCAN_SECS,
        bad_channel_backoff: float = BAD_CHANNEL_BACKOFF_SECS,
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
        # 项目 / 小组频道逐个轮询(D11): "auto" = GET /agent/v1/hall/channels 列出、按 mention_count / unread_count 挑;
        # 列表 = 固定这些 channel_id(入口按 UUID 校验, 非法项进 invalid_channels、不发请求, 其余有效频道照常);
        # "off"/空 = 只打缺省 general(外部工作台的点名会漏, 不建议).
        self.invalid_channels: List[str] = []
        if isinstance(channels, str):
            mode_word = channels.strip().lower()
            if mode_word in ("auto", "off", ""):
                self.channels: Any = mode_word or "off"
            else:
                self.channels = self._validate_channel_list([channels])
        elif channels:
            self.channels = self._validate_channel_list(list(channels))
        else:
            self.channels = "off"
        self.mode = mode if mode in ("mentions", "all") else "mentions"
        self.channel_scan_interval = max(0.0, float(channel_scan_interval))
        self._next_channel_scan = 0.0
        self.last_channels: List[Dict[str, Any]] = []
        # 坏频道隔离(v2): channel_id -> {"code", "count", "first_at", "last_at", "retry_at"}; 到 retry_at 之前不再打这个频道.
        self.bad_channel_backoff = max(0.0, float(bad_channel_backoff))
        self.bad_channels: Dict[str, Dict[str, Any]] = {}
        # 旧接口回落只由缺省调用判定(每轮一次, 频道调用沿用): None = 还没判定过; True = 最近一轮统一口子 404; False = 统一口子在.
        self.legacy_mode: Optional[bool] = None
        # 409 门禁: 只记"已见"; 推断单向(有 409 ⇒ 非 mcp_pull; 无 409 ⇏ mcp_pull), 不替 H1 作答.
        self.saw_409 = False
        self.last_409_at: float = 0.0

    # 不要把 token 带进 repr / 日志。
    def __repr__(self) -> str:  # pragma: no cover - 可读性
        return f"RestInjectSource(api={self.api!r})"

    @property
    def binding_required(self) -> bool:
        """v1 字段名的别名：语义收窄为「已见 409」。不要据此说「席位是 / 不是 mcp_pull 已确认」。"""
        return self.saw_409

    @property
    def note_409(self) -> str:
        return NOTE_409_SEEN if self.saw_409 else NOTE_409_NOT_SEEN

    def _validate_channel_list(self, raw: List[Any]) -> Any:
        good: List[str] = []
        for item in raw:
            text = str(item or "").strip()
            if not text:
                continue
            cid = canonical_channel_id(text)
            if cid is None:
                if text not in self.invalid_channels:
                    self.invalid_channels.append(text)
            elif cid not in good:
                good.append(cid)
        if self.invalid_channels:
            logger.warning(
                "krow rest: krow.channels 里 %d 项不是 UUID、已忽略(服务端按 Uuid 解析, 传名字会 400): %s; 有效频道 %d 个照常轮询",
                len(self.invalid_channels), self.invalid_channels, len(good),
            )
        return good or "off"

    def cooling_down(self) -> bool:
        return self._clock() < self._down_until

    def status(self) -> Dict[str, Any]:
        """给 ``/krow`` 状态用的只读快照（不含 token）。"""
        return {
            "channels": self.channels,
            "invalid_channels": list(self.invalid_channels),
            "bad_channels": {k: dict(v) for k, v in self.bad_channels.items()},
            "legacy_mode": self.legacy_mode,
            "saw_409": self.saw_409,
            "note_409": self.note_409,
            "cooling_down": self.cooling_down(),
            "last_error": self.last_error,
            "calls": self.calls,
        }

    def _get(self, path: str, params: Dict[str, str]) -> Dict[str, Any]:
        url = f"{self.api}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if self.binding:
            headers[JOB_BINDING_HEADER] = self.binding
        self.calls += 1
        _status, body = self._http_get(url, headers, self.timeout)
        return _json_or_empty(body)

    def _record_failure(self, e: BaseException, *, where: str = "inject") -> None:
        """源级失败: 记 ``last_error``; 409 与 5xx / 连不上 进冷却. 频道级错误不走这里(见 :meth:`_quarantine`)."""
        code = getattr(e, "code", "") or ""
        self.last_error = f"{type(e).__name__}: {code}".strip(": ")
        if code == 409:
            # 服务端门禁 require_job_binding(worker_token.rs:486-496): 执行态 && placement_kind != mcp_pull && 无 job binding → 409.
            # 这里只记"已见 409"并冷却(别每轮都撞). 推断是单向的: 有 409 ⇒ 非 mcp_pull; 没见过 409 推不出 mcp_pull.
            self.saw_409 = True
            self.last_409_at = self._clock()
            self.last_error += " (requires job binding → 已见 409 ⇒ 令牌非 mcp_pull, 单向推断; 未见 409 不能反推; H1 仍需 captain 另给)"
            self._down_until = self._clock() + self.cooldown
        elif _is_server_down(e):
            self._down_until = self._clock() + self.cooldown
        logger.warning("krow rest %s failed (%s); %s", where, self.last_error, "cooling down" if self.cooling_down() else "no cooldown")

    # ── 旧接口回落: 只由缺省调用判定 ──────────────────────────────────────

    def _set_legacy(self, flag: bool) -> None:
        if self.legacy_mode is not flag:
            logger.info("krow rest: %s", "统一口子 /agent/v1/inject 404 → 本轮走老口子 /agent/v1/hall/inject(缺省调用判定)" if flag
                        else "统一口子 /agent/v1/inject 可用(缺省调用判定)")
        self.legacy_mode = flag

    def _legacy_params(self, channel_id: Optional[str]) -> Dict[str, str]:
        p = {"max": str(self.max_hall)}
        if channel_id:
            p["channel_id"] = channel_id
        return p

    def _inject_default(self, params: Dict[str, str], *, hall: bool) -> Dict[str, Any]:
        """缺省(general)调用. **只有这里**判定旧接口回落, 每轮一次: 统一口子 404 → ``legacy_mode=True`` 并改打老口子(只有大厅段),
        本轮频道调用沿用; 判定不跨轮粘住(下一轮重新探统一口子). 老口子内容放在 ``{"_legacy": items}``."""
        try:
            data = self._get(INJECT_PATH, params)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            self._set_legacy(True)
            if not hall:
                return {}
            return {"_legacy": _legacy_hall_items(self._get(HALL_INJECT_PATH, self._legacy_params(None)))}
        self._set_legacy(False)
        return data

    def _inject_channel(self, params: Dict[str, str], channel_id: str) -> Dict[str, Any]:
        """显式 ``channel_id`` 调用. 沿用本轮缺省调用判定好的口子, **自己不判定**: 这里的 404 是"频道不存在或不可见"
        (``hall/inject.rs:460-473``), 400 是 id 不是 UUID, 都交给调用方按频道隔离, 绝不触发老口子回落."""
        if self.legacy_mode:
            return {"_legacy": _legacy_hall_items(self._get(HALL_INJECT_PATH, self._legacy_params(channel_id)))}
        return self._get(INJECT_PATH, dict(params, channel_id=channel_id))

    @staticmethod
    def _items(data: Any, *, want_owner: bool, want_leader: bool, want_hall: bool) -> List[InjectItem]:
        if not isinstance(data, dict):
            return []
        if "_legacy" in data:
            return list(data["_legacy"]) if want_hall else []
        return RestInjectSource._parse_unified(data, want_owner=want_owner, want_leader=want_leader, want_hall=want_hall)

    # ── 坏频道隔离 ──────────────────────────────────────────────────────────

    def _quarantined(self, channel_id: str) -> bool:
        bad = self.bad_channels.get(channel_id)
        return bool(bad) and self._clock() < float(bad.get("retry_at") or 0.0)

    def _quarantine(self, channel_id: str, e: urllib.error.HTTPError) -> None:
        """显式频道调用回了频道级错误: 只隔离这个频道(到期重试一次), 其余频道与 ``last_error`` 不受影响. 首次隔离记一条日志."""
        now = self._clock()
        bad = self.bad_channels.get(channel_id)
        if bad is None:
            bad = {"code": int(e.code), "count": 0, "first_at": now}
            self.bad_channels[channel_id] = bad
        bad["code"] = int(e.code)
        bad["count"] = int(bad.get("count") or 0) + 1
        bad["last_at"] = now
        bad["retry_at"] = now + self.bad_channel_backoff
        msg = ("krow rest: 频道 %s 显式拉取回 %s(频道不存在 / 不可见 / id 非法), 隔离 %.0fs 后再试; 其余频道照常轮询"
               % (channel_id, e.code, self.bad_channel_backoff))
        if bad["count"] == 1:
            logger.warning(msg)
        else:
            logger.debug(msg + " (第 %d 次)", bad["count"])

    def _recover(self, channel_id: str) -> None:
        if self.bad_channels.pop(channel_id, None) is not None:
            logger.info("krow rest: 频道 %s 恢复可达, 解除隔离", channel_id)

    def _channel_ids_to_poll(self) -> List[str]:
        """要逐个轮询的频道(不含 general): 固定列表照给; auto 则按 ``hall/channels`` 的计数挑."""
        if isinstance(self.channels, list):
            return list(self.channels)
        if self.channels != "auto":
            return []
        try:
            data = self._get(HALL_CHANNELS_PATH, {})
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            # 老服务端没有 hall/channels: 自动扫描无从谈起, 关掉(要轮询项目频道就用 krow.channels 固定列表), 只提醒一次
            self.channels = "off"
            self.last_error = "hall/channels 404: 服务端不支持频道列表, 项目频道轮询需在 krow.channels 固定填 channel_id"
            logger.warning("krow rest: %s", self.last_error)
            return []
        rows = data.get("channels") if isinstance(data, dict) else None
        self.last_channels = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        out: List[str] = []
        for ch in self.last_channels:
            cid = str(ch.get("id") or "").strip()
            kind = str(ch.get("kind") or "").strip().lower()
            if not cid or kind == "general":
                continue
            mentions = int(ch.get("mention_count") or 0)
            unread = int(ch.get("unread_count") or 0)
            if mentions > 0 or (self.mode == "all" and unread > 0):
                out.append(cid)
        return out

    @staticmethod
    def _tag_channel(items: List[InjectItem], channel_id: str) -> List[InjectItem]:
        for it in items:
            if it.kind != "hall":
                continue
            if it.channel_id is None:
                it.channel_id = channel_id
            if f"channel_id={channel_id}" not in it.text:
                it.text += f"（频道 channel_id={channel_id}）"
        return items

    def fetch(self, *, hall: bool = True, leader: bool = True, owner: bool = True) -> List[InjectItem]:
        if self.cooling_down():
            return []
        params = {"max_owner": str(self.max_owner), "max_hall": str(self.max_hall)}
        if leader:
            params["max_leader"] = str(self.max_leader)
        if not hall:
            params["hall"] = "0"
        out: List[InjectItem] = []
        # 1. 缺省(general)调用: 唯一判定旧接口回落的地方. 失败 = 源级失败(记错 / 冷却), 本轮到此为止.
        try:
            out.extend(self._items(self._inject_default(params, hall=hall), want_owner=owner, want_leader=leader, want_hall=hall))
        except Exception as e:  # noqa: BLE001 — 分类后冷却；调用方 fail-open(已拿到的照样交出去)
            self._record_failure(e, where="inject(default)")
            return out
        self.last_error = ""
        if not (hall and self.channels != "off" and self._clock() >= self._next_channel_scan):
            return out
        # 2. D11: 外部工作台的点名在服务端是 suppressed(not_wakeable) 行, 缺省 general 的跨频道捎带只取 pending 行,
        # 项目 / 小组频道必须各自带 channel_id 轮询才取得到(李靖按 main 读 hall/store.rs INJECT_WHERE 的结论,
        # 服务端行为仍以 captain 实测为准). 频道扫描有节流, 工具边界的高频拉取不会每次都多打 N 个请求.
        self._next_channel_scan = self._clock() + self.channel_scan_interval
        try:
            channel_ids = self._channel_ids_to_poll()
        except Exception as e:  # noqa: BLE001 — hall/channels 的 401 / 409 / 5xx 与注入口子同一道闸: 源级, 冷却
            self._record_failure(e, where="hall/channels")
            return out
        # 3. 逐频道: 坏频道只隔离自己(400/403/404/410/422 → 跳过、记账、到期重试), 循环继续; 源级错误(401/409/429/5xx/连不上)
        # 才停止本轮扫描并按原规则冷却. 两种情况下已拿到的项都照样交出去(fail-open); MCP 路径另有兜底.
        for cid in channel_ids:
            if self._quarantined(cid):
                continue
            try:
                cdata = self._inject_channel(params, cid)
            except urllib.error.HTTPError as e:
                if _is_channel_scoped_error(e):
                    self._quarantine(cid, e)
                    continue
                self._record_failure(e, where=f"inject(channel {cid})")
                break
            except Exception as e:  # noqa: BLE001 — 连不上 / 超时: 源级, 冷却
                self._record_failure(e, where=f"inject(channel {cid})")
                break
            self._recover(cid)
            out.extend(self._tag_channel(self._items(cdata, want_owner=owner, want_leader=leader, want_hall=True), cid))
        return out

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
        # 服务器没有 ``hall_channels``（agent-hub 只给 krow_inbox / krow_hall_read / krow_hall_post …）时，
        # 频道扫描这条 v1 路径永久关掉，点名改由收件箱系统信 + 守候循环承担；只提示一次，不每拍刷警告。
        self.hall_disabled_reason: str = ""

    def _tool(self, name: str, args: Dict[str, Any]) -> str:
        self.calls += 1
        return self._call(name, args) or ""

    def fetch(self, *, hall: bool = True, leader: bool = True, owner: bool = True) -> List[InjectItem]:
        out: List[InjectItem] = []
        if hall and not self.hall_disabled_reason:
            try:
                out.extend(self._fetch_hall())
            except Exception as e:  # noqa: BLE001 — 轮询失败只少听一拍
                if _is_unknown_tool(e):
                    self.hall_disabled_reason = f"{type(e).__name__}: {e}"[:200]
                    logger.info("krow mcp: 服务器没有 hall_channels 工具，停用频道扫描，点名由收件箱系统信承担 (%s)", self.hall_disabled_reason)
                else:
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
