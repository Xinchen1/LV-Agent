"""MCP 工具名解析：逻辑操作 → 服务器上的注册名.

lv-agent 连的是 agent-hub 的 MCP（``services/agent-hub/src/mcp.rs``），工具叫 ``krow_inbox`` / ``krow_read_message`` /
``krow_send_message`` / ``krow_hall_read`` / ``krow_hall_post``（李靖按 origin/main 读码；PR #1264 / 9f834380 新增后两个）；
v1 的 MCP 路径按平台 kit 桥的名字写死为 ``hall_channels`` / ``hall_read`` / ``hall_ack`` / ``team_inbox``。两套名字指的是同一批
服务端口子，这里统一成**逻辑操作名**（沿用 v1 的叫法，老代码一处不改），按服务器 ``tools/list`` 探测实际注册名：

* 有列表：按候选顺序取第一个在列表里的（agent-hub 的 ``krow_*`` 排在前面）；
* 还没拿到列表（MCP 服务器懒启动）：先用第一候选；调用若报「unknown tool / method not found」再顺次换下一候选，成功的记住；
* ``config.krow.tools`` 可以逐项覆盖（``{team_inbox: krow_inbox, ...}``），覆盖项不再探测。

工具内部错误（服务端以文本 ``{"error": ...}`` 返回、不是 JSON-RPC error）由 :meth:`KrowTools.call_json` 识别并抛
:class:`KrowToolError`，调用方按「这次没成」处理，不把错误 JSON 当结果解析。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger("krow.tools")

__all__ = ["KrowToolError", "KrowTools", "TOOL_CANDIDATES", "is_unknown_tool_error"]

#: 逻辑操作名（= v1 用的平台 kit 桥名字）→ 候选注册名，前面的优先（agent-hub 的 ``krow_*`` 在前）。
TOOL_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "team_inbox": ("krow_inbox", "team_inbox", "inbox"),
    "team_read_message": ("krow_read_message", "team_read_message", "read_message"),
    "team_send_message": ("krow_send_message", "team_send_message", "send_message"),
    "hall_read": ("krow_hall_read", "hall_read"),
    "hall_post": ("krow_hall_post", "hall_post"),
    "hall_channels": ("krow_hall_channels", "hall_channels"),
    "hall_ack": ("krow_hall_ack", "hall_ack"),
}

_UNKNOWN_TOOL_RE = re.compile(
    r"(?i)(unknown|unrecognized|unsupported|no such|not found|invalid|undefined)[^\n]{0,24}\btool\b"
    r"|\btool\b[^\n]{0,24}(unknown|not found|does not exist|unsupported|unavailable)"
    r"|-32601|method not found|未知工具|工具不存在|没有这个工具"
)


class KrowToolError(RuntimeError):
    """MCP 工具本身返回了错误（``{"error": ...}`` 文本）或调用链路失败。``text`` 是原始返回。"""

    def __init__(self, message: str, *, tool: str = "", text: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.text = text


def is_unknown_tool_error(exc: BaseException) -> bool:
    """这次失败是不是「服务器上没有这个工具」（该换候选名），而不是网络 / 业务错误。"""
    return bool(_UNKNOWN_TOOL_RE.search(str(exc) or ""))


class KrowTools:
    """``call(op, args) -> str`` / ``call_json(op, args) -> Any``。

    ``call_tool(tool_name, args) -> str`` 是原始调用（bridge 绑到 ``McpManager.call_tool(server, ...)``）；
    ``list_tools() -> Iterable[str]`` 可选，返回服务器当前的工具名（拿不到 / 还没就绪返回空），每次解析缺失项时再问一次。
    """

    def __init__(
        self,
        call_tool: Callable[[str, Dict[str, Any]], str],
        *,
        list_tools: Optional[Callable[[], Iterable[str]]] = None,
        available: Optional[Iterable[str]] = None,
        overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._call = call_tool
        self._list = list_tools
        self._available: List[str] = [str(n) for n in (available or ()) if n]
        self._overrides: Dict[str, str] = {str(k): str(v) for k, v in (overrides or {}).items() if k and v}
        self._resolved: Dict[str, str] = {}
        self.calls = 0
        self.last_error: str = ""

    # ── 解析 ──────────────────────────────────────────────────────────────

    def known_tools(self) -> List[str]:
        if not self._available and self._list is not None:
            try:
                self._available = [str(n) for n in (self._list() or ()) if n]
            except Exception as e:  # noqa: BLE001 — 列表拿不到就按候选顺序试
                logger.debug("krow tools: list_tools failed: %s", e)
        return list(self._available)

    def candidates(self, op: str) -> Tuple[str, ...]:
        return TOOL_CANDIDATES.get(op, (op,))

    def name(self, op: str) -> str:
        """逻辑操作名 → 当前认定的注册名（覆盖 > 已验证 > 列表里第一个候选 > 第一候选）。"""
        if op in self._overrides:
            return self._overrides[op]
        if op in self._resolved:
            return self._resolved[op]
        known = set(self.known_tools())
        if known:
            for cand in self.candidates(op):
                if cand in known:
                    self._resolved[op] = cand
                    return cand
        return self.candidates(op)[0]

    def resolved(self) -> Dict[str, str]:
        """给 ``/krow`` 状态看的「逻辑名 → 注册名」快照。"""
        return {op: self.name(op) for op in TOOL_CANDIDATES}

    # ── 调用 ──────────────────────────────────────────────────────────────

    def call(self, op: str, args: Optional[Dict[str, Any]] = None) -> str:
        """调一次；「没有这个工具」就顺次换候选名（覆盖项不换），其余错误原样抛。"""
        args = dict(args or {})
        tried: List[str] = []
        order: List[str] = [self.name(op)]
        if op not in self._overrides:
            order += [c for c in self.candidates(op) if c not in order]
        last: Optional[BaseException] = None
        for tool in order:
            tried.append(tool)
            self.calls += 1
            try:
                out = self._call(tool, args)
            except Exception as e:  # noqa: BLE001 — 分类后决定换名还是上抛
                last = e
                if op not in self._overrides and is_unknown_tool_error(e) and tool != order[-1]:
                    logger.info("krow tools: %s 不在服务器上(%s), 换候选名", tool, str(e)[:80])
                    continue
                self.last_error = f"{op}→{tool}: {type(e).__name__}: {e}"[:300]
                raise
            if tool != self._resolved.get(op):
                self._resolved[op] = tool
            return out if isinstance(out, str) else ("" if out is None else str(out))
        self.last_error = f"{op}: 候选名都不可用 {tried}: {last}"[:300]
        raise KrowToolError(self.last_error, tool=op)

    def call_json(self, op: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """调一次并解析 JSON；工具返回 ``{"error": ...}`` 抛 :class:`KrowToolError`；非 JSON 文本原样返回字符串。"""
        text = self.call(op, args)
        data = _loads(text)
        if isinstance(data, dict) and _looks_like_error(data):
            err = data.get("error")
            msg = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
            self.last_error = f"{op}: {msg}"[:300]
            raise KrowToolError(f"{op}: {msg}", tool=self.name(op), text=text)
        return data if data is not None else text


def _loads(text: str) -> Any:
    s = (text or "").strip()
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001 — 不是 JSON 就当文本
        return None


def _looks_like_error(data: Mapping[str, Any]) -> bool:
    if "error" not in data or not data.get("error"):
        return False
    # 正常结果里也可能带 error 字段(例如 messages 列表 + 某项的 error 说明): 只有「没有任何正文字段」才算整体失败
    for key in ("messages", "message_id", "id", "channels", "body", "text", "count"):
        if key in data and data.get(key) not in (None, "", []):
            return False
    return True
