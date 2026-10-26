"""离线自检：本地假 krow 服务端（读即消费）+ 假 MCP 桥，端到端走一遍「注入 → 轮询 → 去重 → 重启不丢」.

不碰现网、不读任何真实凭据（token 是本脚本随机生成的假值）。运行::

    python -m agent_project.krow.selfcheck [--json]

退出码 0 = 全部场景通过。覆盖：

1. REST 统一口子：owner / leader / hall 三段解析、口令抬头、``X-Krow-Job-Binding`` 头、读即消费（第二次拉为空）；
2. 旧服务端 404 回落 ``/agent/v1/hall/inject``；
3. 5xx 冷却：窗口内不出网；
4. MCP 轮询：``hall_channels`` → ``hall_read(unread_only)`` → 按 ``@名`` 判点名 → ``hall_ack`` → ``team_inbox``；
5. 两源合流：同一条 #id 只送一次；
6. 取到 ≠ 送到：未 ``mark_delivered`` 的待送项跨「进程重启」（新建 store）仍在；送达后不再出现；
7. ExecutionEngine 集成：工具边界注入 + 轮尾续跑（用假模型后端）；
8. 守护模式：``KrowDaemon.run_once`` 先记账再跑、回合失败不重跑；
9. （v2）REST 坏频道隔离：假服务端按 main 的 ``resolve_channel_access`` 建模——显式 ``channel_id`` 非 UUID → 400、不存在 / 不可见 → 404
   （不回落 general）。固定列表 ``[非 UUID 名字, 不存在的频道, 有效频道]`` **坏的排在前面**，有效频道仍被轮询并取到点名；
   显式 404 不触发老口子回落、不污染 ``last_error``；隔离到期重试、恢复即解除；``auto`` 模式下 ``hall/channels`` 列出却 404 的
   幽灵频道同样只隔离自己；旧接口回落只由缺省调用判定一次（老服务端上统一口子只被探一次）；频道口子 5xx 仍 fail-open；
   ``/krow`` 状态 409 只说「已见 / 未见」；REST 看不见的频道由 MCP 路径兜底且共用一本账。
10. （v3）收件箱守候循环闭环（假 agent-hub ``FakeAgentHub``，按服务端 PR #1264 建模：点名 → 系统信 ``ref.kind=hall_mention``；
    读信翻 wakeups suppressed→injected；带 ``reply_to_id`` 回帖翻 obligations→answered）：上线验证三条口径逐条核对、
    调用顺序 ``inbox → read → hall_read(原文) → hall_read(上下文) → hall_post``、``wait`` 夹在 0..25、点名任务文本是数据不是指令、
    同信重现 / 进程重启只补读不重发、读失败本拍跳过下一拍补上、模型回合失败 / 无 runner 回一句原因不沉默、模型自己已回帖不重发、
    回帖瞬时错误重试 / 超时但服务端已落库查重不重发 / 4xx 不重试记 failed、主人直聊最高优先级并经 ``send_message(to_role=owner)``
    回信、收件箱出错指数退避成功复位 + 空返最小间隔、旧命名服务器（``team_*`` / ``hall_*``）自动回退、崩溃续办
    （read 未发 → 补答；read 已发 → 查大厅确认 / unverified）、``KrowDaemon`` 驱动、全程不自写 ``ref``。
"""
from __future__ import annotations

import json
import logging
import secrets
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .bridge import KrowBridge, KrowConfig
from .items import LEADER_PREFIX_HEAD, OWNER_PREFIX_HEAD
from .sources import McpHallSource, RestInjectSource
from .state import DedupStore
from .wake import KrowDaemon, build_wake_task

CHANNEL = "f76423cc-49a5-4263-b0b7-fe0fcc87a92c"
#: 形态合法、但服务端上不存在 / 对本读者不可见的频道 id(固定列表填错、或合作方 floor 算不出来时就是它) → 显式拉取 404
BAD_CHANNEL = "00000000-0000-4000-8000-00000000bad0"


# ── 假服务端 ─────────────────────────────────────────────────────────────


class FakeKrowServer:
    """``/agent/v1/inject``（读即消费）与 ``/agent/v1/hall/inject``；可切 404 / 503 / 409 模式。

    显式 ``channel_id`` 的门禁按 krowworker main ``hall/agent_api.rs::resolve_channel_access`` 建模（李靖读码）：
    两口子同一道闸，id 不是 UUID → 400（serde ``Uuid``），频道不存在 / 读者不是成员 → 404「频道不存在或不可见」，不回落 general。
    ``ghost_channels``：``hall/channels`` 列出来、显式拉取却 404 的频道（可见性 / floor 对不上）。``channel503``：只有带 ``channel_id`` 的调用 503。
    """

    def __init__(self, token: str) -> None:
        self.token = token
        self.mode = "unified"  # unified | legacy404 | down503 | binding409 | channel503
        self.owner_queue: List[str] = []
        self.leader_queue: List[Dict[str, Any]] = []
        self.hall_queue: List[Dict[str, Any]] = []
        # D11 建模: 项目频道里给外部工作台的点名 = suppressed(not_wakeable) 行, 缺省调用(跨频道只捎 pending 行)取不到,
        # 只有带 channel_id=<该频道> 的调用(按时间线读, 不看唤醒行状态)才给出并翻 injected.
        self.channel_queues: Dict[str, List[Dict[str, Any]]] = {}
        self.ghost_channels: List[str] = []
        self.requests: List[Tuple[str, Dict[str, str]]] = []
        self._lock = threading.Lock()
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:  # 安静
                return

            def do_GET(self) -> None:  # noqa: N802
                server_ref._handle(self)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def api(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()

    def _render_hall(self, msgs: List[Dict[str, Any]]) -> str:
        lines = []
        for m in msgs:
            head = f"#{m['id']} {m['display']}："
            body = ("@你 " if m.get("mentions_me") else "") + m["body"]
            tail = f"（来自别的频道，回复请带 channel_id={m['channel_id']}）" if m.get("channel_id") else ""
            lines.append(head + body + tail)
        return "\n".join(lines)

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        url = urlparse(h.path)
        headers = {k: v for k, v in h.headers.items()}
        with self._lock:
            self.requests.append((h.path, headers))
            if headers.get("Authorization") != f"Bearer {self.token}":
                return self._send(h, 401, {"error": "unauthorized"})
            if self.mode == "down503":
                return self._send(h, 503, {"error": "rolling"})
            if self.mode == "binding409":
                # require_job_binding 同一道闸罩着 inject / hall/inject / hall/channels(李靖核 hall/agent_api.rs:253-257)
                return self._send(h, 409, {"error": "job_binding_required", "message": "request requires an active job binding"})
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/agent/v1/hall/channels":
                chans = [{"id": "general", "name": "大厅", "kind": "general", "unread_count": len(self.hall_queue), "mention_count": len(self.hall_queue)}]
                for gid in self.ghost_channels:  # 列得出、拉不到: 排在前面, 好证明后面的频道不受牵连
                    chans.append({"id": gid, "name": "幽灵频道", "kind": "project", "unread_count": 1, "mention_count": 1})
                for cid, q in self.channel_queues.items():
                    chans.append({"id": cid, "name": "星辰大海", "kind": "project", "unread_count": len(q), "mention_count": sum(1 for m in q if m.get("mentions_me"))})
                return self._send(h, 200, {"channels": chans})
            if url.path == "/agent/v1/inject" and self.mode == "legacy404":
                return self._send(h, 404, {"error": "not found"})  # 老服务端没有统一口子: 路由层 404, 在频道闸之前
            if url.path in ("/agent/v1/inject", "/agent/v1/hall/inject") and qs.get("channel_id"):
                cid = qs["channel_id"]
                try:
                    uuid.UUID(cid)
                except ValueError:
                    return self._send(h, 400, {"error": "Failed to deserialize query string: channel_id: invalid UUID"})
                if cid not in self.channel_queues:
                    return self._send(h, 404, {"error": "频道不存在或不可见"})
                if self.mode == "channel503":
                    return self._send(h, 503, {"error": "rolling (channel path)"})
            if url.path == "/agent/v1/inject":
                max_owner = int(qs.get("max_owner", 3))
                max_hall = int(qs.get("max_hall", 5))
                want_hall = qs.get("hall") != "0"
                want_leader = "max_leader" in qs
                owners, self.owner_queue = self.owner_queue[:max_owner], self.owner_queue[max_owner:]
                payload: Dict[str, Any] = {"owner": {"text": "\n\n".join(owners)}}
                if want_leader:
                    n = int(qs.get("max_leader", 1))
                    items, self.leader_queue = self.leader_queue[:n], self.leader_queue[n:]
                    payload["leader"] = {"items": items}
                if want_hall:
                    msgs = self._take_hall(qs.get("channel_id"), max_hall)
                    payload["hall"] = {"text": self._render_hall(msgs), "remaining": len(self.hall_queue)}
                return self._send(h, 200, payload)
            if url.path == "/agent/v1/hall/inject":
                msgs = self._take_hall(qs.get("channel_id"), int(qs.get("max", 5)))
                return self._send(h, 200, {"items": msgs, "remaining": len(self.hall_queue), "last_read_id": max([m["id"] for m in msgs] or [0])})
            return self._send(h, 404, {"error": "no route"})

    def _take_hall(self, channel_id: Optional[str], n: int) -> List[Dict[str, Any]]:
        """缺省(general) = 本频道 + 跨频道 pending 捎带(hall_queue); 带 channel_id = 只读该频道时间线(含 suppressed 行; 频道闸已在前面过)."""
        if channel_id and channel_id != "general":
            q = self.channel_queues[channel_id]
            msgs, self.channel_queues[channel_id] = q[:n], q[n:]
            return msgs
        msgs, self.hall_queue = self.hall_queue[:n], self.hall_queue[n:]
        return msgs

    @staticmethod
    def _send(h: BaseHTTPRequestHandler, code: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)


class FakeMcpBridge:
    """``call_tool(tool, args) -> str``：模拟 kit ``mcp_bridge`` 的 hall_channels / hall_read / hall_ack / team_inbox。"""

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []  # {id, channel_id, display, time, body}
        self.last_read: Dict[str, int] = {}
        self.inbox: List[Dict[str, Any]] = []
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.self_name = "lv-agent"

    def __call__(self, tool: str, args: Dict[str, Any]) -> str:
        self.calls.append((tool, dict(args)))
        if tool == "hall_channels":
            chans = []
            for cid in sorted({m["channel_id"] for m in self.messages} | {"general"}):
                unread = [m for m in self.messages if m["channel_id"] == cid and m["id"] > self.last_read.get(cid, 0)]
                chans.append({
                    "id": cid, "name": "general" if cid == "general" else "星辰大海", "kind": "general" if cid == "general" else "project",
                    "unread_count": len(unread),
                    "mention_count": sum(1 for m in unread if f"@{self.self_name}" in m["body"]),
                })
            return json.dumps({"channels": chans, "hint": ""}, ensure_ascii=False)
        if tool == "hall_read":
            cid = args.get("channel_id") or "general"
            since = int(args.get("since_id") or 0)
            msgs = [m for m in self.messages if m["channel_id"] == cid and m["id"] > max(since, self.last_read.get(cid, 0) if args.get("unread_only") else 0)]
            msgs = msgs[: int(args.get("limit") or 30)]
            if not msgs:
                return f"大厅没有新消息（channel_id={cid}；last_read_id={self.last_read.get(cid, '-')}）"
            lines = [f"大厅 {len(msgs)} 条（channel_id={cid}；旧→新）。看完标已读：hall.ack(last_read_id={msgs[-1]['id']})。"]
            lines += [f"#{m['id']} [{m['display']}] {m['time']} {m['body']}" for m in msgs]
            return "\n".join(lines)
        if tool == "hall_ack":
            cid = args.get("channel_id") or "general"
            self.last_read[cid] = max(self.last_read.get(cid, 0), int(args["last_read_id"]))
            return json.dumps({"ok": True, "last_read_id": self.last_read[cid]})
        if tool == "team_inbox":
            return json.dumps({"messages": self.inbox, "count": len(self.inbox)}, ensure_ascii=False)
        raise RuntimeError(f"unknown tool {tool}")


class FakeAgentHub:
    """（v3）假 agent-hub MCP：``krow_inbox`` / ``krow_read_message`` / ``krow_send_message`` / ``krow_hall_read`` / ``krow_hall_post``。

    按 krowworker PR #1264 / 9f834380 的服务端半边建模（李靖读码 ``hall/external_mail.rs`` + ``agent-hub/src/mcp.rs``）：
    被 @ → 收件箱投 ``ref.kind=hall_mention`` 系统信 + ``hall_wakeups=suppressed`` + obligation pending；
    ``krow_read_message`` 读那封信 → ``injected``（主人面「叫不醒」→「读到了」）；带 ``reply_to_id`` 回帖 → ``answered``。
    ``naming="legacy"`` 时只认 ``team_*`` / ``hall_*`` 旧名（验证工具名回退）。``fail[op]`` 是下次调用的故障队列。
    """

    KROW = {"krow_inbox": "inbox", "krow_read_message": "read", "krow_send_message": "send", "krow_hall_read": "hall_read", "krow_hall_post": "hall_post"}
    LEGACY = {"team_inbox": "inbox", "team_read_message": "read", "team_send_message": "send", "hall_read": "hall_read", "hall_post": "hall_post"}

    def __init__(self, naming: str = "krow") -> None:
        self.naming = naming
        self.mails: Dict[str, Dict[str, Any]] = {}
        self.hall: Dict[int, Dict[str, Any]] = {}
        self.wakeups: Dict[int, str] = {}
        self.obligations: Dict[int, str] = {}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.sent: List[Dict[str, Any]] = []
        self.fail: Dict[str, List[Any]] = {}
        self._hid = 7960
        self._mid = 0

    def tool_names(self) -> List[str]:
        return list(self.KROW if self.naming == "krow" else self.LEGACY)

    def mention(self, who: str, body: str, *, channel_id: str = CHANNEL, channel_name: str = "星辰大海") -> Tuple[int, str]:
        self._hid += 1
        hid = self._hid
        self.hall[hid] = {"id": hid, "channel_id": channel_id, "display": who, "body": f"@lv-agent {body}", "reply_to_id": None, "created_at": "10-24 15:56"}
        self.wakeups[hid] = "suppressed"
        self.obligations[hid] = "pending"
        self._mid += 1
        mid = f"m-{self._mid}"
        self.mails[mid] = {"id": mid, "from_kind": "worker", "from_worker_id": None, "labels": ["sys:system"], "read": False,
                           "subject": f"大厅 #{hid}：{who} 在「{channel_name}」@ 了你", "body": f"{who}：{body}",
                           "ref": {"kind": "hall_mention", "hall_message_id": hid, "hall_channel_id": channel_id, "no_wake": True},
                           "created_at": f"2026-10-24T15:56:{self._mid:02d}Z"}
        return hid, mid

    def owner_mail(self, subject: str, body: str) -> str:
        self._mid += 1
        mid = f"o-{self._mid}"
        self.mails[mid] = {"id": mid, "from_kind": "owner", "labels": ["sys:owner"], "read": False, "subject": subject, "body": body, "ref": {},
                           "created_at": f"2026-10-24T15:56:{self._mid:02d}Z"}
        return mid

    def ops(self) -> List[str]:
        table = self.KROW if self.naming == "krow" else self.LEGACY
        return [table[t] for t, _ in self.calls if t in table]

    def __call__(self, tool: str, args: Dict[str, Any]) -> str:
        self.calls.append((tool, dict(args)))
        table = self.KROW if self.naming == "krow" else self.LEGACY
        if tool not in table:
            raise RuntimeError(f"MCP error: Unknown tool: {tool}")
        op = table[tool]
        queue = self.fail.get(op)
        special = queue.pop(0) if queue else None
        if isinstance(special, BaseException):
            raise special
        if "ref" in args:
            raise AssertionError("客户端不得往请求的 ref 里写东西")
        if op == "inbox":
            w = int(args.get("wait", 20))
            if not 0 <= w <= 25:
                raise AssertionError(f"wait {w} 超出 0..25")
            rows = [{k: v for k, v in m.items() if k != "read"} for m in self.mails.values() if not m["read"]]
            return json.dumps({"messages": rows, "count": len(rows)}, ensure_ascii=False)
        if op == "read":
            m = self.mails.get(str(args.get("message_id")))
            if m is None:
                return json.dumps({"error": "not found"})
            m["read"] = True
            ref = m.get("ref") or {}
            if ref.get("kind") == "hall_mention" and self.wakeups.get(ref["hall_message_id"]) == "suppressed":
                self.wakeups[ref["hall_message_id"]] = "injected"
            return json.dumps({k: v for k, v in m.items() if k != "read"}, ensure_ascii=False)
        if op == "send":
            self.sent.append(dict(args))
            return json.dumps({"message_id": f"sent-{len(self.sent)}"})
        if op == "hall_read":
            cid = args.get("channel_id")
            if args.get("message_id") is not None:
                rows = [self.hall[int(args["message_id"])]] if int(args["message_id"]) in self.hall else []
            else:
                rows = [m for m in self.hall.values() if cid is None or m["channel_id"] == cid]
                if args.get("since_id") is not None:
                    rows = [m for m in rows if m["id"] > int(args["since_id"])]
                rows = rows[-int(args.get("limit", 30)):]
            if self.naming == "legacy":
                return "\n".join(f"#{m['id']} [{m['display']}] {m['created_at']} {m['body']}" for m in rows) or "没有新消息"
            return json.dumps({"messages": rows}, ensure_ascii=False)
        if op == "hall_post":
            if not str(args.get("body") or "").strip():
                raise AssertionError("hall_post 正文为空")
            self._hid += 1
            mid = self._hid
            reply = args.get("reply_to_id")
            self.hall[mid] = {"id": mid, "channel_id": args.get("channel_id") or CHANNEL, "display": "lv-agent", "body": args["body"],
                              "reply_to_id": int(reply) if reply is not None else None, "created_at": "10-24 16:00"}
            if reply is not None and int(reply) in self.obligations:
                self.obligations[int(reply)] = "answered"
            if special == "post_then_timeout":
                raise TimeoutError("MCP call 'tools/call' timed out on server 'agent-hub'")
            return json.dumps({"message_id": mid, "channel_id": args.get("channel_id") or CHANNEL})
        raise AssertionError(op)


# ── 场景 ─────────────────────────────────────────────────────────────────


class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(r["ok"] for r in self.rows)


def run(verbose: bool = True) -> Report:
    # 自检里有刻意制造的失败回合(场景 8), 其 ERROR 日志是预期内的噪音, 这里压掉
    logging.getLogger("krow").setLevel(logging.CRITICAL)
    rep = Report()
    token = "kwt_fake_" + secrets.token_hex(8)
    srv = FakeKrowServer(token)
    srv.start()
    tmpdir = tempfile.mkdtemp(prefix="krow-selfcheck-")
    state_path = f"{tmpdir}/state.json"
    try:
        cfg = KrowConfig({"enabled": True, "state_path": state_path, "self_names": ["lv-agent"], "poll_interval_s": 5})

        # 1. REST 统一口子
        srv.owner_queue.append("先别动部署，只做本地测试。")
        srv.leader_queue.append({"body": "边界：不要修改 krowworker。", "author_name": "姜尚", "addendum_kind": "boundary",
                                 "task_id": "73877bda-6478-4a59-840f-e3a5c37b0656", "addendum_id": "0a1b2c3d-ffff"})
        srv.hall_queue.extend([
            {"id": 7341, "display": "姜尚#凌烟阁", "body": "请确认运行版本", "mentions_me": True, "channel_id": CHANNEL},
            {"id": 7376, "display": "华佗#Kell", "body": "已派伍连德续办。", "mentions_me": False},
        ])
        rest = RestInjectSource(srv.api, token, binding="job-selfcheck")
        br = KrowBridge(cfg, store=DedupStore(state_path), sources=[rest], nonce="n0nce")
        text = br.interjection_text(None) or ""
        rep.check("rest.unified.sections", text.startswith(OWNER_PREFIX_HEAD) and LEADER_PREFIX_HEAD in text and "#7341" in text and "#7376" in text,
                  f"{len(text)} chars")
        rep.check("rest.unified.nonce", "口令 n0nce" in text)
        rep.check("rest.unified.binding_header", any(h.get("X-Krow-Job-Binding") == "job-selfcheck" for _p, h in srv.requests))
        rep.check("rest.unified.read_consumes", br.interjection_text(None) is None, "second pull empty")
        rep.check("rest.unified.store", br.status()["store"]["seen"] == 4 and br.status()["store"]["pending"] == 0, str(br.status()["store"]))

        # 2. 旧服务端回落
        srv.mode = "legacy404"
        srv.hall_queue.append({"id": 7400, "display": "主人", "body": "在吗", "mentions_me": True, "reply_to": {"id": 7390}})
        items = br.poll()
        rep.check("rest.legacy404.fallback", [i.key for i in items] == ["hall:7400"] and "@你 在吗（回复 #7390）" in items[0].text, str([i.text for i in items]))
        br.mark_delivered(items)
        srv.mode = "unified"

        # 3. 5xx 冷却
        srv.mode = "down503"
        n_before = len(srv.requests)
        br.poll(); br.poll(); br.poll()
        rep.check("rest.503.cooldown", rest.cooling_down() and len(srv.requests) == n_before + 1, f"requests during cooldown: {len(srv.requests) - n_before}")
        srv.mode = "unified"
        rest._down_until = 0.0  # 自检不等 30 秒

        # 3b. D11: 项目频道里的 suppressed 行只有带 channel_id 轮询才取得到
        def _paths(since: int) -> List[str]:
            return [p for p, _h in srv.requests[since:]]

        srv.channel_queues[CHANNEL] = [{"id": 7510, "display": "道哥", "body": "lv-agent 在吗，报下版本", "mentions_me": True}]
        rest_off = RestInjectSource(srv.api, token, channels="off")
        got_off = rest_off.fetch()
        rep.check("rest.d11.default_call_misses_suppressed", got_off == [] and len(srv.channel_queues[CHANNEL]) == 1,
                  "channels=off 只打缺省 general → 项目频道点名取不到(这就是外部工作台被叫不醒的那一环)")
        n0 = len(srv.requests)
        rest_ch = RestInjectSource(srv.api, token, channels="auto", channel_scan_interval=60)
        got = rest_ch.fetch()
        ok = ([i.key for i in got] == ["hall:7510"] and got[0].channel_id == CHANNEL and f"channel_id={CHANNEL}" in got[0].text
              and any(p.startswith("/agent/v1/hall/channels") for p in _paths(n0))
              and any(p.startswith("/agent/v1/inject?") and f"channel_id={CHANNEL}" in p for p in _paths(n0)))
        rep.check("rest.d11.per_channel_delivers", ok, f"keys={[i.key for i in got]} paths={_paths(n0)}")
        n1 = len(srv.requests)
        rest_ch.fetch()
        rep.check("rest.d11.scan_throttled", not any("channel" in p for p in _paths(n1)), f"second fetch within interval: {_paths(n1)}")
        n2 = len(srv.requests)
        srv.channel_queues[CHANNEL].append({"id": 7512, "display": "道哥", "body": "固定频道列表也要能取", "mentions_me": True})
        rest_fixed = RestInjectSource(srv.api, token, channels=[CHANNEL])
        got_fixed = rest_fixed.fetch()
        rep.check("rest.d11.fixed_channel_list", [i.key for i in got_fixed] == ["hall:7512"] and not any(p.startswith("/agent/v1/hall/channels") for p in _paths(n2)),
                  f"keys={[i.key for i in got_fixed]} (不打 hall/channels)")
        srv.mode = "legacy404"
        n3 = len(srv.requests)
        srv.channel_queues[CHANNEL].append({"id": 7511, "display": "道哥", "body": "老服务端也要带 channel_id", "mentions_me": True})
        got_legacy = RestInjectSource(srv.api, token, channels="auto").fetch()
        rep.check("rest.d11.legacy_channel_param", [i.key for i in got_legacy] == ["hall:7511"] and got_legacy[0].channel_id == CHANNEL
                  and any(p.startswith("/agent/v1/hall/inject?") and f"channel_id={CHANNEL}" in p for p in _paths(n3)), f"paths={_paths(n3)}")
        srv.mode = "binding409"
        rest_409 = RestInjectSource(srv.api, token)
        got_409 = rest_409.fetch()
        rep.check("rest.409.binding_flag", got_409 == [] and rest_409.binding_required and rest_409.cooling_down() and "mcp_pull" in rest_409.last_error,
                  rest_409.last_error)
        srv.mode = "unified"

        # 3c. v2 坏频道隔离 —— 核心证明: 坏频道排在前面, 后面的有效频道这一轮仍被轮询并取到点名
        default_path = "/agent/v1/inject?max_owner=3&max_hall=5&max_leader=1"
        now = [1000.0]
        srv.channel_queues[CHANNEL].append({"id": 7520, "display": "道哥", "body": "坏频道排前面, 这条也要到", "mentions_me": True})
        rest_v2 = RestInjectSource(srv.api, token, channels=["星辰大海", BAD_CHANNEL, CHANNEL], clock=lambda: now[0],
                                   channel_scan_interval=10, bad_channel_backoff=300)
        n4 = len(srv.requests)
        got_v2 = rest_v2.fetch()
        p4 = _paths(n4)
        rep.check("rest.v2.bad_channel_first_still_polls_next",
                  [i.key for i in got_v2] == ["hall:7520"] and got_v2[0].channel_id == CHANNEL
                  and rest_v2.invalid_channels == ["星辰大海"] and rest_v2.channels == [BAD_CHANNEL, CHANNEL]
                  and set(rest_v2.bad_channels) == {BAD_CHANNEL} and rest_v2.bad_channels[BAD_CHANNEL]["code"] == 404
                  and p4 == [default_path, f"{default_path}&channel_id={BAD_CHANNEL}", f"{default_path}&channel_id={CHANNEL}"],
                  f"keys={[i.key for i in got_v2]} invalid={rest_v2.invalid_channels}(未发请求) bad={list(rest_v2.bad_channels)}(404→隔离) paths={p4}")
        rep.check("rest.v2.explicit_404_not_legacy",
                  rest_v2.legacy_mode is False and not any("/agent/v1/hall/inject" in p for p in p4) and rest_v2.last_error == "" and not rest_v2.cooling_down(),
                  f"legacy_mode={rest_v2.legacy_mode} last_error={rest_v2.last_error!r} (显式频道 404 没触发老口子回落、没污染 last_error、不冷却)")
        now[0] += 11  # 过了扫描间隔、没到退避: 坏频道不再打, 好频道照常
        n5 = len(srv.requests)
        rest_v2.fetch()
        p5 = _paths(n5)
        rep.check("rest.v2.bad_channel_quarantined", not any(BAD_CHANNEL in p for p in p5) and any(CHANNEL in p for p in p5), f"paths={p5}")
        now[0] += 300  # 退避到期: 重试; 此时频道已建好(可见) → 解除隔离并取到消息
        srv.channel_queues[BAD_CHANNEL] = [{"id": 7530, "display": "主人", "body": "频道建好了, 现在能看到吗", "mentions_me": True}]
        n6 = len(srv.requests)
        got_rec = rest_v2.fetch()
        rep.check("rest.v2.bad_channel_recovers",
                  [i.key for i in got_rec] == ["hall:7530"] and got_rec[0].channel_id == BAD_CHANNEL and BAD_CHANNEL not in rest_v2.bad_channels
                  and any(BAD_CHANNEL in p for p in _paths(n6)),
                  f"keys={[i.key for i in got_rec]} bad={list(rest_v2.bad_channels)}")
        del srv.channel_queues[BAD_CHANNEL]
        # auto 模式: hall/channels 列出的幽灵频道(排在前面)显式拉取 404 → 只隔离它, 后面的真频道照常
        srv.ghost_channels.append(BAD_CHANNEL)
        srv.channel_queues[CHANNEL].append({"id": 7540, "display": "道哥", "body": "auto 模式幽灵频道排前面", "mentions_me": True})
        rest_auto = RestInjectSource(srv.api, token, channels="auto")
        n7 = len(srv.requests)
        got_auto = rest_auto.fetch()
        p7 = _paths(n7)
        rep.check("rest.v2.auto_ghost_channel_isolated",
                  [i.key for i in got_auto] == ["hall:7540"] and BAD_CHANNEL in rest_auto.bad_channels and rest_auto.last_error == ""
                  and p7.index(f"{default_path}&channel_id={BAD_CHANNEL}") < p7.index(f"{default_path}&channel_id={CHANNEL}"),
                  f"keys={[i.key for i in got_auto]} bad={list(rest_auto.bad_channels)} paths={p7}")
        srv.ghost_channels.clear()
        # 老服务端: 回落只由缺省调用判定一次; 频道调用直接走老口子(不逐频道再探统一口子); 老口子上坏频道 404 同样只隔离
        srv.mode = "legacy404"
        srv.channel_queues[CHANNEL].append({"id": 7550, "display": "道哥", "body": "老服务端上坏频道也只隔离", "mentions_me": True})
        rest_leg = RestInjectSource(srv.api, token, channels=[BAD_CHANNEL, CHANNEL])
        n8 = len(srv.requests)
        got_leg = rest_leg.fetch()
        p8 = _paths(n8)
        unified_calls = [p for p in p8 if p.startswith("/agent/v1/inject?")]
        legacy_calls = [p for p in p8 if p.startswith("/agent/v1/hall/inject?")]
        rep.check("rest.v2.legacy_decided_once_by_default_call",
                  [i.key for i in got_leg] == ["hall:7550"] and rest_leg.legacy_mode is True
                  and unified_calls == [default_path]
                  and legacy_calls == ["/agent/v1/hall/inject?max=5", f"/agent/v1/hall/inject?max=5&channel_id={BAD_CHANNEL}",
                                       f"/agent/v1/hall/inject?max=5&channel_id={CHANNEL}"]
                  and BAD_CHANNEL in rest_leg.bad_channels and rest_leg.last_error == "",
                  f"unified={unified_calls} legacy={legacy_calls}")
        srv.mode = "unified"
        # 频道口子 5xx: 源级 → 冷却、停止本轮扫描, 但缺省调用已拿到的主人插话照样交出去(fail-open 保留)
        srv.mode = "channel503"
        srv.owner_queue.append("频道口子滚动发布中也要把这句送到。")
        rest_fo = RestInjectSource(srv.api, token, channels=[CHANNEL, BAD_CHANNEL])
        n9 = len(srv.requests)
        got_fo = rest_fo.fetch()
        p9 = _paths(n9)
        rep.check("rest.v2.channel_5xx_fail_open",
                  [i.kind for i in got_fo] == ["owner"] and rest_fo.cooling_down() and "503" in rest_fo.last_error
                  and not rest_fo.bad_channels and not any(BAD_CHANNEL in p for p in p9),
                  f"kinds={[i.kind for i in got_fo]} last_error={rest_fo.last_error!r} (5xx 不标坏频道; 本轮停扫, 第二个频道没打)")
        srv.mode = "unified"
        # /krow 状态: 409 只说"已见 / 未见", 不据此断言 H1 已回答; 坏频道 / 非法项 / 旧接口判定可见
        st_409 = KrowBridge(cfg, store=DedupStore(state_path), sources=[rest_409], nonce="n0nce").status()
        st_ok = KrowBridge(cfg, store=DedupStore(state_path), sources=[rest_v2], nonce="n0nce").status()
        rep.check("rest.v2.409_status_semantics",
                  st_409["rest_409_seen"] is True and "已见 409" in st_409["rest_409_note"] and "单向" in st_409["rest_409_note"] and "H1" in st_409["rest_409_note"]
                  and st_ok["rest_409_seen"] is False and "未见 409" in st_ok["rest_409_note"] and "不等于 mcp_pull 已确认" in st_ok["rest_409_note"]
                  and "rest_binding_required" not in st_ok and st_ok["rest_invalid_channels"] == ["星辰大海"] and st_ok["rest_legacy_mode"] is False,
                  f"seen: {st_409['rest_409_note'][:40]}… / not seen: {st_ok['rest_409_note'][:40]}…")
        # REST 看不见的频道由 MCP 路径兜底(共用一本账)
        mcp_fb = FakeMcpBridge()
        mcp_fb.messages = [{"id": 7560, "channel_id": BAD_CHANNEL, "display": "道哥", "time": "10-24 08:30", "body": "@lv-agent REST 看不见这个频道, MCP 能"}]
        rest_fb = RestInjectSource(srv.api, token, channels=[BAD_CHANNEL])
        br_fb = KrowBridge(cfg, store=DedupStore(f"{tmpdir}/fb.json"),
                           sources=[rest_fb, McpHallSource(mcp_fb, self_names=["lv-agent"], inbox=False)], nonce="n0nce")
        got_fb = br_fb.poll()
        rep.check("rest.v2.mcp_covers_rest_bad_channel",
                  [i.key for i in got_fb] == ["hall:7560"] and got_fb[0].source == "mcp" and BAD_CHANNEL in rest_fb.bad_channels
                  and br_fb.status()["source_errors"]["rest"] == "",
                  f"keys={[i.key for i in got_fb]} via={got_fb[0].source if got_fb else None} rest_bad={list(rest_fb.bad_channels)}")

        # 4. MCP 轮询
        mcp = FakeMcpBridge()
        mcp.messages = [
            {"id": 7329, "channel_id": CHANNEL, "display": "王玄策#凌烟阁", "time": "10-24 06:55", "body": "已到岗。"},
            {"id": 7341, "channel_id": CHANNEL, "display": "姜尚#凌烟阁", "time": "10-24 07:01", "body": "@lv-agent#默认小组#captain 请确认运行版本"},
            {"id": 7342, "channel_id": CHANNEL, "display": "captain", "time": "10-24 07:04", "body": "@lv-agent 2#默认小组#captain 这条点的是另一个人"},
            {"id": 7350, "channel_id": "general", "display": "白起#凌烟阁", "time": "10-24 07:05", "body": "general 里的闲聊"},
        ]
        mcp.inbox = [{"id": "f3b613cd-45f3-442b-859a-8f2edf14ae13", "subject": "复核裁定及LV-Agent研发直派", "body": "任务 73877bda 直派给你。",
                      "from_worker_name": "姜尚", "labels": ["sys:internal"], "created_at": "2026-10-24T07:15:54Z"}]
        store2 = DedupStore(state_path)
        mcp_src = McpHallSource(mcp, self_names=["lv-agent"], inbox=True, cursor_get=store2.cursor, cursor_set=store2.set_cursor)
        br2 = KrowBridge(cfg, store=store2, sources=[rest, mcp_src], nonce="n0nce")
        items = br2.poll()
        keys = [i.key for i in items]
        rep.check("mcp.poll.mentions_only", keys == ["hall:7342", "mail:f3b613cd-45f3-442b-859a-8f2edf14ae13"],
                  f"keys={keys} (7341 已由 REST 送过 → 去重; 7329 无点名; general 无点名)")
        rep.check("mcp.poll.ack", mcp.last_read.get(CHANNEL) == 7342 and store2.cursor(f"hall:{CHANNEL}") == 7342, str(mcp.last_read))
        rep.check("mcp.poll.tools", [c[0] for c in mcp.calls] == ["hall_channels", "hall_read", "hall_ack", "team_inbox"], str([c[0] for c in mcp.calls]))

        # 5/6. 取到 ≠ 送到：重启后 pending 仍在；送达后不再出现；再次轮询 inbox 同一封信不重复
        store3 = DedupStore(state_path)
        pend = [p.key for p in store3.take_pending()]
        rep.check("state.pending_survives_restart", pend == keys, str(pend))
        br3 = KrowBridge(cfg, store=store3, sources=[McpHallSource(mcp, self_names=["lv-agent"], inbox=True, cursor_get=store3.cursor, cursor_set=store3.set_cursor)], nonce="n0nce")
        again = br3.poll()
        rep.check("state.no_duplicate_before_delivery", [i.key for i in again] == keys and br3.last_new == 0, f"last_new={br3.last_new}")
        wake = build_wake_task(again, nonce="n0nce", tool_prefix="mcp_krow-platform_")
        rep.check("wake.task_text", "#7342" in wake and "mcp_krow-platform_hall_post" in wake and "team_read_message" in wake)
        br3.mark_delivered(again)
        rep.check("state.delivered_not_repolled", br3.poll() == [] and DedupStore(state_path).stats()["pending"] == 0)

        # 7. ExecutionEngine 集成（假后端）
        try:
            from ..config import AgentConfig
            from ..execution_engine import ExecutionContext, ExecutionEngine
            from ..policies import ReActPolicy
            from ..tools import TOOLS_REGISTRY
            from ..tools.core import ToolResult

            class _Fake:
                name = "web_search"
                description = "fake"
                parameters = {"type": "object", "properties": {}}

                def execute(self, query: str = "", **kw: Any) -> ToolResult:
                    return ToolResult(success=True, output='[{"title": "ok"}]')

            old_tool = TOOLS_REGISTRY.get("web_search")
            TOOLS_REGISTRY._tools["web_search"] = _Fake()
            try:
                srv.hall_queue.append({"id": 7500, "display": "姜尚#凌烟阁", "body": "边界提醒：不推送", "mentions_me": True, "channel_id": CHANNEL})
                prompts: List[str] = []

                class FB:
                    def __init__(self) -> None:
                        self.c = 0

                    def generate(self, prompt: str, **kw: Any) -> str:
                        prompts.append(prompt)
                        self.c += 1
                        if self.c == 1:
                            return '{"action": "web_search", "args": {"query": "a"}}'
                        if self.c == 2:
                            # 主人在模型准备收口的那一刻才插话(工具边界那次拉取已经过去) → 只能靠轮尾拉取接住
                            srv.owner_queue.append("收口前补一句版本号。")
                            return '{"final_answer": "版本是 986f5bf"}'
                        return '{"final_answer": "版本是 986f5bf，已按主人要求补充。"}'

                br4 = KrowBridge(cfg, store=DedupStore(state_path), sources=[rest], nonce="n0nce")
                acfg = AgentConfig()
                eng = ExecutionEngine(model_backend=FB(), config=acfg, interjection_provider=br4)
                ctx = ExecutionContext(task="查版本", available_tools=TOOLS_REGISTRY.get_tools_dict(), config=acfg, max_steps=8)
                ctx.monitor_enabled = False
                tr = eng.run(ReActPolicy(), ctx)
                rep.check("engine.contract_in_prompt", "krow 运行中注入契约" in prompts[0])
                rep.check("engine.tool_boundary_inject", len(prompts) >= 2 and "#7500" in prompts[1] and "#7500" not in prompts[0])
                rep.check("engine.stop_continue", tr.final_answer == "版本是 986f5bf，已按主人要求补充。" and len(prompts) == 3 and "收口前补一句版本号" in prompts[2],
                          f"final={tr.final_answer!r} prompts={len(prompts)}")
            finally:
                TOOLS_REGISTRY._tools["web_search"] = old_tool
        except Exception as e:  # noqa: BLE001 — 自检要把失败写进报告，不是抛出去
            rep.check("engine.integration", False, f"{type(e).__name__}: {e}")

        # 8. 守护模式
        srv.hall_queue.append({"id": 7600, "display": "主人", "body": "lv-agent 在吗", "mentions_me": True})
        br5 = KrowBridge(cfg, store=DedupStore(state_path), sources=[rest], nonce="n0nce")
        ran: List[str] = []

        def run_task(task: str) -> Dict[str, Any]:
            ran.append(task)
            if br5.status()["store"]["pending"] != 0:
                raise AssertionError("pending should be 0 while running (记账先于执行)")
            raise RuntimeError("model exploded")

        d = KrowDaemon(br5, run_task, interval_s=1)
        out = d.run_once()
        rep.check("daemon.mark_before_run", out["status"] == "error" and "model exploded" in out["error"] and len(ran) == 1 and "#7600" in ran[0], str(out))
        rep.check("daemon.no_rerun_same_item", d.run_once() == {"status": "idle"})
        rep.check("daemon.busy_cleared", not br5.busy)
    finally:
        srv.stop()

    # 10. （v3）守候循环: 收件箱系统信 → 真读翻 injected → hall_read 原文 → 作答 → 带 reply_to_id 回帖翻 answered
    _v3_scenarios(rep, tmpdir)

    if verbose:
        width = max(len(r["name"]) for r in rep.rows) + 2
        for r in rep.rows:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{mark}] {r['name']:<{width}} {r['detail']}")
        print(f"\n{sum(r['ok'] for r in rep.rows)}/{len(rep.rows)} passed; state file: {state_path}")
    return rep


def _v3_scenarios(rep: Report, tmpdir: str) -> None:
    """（v3）守候循环闭环。上线验证的三条口径（李靖 03125e84）在假 agent-hub 上逐条核对：
    ① 被 @ 后收件箱出现 hall_mention 信；② 读信后 wakeups 翻 injected；③ 带 reply_to_id 回帖后 obligations 翻 answered。"""
    from .mailloop import KrowMailLoop
    from .tools import KrowTools

    def make_loop(hub: FakeAgentHub, store: DedupStore, run_task: Any, *, listed: bool = True, **kw: Any) -> KrowMailLoop:
        tools = KrowTools(hub, list_tools=hub.tool_names if listed else None)
        kw.setdefault("self_names", ["lv-agent"])
        kw.setdefault("backoff_base_s", 0.01)
        kw.setdefault("backoff_max_s", 0.02)
        return KrowMailLoop(tools, store, run_task=run_task, min_loop_s=0.0, jitter=False, nonce="n0nce", **kw)

    try:
        # a. 闭环三条口径 + 调用顺序 + 工具名解析(krow_*)
        hub = FakeAgentHub()
        hub.hall[7950] = {"id": 7950, "channel_id": CHANNEL, "display": "李靖", "body": "之前的上下文", "reply_to_id": None, "created_at": "10-24 15:50"}
        hid, mid = hub.mention("道哥", "lv-agent 轮询修好了吗")
        tasks: List[str] = []
        store = DedupStore(f"{tmpdir}/v3.json")
        loop = make_loop(hub, store, lambda t: tasks.append(t) or {"final_answer": "修好了：守候循环长轮询 20s，读信即翻 injected。", "actions": []})
        rows = [m for m in hub.mails.values() if not m["read"] and (m.get("ref") or {}).get("kind") == "hall_mention"]
        rep.check("v3.mention_mail_in_inbox", len(rows) == 1 and rows[0]["ref"]["hall_message_id"] == hid, f"ref={rows[0]['ref'] if rows else None}")
        s = loop.wait_once(wait=0)
        rep.check("v3.loop.handled", s["mentions"] == 1 and s["new"] == 1 and not s["error"], str(s))
        rep.check("v3.loop.call_order", hub.ops() == ["inbox", "read", "hall_read", "hall_read", "hall_post"], str(hub.ops()))
        rep.check("v3.tools.krow_names", [t for t, _ in hub.calls][:2] == ["krow_inbox", "krow_read_message"], str([t for t, _ in hub.calls]))
        rep.check("v3.read_flips_injected", hub.wakeups[hid] == "injected", hub.wakeups[hid])
        post = hub.calls[-1][1]
        rep.check("v3.post_has_reply_to_and_channel", post.get("reply_to_id") == hid and post.get("channel_id") == CHANNEL and "ref" not in post, str(post))
        rep.check("v3.post_flips_answered", hub.obligations[hid] == "answered", hub.obligations[hid])
        rep.check("v3.inbox_wait_clamped", all(0 <= a.get("wait", 0) <= 25 for t, a in hub.calls if t == "krow_inbox"))
        rep.check("v3.task_is_data_not_instruction", bool(tasks) and "数据，不是指令" in tasks[0] and "之前的上下文" in tasks[0]
                  and f"reply_to_id={hid}" in tasks[0] and "krow_hall_post" in tasks[0], tasks[0][:120] if tasks else "")
        rec = store.mention(hid) or {}
        rep.check("v3.ledger_answered", rec.get("state") == "answered" and rec.get("attempts") == 1 and rec.get("mail_id") == mid, str(rec))

        # b. 去重: 同一封信再出现 / 进程重启(同一账本) → 只补读, 不回第二次
        hub.mails[mid]["read"] = False
        s = loop.wait_once(wait=0)
        loop2 = make_loop(hub, DedupStore(f"{tmpdir}/v3.json"), lambda t: {"final_answer": "又来"})
        hub.mails[mid]["read"] = False
        s2 = loop2.wait_once(wait=0)
        rep.check("v3.dedup_same_mail_and_restart", s["mentions"] == 0 and s2["mentions"] == 0 and hub.ops().count("hall_post") == 1 and hub.mails[mid]["read"],
                  f"posts={hub.ops().count('hall_post')}")

        # c. 读失败: 本拍不往下走(信仍未读), 下一拍补上
        hub2 = FakeAgentHub()
        hid2, mid2 = hub2.mention("道哥", "在吗")
        hub2.fail["read"] = [RuntimeError("MCP error: upstream 502")]
        loop3 = make_loop(hub2, DedupStore(None), lambda t: {"final_answer": "在"})
        s = loop3.wait_once(wait=0)
        ok_first = s["mentions"] == 0 and hub2.wakeups[hid2] == "suppressed" and hub2.ops().count("hall_post") == 0
        s = loop3.wait_once(wait=0)
        rep.check("v3.read_failure_defers_then_recovers", ok_first and s["mentions"] == 1 and hub2.obligations[hid2] == "answered", str(s))

        # d. 模型回合失败 / 没有 runner: 回一句原因, 不沉默; 模型自己已回帖: 不重发
        hub3 = FakeAgentHub()
        h_a, _ = hub3.mention("道哥", "一")

        def boom(_t: str) -> Any:
            raise RuntimeError("model exploded")

        loop4 = make_loop(hub3, DedupStore(None), boom)
        loop4.wait_once(wait=0)
        body_a = hub3.calls[-1][1].get("body", "")
        h_b, _ = hub3.mention("道哥", "二")
        loop4.run_task = None
        loop4.wait_once(wait=0)
        body_b = hub3.calls[-1][1].get("body", "")
        rep.check("v3.failure_posts_reason", "自动回执" in body_a and "model exploded" in body_a and "没有可用的模型回合" in body_b
                  and hub3.obligations[h_a] == "answered" and hub3.obligations[h_b] == "answered", f"{body_a[:60]} | {body_b[:60]}")
        h_c, _ = hub3.mention("道哥", "三")

        def self_post(_t: str) -> Any:
            hub3("krow_hall_post", {"body": "我自己回", "reply_to_id": h_c, "channel_id": CHANNEL})
            return {"final_answer": "我自己回", "actions": [{"tool_name": "mcp_agent-hub_krow_hall_post", "arguments": {"reply_to_id": h_c}}]}

        loop4.run_task = self_post
        n_before = hub3.ops().count("hall_post")
        loop4.wait_once(wait=0)
        rep.check("v3.model_self_post_not_duplicated", hub3.ops().count("hall_post") == n_before + 1 and (loop4.store.mention(h_c) or {}).get("answered_by") == "model")

        # e. 回帖瞬时错误重试 / 超时但服务端已落库 → 查重不重发 / 4xx 不重试记 failed
        hub4 = FakeAgentHub()
        h1, _ = hub4.mention("道哥", "一")
        hub4.fail["hall_post"] = [RuntimeError("MCP error: upstream 503")]
        loop5 = make_loop(hub4, DedupStore(None), lambda t: {"final_answer": "答"})
        loop5.wait_once(wait=0)
        ok_retry = hub4.ops().count("hall_post") == 2 and hub4.obligations[h1] == "answered"
        h2, _ = hub4.mention("道哥", "二")
        hub4.fail["hall_post"] = ["post_then_timeout"]
        loop5.wait_once(wait=0)
        replies_h2 = [m for m in hub4.hall.values() if m["reply_to_id"] == h2]
        ok_timeout = len(replies_h2) == 1 and (loop5.store.mention(h2) or {}).get("note") == "confirmed_by_hall_read"
        h3, _ = hub4.mention("道哥", "三")
        hub4.fail["hall_post"] = [RuntimeError("MCP error: 404 频道不存在或不可见")]
        n = hub4.ops().count("hall_post")
        loop5.wait_once(wait=0)
        ok_4xx = hub4.ops().count("hall_post") == n + 1 and (loop5.store.mention(h3) or {}).get("state") == "failed"
        rep.check("v3.post_retry_timeout_dedup_4xx", ok_retry and ok_timeout and ok_4xx, f"retry={ok_retry} timeout={ok_timeout} 4xx={ok_4xx}")

        # f. 主人直聊最高优先级: 同一批里先读主人信、跑回合、krow_send_message(to_role=owner, in_reply_to)
        hub5 = FakeAgentHub()
        h5, _ = hub5.mention("李靖", "在吗")
        o5 = hub5.owner_mail("先别发布", "lv-agent 先别推 main")
        seen: List[str] = []
        loop6 = make_loop(hub5, DedupStore(None), lambda t: seen.append(t) or {"final_answer": "收到，不推。", "actions": []})
        s = loop6.wait_once(wait=0)
        reads = [a["message_id"] for t, a in hub5.calls if t == "krow_read_message"]
        rep.check("v3.owner_first_and_replied", s["owner"] == 1 and s["mentions"] == 1 and reads[0] == o5 and seen[0].startswith(OWNER_PREFIX_HEAD)
                  and hub5.sent == [{"to_role": "owner", "in_reply_to": o5, "body": "收到，不推。"}] and hub5.obligations[h5] == "answered",
                  f"reads={reads} sent={hub5.sent}")

        # g. 收件箱出错指数退避并在成功后复位; 瞬间空返补最小间隔
        hub6 = FakeAgentHub()
        hub6.fail["inbox"] = [RuntimeError("MCP error: 503")] * 3
        loop7 = make_loop(hub6, DedupStore(None), None, backoff_base_s=2.0, backoff_max_s=5.0)
        delays = [loop7.wait_once(wait=0)["delay"] for _ in range(3)]
        s = loop7.wait_once(wait=0)
        loop7.min_loop_s = 5.0
        s_idle = loop7.wait_once(wait=0)
        rep.check("v3.backoff_and_min_interval", delays == [2.0, 4.0, 5.0] and not s["error"] and loop7.status()["consecutive_errors"] == 0 and s_idle["delay"] == 5.0,
                  f"delays={delays} idle_delay={s_idle['delay']}")

        # h. 旧命名服务器: krow_* 被拒后回退 team_* / hall_*, 闭环照样成立; hall_read 文本解析
        hub7 = FakeAgentHub(naming="legacy")
        h7, _ = hub7.mention("道哥", "在吗")
        loop8 = make_loop(hub7, DedupStore(None), lambda t: {"final_answer": "在"}, listed=False)
        s = loop8.wait_once(wait=0)
        rep.check("v3.legacy_names_fallback", s["mentions"] == 1 and hub7.obligations[h7] == "answered" and hub7.ops() == ["inbox", "read", "hall_read", "hall_read", "hall_post"]
                  and loop8.tool_names()["hall_post"] == "hall_post", f"ops={hub7.ops()} tools={loop8.tool_names()}")

        # i. 崩溃续办: read 且未发 → 重新作答回帖; read 且发过 → 查大厅确认 / unverified
        hub8 = FakeAgentHub()
        h8a, m8a = hub8.mention("道哥", "一")
        h8b, m8b = hub8.mention("道哥", "二")
        st8 = DedupStore(f"{tmpdir}/v3-resume.json")
        hub8("krow_read_message", {"message_id": m8a})
        st8.set_mention(h8a, state="read", mail_id=m8a, channel_id=CHANNEL, attempts=0)
        st8.set_mention(h8b, state="read", mail_id=m8b, channel_id=CHANNEL, attempts=1)
        loop9 = make_loop(hub8, DedupStore(f"{tmpdir}/v3-resume.json"), lambda t: {"final_answer": "补回"})
        out = loop9.resume_pending()
        rep.check("v3.resume_pending", out == {"resumed": 1, "settled": 1} and hub8.obligations[h8a] == "answered"
                  and (loop9.store.mention(h8b) or {}).get("state") == "unverified" and hub8.ops().count("hall_post") == 1, f"{out}")

        # j. 守护驱动: KrowDaemon.run_once 先跑守候循环一拍再 poll; 空转时不固定睡(长轮询即等待)
        hub9 = FakeAgentHub()
        h9, _ = hub9.mention("道哥", "在吗")
        tools9 = KrowTools(hub9, list_tools=hub9.tool_names)
        st9 = DedupStore(None)
        cfg9 = KrowConfig({"enabled": True, "self_names": ["lv-agent"], "poll_interval_s": 5})
        br9 = KrowBridge(cfg9, store=st9, sources=[McpHallSource(tools9.call, self_names=["lv-agent"], inbox=False)])
        br9.tools = tools9
        br9.attach_mail_loop(KrowMailLoop(tools9, st9, backoff_base_s=0.01, backoff_max_s=0.02, min_loop_s=0.0, jitter=False, self_names=["lv-agent"]))
        d9 = KrowDaemon(br9, lambda t: {"final_answer": "在"}, interval_s=5, wait_s=0)
        out1 = d9.run_once()
        out2 = d9.run_once()
        rep.check("v3.daemon_drives_loop", out1["status"] == "woken" and out1["mail"]["mentions"] == 1 and hub9.obligations[h9] == "answered"
                  and out2["status"] == "idle" and out2["mail"]["delay"] == 0.0 and not br9.busy, f"{out1['status']}/{out2['status']}")
        # 没有任何请求带 ref(hall_mention 是服务端保留值)
        rep.check("v3.never_writes_ref", all("ref" not in a for h in (hub, hub2, hub3, hub4, hub5, hub6, hub7, hub8, hub9) for _, a in h.calls))
    except Exception as e:  # noqa: BLE001 — 自检要把失败写进报告
        rep.check("v3.scenarios", False, f"{type(e).__name__}: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    rep = run(verbose="--json" not in args)
    if "--json" in args:
        print(json.dumps({"passed": rep.passed, "checks": rep.rows}, ensure_ascii=False, indent=2))
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
