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
8. 守护模式：``KrowDaemon.run_once`` 先记账再跑、回合失败不重跑。
"""
from __future__ import annotations

import json
import logging
import secrets
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .bridge import KrowBridge, KrowConfig
from .items import LEADER_PREFIX_HEAD, OWNER_PREFIX_HEAD
from .sources import McpHallSource, RestInjectSource
from .state import DedupStore
from .wake import KrowDaemon, build_wake_task

CHANNEL = "f76423cc-49a5-4263-b0b7-fe0fcc87a92c"


# ── 假服务端 ─────────────────────────────────────────────────────────────


class FakeKrowServer:
    """``/agent/v1/inject``（读即消费）与 ``/agent/v1/hall/inject``；可切 404 / 503 模式。"""

    def __init__(self, token: str) -> None:
        self.token = token
        self.mode = "unified"  # unified | legacy404 | down503
        self.owner_queue: List[str] = []
        self.leader_queue: List[Dict[str, Any]] = []
        self.hall_queue: List[Dict[str, Any]] = []
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
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/agent/v1/inject":
                if self.mode == "legacy404":
                    return self._send(h, 404, {"error": "not found"})
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
                    msgs, self.hall_queue = self.hall_queue[:max_hall], self.hall_queue[max_hall:]
                    payload["hall"] = {"text": self._render_hall(msgs), "remaining": len(self.hall_queue)}
                return self._send(h, 200, payload)
            if url.path == "/agent/v1/hall/inject":
                n = int(qs.get("max", 5))
                msgs, self.hall_queue = self.hall_queue[:n], self.hall_queue[n:]
                return self._send(h, 200, {"items": msgs, "remaining": len(self.hall_queue), "last_read_id": max([m["id"] for m in msgs] or [0])})
            return self._send(h, 404, {"error": "no route"})

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

    if verbose:
        width = max(len(r["name"]) for r in rep.rows) + 2
        for r in rep.rows:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{mark}] {r['name']:<{width}} {r['detail']}")
        print(f"\n{sum(r['ok'] for r in rep.rows)}/{len(rep.rows)} passed; state file: {state_path}")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    rep = run(verbose="--json" not in args)
    if "--json" in args:
        print(json.dumps({"passed": rep.passed, "checks": rep.rows}, ensure_ascii=False, indent=2))
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
