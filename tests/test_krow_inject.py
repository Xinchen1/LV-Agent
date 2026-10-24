"""krow 原生点名注入 / 轮询 / 防重复消费 —— 离线回归(不碰网络、不碰现网)."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error

import pytest

from agent_project.config import AgentConfig
from agent_project.execution_engine import ExecutionContext, ExecutionEngine
from agent_project.krow import (
    DedupStore,
    InjectItem,
    KrowBridge,
    KrowConfig,
    KrowDaemon,
    build_bridge,
    build_wake_task,
    contract_text,
    render_items,
)
from agent_project.krow.items import HALL_PREFIX, LEADER_PREFIX_HEAD, MAIL_PREFIX, OWNER_PREFIX_HEAD
from agent_project.krow.sources import (
    McpHallSource,
    RestInjectSource,
    parse_hall_read,
    parse_inject_hall_text,
    resolve_rest_env,
)
from agent_project.krow.wake import WAKE_HEAD
from agent_project.policies import ReActPolicy
from agent_project.tools import TOOLS_REGISTRY
from agent_project.tools.core import ToolResult

CHANNEL = "f76423cc-49a5-4263-b0b7-fe0fcc87a92c"

# ---------------------------------------------------------------------------
# 假数据
# ---------------------------------------------------------------------------

UNIFIED_PAYLOAD = {
    "owner": {"text": "先别动部署, 只做本地测试。"},
    "leader": {
        "items": [
            {
                "body": "边界: 不要修改 krowworker。",
                "author_name": "姜尚",
                "addendum_kind": "boundary",
                "task_id": "73877bda-6478-4a59-840f-e3a5c37b0656",
                "addendum_id": "0a1b2c3d-ffff-0000-1111-222233334444",
            }
        ]
    },
    "hall": {
        "text": (
            f"#7360 【大厅 · krowwork 联调 · 合作方·马修权】@你 请分析批量复盘（来自别的频道，回复请带 channel_id={CHANNEL}）\n"
            "#7376 【大厅 · 华佗#Kell】已派伍连德续办登记缺口。"
        )
    },
}

HALL_READ_TEXT = (
    f"大厅 3 条（channel_id={CHANNEL}；你已读到 #7300；旧→新）。看完标已读：hall.ack(last_read_id=7342)。\n"
    "#7329 [王玄策#凌烟阁] 10-24 06:55 回 #7164：已到岗。（回复 #7164）\n"
    "#7341 [姜尚#凌烟阁] 10-24 07:01 @lv-agent#默认小组#captain 请确认运行版本〔待回应〕\n"
    "  （附件：排查报告.md → /workspace/inbox/hall/7341/排查报告.md）\n"
    "#7342 [captain] 10-24 07:04 @lv-agent 2#默认小组#captain 这条点的是另一个人\n"
)

CHANNELS_JSON = json.dumps(
    {
        "channels": [
            {"id": "general", "name": "general", "kind": "general", "unread_count": 2, "mention_count": 0},
            {"id": CHANNEL, "name": "星辰大海", "kind": "project", "unread_count": 3, "mention_count": 1},
        ],
        "hint": "",
    }
)

INBOX_JSON = json.dumps(
    {
        "messages": [
            {
                "id": "f3b613cd-45f3-442b-859a-8f2edf14ae13",
                "subject": "复核裁定及LV-Agent研发直派",
                "body": "任务 73877bda 直派给你。\n先核版本。",
                "from_worker_name": "姜尚",
                "labels": ["sys:internal"],
                "created_at": "2026-10-24T07:15:54Z",
                "attachments": [],
            }
        ],
        "count": 1,
    }
)


class FakeHttp:
    """``http_get`` 替身: 按路径返回预设 (status, body) 或抛 HTTPError."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers)))
        path = url.split("?", 1)[0]
        for prefix, resp in self.routes.items():
            if path.endswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                status, body = resp
                return status, body
        raise urllib.error.HTTPError(url, 404, "nf", {}, None)


class FakeMcp:
    """``call_tool(tool, args) -> str`` 替身, 记录调用."""

    def __init__(self, *, hall_text=HALL_READ_TEXT, channels=CHANNELS_JSON, inbox=INBOX_JSON):
        self.calls = []
        self.hall_text = hall_text
        self.channels = channels
        self.inbox = inbox

    def __call__(self, tool, args):
        self.calls.append((tool, dict(args)))
        if tool == "hall_channels":
            return self.channels
        if tool == "hall_read":
            return self.hall_text
        if tool == "hall_ack":
            return json.dumps({"ok": True, "last_read_id": args.get("last_read_id")})
        if tool == "team_inbox":
            return self.inbox
        raise RuntimeError(f"unexpected tool {tool}")


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


# ---------------------------------------------------------------------------
# items: 文案契约
# ---------------------------------------------------------------------------

def test_render_items_sections_order_and_nonce():
    items = [
        InjectItem(key="hall:1", kind="hall", text="#1 甲：@你 在吗"),
        InjectItem(key="owner:x", kind="owner", text="改走方案 B"),
        InjectItem(key="mail:m1", kind="mail", text="信 m1 来自 乙：主题 — 正文"),
        InjectItem(key="leader:a", kind="leader", text="验收: 加回归用例",
                   meta={"author_name": "姜尚", "addendum_kind": "acceptance", "task_id": "73877bda-x", "addendum_id": "abcdef12-y"}),
    ]
    text = render_items(items, "9d79260a")
    i_owner = text.index(OWNER_PREFIX_HEAD)
    i_leader = text.index(LEADER_PREFIX_HEAD)
    i_hall = text.index(HALL_PREFIX.rstrip("\n"))
    i_mail = text.index(MAIL_PREFIX.rstrip("\n"))
    assert i_owner < i_leader < i_hall < i_mail
    assert "口令 9d79260a" in text
    assert "姜尚 · 验收 · 任务 73877bda · addendum abcdef12" in text
    assert "改走方案 B" in text and "#1 甲：@你 在吗" in text
    # 契约块同口令、同抬头
    contract = contract_text("9d79260a")
    assert f"{OWNER_PREFIX_HEAD}，口令 9d79260a）" in contract
    assert "hall_post" in contract


def test_render_items_clips_hall_first_and_never_drops_owner():
    long_hall = [InjectItem(key=f"hall:{i}", kind="hall", text=f"#{i} 甲：" + "x" * 3000) for i in range(5)]
    owner = InjectItem(key="owner:1", kind="owner", text="主人的话" * 50)
    text = render_items([owner] + long_hall, "n", max_total_chars=4000)
    assert len(text) <= 4000
    assert "主人的话" in text
    assert text.startswith(OWNER_PREFIX_HEAD)


def test_render_items_empty():
    assert render_items([], "n") == ""
    assert render_items([InjectItem(key="k", kind="hall", text="   ")], "n") == ""


# ---------------------------------------------------------------------------
# state: 防重复消费
# ---------------------------------------------------------------------------

def test_dedup_store_pending_then_seen_persists(tmp_path):
    path = tmp_path / "krow_state.json"
    st = DedupStore(str(path))
    a = InjectItem(key="hall:1", kind="hall", text="a")
    b = InjectItem(key="hall:2", kind="hall", text="b")
    assert [i.key for i in st.add_pending([a, b, a])] == ["hall:1", "hall:2"]
    assert st.add_pending([a]) == []  # 已在待送队列
    # 进程重启: pending 还在
    st2 = DedupStore(str(path))
    assert [i.key for i in st2.take_pending()] == ["hall:1", "hall:2"]
    st2.mark_delivered(["hall:1"])
    st3 = DedupStore(str(path))
    assert [i.key for i in st3.take_pending()] == ["hall:2"]
    assert st3.is_known("hall:1") and st3.filter_new([a]) == []
    st3.set_cursor("hall:general", 7342)
    assert DedupStore(str(path)).cursor("hall:general") == 7342
    assert not list(tmp_path.glob(".krow_state.*.tmp")), "临时文件应被 os.replace 掉"


def test_dedup_store_trims_seen(tmp_path):
    st = DedupStore(str(tmp_path / "s.json"), max_seen=100)
    st.mark_delivered([f"k{i}" for i in range(150)])
    assert st.stats()["seen"] == 100


def test_dedup_store_survives_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    st = DedupStore(str(p))
    assert st.stats()["pending"] == 0
    st.mark_delivered(["x"])
    assert json.loads(p.read_text(encoding="utf-8"))["seen"].keys() == {"x"}


# ---------------------------------------------------------------------------
# sources: REST 统一口子
# ---------------------------------------------------------------------------

def test_resolve_rest_env_rules():
    assert resolve_rest_env({}) == {}
    assert resolve_rest_env({"KW_API_ENDPOINT": "http://a/", "KW_WORKER_TOKEN": ""}) == {}
    env = resolve_rest_env({"KW_API_ENDPOINT": "http://a/", "KW_WORKER_TOKEN": "t1", "KW_INJECT_NONCE": "abcd"})
    assert env == {"api": "http://a", "token": "t1", "binding": "", "nonce": "abcd"}
    conv = resolve_rest_env({"KW_API_ENDPOINT": "http://a", "KW_WORKER_TOKEN": "t1",
                             "KW_WORKER_TOKEN_CONVERSATION": "t2", "KROW_TASK_SLOT": "conversation"})
    assert conv["token"] == "t2"
    assert resolve_rest_env({"KW_API_ENDPOINT": "http://a", "KW_WORKER_TOKEN": "t1", "KW_INJECT_HOOK": "0"}) == {}


def test_rest_source_parses_unified_payload():
    http = FakeHttp({"/agent/v1/inject": (200, json.dumps(UNIFIED_PAYLOAD))})
    src = RestInjectSource("http://api", "secret-token", binding="job-1", http_get=http)
    items = src.fetch()
    kinds = [i.kind for i in items]
    assert kinds == ["owner", "leader", "hall", "hall"]
    assert items[0].key.startswith("owner:") and items[1].key == "leader:0a1b2c3d-ffff-0000-1111-222233334444"
    hall = items[2]
    assert hall.key == "hall:7360" and hall.message_id == 7360 and hall.mentions_me and hall.channel_id == CHANNEL
    assert items[3].key == "hall:7376" and not items[3].mentions_me
    url, headers = http.calls[0]
    assert "max_owner=3" in url and "max_hall=5" in url and "max_leader=1" in url
    assert headers["Authorization"] == "Bearer secret-token" and headers["X-Krow-Job-Binding"] == "job-1"
    assert "secret-token" not in repr(src)
    # hall=0 时不传大厅、不传 max_leader
    src.fetch(hall=False, leader=False)
    url2, _ = http.calls[1]
    assert "hall=0" in url2 and "max_leader" not in url2


def test_rest_source_falls_back_to_legacy_hall_inject_on_404():
    legacy = {"items": [{"id": 7, "display": "主人", "body": "在吗", "mentions_me": True, "reply_to": {"id": 3}}],
              "remaining": 0, "last_read_id": 7}
    http = FakeHttp({"/agent/v1/inject": _http_error(404), "/agent/v1/hall/inject": (200, json.dumps(legacy))})
    src = RestInjectSource("http://api", "t", http_get=http)
    items = src.fetch()
    assert len(items) == 1 and items[0].key == "hall:7" and items[0].mentions_me
    assert items[0].text == "#7 主人：@你 在吗（回复 #3）"
    assert http.calls[1][0].endswith("/agent/v1/hall/inject?max=5")


def test_rest_source_cooldown_on_5xx_and_recovery():
    now = [1000.0]
    http = FakeHttp({"/agent/v1/inject": _http_error(503)})
    src = RestInjectSource("http://api", "t", http_get=http, clock=lambda: now[0])
    assert src.fetch() == [] and src.cooling_down() and len(http.calls) == 1
    assert src.fetch() == [] and len(http.calls) == 1, "冷却窗口内不出网"
    now[0] += 31
    http.routes["/agent/v1/inject"] = (200, json.dumps({"owner": {"text": "回来了"}}))
    items = src.fetch()
    assert [i.kind for i in items] == ["owner"] and not src.cooling_down()


def test_rest_source_4xx_does_not_cooldown():
    http = FakeHttp({"/agent/v1/inject": _http_error(401)})
    src = RestInjectSource("http://api", "t", http_get=http)
    assert src.fetch() == [] and not src.cooling_down()
    assert "401" in src.last_error and "t" != src.last_error


def test_parse_inject_hall_text_multiline_and_fallback():
    items = parse_inject_hall_text("#1 甲：第一行\n  第二行续\n#2 乙：第二条")
    assert [i.message_id for i in items] == [1, 2]
    assert items[0].text == "#1 甲：第一行\n  第二行续"
    only = parse_inject_hall_text("没有编号的一段")
    assert len(only) == 1 and only[0].key.startswith("hall:") and only[0].message_id is None


# ---------------------------------------------------------------------------
# sources: MCP 轮询
# ---------------------------------------------------------------------------

def test_parse_hall_read_lines():
    items = parse_hall_read(HALL_READ_TEXT, channel_id=CHANNEL)
    assert [i.message_id for i in items] == [7329, 7341, 7342]
    assert items[1].meta["display"] == "姜尚#凌烟阁" and items[1].created_at == "10-24 07:01"
    assert "附件：排查报告.md" in items[1].text, "缩进的附件行并入上一条"
    assert all(i.channel_id == CHANNEL for i in items)


def test_mcp_source_mentions_mode_filters_by_self_name_and_acks():
    mcp = FakeMcp()
    cursors = {}
    src = McpHallSource(mcp, self_names=["lv-agent#默认小组", "lv-agent"], mode="mentions", inbox=True,
                        cursor_get=cursors.get, cursor_set=cursors.__setitem__)
    items = src.fetch()
    halls = [i for i in items if i.kind == "hall"]
    mails = [i for i in items if i.kind == "mail"]
    # general 的 mention_count=0 不读; 项目频道读到 3 条, 7341 与 7342 都含 "@lv-agent"(7342 是 "lv-agent 2" 的前缀——
    # 这正是平台 PR #1240 要修的错投形态, 客户端按子串匹配同样会命中, 由模型判断是否点的是自己)
    assert [h.message_id for h in halls] == [7341, 7342]
    assert all(h.mentions_me for h in halls)
    assert f"channel_id={CHANNEL}" in halls[0].text
    tools = [c[0] for c in mcp.calls]
    assert tools == ["hall_channels", "hall_read", "hall_ack", "team_inbox"]
    read_args = mcp.calls[1][1]
    assert read_args == {"unread_only": True, "limit": 30, "channel_id": CHANNEL}
    assert mcp.calls[2][1] == {"last_read_id": 7342, "channel_id": CHANNEL}
    assert cursors[f"hall:{CHANNEL}"] == 7342
    assert len(mails) == 1 and mails[0].key == "mail:f3b613cd-45f3-442b-859a-8f2edf14ae13"
    assert "姜尚" in mails[0].text and "[sys:internal]" in mails[0].text and "\n" not in mails[0].text
    # 第二次轮询带 since_id 游标
    src.fetch()
    assert mcp.calls[5][1]["since_id"] == 7342


def test_mcp_source_without_self_names_passes_whole_mentioned_channel():
    src = McpHallSource(FakeMcp(), self_names=[], mode="mentions", inbox=False)
    items = src.fetch()
    assert [i.message_id for i in items] == [7329, 7341, 7342]


def test_mcp_source_all_mode_reads_unread_channels():
    mcp = FakeMcp()
    src = McpHallSource(mcp, self_names=["lv-agent"], mode="all", inbox=False, ack=False)
    items = src.fetch()
    # general(unread 2) 与项目频道(unread 3)都读; general 不带 channel_id
    reads = [c for c in mcp.calls if c[0] == "hall_read"]
    assert len(reads) == 2 and "channel_id" not in reads[0][1] and reads[1][1]["channel_id"] == CHANNEL
    assert not any(c[0] == "hall_ack" for c in mcp.calls)
    assert len(items) == 6


def test_mcp_source_tool_failure_is_soft():
    def boom(tool, args):
        raise RuntimeError("krow MCP server not found/ready")

    src = McpHallSource(boom, self_names=["x"])
    assert src.fetch() == []
    assert "not found" in src.last_error


# ---------------------------------------------------------------------------
# bridge: 两源合流去重 / 送达记账 / 空闲轮询
# ---------------------------------------------------------------------------

def _bridge(tmp_path, *sources, **cfg):
    conf = KrowConfig({"enabled": True, "state_path": str(tmp_path / "state.json"), **cfg})
    return KrowBridge(conf, store=DedupStore(conf.state_path), sources=list(sources), nonce="n0nce")


def test_bridge_dedups_across_sources_and_marks_delivered(tmp_path):
    # REST 与 MCP 都看到 #7341: REST 的 hall.text 写法 vs hall_read 渲染写法, 键同为 hall:7341
    rest_payload = {"hall": {"text": "#7341 姜尚#凌烟阁：@你 请确认运行版本"}}
    rest = RestInjectSource("http://api", "t", http_get=FakeHttp({"/agent/v1/inject": (200, json.dumps(rest_payload))}))
    mcp = McpHallSource(FakeMcp(), self_names=["lv-agent"], inbox=False)
    br = _bridge(tmp_path, rest, mcp)
    items = br.poll()
    keys = [i.key for i in items]
    assert keys == ["hall:7341", "hall:7342"], keys
    assert items[0].source == "rest", "先到先得: REST 版本保留, MCP 的同 id 被去掉"
    # 没送达之前再 poll 还会拿到(取到 ≠ 送到)
    again = br.poll()
    assert [i.key for i in again] == keys and br.last_new == 0
    br.mark_delivered(items)
    assert br.poll() == [] and br.delivered == 2
    assert br.status()["store"]["seen"] == 2


def test_bridge_interjection_text_renders_and_consumes(tmp_path):
    rest = RestInjectSource("http://api", "t", http_get=FakeHttp({"/agent/v1/inject": (200, json.dumps(UNIFIED_PAYLOAD))}))
    br = _bridge(tmp_path, rest)
    text = br.interjection_text(None)
    assert text and text.startswith(OWNER_PREFIX_HEAD) and "口令 n0nce" in text and "#7360" in text
    assert br.interjection_text(None) is None, "同一批不重复送"
    assert br.status()["store"]["pending"] == 0


def test_bridge_owner_only_text_uses_rest_owner_only(tmp_path):
    rest = RestInjectSource("http://api", "t", http_get=FakeHttp({"/agent/v1/inject": (200, json.dumps(UNIFIED_PAYLOAD))}))
    mcp = McpHallSource(FakeMcp(), self_names=["lv-agent"])
    br = _bridge(tmp_path, rest, mcp)
    text = br.owner_only_text(None)
    assert text and text.startswith(OWNER_PREFIX_HEAD) and "#7360" not in text and LEADER_PREFIX_HEAD not in text
    # REST 的 hall=0 调用: 大厅 / 任务单没被消费; MCP 源完全没被碰
    assert all(c[0] not in ("hall_channels", "hall_read") for c in mcp._call.calls)
    mcp_only = _bridge(tmp_path / "b", McpHallSource(FakeMcp(), self_names=["lv-agent"]))
    assert mcp_only.owner_only_text(None) is None


def test_bridge_disabled_when_no_sources(tmp_path):
    br = _bridge(tmp_path)
    assert not br.enabled and br.interjection_text(None) is None and br.start_idle_poller(lambda i: None) is None


def test_bridge_idle_poller_respects_busy_and_idle(tmp_path):
    mcp = FakeMcp()
    br = _bridge(tmp_path, McpHallSource(mcp, self_names=["lv-agent"], inbox=False), poll_interval_s=5)
    got = []
    idle = {"v": False}
    br.set_busy(True)
    th = br.start_idle_poller(lambda items: got.append([i.key for i in items]), is_idle=lambda: idle["v"], interval_s=0.05)
    assert th is not None and br.start_idle_poller(lambda i: None) is th, "重复起线程是 no-op"
    time.sleep(0.2)
    assert not mcp.calls, "忙时不轮询"
    br.set_busy(False)
    time.sleep(0.2)
    assert not mcp.calls, "不空闲也不轮询"
    idle["v"] = True
    deadline = time.time() + 2
    while not got and time.time() < deadline:
        time.sleep(0.02)
    br.stop()
    assert got and got[0] == ["hall:7341", "hall:7342"]


def test_build_bridge_from_config_env_and_mcp(tmp_path, monkeypatch):
    class FakeConn:
        def __init__(self, tools):
            self.tools = tools
            self._ready = threading.Event()
            self._ready.set()

    class FakeManager:
        def __init__(self):
            self.servers = {"filesystem": FakeConn([{"name": "read_file"}]), "my-krow": FakeConn([{"name": "hall_read"}])}
            self.calls = []

        def call_tool(self, server, tool, args):
            self.calls.append((server, tool, dict(args)))
            return FakeMcp()(tool, args)

    cfg = AgentConfig(krow={"enabled": True, "state_path": str(tmp_path / "s.json"), "self_names": ["lv-agent"]},
                      mcp={"enabled": True, "servers": {"my-krow": {}}})
    env = {"KW_API_ENDPOINT": "http://api", "KW_WORKER_TOKEN": "tok", "KW_INJECT_NONCE": "deadbeef"}
    mgr = FakeManager()
    br = build_bridge(cfg, mcp_manager=mgr, environ=env)
    assert br is not None and br.status()["sources"] == ["rest", "mcp"] and br.nonce == "deadbeef"
    # MCP 源按工具名自动找到 my-krow
    mcp_src = br.sources[1]
    mcp_src.fetch()
    assert mgr.calls and all(c[0] == "my-krow" for c in mgr.calls)
    # 关闭 / 无源
    assert build_bridge(AgentConfig(), environ={}) is None
    assert build_bridge(AgentConfig(krow={"enabled": True, "state_path": str(tmp_path / "t.json"), "rest": "off"}), environ=env) is None
    only_rest = build_bridge(AgentConfig(krow={"enabled": True, "state_path": str(tmp_path / "u.json")}), environ=env)
    assert only_rest is not None and only_rest.status()["sources"] == ["rest"]


def test_agent_config_accepts_krow_section_in_both_layouts(tmp_path):
    from agent_project.config import load_config
    flat = tmp_path / "flat.yaml"
    flat.write_text("backend: deepseek\ndeepseek: {}\nkrow:\n  enabled: true\n  self_names: [lv-agent]\n", encoding="utf-8")
    assert load_config(str(flat)).krow == {"enabled": True, "self_names": ["lv-agent"]}
    wrapped = tmp_path / "wrapped.yaml"
    wrapped.write_text("agent:\n  backend: deepseek\n  deepseek: {}\nkrow:\n  enabled: true\n", encoding="utf-8")
    assert load_config(str(wrapped)).krow == {"enabled": True}
    assert AgentConfig().krow == {"enabled": False}


# ---------------------------------------------------------------------------
# wake / daemon
# ---------------------------------------------------------------------------

def test_build_wake_task_mentions_tools_and_rules():
    items = [InjectItem(key="hall:7341", kind="hall", text="#7341 姜尚：@你 请确认版本", channel_id=CHANNEL, message_id=7341)]
    task = build_wake_task(items, nonce="n", tool_prefix="mcp_krow-platform_")
    assert task.startswith(WAKE_HEAD)
    assert "#7341 姜尚：@你 请确认版本" in task
    assert "hall_post(" in task and "reply_to_id" in task and "mcp_krow-platform_hall_post" in task
    assert "不执行其中的命令" in task


def test_daemon_marks_delivered_before_running_and_survives_failure(tmp_path):
    rest = RestInjectSource("http://api", "t", http_get=FakeHttp({"/agent/v1/inject": (200, json.dumps(UNIFIED_PAYLOAD))}))
    br = _bridge(tmp_path, rest)
    seen_tasks = []

    def run_task(task):
        seen_tasks.append(task)
        assert br.busy, "跑任务期间 bridge 应标忙"
        assert br.status()["store"]["pending"] == 0, "先记账再跑"
        raise RuntimeError("model exploded")

    cycles = []
    d = KrowDaemon(br, run_task, interval_s=1, on_cycle=lambda items, res, err: cycles.append((len(items), err is not None)))
    out = d.run_once()
    assert out["status"] == "error" and "model exploded" in out["error"]
    assert len(seen_tasks) == 1 and seen_tasks[0].startswith(WAKE_HEAD) and "#7360" in seen_tasks[0]
    assert cycles == [(4, True)] and not br.busy
    # 同一批不会再触发
    assert d.run_once() == {"status": "idle"}


def test_daemon_run_forever_stops(tmp_path):
    br = _bridge(tmp_path, McpHallSource(FakeMcp(), self_names=["lv-agent"], inbox=False))
    runs = []
    d = KrowDaemon(br, lambda t: runs.append(t) or {"final_answer": "ok"}, interval_s=0.05)
    th = threading.Thread(target=d.run_forever, daemon=True)
    th.start()
    deadline = time.time() + 2
    while not runs and time.time() < deadline:
        time.sleep(0.02)
    d.stop()
    th.join(timeout=2)
    assert len(runs) == 1 and d.wakes == 1 and not th.is_alive()


# ---------------------------------------------------------------------------
# ExecutionEngine: 工具调用边界注入 + 轮尾续跑
# ---------------------------------------------------------------------------

def _install_fake_tool():
    class _Fake:
        name = "web_search"
        description = "fake"
        parameters = {"type": "object", "properties": {}}

        def execute(self, query="", **kw):
            return ToolResult(success=True, output=f'[{{"title": "ok:{query}"}}]')

    old = TOOLS_REGISTRY.get("web_search")
    TOOLS_REGISTRY._tools["web_search"] = _Fake()
    return lambda: TOOLS_REGISTRY._tools.__setitem__("web_search", old)


class _Provider:
    """最小注入提供者: 第一次工具边界给一段大厅文本; 轮尾给一段主人插话(只给一次)."""

    def __init__(self, hall_text="【大厅】有人在大厅说话：\n#7341 姜尚：@你 请确认版本", owner_text=None):
        self.hall_text = hall_text
        self.owner_text = owner_text
        self.interjection_calls = 0
        self.owner_calls = 0

    def contract_text(self):
        return "## krow 运行中注入契约（LV-Agent 原生）\n- 口令 n0nce"

    def interjection_text(self, ctx):
        self.interjection_calls += 1
        if self.interjection_calls == 1:
            return self.hall_text
        return None

    def owner_only_text(self, ctx):
        self.owner_calls += 1
        if self.owner_calls == 1:
            return self.owner_text
        return None


def test_engine_injects_at_tool_boundary_and_contract_in_context():
    cfg = AgentConfig()
    restore = _install_fake_tool()
    try:
        prompts = []

        class FB:
            def __init__(self):
                self.c = 0

            def generate(self, prompt, **kw):
                prompts.append(prompt)
                self.c += 1
                if self.c == 1:
                    return '{"action": "web_search", "args": {"query": "a"}}'
                if self.c == 2:
                    return '{"action": "web_search", "args": {"query": "b"}}'
                return '{"final_answer": "done"}'

        prov = _Provider()
        eng = ExecutionEngine(model_backend=FB(), config=cfg, interjection_provider=prov)
        eng.logger = logging.getLogger("test")
        ctx = ExecutionContext(task="搜一下", available_tools=TOOLS_REGISTRY.get_tools_dict(), config=cfg, max_steps=8)
        ctx.monitor_enabled = False
        tr = eng.run(ReActPolicy(), ctx)
        assert tr.final_answer == "done"
        assert "krow 运行中注入契约" in ctx.extra_context and "krow 运行中注入契约" in prompts[0]
        # 第一次工具调用结束后拉到大厅文本 → 出现在第二步 prompt 里, 且只出现这一次
        assert "#7341 姜尚：@你 请确认版本" in prompts[1]
        assert "#7341" not in prompts[0] and "#7341" not in prompts[2]
        assert prov.interjection_calls == 2, "每个工具边界拉一次(两次工具调用)"
        assert ctx.interjections == []
    finally:
        restore()


def test_engine_stop_interjection_continues_once():
    cfg = AgentConfig()
    restore = _install_fake_tool()
    try:
        prompts = []

        class FB:
            def __init__(self):
                self.c = 0

            def generate(self, prompt, **kw):
                prompts.append(prompt)
                self.c += 1
                if self.c == 1:
                    return '{"action": "web_search", "args": {"query": "a"}}'
                if self.c == 2:
                    return '{"final_answer": "第一版答案"}'
                return '{"final_answer": "按主人意见修正后的答案"}'

        prov = _Provider(hall_text=None, owner_text="📢 主人插话（经 LV-Agent krow 注入送达，口令 n0nce）——先别交, 补上版本号")
        eng = ExecutionEngine(model_backend=FB(), config=cfg, interjection_provider=prov)
        eng.logger = logging.getLogger("test")
        # 任务文本不能命中引擎的研究类任务正则(含「报告」等词会触发短答案强制重生成, 与本测试无关)
        ctx = ExecutionContext(task="查一下版本号", available_tools=TOOLS_REGISTRY.get_tools_dict(), config=cfg, max_steps=8)
        ctx.monitor_enabled = False
        tr = eng.run(ReActPolicy(), ctx)
        assert tr.final_answer == "按主人意见修正后的答案"
        assert "补上版本号" in prompts[2] and "第一版答案" in prompts[2]
        assert ctx.interjection_stop_used and prov.owner_calls == 1, "轮尾只拉一次, 第二次收口不再拉"
        assert len(tr.steps) == 3
    finally:
        restore()


def test_engine_provider_failure_is_swallowed():
    cfg = AgentConfig()
    restore = _install_fake_tool()
    try:
        class Boom:
            def contract_text(self):
                raise RuntimeError("no contract")

            def interjection_text(self, ctx):
                raise RuntimeError("network down")

            def owner_only_text(self, ctx):
                raise RuntimeError("network down")

        class FB:
            def __init__(self):
                self.c = 0

            def generate(self, prompt, **kw):
                self.c += 1
                if self.c == 1:
                    return '{"action": "web_search", "args": {"query": "a"}}'
                return '{"final_answer": "ok"}'

        eng = ExecutionEngine(model_backend=FB(), config=cfg, interjection_provider=Boom())
        eng.logger = logging.getLogger("test")
        ctx = ExecutionContext(task="x", available_tools=TOOLS_REGISTRY.get_tools_dict(), config=cfg, max_steps=6)
        ctx.monitor_enabled = False
        tr = eng.run(ReActPolicy(), ctx)
        assert tr.final_answer == "ok" and tr.success
    finally:
        restore()


def test_engine_without_provider_unchanged():
    cfg = AgentConfig()
    restore = _install_fake_tool()
    try:
        class FB:
            def __init__(self):
                self.c = 0

            def generate(self, prompt, **kw):
                self.c += 1
                if self.c == 1:
                    return '{"action": "web_search", "args": {"query": "a"}}'
                return '{"final_answer": "ok"}'

        eng = ExecutionEngine(model_backend=FB(), config=cfg)
        assert eng.interjection_provider is None
        ctx = ExecutionContext(task="x", available_tools=TOOLS_REGISTRY.get_tools_dict(), config=cfg, max_steps=6)
        ctx.monitor_enabled = False
        tr = eng.run(ReActPolicy(), ctx)
        assert tr.final_answer == "ok" and ctx.extra_context == "" and len(tr.steps) == 2
    finally:
        restore()
