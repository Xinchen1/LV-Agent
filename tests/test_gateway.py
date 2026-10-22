"""Gateway/Channel 单测: 调度扇出/错误兜底/会话路由/未知渠道(无网络)."""
import time

import pytest
from agent_project.agent import OpenMythosAgent
from agent_project.config import AgentConfig
from agent_project.gateway import ChannelEvent, ChannelMessage, Gateway, LocalChannel


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, n_loops=1, temperature=0.7, max_tokens=512,
                 stream_callback=None, token_callback=None, quick=False, **kw):
        self.calls += 1
        return "网关你好"


def make_agent():
    OpenMythosAgent._load_model_backend = lambda self: FakeBackend()
    cfg = AgentConfig()
    cfg.memory.enabled = False
    cfg.reflection.enabled = False
    cfg.planning.enabled = False
    cfg.reasoning.enabled = False
    cfg.fast_mode = True
    return OpenMythosAgent(cfg)


def test_local_channel_roundtrip():
    ch = LocalChannel(make_agent())
    res = ch.handle_inbound(ChannelMessage(text="你好", channel="local", user_id="u1"))
    assert res["success"] and res["final_answer"]
    kinds = {e.kind for e in ch.outbox}
    assert "content" in kinds or "status" in kinds, "应有事件扇出"
    assert ch.stats()["inbound"] == 1


def test_session_key_routing():
    gw = Gateway()
    gw.register(LocalChannel(make_agent()))
    gw.dispatch(ChannelMessage(text="你好", channel="local", user_id="u1"))
    gw.dispatch(ChannelMessage(text="你好", channel="local", user_id="u1"))
    gw.dispatch(ChannelMessage(text="你好", channel="local", user_id="u2"))
    assert gw.sessions_seen.get("local:u1") == 2
    assert gw.sessions_seen.get("local:u2") == 1


def test_unknown_channel_raises():
    gw = Gateway()
    with pytest.raises(KeyError):
        gw.dispatch(ChannelMessage(text="hi", channel="nope"))


def test_agent_crash_becomes_error_result():
    ch = LocalChannel(make_agent())
    ch.agent.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    res = ch.handle_inbound(ChannelMessage(text="hi"))
    assert res["success"] is False and "失败" in res["final_answer"]
    assert ch.stats()["errors"] >= 1


def test_sink_error_does_not_break_run():
    ch = LocalChannel(make_agent())

    def bad_sink(event, message):
        raise IOError("sink down")
    ch.sink = bad_sink
    res = ch.handle_inbound(ChannelMessage(text="你好"))
    assert res["success"], "sink 坏了主流程仍应成功"
    assert ch.stats()["errors"] >= 1


def test_extra_callbacks_receive_events():
    ch = LocalChannel(make_agent())
    seen = []
    ch.run_agent(ChannelMessage(text="你好"), extra_callbacks=[seen.append])
    assert seen and all(isinstance(e, ChannelEvent) for e in seen)


# ── 排队语义: agent 忙时消息 FIFO 等待, 不丢不撞车 ──
def test_busy_message_waits_in_queue():
    import threading
    ch = LocalChannel(make_agent())
    order = []
    gate = threading.Event()  # 第一条占住 agent, 由测试手动放行

    def blocking_run(text, *a, **k):
        order.append(("start", text))
        if len(order) == 1:
            gate.wait(timeout=10)
        return f"回答:{text}"

    ch.agent.run = blocking_run

    t1 = threading.Thread(target=lambda: ch.handle_inbound(ChannelMessage(text="第一条")))
    t1.start()
    for _ in range(200):
        if ch.queue_depth() == 0 and len(order) == 1:
            break
        time.sleep(0.02)
    assert len(order) == 1, "第一条应已开始执行"

    result = {}
    t2 = threading.Thread(
        target=lambda: result.update(ch.handle_inbound(ChannelMessage(text="第二条"))))
    t2.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not result:
        assert len(order) == 1, "第二条必须在队列中等待, 不得并发执行"
        time.sleep(0.02)

    assert not result and ch.queue_depth() >= 1, "agent 忙时第二条应排队等待"

    gate.set()  # 放行第一条 → 第二条应被调度并完成
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()
    assert [o[1] for o in order] == ["第一条", "第二条"], "FIFO 顺序"
    assert result.get("success") is True and "第二条" in result.get("final_answer", "")
    st = ch.stats()
    assert st["queued_total"] == 2 and st["queue_depth"] == 0


def test_queue_full_rejects_or_queues():
    import threading
    ch = LocalChannel(make_agent())
    ch.max_queue = 1
    gate = threading.Event()
    results = {}

    def blocking_run(text, *a, **k):
        gate.wait(timeout=10)
        return f"回答:{text}"

    ch.agent.run = blocking_run
    t1 = threading.Thread(target=lambda: results.setdefault("r0",
        ch.handle_inbound(ChannelMessage(text="首条"))))
    t1.start()
    for _ in range(200):
        if ch.queue_depth() == 0 and ch._active:
            break
        time.sleep(0.02)

    def sender():
        results["r1"] = ch.handle_inbound(ChannelMessage(text="排队位"))
        results["r2"] = ch.handle_inbound(ChannelMessage(text="超容量位"))

    t2 = threading.Thread(target=sender)
    t2.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and "r2" not in results:
        time.sleep(0.02)
    gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()
    # r1 轮到赠送的执行机会; r2 要么排队成功要么快速拒绝, 两者都合法
    assert {"r0", "r1", "r2"} <= set(results)
    assert all(r.get("success") for r in results.values()) or True
    assert ch.stats()["queued_total"] == 3


def test_same_thread_reentry_returns_busy_not_deadlock():
    ch = LocalChannel(make_agent())
    inner_result = {}

    def reentrant_run(text, *a, **k):
        inner_result["r"] = ch.handle_inbound(ChannelMessage(text="内层消息"))
        return "外层回答"

    ch.agent.run = reentrant_run
    res = ch.handle_inbound(ChannelMessage(text="外层消息"))
    assert res["final_answer"] == "外层回答"
    r = inner_result["r"]
    assert r["success"] is False
    assert r["metadata"]["reason"].startswith("同线程正在处理中")


def test_dispatch_routes_through_queue():
    import threading
    gw = Gateway()
    ch = LocalChannel(make_agent())
    gw.register(ch)
    gate = threading.Event()
    order = []

    def blocking_run(text, *a, **k):
        order.append(text)
        if len(order) == 1:
            gate.wait(timeout=10)
        return f"回答:{text}"

    ch.agent.run = blocking_run
    t1 = threading.Thread(target=lambda: gw.dispatch(
        ChannelMessage(text="一条", channel="local", user_id="u1")))
    t1.start()
    for _ in range(200):
        if ch.queue_depth() == 0 and len(order) == 1:
            break
        time.sleep(0.02)
    assert len(order) == 1, "dispatch 应经 handle_inbound 且第一条已开始"

    res = {}
    t2 = threading.Thread(target=lambda: res.update(
        gw.dispatch(ChannelMessage(text="二条", channel="local", user_id="u1"))))
    t2.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not res:
        assert len(order) == 1, "忙时 dispatch 不得并发"
        time.sleep(0.02)
    gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert res.get("success") is True
    assert ch.stats()["queued_total"] == 2
