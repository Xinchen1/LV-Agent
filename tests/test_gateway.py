"""Gateway/Channel 单测: 调度扇出/错误兜底/会话路由/未知渠道(无网络)."""
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
