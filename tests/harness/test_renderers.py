"""StreamCallbackAdapter mapping and ConsoleRenderer smoke."""

from agent_project.harness import events as ev
from agent_project.harness.errors import ErrorKind, ErrorRecord
from agent_project.harness.renderers import ConsoleRenderer, StreamCallbackAdapter
from agent_project.harness.stream import Delta, EventBus


def _adapter():
    calls = []
    tokens = []
    adapter = StreamCallbackAdapter(
        lambda kind, text: calls.append((kind, text)),
        token_callback=tokens.append,
    )
    return adapter, calls, tokens


def test_delta_mapping():
    adapter, calls, _tokens = _adapter()
    adapter(Delta(kind="token", text="hello"))
    adapter(Delta(kind="reasoning", text="thinking"))
    adapter(Delta(kind="status", text="working"))
    adapter(Delta(kind="unknown", text="dropped"))
    assert calls == [
        ("content", "hello"),
        ("reasoning", "thinking"),
        ("status", "working"),
    ]


def test_event_mapping():
    adapter, calls, tokens = _adapter()
    adapter(ev.EffectRequested(effect_id="k", tool_name="grep", arguments={"pattern": "x"}))
    adapter(ev.EffectCompleted(effect_id="k", output="found it"))
    adapter(ev.EffectFailed(effect_id="k2", error=ErrorRecord(
        kind=ErrorKind.TOOL_FAILED, retriable=False, message="broke")))
    adapter(ev.EffectDenied(effect_id="k3", tool_name="bash_exec", reason="no shell"))
    adapter(ev.ModelResponded(call_id="c", text="t", prompt_tokens=7, completion_tokens=3))

    kinds = [c[0] for c in calls]
    assert kinds == ["tool_call", "tool_result", "tool_result", "tool_result"]
    assert "grep" in calls[0][1]
    assert calls[1][1] == "found it"
    assert "broke" in calls[2][1]
    assert "denied" in calls[3][1]
    assert tokens == [10]


def test_context_and_circuit_events_become_status():
    adapter, calls, _ = _adapter()
    adapter(ev.ContextCompacted(dropped_messages=5, kept_messages=8, summary="s"))
    adapter(ev.CircuitTripped(breaker="stagnation", reason="loop"))
    assert calls[0][0] == "status" and "5" in calls[0][1]
    assert calls[1][0] == "status" and "stagnation" in calls[1][1]


def test_attach_detach_via_bus():
    bus = EventBus()
    adapter, calls, _ = _adapter()
    unsub = adapter.attach(bus)
    bus.emit_delta("token", "live")
    unsub()
    bus.emit_delta("token", "gone")
    assert [c[1] for c in calls] == ["live"]


def test_console_renderer_never_raises(capsys):
    renderer = ConsoleRenderer()
    bus = EventBus()
    renderer.attach(bus)
    bus.emit_delta("token", "answer text")
    bus.emit_event(ev.EffectRequested(effect_id="k", tool_name="grep", arguments={}))
    bus.emit_event(ev.EffectCompleted(effect_id="k", output="out"))
    out = capsys.readouterr().out
    assert "answer text" in out and "grep" in out
