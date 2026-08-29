#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research





from agent_project.harness import events as ev


def test_fold_counts_model_and_tool_activity():
    events = [
        ev.SessionStarted(task="do something"),
        ev.TurnStarted(turn_index=0),
        ev.ModelResponded(call_id="c1", text="hi", prompt_tokens=10, completion_tokens=5),
        ev.EffectRequested(effect_id="k1", tool_name="grep", arguments={"pattern": "x"}),
        ev.EffectCompleted(effect_id="k1", output="found"),
        ev.EffectDenied(effect_id="k2", tool_name="bash_exec", reason="denied"),
        ev.FinalAnswer(text="done"),
        ev.SessionFinished(status="completed"),
    ]
    state = ev.fold(events)
    assert state.task == "do something"
    assert state.model_calls == 1
    assert state.tool_calls == 1
    assert state.total_tokens == 15
    assert state.denials == 1
    assert state.final_text == "done"
    assert state.finished and state.finish_status == "completed"


def test_fold_accumulates_errors_and_breakers():
    from agent_project.harness.errors import ErrorKind, ErrorRecord

    r1 = ErrorRecord(kind=ErrorKind.MODEL_TRANSIENT, retriable=True, message="boom")
    state = ev.fold([
        ev.ModelFailed(call_id="c", error=r1),
        ev.CircuitTripped(breaker="stagnation", reason="loop"),
    ])
    assert state.errors == [r1]
    assert state.breakers == ["stagnation"]


def test_registry_covers_all_event_types():
    from agent_project.harness.events import EVENT_REGISTRY, Event

    expected = {
        "SessionStarted", "SessionFinished", "ModelRequested", "ModelResponded",
        "ModelFailed", "EffectRequested", "EffectCompleted", "EffectFailed",
        "EffectDenied", "ContextAssembled", "ContextCompacted", "BudgetConsumed",
        "CircuitTripped", "TurnStarted", "TurnFinished", "LoopPaused",
        "CheckpointWritten", "FinalAnswer",
    }
    assert set(EVENT_REGISTRY) == expected
    for kind, cls in EVENT_REGISTRY.items():
        assert issubclass(cls, Event)
        assert cls.model_config.get("frozen") is True
        assert cls.__name__ == kind


def test_events_roundtrip_through_json():
    from agent_project.harness.errors import ErrorKind, ErrorRecord

    samples = [
        ev.SessionStarted(task="t"),
        ev.ModelResponded(call_id="c", text="x", prompt_tokens=1, completion_tokens=2),
        ev.EffectRequested(effect_id="k", tool_name="grep", arguments={"p": 1}),
        ev.EffectFailed(effect_id="k", error=ErrorRecord(
            kind=ErrorKind.TOOL_FAILED, retriable=False, message="m")),
        ev.TurnFinished(turn_index=3, stop_reason="continued"),
    ]
    for sample in samples:
        restored = type(sample).model_validate(sample.model_dump(mode="json"))
        assert restored == sample
        assert restored.kind == type(sample).__name__


def test_with_seq_produces_new_instance():
    e1 = ev.SessionStarted(task="t")
    e2 = e1.with_seq(7)
    assert e1.seq == -1 and e2.seq == 7 and e1.event_id == e2.event_id
