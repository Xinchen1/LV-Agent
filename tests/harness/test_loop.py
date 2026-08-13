"""End-to-end loop behaviour with a scripted fake sampler.

These tests are the architecture's proof: a full agent run -- model calls,
tool dispatch, policy denials, budgets, pause/resume -- driven entirely
through the journal, with no real LLM and no real tools.
"""

import asyncio

import pytest

from agent_project.harness import events as ev
from agent_project.harness.budget import Ledger, Limits
from agent_project.harness.errors import (
    BudgetExhaustedError,
    ModelTransientError,
    StagnationError,
)
from agent_project.harness.journal import Journal
from agent_project.harness.kernel import Decision, Kernel, Rule
from agent_project.harness.loop import AgentLoop, SampleResult, rebuild_messages
from agent_project.harness.scheduler import Scheduler
from agent_project.harness.stream import EventBus


class ScriptedSampler:
    """Plays back queued responses; each call pops one."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(self, messages, tools):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def make_loop(tmp_path, sampler, *, policy=None, limits=None, pause_check=None,
              executor=None):
    journal = Journal(tmp_path / "run.jsonl")
    kernel = Kernel(
        policy or [Rule(Decision.ALLOW, reason="test allow")],
        executor=executor or (lambda effect: f"out:{effect.tool_name}"),
    )
    loop = AgentLoop(
        kernel=kernel,
        scheduler=Scheduler(),
        ledger=Ledger(limits or Limits()),
        journal=journal,
        bus=EventBus(),
        sampler=sampler,
        pause_check=pause_check,
    )
    return loop, journal


def run(coro):
    return asyncio.run(coro)


# ---------- happy paths ----------

def test_direct_final_answer_no_tools(tmp_path):
    sampler = ScriptedSampler([SampleResult(text="the answer is 42")])
    loop, journal = make_loop(tmp_path, sampler)
    assert run(loop.run("what is the answer?")) == "the answer is 42"

    kinds = [e.kind for e in journal.read_all()]
    assert kinds[:2] == ["SessionStarted", "TurnStarted"]
    assert "FinalAnswer" in kinds and kinds[-1] == "SessionFinished"
    state = ev.fold(journal.read_all())
    assert state.finished and state.finish_status == "completed"
    assert state.final_text == "the answer is 42"


def test_tool_call_then_answer(tmp_path):
    sampler = ScriptedSampler([
        SampleResult(text="let me check", tool_calls=(("grep", {"pattern": "x"}),)),
        SampleResult(text="found it, done"),
    ])
    loop, journal = make_loop(tmp_path, sampler)
    assert run(loop.run("find x")) == "found it, done"

    kinds = [e.kind for e in journal.read_all()]
    assert "EffectRequested" in kinds and "EffectCompleted" in kinds
    # second sampler call saw the tool observation folded into messages
    second_call_messages = sampler.calls[1]
    assert any("obs:grep" in str(m.get("content", "")) for m in second_call_messages)


def test_denied_effect_feeds_back_to_model(tmp_path):
    sampler = ScriptedSampler([
        SampleResult(text="run it", tool_calls=(("bash_exec", {"command": "rm -rf /"}),)),
        SampleResult(text="policy stopped me; answering directly"),
    ])
    policy = [Rule(Decision.DENY, tool="bash_exec", reason="no shell ever")]
    loop, journal = make_loop(tmp_path, sampler, policy=policy)
    answer = run(loop.run("try shell"))
    assert "policy stopped me" in answer
    denied = [e for e in journal.read_all() if isinstance(e, ev.EffectDenied)]
    assert len(denied) == 1 and denied[0].reason == "no shell ever"
    assert any("DENIED" in str(m.get("content", "")) for m in sampler.calls[1])


# ---------- retries and budgets ----------

def test_transient_model_error_retries_then_succeeds(tmp_path):
    sampler = ScriptedSampler([
        ModelTransientError("rate limited"),
        SampleResult(text="recovered"),
    ])
    loop, journal = make_loop(tmp_path, sampler)
    loop.model_retry_base_s = 0.01
    assert run(loop.run("hi")) == "recovered"
    failed = [e for e in journal.read_all() if isinstance(e, ev.ModelFailed)]
    assert len(failed) == 1


def test_fatal_model_error_does_not_retry(tmp_path):
    from agent_project.harness.errors import ModelFatalError

    sampler = ScriptedSampler([ModelFatalError("bad key")])
    loop, _journal = make_loop(tmp_path, sampler)
    with pytest.raises(ModelFatalError):
        run(loop.run("hi"))
    assert len(sampler.calls) == 1


def test_turn_budget_stops_loop(tmp_path):
    sampler = ScriptedSampler([
        SampleResult(text="", tool_calls=(("grep", {"pattern": "x"}),)),
        SampleResult(text="", tool_calls=(("grep", {"pattern": "y"}),)),
        SampleResult(text="should never reach"),
    ])
    loop, journal = make_loop(tmp_path, sampler, limits=Limits(max_turns=2))
    with pytest.raises(BudgetExhaustedError):
        run(loop.run("loop forever"))
    state = ev.fold(journal.read_all())
    assert state.finish_status == "budget_exhausted"


def test_stagnation_breaker_trips_on_identical_effects(tmp_path):
    sampler = ScriptedSampler([
        SampleResult(text="", tool_calls=(("grep", {"pattern": "same"}),))
        for _ in range(4)
    ] + [SampleResult(text="late")])
    loop, journal = make_loop(
        tmp_path, sampler,
        limits=Limits(max_identical_effects=3),
        executor=lambda effect: "same output",
    )
    with pytest.raises(StagnationError):
        run(loop.run("stuck"))
    assert any(isinstance(e, ev.CircuitTripped) for e in journal.read_all())


# ---------- pause / resume ----------

def test_pause_writes_continuation_and_resume_finishes(tmp_path):
    pauses = {"n": 0}

    def pause_check():
        pauses["n"] += 1
        return pauses["n"] == 2  # pause at the second turn boundary

    sampler = ScriptedSampler([
        SampleResult(text="", tool_calls=(("grep", {"pattern": "x"}),)),
        SampleResult(text="resumed answer"),
    ])
    loop, journal = make_loop(tmp_path, sampler, pause_check=pause_check)
    paused = run(loop.run("task"))
    assert paused.startswith("[paused:")

    fresh_loop, _ = make_loop(tmp_path, sampler)  # same journal path
    assert run(fresh_loop.resume()) == "resumed answer"
    state = ev.fold(journal.read_all())
    assert state.finished and state.finish_status == "completed"


def test_resume_without_pause_raises(tmp_path):
    loop, _journal = make_loop(tmp_path, ScriptedSampler([]))
    with pytest.raises(RuntimeError):
        run(loop.resume())


# ---------- message derivation ----------

def test_rebuild_messages_from_events():
    events = [
        ev.SessionStarted(task="do it"),
        ev.ModelResponded(call_id="c", text="working"),
        ev.EffectRequested(effect_id="k", tool_name="grep", arguments={}),
        ev.EffectCompleted(effect_id="k", output="hit"),
        ev.EffectDenied(effect_id="d", tool_name="bash_exec", reason="nope"),
    ]
    msgs = rebuild_messages(events, "SYS")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "do it"}
    assert msgs[2]["role"] == "assistant"
    assert any("obs:grep" in m["content"] for m in msgs)
    assert any("DENIED nope" in m["content"] for m in msgs)
