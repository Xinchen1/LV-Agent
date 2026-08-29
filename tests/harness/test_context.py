#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research





import asyncio

from agent_project.harness import events as ev
from agent_project.harness.context import (
    ContextAssembler,
    TruncatingCompactor,
    approx_tokens,
    messages_tokens,
)
from agent_project.harness.loop import AgentLoop, SampleResult, rebuild_messages
from agent_project.harness.budget import Ledger, Limits
from agent_project.harness.journal import Journal
from agent_project.harness.kernel import Decision, Kernel, Rule
from agent_project.harness.scheduler import Scheduler
from agent_project.harness.stream import EventBus


def _msgs(n, width=40):
    base = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "TASK"},
    ]
    for i in range(n):
        base.append({"role": "user", "content": f"[obs:grep] {'x' * width} {i}"})
    return base


def test_approx_tokens_heuristic():
    assert approx_tokens("") == 0
    assert approx_tokens("abcd" * 10) == 10
    assert messages_tokens([{"content": "abcd"}]) == 1


def test_under_budget_passes_through():
    events = [ev.SessionStarted(task="t")]
    asm = ContextAssembler("SYS", budget_tokens=10_000)
    out = asm.assemble(events, rebuild_messages)
    assert not out.compacted and out.dropped_messages == 0
    assert out.messages[0]["content"] == "SYS"


def test_over_budget_compacts_middle_and_keeps_head_tail():
    messages = _msgs(n=20, width=200)  # ~1000 tokens, over budget of 200
    compactor = TruncatingCompactor(keep_tail=4)
    rewritten, summary = compactor(messages, budget_tokens=200)

    assert rewritten[0]["content"] == "SYS"
    assert rewritten[1]["content"] == "TASK"
    assert any("compacted" in str(m["content"]) for m in rewritten)
    assert "grep" in summary  # tool trace survives compaction
    assert rewritten[-1]["content"].endswith("19")  # newest tail preserved
    assert messages_tokens(rewritten) < messages_tokens(messages)


def test_assembler_reports_dropped_count():
    events = [ev.SessionStarted(task="t")] + [
        ev.EffectRequested(effect_id=f"k{i}", tool_name="grep", arguments={})
        for i in range(15)
    ] + [
        ev.EffectCompleted(effect_id=f"k{i}", output="y" * 400) for i in range(15)
    ]
    asm = ContextAssembler("SYS", budget_tokens=150)
    out = asm.assemble(events, rebuild_messages)
    assert out.compacted and out.dropped_messages > 0
    assert "compacted" in out.summary


class _Script:
    def __init__(self):
        self.items = [
            SampleResult(text="", tool_calls=(("grep", {"pattern": f"p{i}"}),))
            for i in range(3)
        ] + [SampleResult(text="final")]

    async def __call__(self, messages, tools):
        return self.items.pop(0)


def test_loop_journals_assembly_and_compaction(tmp_path):
    journal = Journal(tmp_path / "run.jsonl")
    kernel = Kernel([Rule(Decision.ALLOW, reason="ok")],
                    executor=lambda e: "z" * 2000)
    from agent_project.harness.context import ContextAssembler

    loop = AgentLoop(
        kernel=kernel,
        scheduler=Scheduler(),
        ledger=Ledger(Limits()),
        journal=journal,
        bus=EventBus(),
        sampler=_Script(),
        context=ContextAssembler("SYS", budget_tokens=200),
    )
    assert asyncio.run(loop.run("t")) == "final"

    kinds = [e.kind for e in journal.read_all()]
    assert "ContextAssembled" in kinds
    assert "ContextCompacted" in kinds
    compacted = [e for e in journal.read_all() if isinstance(e, ev.ContextCompacted)]
    assert compacted[-1].kept_messages > 0
