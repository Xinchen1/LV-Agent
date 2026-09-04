#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# P4 回归: P2 上下文预算(离线)

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.context_engine import (
    ContextCompressor,
    ContextEngine,
    WorkingMemory,
    WorkingMemoryEvent,
    _estimate_tokens,
)


def make_engine():
    ce = ContextEngine.__new__(ContextEngine)
    ce.enabled = True
    ce.working_memory = WorkingMemory(max_events=200)
    ce.working_budget = 1500
    ce.compressor = ContextCompressor(llm_client=None)
    return ce


def test_first_round_protected_on_compress():
    ce = make_engine()
    ce.working_memory.add("user", "原始需求:做贪吃蛇")
    for i in range(40):
        ce.working_memory.add("user", f"第{i}问")
        ce.working_memory.add("assistant", f"第{i}答" + "x" * 200)
    n = ce.compress_working_memory(target_tokens=300)
    assert n > 0
    evs = ce.working_memory.get_events()
    first_users = [e.content for e in evs if e.role == "user" and e.event_type == "message"]
    assert first_users[0] == "原始需求:做贪吃蛇"
    summaries = [e for e in evs if e.event_type == "summary"]
    assert summaries and evs.index(summaries[0]) == 1


def test_add_overflow_keeps_first():
    wm = WorkingMemory(max_events=20)
    wm.add("user", "首轮需求")
    for i in range(30):
        wm.add("assistant", f"r{i}")
    assert wm.get_events()[0].content == "首轮需求"


def test_short_text_never_bloats():
    cc = ContextCompressor(llm_client=None)
    evs = [
        WorkingMemoryEvent(role="user", content="查发布时间"),
        WorkingMemoryEvent(role="assistant", content="2025-10-07" + "补" * 100),
    ]
    out = cc.compress_events(evs, target_tokens=150)
    assert _estimate_tokens(str(out)) <= 150


def test_narrative_has_structure():
    cc = ContextCompressor(llm_client=None)
    evs = [
        WorkingMemoryEvent(role="user", content="查发布时间"),
        WorkingMemoryEvent(role="assistant", content="2025-10-07发布"),
    ]
    out = cc.summarize_narrative(evs, target_tokens=150)
    assert "Facts" in str(out)


def test_dynamic_budget():
    assert ContextEngine._task_budget_multiplier("分析一下项目架构") == 1.5
    assert ContextEngine._task_budget_multiplier("写个排序函数") == 1.5
    assert ContextEngine._task_budget_multiplier("你好") == 1.0


def test_empty_events():
    cc = ContextCompressor(llm_client=None)
    assert cc.compress_events([], target_tokens=150) == ""
