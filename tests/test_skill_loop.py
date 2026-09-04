#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# P4 回归: P1 学习闭环(离线, 隔离目录)

import logging
import sys
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.agent import OpenMythosAgent
from agent_project.execution_engine import StepRecord, ToolCallRequest
from agent_project.memskill import MemSkillEngine, MemorySkill
from agent_project.policies import ReActPolicy


def make_engine():
    d = tempfile.mkdtemp()
    eng = MemSkillEngine(skills_dir=d, llm_call=lambda p, t=0.3, m=512: "")
    return eng


def test_cjk_skill_selected_and_injected():
    eng = make_engine()
    eng.bank.save(MemorySkill(
        name="amd-search", description="AMD平台搜索用中文关键词加年份",
        content="搜索时用中文关键词并补年份", source="learned"))
    sel = eng.controller.select("搜索一下Python 3.14发布时间")
    assert any(s.name == "amd-search" for s, _ in sel)
    a = OpenMythosAgent.__new__(OpenMythosAgent)
    a.skill_engine = None
    a.memskill_engine = eng
    ctx = a._get_skill_context("搜索一下Python 3.14发布时间")
    assert "amd-search" in ctx


def test_irrelevant_skill_not_injected():
    eng = make_engine()
    eng.bank.save(MemorySkill(
        name="french-poetry", description="translate french poetry",
        content="french verse translation", source="learned"))
    a = OpenMythosAgent.__new__(OpenMythosAgent)
    a.skill_engine = None
    a.memskill_engine = eng
    ctx = a._get_skill_context("搜索一下Python 3.14发布时间")
    assert "french-poetry" not in ctx


def test_failure_lowers_score():
    eng = make_engine()
    eng.bank.save(MemorySkill(
        name="s1", description="测试技能描述", content="测试内容", source="learned"))
    eng.bank.record_skill_usage("s1", success=True)
    before = eng.bank.get_skill_score("s1")
    assert before > 0
    eng.bank.record_skill_usage("s1", success=False)
    eng.bank.record_skill_usage("s1", success=False)
    assert eng.bank.get_skill_score("s1") < before


def test_history_includes_action():
    p = ReActPolicy()
    steps = [type("S", (), {
        "reasoning": "先写文件",
        "tool_calls": [ToolCallRequest(tool_name="file_ops", arguments={"action": "write"})],
        "observations": ["写入成功"],
    })()]
    h = p._format_react_history(steps)
    assert "Action:" in h and "file_ops" in h
    assert "Observation:" in h
