#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# P4 回归: P3 任务感知工具子集(离线)

import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.tools import TOOLS_REGISTRY, select_tools_for_task, CORE_TOOLS


def all_tools():
    return TOOLS_REGISTRY.get_tools_dict()


def test_core_always_present():
    sub = select_tools_for_task("随便聊聊", all_tools())
    assert CORE_TOOLS <= set(sub)


def test_weather_hint():
    sub = select_tools_for_task("今天北京天气怎么样", all_tools())
    assert "weather" in sub
    assert CORE_TOOLS <= set(sub)


def test_git_hint():
    sub = select_tools_for_task("用git提交代码", all_tools())
    assert "git" in sub


def test_pdf_hint():
    sub = select_tools_for_task("读这篇论文pdf", all_tools())
    assert "pdf_tool" in sub


def test_empty_falls_back_to_all():
    assert select_tools_for_task("", all_tools()) == all_tools()


def test_subset_saves_tokens():
    full = TOOLS_REGISTRY.get_openai_tools()
    sub = TOOLS_REGISTRY.get_openai_tools(names=set(list(all_tools())[:8]))
    full_tok = len(json.dumps(full, ensure_ascii=False)) // 4
    sub_tok = len(json.dumps(sub, ensure_ascii=False)) // 4
    assert sub_tok < full_tok * 0.6, f"subset {sub_tok} vs full {full_tok}"
    assert len(sub) == 8


def test_subset_preserves_order():
    sub = select_tools_for_task("天气", all_tools())
    keys = list(sub.keys())
    assert keys == [k for k in all_tools() if k in set(keys)]
