#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# P4 回归: 定位快路防劫持(离线, mock GlobTool)

import logging
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

import agent_project.agent as A
from agent_project.tools import ToolResult


class FakeGlob:
    def __init__(self, hits=()):
        self.hits = hits
        self.seen = []

    def execute(self, pattern="", path="", max_results=40):
        self.seen.append(pattern)
        if self.hits:
            return ToolResult(success=True, output="f1\nf2",
                              metadata={"count": len(self.hits)})
        return ToolResult(success=True, output="Found 0 file(s)",
                          metadata={"count": 0})


def make_agent(glob):
    # 函数内是 from .tools import GlobTool, 必须补目标模块
    import agent_project.tools as T
    T.GlobTool = lambda *a, **k: glob  # noqa: E731
    a = A.OpenMythosAgent.__new__(A.OpenMythosAgent)
    a.logger = logging.getLogger("test_locate")
    a._status = lambda cb, msg: None
    return a


def test_compound_task_falls_through():
    g = FakeGlob(hits=("x",))
    a = make_agent(g)
    assert a._try_location_fast_path("看下桌面的grok-build文件夹，修改它为不要登录可以使用") is None


def test_search_task_not_hijacked():
    g = FakeGlob()
    a = make_agent(g)
    assert a._try_location_fast_path("搜索一下Python 3.14发布时间") is None


def test_memory_task_not_hijacked():
    g = FakeGlob()
    a = make_agent(g)
    assert a._try_location_fast_path("你记得我吗") is None


def test_empty_results_not_success():
    g = FakeGlob(hits=())
    a = make_agent(g)
    # 纯定位但无命中 → 回None走正常循环, 不冒充成功
    assert a._try_location_fast_path("找一下桌面的不存在xyz123文件夹") is None


def test_change_to_compound_falls_through():
    # "改为/不用登录" 等后继动作 → 不走定位快路
    g = FakeGlob(hits=("x",))
    a = make_agent(g)
    assert a._try_location_fast_path("看下桌面的grok文件夹，需要改为不用登录就可以使用") is None
    assert a._try_location_fast_path("找下桌面的app文件夹，改成免登录版") is None


def test_real_hit_returns_success():
    g = FakeGlob(hits=("a", "b"))
    a = make_agent(g)
    r = a._try_location_fast_path("找一下桌面的agent_project文件夹")
    assert isinstance(r, dict) and r.get("success") is True
    # 提取的是干净目标名, 不是整句
    assert "修改" not in g.seen[0] and "一下" not in g.seen[0]
