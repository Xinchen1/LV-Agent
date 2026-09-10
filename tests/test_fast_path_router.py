#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for FastPathRouter, LocationFastPath, and SimpleQueryClassifier.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.fast_path import (
    FastPathRouter,
    LocationFastPath,
    SimpleQueryClassifier,
    is_simple_query,
)
from agent_project.tools import ToolResult


class FakeGlob:
    """Mock GlobTool for location fast path testing."""
    def __init__(self, hits=("match1.py",)):
        self.hits = hits
        self.seen_patterns = []

    def execute(self, pattern="", path="", max_results=40):
        self.seen_patterns.append(pattern)
        if self.hits:
            return ToolResult(
                success=True,
                output="\n".join(self.hits),
                metadata={"count": len(self.hits)},
            )
        return ToolResult(
            success=True,
            output="Found 0 file(s)",
            metadata={"count": 0},
        )


# =====================================================================
# LocationFastPath Tests
# =====================================================================

def test_location_fast_path_target_extraction():
    loc = LocationFastPath()

    # Explicit alias
    res = loc.extract_target_and_root("找一下桌面的agent_project文件夹")
    assert res is not None
    target, root = res
    assert target == "agent_project"
    assert "Desktop" in root

    # Downloads folder
    res2 = loc.extract_target_and_root("下载文件夹的claude code")
    assert res2 is not None
    target2, root2 = res2
    assert target2 == "claude code"
    assert "Downloads" in root2

    # Documents folder
    res3 = loc.extract_target_and_root("看下文档里面的report")
    assert res3 is not None
    target3, root3 = res3
    assert target3 == "report"
    assert "Documents" in root3


def test_location_fast_path_compound_task_rejected():
    loc = LocationFastPath()
    assert loc.extract_target_and_root("看下桌面的grok-build文件夹，修改它为不要登录可以使用") is None
    assert loc.extract_target_and_root("看下桌面的grok文件夹，需要改为不用登录就可以使用") is None
    assert loc.extract_target_and_root("找下桌面的app文件夹，改成免登录版") is None


def test_location_fast_path_search_news_rejected():
    loc = LocationFastPath()
    assert loc.extract_target_and_root("搜索一下Python 3.14发布时间") is None
    assert loc.extract_target_and_root("查下最新的 ai 新闻") is None
    assert loc.extract_target_and_root("搜索AI相关新闻") is None


def test_location_fast_path_doc_reading_rejected():
    loc = LocationFastPath()
    assert loc.extract_target_and_root("看下这篇调研报告") is None
    assert loc.extract_target_and_root("读一下这篇笔记") is None


def test_location_fast_path_execution_success():
    loc = LocationFastPath()
    fake_glob = FakeGlob(hits=["/Users/mac/Desktop/agent_project/main.py"])
    res = loc.execute(
        task="找一下桌面的agent_project文件夹",
        glob_tool=fake_glob,
    )
    assert res is not None
    assert res.get("success") is True
    assert "agent_project" in res.get("final_answer", "")
    assert res.get("outer_loops") == 1


def test_location_fast_path_execution_no_hits_falls_through():
    loc = LocationFastPath()
    fake_glob = FakeGlob(hits=())
    res = loc.execute(
        task="找一下桌面的不存在文件夹_xyz123",
        glob_tool=fake_glob,
    )
    # 无命中时应返回 None，交回主循环处理，不可冒充成功
    assert res is None


# =====================================================================
# SimpleQueryClassifier Tests
# =====================================================================

def test_simple_classifier_greetings_and_courtesy():
    clf = SimpleQueryClassifier()
    assert clf.is_simple_query("你好") is True
    assert clf.is_simple_query("hi") is True
    assert clf.is_simple_query("hello!") is True
    assert clf.is_simple_query("谢谢") is True
    assert clf.is_simple_query("好的") is True
    assert clf.is_simple_query("ok") is True
    assert clf.is_simple_query("在吗") is True


def test_simple_classifier_qa_exemptions():
    clf = SimpleQueryClassifier()
    assert clf.is_simple_query("什么是图灵机") is True
    assert clf.is_simple_query("Agent是什么") is True
    assert clf.is_simple_query("简单解释一下强化学习概念") is True


def test_simple_classifier_code_modifications_rejected():
    clf = SimpleQueryClassifier()
    assert clf.is_simple_query("修改main.py文件增加登录检查") is False
    assert clf.is_simple_query("给贪吃蛇加个音效") is False
    assert clf.is_simple_query("把test.py移动到tests目录") is False
    assert clf.is_simple_query("优化一下排序算法代码") is False


def test_simple_classifier_deep_tasks_rejected():
    clf = SimpleQueryClassifier()
    assert clf.is_simple_query("搜索一下最新的AI新闻") is False
    assert clf.is_simple_query("写一段快速排序代码并运行") is False
    assert clf.is_simple_query("部署这个项目到本地") is False


# =====================================================================
# FastPathRouter Integration Tests
# =====================================================================

def test_fast_path_router_delegation():
    router = FastPathRouter()
    assert router.is_simple_query("你好啊") is True
    assert router.is_simple_query("搜索DeepSeek最新发布") is False

    fake_glob = FakeGlob(hits=["/mock/path/project"])
    res = router.try_location_fast_path(
        task="找一下桌面的project文件夹",
        glob_tool=fake_glob,
    )
    assert res is not None
    assert res["success"] is True


def test_agent_fast_path_integration(monkeypatch):
    from agent_project.agent import OpenMythosAgent
    import agent_project.tools as T

    # Mock GlobTool on tools module (monkeypatch 自动还原, 防 FakeGlob 污染全局注册表)
    fake_glob = FakeGlob(hits=["/mock/desk/my_project"])
    monkeypatch.setattr(T, "GlobTool", lambda *a, **k: fake_glob)

    agent = OpenMythosAgent.__new__(OpenMythosAgent)
    agent.logger = None
    agent._status = lambda cb, msg: None

    # Verify agent methods delegate correctly to FastPathRouter
    assert agent._is_simple_query("你好") is True
    assert agent._is_simple_query("修改代码文件") is False

    loc_res = agent._try_location_fast_path("找一下桌面的my_project文件夹")
    assert loc_res is not None
    assert loc_res["success"] is True

    # Compound task should fall through to None
    compound_res = agent._try_location_fast_path("找一下桌面的my_project文件夹，修改它")
    assert compound_res is None
