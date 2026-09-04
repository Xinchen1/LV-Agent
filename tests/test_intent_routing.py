#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# P4 回归: 意图路由(启发式层, 离线)

import logging
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.intent_classifier import IntentClassifier


def make_clf():
    return IntentClassifier(
        correct_tool_name=lambda x: x,
        logger=logging.getLogger("test_intent"),
    )


def classify(text):
    return make_clf().classify(text, None)


def tool_of(text):
    r = classify(text)
    return r[0] if isinstance(r, tuple) else r


def test_memory_query():
    assert tool_of("你记得我叫什么名字吗") == "__memory_query__"


def test_folder_desktop():
    assert tool_of("看看桌面上的agent_project文件夹") == "file_ops"


def test_search_beats_python_keyword():
    # 显式搜索动词优先于 python 裸关键词
    assert tool_of("搜索一下Python 3.14发布时间") == "web_search"


def test_calculator():
    assert tool_of("计算 12*34 等于多少") == "calculator"


def test_move_file():
    assert tool_of("把/tmp/a.txt移动到/tmp/b/") == "bash_exec"


def test_weather():
    assert tool_of("今天北京天气怎么样") == "weather"


def test_ambiguous_returns_none():
    assert classify("帮我看一下那个东西") is None
    assert classify("嗯") is None
