#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research




import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.config import load_config


def test_flat_top_level_layout_loads_backend():
    """仓库自带平铺式 config.yaml(backend/deepseek 在顶层)不应被静默忽略."""
    flat = """backend: deepseek
model_registry_path: agent_project/config/models.yaml
deepseek:
  api_key: null
  base_url: https://api.deepseek.com
  model: deepseek-chat
  temperature: 0.7
openai:
  api_key: null
  base_url: null
  model: gpt-4o-mini
"""
    td = tempfile.mkdtemp()
    p = Path(td) / "config.yaml"
    p.write_text(flat, encoding="utf-8")
    c = load_config(str(p))
    assert c.backend == "deepseek"
    assert c.deepseek.get("model") == "deepseek-chat", c.deepseek
    assert c.deepseek.get("base_url") == "https://api.deepseek.com"
    assert c.openai.get("model") == "gpt-4o-mini"


def test_flat_layout_openai_backend():
    """平铺式 backend: openai 应真正生效(之前 bug: 被默认 deepseek 覆盖)."""
    flat = """backend: openai
openai:
  api_key: sk-test-123
  base_url: https://api.test.com/v1
  model: test-model
"""
    td = tempfile.mkdtemp()
    p = Path(td) / "config.yaml"
    p.write_text(flat, encoding="utf-8")
    c = load_config(str(p))
    assert c.backend == "openai", f"backend 应取 openai, 实际 {c.backend}"
    assert c.openai.get("api_key") == "sk-test-123"
    assert c.openai.get("model") == "test-model"


def test_agent_wrapper_layout_still_works():
    """config.example.yaml 的 agent 包装式布局不受影响."""
    wrapped = """agent:
  backend: deepseek
  deepseek:
    api_key: sk-wrapped
    base_url: https://api.deepseek.com
    model: deepseek-v4-flash
tools:
  bash_exec:
    enabled: true
"""
    td = tempfile.mkdtemp()
    p = Path(td) / "config.yaml"
    p.write_text(wrapped, encoding="utf-8")
    c = load_config(str(p))
    assert c.backend == "deepseek"
    assert c.deepseek.get("model") == "deepseek-v4-flash", c.deepseek
    assert c.deepseek.get("api_key") == "sk-wrapped"
    assert c.tools.bash_exec.get("enabled") is True
