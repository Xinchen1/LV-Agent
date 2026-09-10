#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: AGPL-3.0-only
# P4 回归: 临时写工具 + 临时热插拔工具能力

import sys

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.harness.hotswap import (
    CapabilitySlot,
    ModuleRegistry,
    HotSwapKernel,
    HealthAwareRegistry,
    SwapAuditLog,
    SwapError,
    SlotState,
)
from agent_project.tools import BaseTool, ToolResult, ToolRegistry, TOOLS_REGISTRY


# ---------------------------------------------------------------------------
# 辅助: 假工具
# ---------------------------------------------------------------------------

class DummyTool(BaseTool):
    name = "dummy"
    description = "a dummy tool"
    parameters = {"type": "object", "properties": {"x": {"type": "integer"}}}

    def __init__(self, tag=""):
        self.tag = tag
        self.calls = 0

    def execute(self, **kwargs):
        self.calls += 1
        return ToolResult(success=True, output=f"dummy:{self.tag}:{kwargs.get('x')}")


class SwappableModule:
    """模拟可热插拔的能力模块 (带 dispose)."""

    def __init__(self, version, *, fail_dispose=False):
        self.version = version
        self.calls = 0
        self.disposed = False
        self._fail_dispose = fail_dispose

    def invoke(self, x=1):
        self.calls += 1
        return f"{self.version}:{x}"

    def dispose(self):
        if self._fail_dispose:
            raise RuntimeError("dispose failure")
        self.disposed = True


class BrokenLoadModule(SwappableModule):
    """带 generate 属性(getter 抛错) 的模块, 触发 _wrap_module 抛错 → 回滚."""

    @property
    def generate(self):
        raise RuntimeError("simulated load failure")


# ---------------------------------------------------------------------------
# 临时写工具: ToolRegistry 注册/反注册
# ---------------------------------------------------------------------------

def test_temp_tool_register_get_unregister():
    reg = ToolRegistry()
    t = DummyTool("v1")
    reg.register(t)
    assert reg.get("dummy") is t
    assert "dummy" in reg.list_tools()
    # 反注册后消失
    assert reg.unregister("dummy") is True
    assert reg.get("dummy") is None
    assert "dummy" not in reg.list_tools()
    # 二次反注册返回 False
    assert reg.unregister("dummy") is False


def test_temp_tool_overwrite_by_register():
    reg = ToolRegistry()
    reg.register(DummyTool("v1"))
    reg.register(DummyTool("v2"))
    got = reg.get("dummy")
    assert got.tag == "v2"
    assert got.execute(x=3).output == "dummy:v2:3"


def test_temp_tool_swap_restores_original():
    """临时写工具: 替换后恢复原工具, 不影响注册表其他工具."""
    original = DummyTool("orig")
    TOOLS_REGISTRY.register(original)
    names_before = set(TOOLS_REGISTRY.list_tools())
    try:
        TOOLS_REGISTRY.register(DummyTool("temp"))
        assert TOOLS_REGISTRY.get("dummy").tag == "temp"
        assert TOOLS_REGISTRY.get("dummy").execute(x=1).output == "dummy:temp:1"
    finally:
        TOOLS_REGISTRY.register(original)
    assert TOOLS_REGISTRY.get("dummy") is original
    assert set(TOOLS_REGISTRY.list_tools()) == names_before


def test_temp_tool_unregister_restores_registry():
    """临时写工具: unregister 后不再出现在工具描述里."""
    TOOLS_REGISTRY.unregister("dummy")
    desc = TOOLS_REGISTRY.get_prompt_description()
    assert "dummy" not in desc
    assert "web_search" in desc or "file_ops" in desc


# ---------------------------------------------------------------------------
# 临时热插拔工具: CapabilitySlot / ModuleRegistry 的 register + swap
# ---------------------------------------------------------------------------

def test_hotswap_register_install_active():
    slot = CapabilitySlot(name="cap")
    m = SwappableModule("v1")
    slot.register(m, version="v1")
    assert slot.state is SlotState.ACTIVE
    assert slot.get() is not None
    assert slot.get().invoke(x=2) == "v1:2"


def test_hotswap_swap_swaps_module_and_versiosn():
    slot = CapabilitySlot(name="cap")
    slot.register(SwappableModule("v1"), version="v1")
    slot.swap(SwappableModule("v2"), version="v2")
    assert slot.get().invoke() == "v2:1"
    assert slot.version == "v2"
    assert slot.state is SlotState.ACTIVE


def test_hotswap_swap_disposes_old():
    old = SwappableModule("v1")
    slot = CapabilitySlot(name="cap")
    slot.register(old, version="v1")
    slot.swap(SwappableModule("v2"), version="v2")
    assert old.disposed is True


def test_hotswap_swap_rolls_back_on_load_failure():
    """临时热插拔: 换入坏模块自动回滚到旧模块并抛 SwapError."""
    good = SwappableModule("v1")
    slot = CapabilitySlot(name="cap")
    slot.register(good, version="v1")
    bad = BrokenLoadModule("v2")
    with pytest.raises(SwapError):
        slot.swap(bad, version="v2")
    # 回滚: 仍是旧模块, 且状态 ACTIVE
    assert slot.get().invoke() == "v1:1"
    assert slot.version == "v1"
    assert slot.state is SlotState.ACTIVE


def test_hotswap_swap_keeps_fallback_after_circuit_open():
    slot = CapabilitySlot(name="cap")
    slot.register(SwappableModule("v1"), version="v1")
    slot.set_fallback(SwappableModule("fallback"))
    # 触发熔断: 连续错误超阈值
    for _ in range(slot._cb_threshold):
        slot.record_error()
    assert slot.is_circuit_open() is True
    # 熔断打开 -> get() 路由到 fallback
    assert slot.get().invoke(x=9) == "fallback:9"


def test_hotswap_circuit_closed_without_errors():
    slot = CapabilitySlot(name="cap")
    slot.register(SwappableModule("v1"), version="v1")
    assert slot.is_circuit_open() is False


def test_hotswap_metrics_record_calls_and_errors():
    slot = CapabilitySlot(name="cap")
    slot.register(SwappableModule("v1"), version="v1")
    for _ in range(3):
        slot.record_call(success=True)
    slot.record_call(success=False)
    st = slot.status()
    assert st["active"]["calls"] == 4
    assert st["active"]["errors"] == 1


def test_hotswap_status_reflects_state():
    slot = CapabilitySlot(name="cap")
    slot.register(SwappableModule("v1"), version="v1")
    st = slot.status()
    assert st["state"] == "active"
    assert st["current_version"] == "v1"
    assert st["active"]["type"] == "SwappableModule"
    assert st["version_graph_size"] >= 1
    assert st["has_dispose"] is True


# ---------------------------------------------------------------------------
# ModuleRegistry 级别的临时热插拔
# ---------------------------------------------------------------------------

def test_registry_register_swap_get():
    reg = ModuleRegistry()
    reg.register("cap", SwappableModule("v1"), version="v1")
    assert reg.get("cap").invoke() == "v1:1"
    reg.swap("cap", SwappableModule("v2"), version="v2")
    assert reg.get("cap").invoke() == "v2:1"
    assert "cap" in reg.list_capabilities()


def test_registry_swap_unknown_capability_raises():
    reg = ModuleRegistry()
    with pytest.raises(KeyError):
        reg.swap("nope", SwappableModule("v1"))


def test_registry_force_swap():
    reg = ModuleRegistry()
    reg.register("cap", SwappableModule("v1"), version="v1")
    reg.force_swap("cap", SwappableModule("v2"), version="v2")
    assert reg.get("cap").invoke() == "v2:1"


def test_registry_shadow_promote_after_min_calls():
    reg = ModuleRegistry()
    reg.register("cap", SwappableModule("v1"), version="v1")
    reg.set_shadow("cap", SwappableModule("v2"), version="v2")
    # 尚未达到最小调用次数, 不能 promote
    assert reg.promote_shadow("cap") is False
    assert reg.get("cap").invoke() == "v1:1"
    # 给 shadow 记录调用到阈值
    slot = reg.get_slot("cap")
    for _ in range(slot.shadow_min_calls):
        slot.record_call(success=True, for_shadow=True)
    assert reg.promote_shadow("cap") is True
    assert reg.get("cap").invoke() == "v2:1"


# ---------------------------------------------------------------------------
# HotSwapKernel 桥接
# ---------------------------------------------------------------------------

def test_hotswap_kernel_register_swap():
    hk = HotSwapKernel(kernel=None)
    hk.register_capability("cap", SwappableModule("v1"), version="v1")
    assert hk.get_module("cap").invoke() == "v1:1"
    hk.swap("cap", SwappableModule("v2"), version="v2")
    assert hk.get_module("cap").invoke() == "v2:1"
    assert hk.status("cap")["cap"]["current_version"] == "v2"


# ---------------------------------------------------------------------------
# SwapAuditLog 审计
# ---------------------------------------------------------------------------

def test_swap_audit_log_records_events(tmp_path):
    log = SwapAuditLog(tmp_path / "swaps.jsonl")
    log.record("register", "cap", version="v1")
    log.record("swap", "cap", version="v2", old_version="v1", duration_ms=5)
    log.record("swap", "cap", version="v3", old_version="v2", duration_ms=7)
    recs = log.replay(capability="cap")
    assert len(recs) == 3
    swaps = [r for r in recs if r["event"] == "swap"]
    assert len(swaps) == 2
    stats = log.stats("cap")
    assert stats["total_swaps"] == 2
    assert stats["avg_duration_ms"] == 6


def test_swap_audit_log_replay_filter_event(tmp_path):
    log = SwapAuditLog(tmp_path / "swaps.jsonl")
    log.record("swap", "cap", version="v2", old_version="v1")
    log.record("error", "cap", version="v2")
    assert len(log.replay(capability="cap", event="error")) == 1


# ---------------------------------------------------------------------------
# HealthAwareRegistry
# ---------------------------------------------------------------------------

def test_health_aware_registry_swaps_if_unhealthy():
    reg = HealthAwareRegistry()
    reg.register("cap", SwappableModule("v1"), version="v1")

    def checker(mod):
        return {"healthy": False}

    reg.set_health_checker(checker)
    assert reg.swap_if_unhealthy("cap", SwappableModule("v2")) is True
    assert reg.reg.get("cap").invoke() == "v2:1"

    def healthy_checker(mod):
        return {"healthy": True}

    reg.set_health_checker(healthy_checker)
    assert reg.swap_if_unhealthy("cap", SwappableModule("v3")) is False
    assert reg.reg.get("cap").invoke() == "v2:1"