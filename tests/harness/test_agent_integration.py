"""Legacy agent <-> kernel wiring: config gating and the _execute_tool guard.

The agent class is a god object; these tests exercise the harness seam
without booting the full module stack (backend, memory, planners).
"""

import logging
from types import SimpleNamespace

from agent_project.agent import OpenMythosAgent
from agent_project.config import HarnessConfig
from agent_project.harness.kernel import Kernel, permissive_policy, safe_default_policy
from agent_project.tools import ToolCall


def _bare_agent(**attrs):
    agent = object.__new__(OpenMythosAgent)
    agent.logger = logging.getLogger("test.harness")
    agent._harness_kernel = None
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def test_config_defaults_disabled():
    assert HarnessConfig().enabled is True
    assert HarnessConfig().policy == "safe"


def test_build_kernel_returns_none_when_disabled():
    agent = _bare_agent(config=SimpleNamespace(harness=HarnessConfig(enabled=False)))
    assert agent._build_harness_kernel() is None


def test_build_kernel_safe_policy(tmp_path):
    agent = _bare_agent(
        config=SimpleNamespace(
            harness=HarnessConfig(enabled=True, workspace_root=str(tmp_path))
        )
    )
    kernel = agent._build_harness_kernel()
    assert isinstance(kernel, Kernel)
    from agent_project.harness.effects import make_effect
    from agent_project.harness.kernel import Decision

    adm = kernel.evaluate(make_effect("bash_exec", {"command": "sudo rm -rf /"}))
    assert adm.decision is Decision.DENY


def test_execute_tool_denied_by_policy(tmp_path):
    agent = _bare_agent(
        _harness_kernel=Kernel(safe_default_policy(str(tmp_path)))
    )
    result = agent._execute_tool(
        ToolCall(tool_name="bash_exec", arguments={"command": "sudo rm -rf /"})
    )
    assert result.success is False
    assert "Denied by harness policy" in result.error


def test_execute_tool_allowed_runs_for_real(tmp_path):
    agent = _bare_agent(
        _harness_kernel=Kernel(safe_default_policy(str(tmp_path)))
    )
    result = agent._execute_tool(
        ToolCall(tool_name="calculator", arguments={"expression": "6*7"})
    )
    assert result.success is True
    assert "42" in result.output


def test_execute_tool_permissive_allows_harmless_exec(tmp_path):
    agent = _bare_agent(_harness_kernel=Kernel(permissive_policy()))
    result = agent._execute_tool(
        ToolCall(tool_name="bash_exec", arguments={"command": "echo harness-ok"})
    )
    assert result.success is True
    assert "harness-ok" in result.output


def test_kernel_none_preserves_legacy_path():
    agent = _bare_agent(_harness_kernel=None)
    result = agent._execute_tool(
        ToolCall(tool_name="calculator", arguments={"expression": "1+1"})
    )
    assert result.success is True
