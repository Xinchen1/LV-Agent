"""Regression tests for SuperAgentPolicy reflection meta-loop (policies.py)."""

from agent_project.execution_engine import ExecutionContext
from agent_project.policies import SuperAgentPolicy


class _Trace:
    def __init__(self, score, obs, tools):
        self.quality_score = score
        self.observations = obs
        self.tools_used = tools


class _Engine:
    """Fake engine: first run executes a real call; re-runs are all deduped."""

    def __init__(self):
        self.calls = 0

    def run(self, policy, ctx):
        self.calls += 1
        if self.calls == 1:
            ctx.executed_calls = {"file_ops:{}": "x"}  # a genuinely new call
            return _Trace(0.5, ["tool error: file not found"], ["file_ops"])
        # reflection re-run hit nothing new (everything deduped)
        return _Trace(0.5, ["system stop: you already executed file_ops"], [])


def test_run_meta_loop_stops_when_reflection_yields_no_new_calls():
    p = SuperAgentPolicy.__new__(SuperAgentPolicy)
    p.model = object()
    p.inner = None  # run_meta_loop calls engine.run(self.inner, ...); fake engine ignores it
    reflect_calls = []
    p._reflect_and_replan = lambda trace, ctx: reflect_calls.append(1) or "plan"
    p._has_actionable_errors = lambda obs: True  # force the reflect path

    eng = _Engine()
    ctx = ExecutionContext(task="do x", available_tools=None, config=None, max_steps=16)
    trace = p.run_meta_loop(eng, ctx)

    # Without the early-break, this would loop to max_attempts (3 engine runs);
    # with it, the 2nd run produces no new calls so we stop after 2 runs.
    assert eng.calls == 2
    assert len(reflect_calls) == 1
    assert trace.quality_score == 0.5


def test_has_actionable_errors_ignores_dedup():
    p = SuperAgentPolicy.__new__(SuperAgentPolicy)
    # dedup-only observations must NOT be "actionable" -> no wasted reflection
    assert p._has_actionable_errors(["system stop: you already executed file_ops"]) is False
    # real errors are actionable
    assert p._has_actionable_errors(["tool error: file not found", "syntax error in code"]) is True
