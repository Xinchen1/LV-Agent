"""BatchRunner 单测: FakeBackend 跑三条链路 + diff 检退化(无网络)."""
import json
from agent_project.agent import OpenMythosAgent
from agent_project.batch_runner import BatchCase, BatchRunner, check_case, diff_reports
from agent_project.config import AgentConfig


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, n_loops=1, temperature=0.7, max_tokens=512,
                 stream_callback=None, token_callback=None, quick=False, **kw):
        self.calls += 1
        if self.calls == 1:
            return "你好！有什么可以帮你的吗？"
        return "收到，已处理。"


def make_agent():
    OpenMythosAgent._load_model_backend = lambda self: FakeBackend()
    cfg = AgentConfig()
    cfg.memory.enabled = False
    cfg.reflection.enabled = False
    cfg.planning.enabled = False
    cfg.reasoning.enabled = False
    cfg.fast_mode = True
    return OpenMythosAgent(cfg)


def test_check_case_pass_and_fail():
    ok = {"success": True, "final_answer": "你好世界",
          "actions": [{"tool_name": "file_ops"}], "metadata": {"strategy": "direct"}}
    r = check_case(BatchCase(name="a", task="t", must_use_tools=["file_ops"],
                             expect_strategy="direct", min_answer_len=2), ok)
    assert r.passed, r.failures
    bad = dict(ok, success=False)
    r2 = check_case(BatchCase(name="a", task="t"), bad)
    assert not r2.passed and any("success" in f for f in r2.failures)


def test_forbid_tools():
    res = {"success": True, "final_answer": "hi",
           "actions": [{"tool_name": "bash_exec"}], "metadata": {}}
    r = check_case(BatchCase(name="x", task="t", forbid_tools=["bash_exec"]), res)
    assert not r.passed


def test_runner_writes_report(tmp_path):
    cases = [BatchCase(name="greet", task="你好", min_answer_len=2)]
    rep = BatchRunner(make_agent, out_dir=str(tmp_path)).run(cases, run_name="r1")
    assert rep["passed"] == 1 and rep["total"] == 1
    assert (tmp_path / "r1" / "report.json").exists()
    assert (tmp_path / "r1" / "greet.json").exists()
    saved = json.loads((tmp_path / "r1" / "report.json").read_text(encoding="utf-8"))
    assert saved["passed"] == 1


def test_runner_survives_backend_crash(tmp_path):
    """后端崩了快车道应降级为友好提示(成功返回), 而非掀翻整批."""
    def boom_factory():
        a = make_agent()
        a.backend.generate = lambda *a_, **k: (_ for _ in ()).throw(RuntimeError("down"))
        return a
    rep = BatchRunner(boom_factory, out_dir=str(tmp_path)).run(
        [BatchCase(name="c", task="hi", min_answer_len=2)], run_name="r2")
    assert rep["passed"] == 1, rep
    traj = json.loads((tmp_path / "r2" / "c.json").read_text(encoding="utf-8"))
    assert "失败" in traj["result"]["final_answer"] or traj["result"]["success"]


def test_diff_reports_spots_regression():
    old = {"passed": 2, "total": 2, "cases": [
        {"name": "a", "passed": True, "failures": []},
        {"name": "b", "passed": True, "failures": []}]}
    new = {"passed": 1, "total": 2, "cases": [
        {"name": "a", "passed": False, "failures": ["success=False"]},
        {"name": "b", "passed": True, "failures": []}]}
    d = diff_reports(old, new)
    assert [r["case"] for r in d["regressions"]] == ["a"]
    assert d["fixes"] == [] and d["new_failures"] == []
