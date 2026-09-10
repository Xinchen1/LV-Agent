"""Batch trajectory runner (对标 Hermes batch_runner + trajectory).

批量跑一批任务, 落盘 JSONL 轨迹 + report.json, 做结构化断言
(只比结构: success/strategy/工具序列, 不比逐字文本, LLM 输出天然波动).
两次 report 可 diff 做回归( pass->fail 即退化 ).

后端无关: 调用方传 make_agent() 工厂(单测注 FakeBackend, 真跑注真实配置).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BatchCase:
    """一条评测用例: 只约束结构, 不约束逐字文本."""

    name: str
    task: str
    expect_success: bool = True
    expect_strategy: Optional[str] = None  # 如 'direct' / 'react', None 不约束
    must_use_tools: List[str] = field(default_factory=list)
    forbid_tools: List[str] = field(default_factory=list)
    min_answer_len: int = 1


@dataclass
class CaseResult:
    name: str
    task: str
    passed: bool
    failures: List[str]
    strategy: str
    success: bool
    tools_used: List[str]
    answer_len: int
    duration_ms: int


def check_case(case: BatchCase, result: Dict[str, Any]) -> CaseResult:
    """对单条 agent.run() 结果做结构化断言."""
    failures: List[str] = []
    meta = result.get("metadata", {}) or {}
    strategy = str(meta.get("strategy", "unknown"))
    success = bool(result.get("success"))
    actions = result.get("actions", []) or []
    tools_used = [a.get("tool_name", "?") for a in actions if isinstance(a, dict)]
    answer = result.get("final_answer") or ""

    if success != case.expect_success:
        failures.append(f"success={success} 期望 {case.expect_success}")
    if case.expect_strategy and strategy != case.expect_strategy:
        failures.append(f"strategy={strategy} 期望 {case.expect_strategy}")
    for t in case.must_use_tools:
        if t not in tools_used:
            failures.append(f"缺必需工具 {t} (实际 {tools_used})")
    for t in case.forbid_tools:
        if t in tools_used:
            failures.append(f"禁用工具被调用 {t}")
    if len(answer) < case.min_answer_len:
        failures.append(f"答案过短 {len(answer)} < {case.min_answer_len}")
    return CaseResult(
        name=case.name, task=case.task, passed=not failures, failures=failures,
        strategy=strategy, success=success, tools_used=tools_used,
        answer_len=len(answer), duration_ms=int(meta.get("duration_ms", 0)),
    )


class BatchRunner:
    """批量执行 + 落盘 + 打分."""

    def __init__(self, make_agent: Callable[[], Any], out_dir: str = "eval_runs"):
        self.make_agent = make_agent
        self.out_dir = Path(out_dir)

    def run(self, cases: List[BatchCase], run_name: Optional[str] = None) -> Dict[str, Any]:
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        agent = self.make_agent()
        results: List[CaseResult] = []
        for case in cases:
            started = time.time()
            try:
                raw = agent.run(case.task)
            except Exception as e:  # noqa: BLE001 — 单条失败不掀翻整批, 记分即可
                raw = {"success": False, "final_answer": "",
                       "actions": [], "metadata": {"error": f"{type(e).__name__}: {e}"}}
            checked = check_case(case, raw)
            checked.duration_ms = checked.duration_ms or int((time.time() - started) * 1000)
            results.append(checked)
            traj = {
                "name": case.name, "task": case.task,
                "result": {
                    "success": raw.get("success"), "strategy": checked.strategy,
                    "tools_used": checked.tools_used,
                    "final_answer": (raw.get("final_answer") or "")[:2000],
                    "thinking_steps": raw.get("thinking_steps"),
                    "outer_loops": raw.get("outer_loops"),
                },
            }
            (run_dir / f"{case.name}.json").write_text(
                json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        passed = sum(1 for r in results if r.passed)
        report = {
            "run": run_name, "total": len(results), "passed": passed,
            "failed": len(results) - passed,
            "cases": [asdict(r) for r in results],
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.out_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run": run_name, "passed": passed,
                                "total": len(results)}, ensure_ascii=False) + "\n")
        return report


def diff_reports(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """对比两次 report: 揪出 pass->fail 退化与新失败."""
    old_map = {c["name"]: c for c in old.get("cases", [])}
    regressions, fixes, new_failures = [], [], []
    for c in new.get("cases", []):
        o = old_map.get(c["name"])
        if o is None:
            if not c["passed"]:
                new_failures.append(c["name"])
            continue
        if o["passed"] and not c["passed"]:
            regressions.append({"case": c["name"], "old_failures": o["failures"],
                                "new_failures": c["failures"]})
        elif not o["passed"] and c["passed"]:
            fixes.append(c["name"])
    return {"regressions": regressions, "fixes": fixes, "new_failures": new_failures,
            "old": f'{old.get("passed")}/{old.get("total")}',
            "new": f'{new.get("passed")}/{new.get("total")}'}
