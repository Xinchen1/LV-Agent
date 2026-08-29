"""
Task Planning - lightweight LLM-based planner and plan structures.

NOTE: agent.py imports Planner, PlanningStrategy and create_simple_plan from
this module.

This module previously also contained a Knowledge-Graph / Episodic-Memory /
ContextCompressor subsystem that duplicated agent_project.memory. That dead
code has been removed. The shared dataclasses it relied on (MemoryNode /
MemoryEdge / MemoryRecord) are kept here because agent_project.memory imports
them.
"""

from __future__ import annotations

import uuid
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from pydantic import BaseModel, Field


# ============ Types ============

@dataclass
class MemoryNode:
    """
    A node in the knowledge graph representing an entity or concept
    """
    label: str = "entity"  # entity type (person, organization, concept, task, etc.)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def access(self):
        """Update access timestamp"""
        self.last_accessed = datetime.now().isoformat()


@dataclass
class MemoryEdge:
    """
    An edge representing a relationship between two nodes
    """
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0  # strength of relationship 0-1
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class MemoryRecord:
    """
    A complete memory entry (like an episode or experience)
    """
    content: str
    summary: str
    task_type: str
    nodes: List[str] = field(default_factory=list)  # linked node IDs
    edges: List[str] = field(default_factory=list)  # linked edge IDs
    importance: float = 0.5  # 0-1
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


# ============ Planning ============

class PlanningStrategy(Enum):
    """Supported planning strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class PlanNode:
    """A single node in an execution plan."""
    node_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    assigned_loops: int = 2
    tool_hint: Optional[str] = None
    expected_output: str = ""


class Plan:
    """Simple DAG-based plan."""

    def __init__(self, nodes: Dict[str, PlanNode]):
        self.nodes = nodes

    def topological_sort(self) -> List[str]:
        """Return node IDs in dependency order."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for dep in node.dependencies:
                    visit(dep)
            order.append(nid)

        for nid in self.nodes:
            visit(nid)
        return order

    def to_markdown(self) -> str:
        lines = ["## Plan"]
        for idx, nid in enumerate(self.topological_sort(), 1):
            node = self.nodes[nid]
            deps = f" (after: {', '.join(node.dependencies)})" if node.dependencies else ""
            lines.append(f"{idx}. [{nid}] {node.description}{deps}")
        return "\n".join(lines)


class Planner:
    """Lightweight LLM-based task planner."""

    def __init__(self, model_backend: Any, tokenizer: Any = None, config: Any = None):
        self.model = model_backend
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger("Planner")

    def create_plan(
        self,
        task: str,
        strategy: PlanningStrategy = PlanningStrategy.ADAPTIVE,
        optimize: bool = True,
        max_subtasks: int = 10
    ) -> Plan:
        """Create a plan by asking the LLM to decompose the task."""
        prompt = self._build_plan_prompt(task, strategy, max_subtasks)
        try:
            raw = self.model.generate(
                prompt=prompt,
                n_loops=1,
                temperature=0.2,
                max_tokens=1024,
            ).strip()
        except Exception as e:
            self.logger.warning(f"LLM plan generation failed: {e}")
            return self._fallback_plan(task)

        nodes = self._parse_plan_text(raw, max_subtasks)
        if not nodes:
            return self._fallback_plan(task)

        if optimize:
            nodes = self._optimize(nodes, strategy)

        return Plan(nodes)

    def _build_plan_prompt(self, task: str, strategy: PlanningStrategy, max_subtasks: int) -> str:
        strategy_desc = {
            PlanningStrategy.SEQUENTIAL: "Break the task into sequential steps that must run in order.",
            PlanningStrategy.PARALLEL: "Identify independent subtasks that can be executed in parallel.",
            PlanningStrategy.HIERARCHICAL: "Create a high-level plan with nested sub-goals.",
            PlanningStrategy.ADAPTIVE: "Choose sequential or parallel decomposition based on the task.",
        }.get(strategy, "Break the task into clear, actionable steps.")

        return (
            "You are a task planner. Decompose the user's task into a small number of concrete steps.\n\n"
            f"Task: {task}\n\n"
            f"Strategy: {strategy.value}\n"
            f"{strategy_desc}\n\n"
            "Rules:\n"
            "- Output ONLY a JSON object with key 'steps' containing a list of objects.\n"
            "- Each step object must have 'description' (string), 'dependencies' (list of step ids), and optional 'tool_hint' (string).\n"
            "- Use step ids like 'task_0', 'task_1', etc. task_0 is the root step.\n"
            f"- Limit to at most {max_subtasks} steps.\n"
            "- Do not include any explanation outside the JSON.\n\n"
            "Example:\n"
            '{"steps": [{"id": "task_0", "description": "Locate the target file or directory", "dependencies": []}, '
            '{"id": "task_1", "description": "Read and analyze the relevant contents", "dependencies": ["task_0"]}]}'
        )

    def _parse_plan_text(self, text: str, max_subtasks: int) -> Dict[str, PlanNode]:
        """Parse plan from LLM output (JSON or numbered list)."""
        # Try JSON first.
        try:
            start = text.find('{')
            if start != -1:
                payload = json.loads(text[start:])
                steps = payload.get("steps") or payload.get("tasks") or payload.get("plan")
                if isinstance(steps, list):
                    nodes: Dict[str, PlanNode] = {}
                    for i, s in enumerate(steps[:max_subtasks]):
                        sid = s.get("id") if s.get("id") else f"task_{i}"
                        nodes[sid] = PlanNode(
                            node_id=sid,
                            description=s.get("description", s.get("task", s.get("name", "unknown"))),
                            dependencies=[str(d) for d in s.get("dependencies", []) if d],
                            assigned_loops=max(1, min(8, int(s.get("loops", 2)))),
                            tool_hint=s.get("tool_hint") or s.get("tool"),
                            expected_output=s.get("expected_output", ""),
                        )
                    return nodes
        except Exception as e:
            self.logger.debug(f"Plan JSON parse failed: {e}")

        # Fallback: parse numbered lines.
        nodes: Dict[str, PlanNode] = {"task_0": PlanNode("task_0", f"Complete: {text[:120]}")}
        line_re = re.compile(r"^\s*(?:\d+|[\-\*])\.\s*(.+)$")
        count = 1
        prev_ids: List[str] = []
        for line in text.splitlines():
            m = line_re.match(line)
            if not m:
                continue
            sid = f"task_{count}"
            nodes[sid] = PlanNode(
                node_id=sid,
                description=m.group(1).strip(),
                dependencies=prev_ids[-1:] if prev_ids else [],
                assigned_loops=2,
            )
            prev_ids.append(sid)
            count += 1
            if count > max_subtasks:
                break
        return nodes

    def _optimize(self, nodes: Dict[str, PlanNode], strategy: PlanningStrategy) -> Dict[str, PlanNode]:
        """Basic optimization: cap loops and ensure root exists."""
        if "task_0" not in nodes:
            root_desc = "Overall task execution"
            nodes["task_0"] = PlanNode(
                node_id="task_0",
                description=root_desc,
                dependencies=[],
                assigned_loops=2,
            )

        for nid, node in nodes.items():
            # Keep loop assignments modest.
            node.assigned_loops = max(1, min(node.assigned_loops, 6))
            # Sequential fallback for unknown strategies.
            if strategy == PlanningStrategy.SEQUENTIAL and node.dependencies:
                pass
        return nodes

    def _fallback_plan(self, task: str) -> Plan:
        """Minimal fallback plan when LLM planning fails."""
        return Plan({
            "task_0": PlanNode("task_0", task, assigned_loops=2),
            "task_1": PlanNode("task_1", "Execute the task and verify the result", dependencies=["task_0"], assigned_loops=2),
        })


def create_simple_plan(task: str) -> Plan:
    """Create a lightweight plan for file/folder analysis tasks without LLM."""
    task_lower = task.lower()
    nodes: Dict[str, PlanNode] = {
        "task_0": PlanNode("task_0", f"Locate and identify the target for: {task[:120]}", assigned_loops=1),
        "task_1": PlanNode("task_1", "Gather relevant files and context", dependencies=["task_0"], assigned_loops=2),
        "task_2": PlanNode("task_2", "Analyze contents and synthesize the answer", dependencies=["task_1"], assigned_loops=2),
    }

    if any(kw in task_lower for kw in ["fix", "修改", "修复", "apply", "patch"]):
        nodes["task_3"] = PlanNode(
            "task_3",
            "Apply the required changes and verify syntax",
            dependencies=["task_2"],
            assigned_loops=2,
            tool_hint="file_ops",
        )
    elif any(kw in task_lower for kw in ["test", "测试", "run", "执行"]):
        nodes["task_3"] = PlanNode(
            "task_3",
            "Run tests or commands to verify behavior",
            dependencies=["task_2"],
            assigned_loops=2,
            tool_hint="bash_exec",
        )

    return Plan(nodes)


# Backwards-compatible aliases used by older tests and external callers.
TaskNode = PlanNode


class DynamicReplanner:
    """运行时根据工具观察结果动态修订 DAG 计划.

    触发: 当最近一次工具返回呈现"前置条件失败"(文件不存在/无权限/模块缺失/
    404/连接拒绝等)时, 调用 LLM 基于当前计划与失败观察提出修订(增/删/重排节点),
    并就地生成新的 Plan。修订仅在确实出现失败信号时发生, 且受 max_replans 预算限制。
    """

    FAILURE_SIGNALS = (
        "no such file", "not found", "does not exist", "permission denied",
        "modulenotfounderror", "importerror", "filenotfounderror", "404",
        "connection refused", "command not found", "拒绝访问", "找不到",
        "不存在", "denied", "traceback (most recent call last)",
    )

    def __init__(self, model_backend: Any, config: Any = None, max_replans: int = 2):
        self.model = model_backend
        self.config = config
        self.max_replans = max_replans

    def should_replan(self, observations: List[str]) -> bool:
        """Heuristic: 最近观察是否暴露了前置条件失败."""
        if not observations:
            return False
        recent = " ".join(observations[-2:]).lower()
        return any(sig in recent for sig in self.FAILURE_SIGNALS)

    def _build_replan_prompt(self, plan: "Plan", task: str, broken: str) -> str:
        plan_md = plan.to_markdown()
        return (
            "You are a planning adapter. The execution of the following plan hit a blocker.\n\n"
            f"TASK: {task}\n\n"
            f"CURRENT PLAN:\n{plan_md}\n\n"
            f"BLOCKER OBSERVATION:\n{broken}\n\n"
            "Propose minimal changes so the plan can proceed. You may add prerequisite/recovery "
            "steps, remove impossible steps, or reorder. Keep existing node ids that stay.\n"
            "Return ONLY a JSON object:\n"
            '{"analysis":"...",'
            '"add":[{"id":"task_N","description":"...","dependencies":["task_0"],"tool_hint":"..."}],'
            '"remove":["task_X"],'
            '"note":"short summary of changes"}\n'
            "Use new unique ids (e.g. task_7) for added nodes."
        )

    def replan(self, plan: "Plan", task: str, broken: str) -> "tuple":
        """Return (updated_plan, note). Falls back to (original_plan, '') on any failure."""
        # 快速路径: 高置信度失败模式 → 直接套用确定性修订, 免一次 LLM 往返(显著缩短重试耗时)
        fast = self._fast_fix(plan, broken)
        if fast is not None:
            return fast
        prompt = self._build_replan_prompt(plan, task, broken)
        try:
            raw = self.model.generate(prompt, n_loops=1, temperature=0.2, max_tokens=350)
        except Exception:
            return plan, ""
        if isinstance(raw, str):
            text = raw
        else:
            try:
                text = self.model.tokenizer.decode(raw)
            except Exception:
                text = str(raw)
        try:
            s = text.find("{")
            e = text.rfind("}") + 1
            data = json.loads(text[s:e]) if s != -1 and e > s else {}
        except Exception:
            return plan, ""
        if not isinstance(data, dict):
            return plan, ""
        new_plan = self._apply_changes(plan, data)
        note = data.get("note", "plan adjusted based on observation")
        return new_plan, note

    def _fast_fix(self, plan: "Plan", broken: str):
        """对常见失败做零延迟的确定性修订; 无法高置信匹配时返回 None 走 LLM 路径."""
        b = (broken or "").lower()
        additions = []
        if "command not found" in b:
            additions.append(("task_path", "Locate the command and fix PATH or use its absolute path", "bash"))
        elif any(s in b for s in ("no such file", "file not found", "找不到", "不存在", "filenotfounderror", "no such directory")):
            additions.append(("task_locate", "Locate the target path with glob/grep, then read the correct file", "glob"))
        elif any(s in b for s in ("modulenotfounderror", "importerror", "module not found", "无法导入")):
            additions.append(("task_dep", "Verify the dependency is installed (pip/venv/PATH) and the import path is correct", "bash"))
        elif any(s in b for s in ("permission denied", "拒绝访问", "denied")):
            additions.append(("task_perm", "Fix file/command permissions (chmod) or run with the correct user", "bash"))
        if not additions:
            return None
        import copy
        nodes = {nid: copy.copy(n) for nid, n in plan.nodes.items()}
        existing = set(nodes.keys())
        used = []
        for nid, desc, hint in additions:
            rid = nid
            while rid in existing:
                rid = rid + "_"
            nodes[rid] = PlanNode(node_id=rid, description=desc, dependencies=[], assigned_loops=2, tool_hint=hint)
            existing.add(rid)
            used.append(rid)
        note = "auto recovery (no LLM): " + "; ".join(f"added {rid}" for rid in used)
        return Plan(nodes), note

    def _apply_changes(self, plan: "Plan", data: Dict[str, Any]) -> "Plan":
        import copy

        nodes = {nid: copy.copy(n) for nid, n in plan.nodes.items()}
        for rid in data.get("remove", []) or []:
            nodes.pop(rid, None)
        # 清理指向已删除节点的悬挂依赖
        existing = set(nodes.keys())
        for n in nodes.values():
            n.dependencies = [d for d in n.dependencies if d in existing]
        # 新增节点(确保 id 唯一)
        for item in data.get("add", []) or []:
            nid = item.get("id")
            if not nid or nid in nodes:
                continue
            deps = [d for d in item.get("dependencies", []) if d in existing]
            nodes[nid] = PlanNode(
                node_id=nid,
                description=item.get("description", "recovery step"),
                dependencies=deps,
                assigned_loops=int(item.get("loops", 2)),
                tool_hint=item.get("tool_hint"),
            )
        return Plan(nodes)
