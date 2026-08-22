"""
Topology-based strategy and tool scheduler.

Nodes represent capabilities: strategies, tools, memory ops.
Edges represent dependencies with Captain OS weights as heuristics.
Supports DAG traversal, scoring, parallel execution groups, and rollback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable
from collections import defaultdict, deque
import heapq

@dataclass
class Node:
    id: str
    type: str  # "strategy"|"tool"|"memory"
    name: str
    weight: float = 1.0
    cost: float = 1.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

@dataclass
class Edge:
    src: str
    dst: str
    relation: str = "depends"  # depends | enables | prefers
    weight: float = 1.0

class TopologyGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, List[Edge]] = defaultdict(list)
        self.rev_adj: Dict[str, List[Edge]] = defaultdict(list)

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, src: str, dst: str, relation: str = "depends", weight: float = 1.0):
        e = Edge(src, dst, relation, weight)
        self.adj[src].append(e)
        self.rev_adj[dst].append(e)

    def topological_order(self) -> List[str]:
        indeg = {nid: 0 for nid in self.nodes}
        for src, edges in self.adj.items():
            for e in edges:
                indeg[e.dst] += 1
        q = deque([n for n, d in indeg.items() if d == 0])
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for e in self.adj.get(n, []):
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    q.append(e.dst)
        return order

    def reachable_from(self, start_ids: Set[str]) -> Set[str]:
        visited = set()
        stack = list(start_ids)
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            for e in self.adj.get(n, []):
                if e.relation in ("depends", "enables"):
                    stack.append(e.dst)
        return visited

class TopologyScheduler:
    def __init__(self, graph: TopologyGraph, captain_rules: Optional[Dict] = None):
        self.graph = graph
        self.captain_rules = captain_rules or {}
        self.execution_log: List[Tuple[str, str]] = []

    def score_node(self, node: Node, task: str, context: Dict) -> float:
        base = node.weight / max(0.1, node.cost)
        # Captain OS heuristic: turn slow / straight fast
        heuristic = 1.0
        if "critical" in node.tags or "strategy" in node.type:
            heuristic *= self.captain_rules.get("slow_turn", 1.2)
        if "tool" in node.type and "quick" in node.tags:
            heuristic *= self.captain_rules.get("fast_straight", 1.3)
        # task keyword overlap
        task_low = task.lower()
        overlap = sum(1 for t in node.tags if t in task_low)
        overlap_bonus = 1 + 0.1 * overlap
        return base * heuristic * overlap_bonus

    def select_plan(self, task: str, entry_points: List[str], context: Dict) -> List[List[str]]:
        # Build subgraph reachable from entries
        reachable = self.graph.reachable_from(set(entry_points))
        # Score nodes
        scored = []
        for nid in reachable:
            node = self.graph.nodes[nid]
            s = self.score_node(node, task, context)
            scored.append((s, nid))
        scored.sort(reverse=True)
        # Group into parallel layers by topological order
        order = [n for n in self.graph.topological_order() if n in reachable]
        layers: List[List[str]] = []
        seen = set()
        for nid in order:
            # collect nodes whose dependencies are satisfied
            deps = {e.src for e in self.graph.rev_adj.get(nid, []) if e.relation == "depends"}
            if deps.issubset(seen):
                # allow parallelism within same layer
                # simple bucket by depth
                layer_idx = 0
                # find layer index based on max depth of deps
                # For simplicity, place node in first layer where deps satisfied
                placed = False
                for i, layer in enumerate(layers):
                    if all(d in seen for d in deps):
                        layer.append(nid)
                        placed = True
                        break
                if not placed:
                    layers.append([nid])
                seen.add(nid)
        # If no layering worked, fallback to scored order
        if not layers:
            layers = [[nid] for _, nid in scored]
        return layers

    def execute_plan(self, plan: List[List[str]], executor: Callable[[str], bool]) -> Dict:
        results = {"success": True, "executed": [], "failed": []}
        for layer in plan:
            # parallel layer
            layer_results = []
            for nid in layer:
                try:
                    ok = executor(nid)
                    results["executed"].append(nid)
                    self.execution_log.append((nid, "ok" if ok else "fail"))
                    if not ok:
                        results["failed"].append(nid)
                except Exception:
                    results["failed"].append(nid)
                    results["success"] = False
            if results["failed"]:
                # rollback: execute reverse order of successful nodes
                for nid in reversed(results["executed"]):
                    try:
                        executor(f"rollback:{nid}")
                    except Exception:
                        pass
                break
        return results
