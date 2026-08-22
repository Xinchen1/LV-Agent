#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/mac/Desktop/agent_project')
from agent_project.topology_builder import build_default_graph
from agent_project.topology_scheduler import TopologyScheduler

def test_plan():
    graph = build_default_graph()
    scheduler = TopologyScheduler(graph, {"slow_turn":1.2, "fast_straight":1.3})
    task = "请做一个关键战略决策，分析要不要投资AI"
    plan = scheduler.select_plan(task, ["strat_mcts"], {})
    print("Plan layers:", plan)
    # verify strategy node present
    nodes = [nid for layer in plan for nid in layer]
    assert "strat_mcts" in nodes, "MCTS should be selected"
    assert "tool_web" in nodes or "tool_file" in nodes, "Tool should be reachable"
    print("Test passed")

if __name__ == "__main__":
    test_plan()
