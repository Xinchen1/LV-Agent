from .topology_scheduler import TopologyGraph, Node

def build_default_graph() -> TopologyGraph:
    g = TopologyGraph()
    # Strategies
    g.add_node(Node(id="strat_fast", type="strategy", name="FastDirect", weight=1.2, cost=0.8, tags={"quick","answer"}))
    g.add_node(Node(id="strat_deep", type="strategy", name="DeepReason", weight=1.5, cost=1.5, tags={"critical","strategy","slow_turn"}))
    g.add_node(Node(id="strat_mcts", type="strategy", name="MCTS", weight=1.8, cost=1.6, tags={"critical","strategy","decision"}))

    # Tools
    g.add_node(Node(id="tool_web", type="tool", name="web_search", weight=1.1, cost=1.0, tags={"quick","search"}))
    g.add_node(Node(id="tool_file", type="tool", name="file_ops", weight=1.0, cost=0.7, tags={"quick","read"}))
    g.add_node(Node(id="tool_code", type="tool", name="code_exec", weight=1.3, cost=1.2, tags={"tool","code"}))

    # Memory
    g.add_node(Node(id="mem_skill", type="memory", name="memskill", weight=1.0, cost=0.9, tags={"memory"}))

    # Dependencies
    g.add_edge("strat_deep", "tool_web", relation="enables", weight=1.0)
    g.add_edge("strat_mcts", "tool_web", relation="enables", weight=1.0)
    g.add_edge("strat_mcts", "tool_file", relation="enables", weight=0.8)
    g.add_edge("tool_web", "mem_skill", relation="depends", weight=1.0)
    g.add_edge("strat_fast", "tool_file", relation="enables", weight=0.9)

    return g
