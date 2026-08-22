from .topology_scheduler import TopologyGraph, Node

def build_default_graph() -> TopologyGraph:
    g = TopologyGraph()
    # Strategies with mode/code attributes
    g.add_node(Node(id="strat_fast", type="strategy", name="FastDirect", weight=1.2, cost=0.8, tags={"quick","answer"}, metadata={"mode":"default","code_mode":False}))
    g.add_node(Node(id="strat_deep", type="strategy", name="DeepReason", weight=1.5, cost=1.5, tags={"critical","strategy","slow_turn"}, metadata={"mode":"deep","code_mode":False}))
    g.add_node(Node(id="strat_mcts", type="strategy", name="MCTS", weight=1.8, cost=1.6, tags={"critical","strategy","decision"}, metadata={"mode":"critical","code_mode":False}))
    g.add_node(Node(id="strat_code", type="strategy", name="CodeAgent", weight=1.6, cost=1.4, tags={"code","implement"}, metadata={"mode":"code","code_mode":True}))

    # Tools
    g.add_node(Node(id="tool_web", type="tool", name="web_search", weight=1.1, cost=1.0, tags={"quick","search"}, metadata={"mode":"default"}))
    g.add_node(Node(id="tool_file", type="tool", name="file_ops", weight=1.0, cost=0.7, tags={"quick","read"}, metadata={"mode":"default"}))
    g.add_node(Node(id="tool_code", type="tool", name="code_exec", weight=1.3, cost=1.2, tags={"tool","code"}, metadata={"mode":"code","code_mode":True}))

    # Memory
    g.add_node(Node(id="mem_skill", type="memory", name="memskill", weight=1.0, cost=0.9, tags={"memory"}, metadata={"mode":"default"}))

    # Dependencies
    g.add_edge("strat_deep", "tool_web", relation="enables", weight=1.0)
    g.add_edge("strat_mcts", "tool_web", relation="enables", weight=1.0)
    g.add_edge("strat_mcts", "tool_file", relation="enables", weight=0.8)
    g.add_edge("tool_web", "mem_skill", relation="depends", weight=1.0)
    g.add_edge("strat_fast", "tool_file", relation="enables", weight=0.9)
    g.add_edge("strat_code", "tool_code", relation="enables", weight=1.2)

    return g
