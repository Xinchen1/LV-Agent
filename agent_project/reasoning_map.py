from .reasoning import ReasoningStrategy

NODE_TO_REASONING = {
    "strat_fast": ReasoningStrategy.ZERO_SHOT,
    "strat_deep": ReasoningStrategy.SUPER_AGENT,
    "strat_mcts": ReasoningStrategy.MONTE_CARLO,
    "strat_code": ReasoningStrategy.SUPER_AGENT,
}

def map_node_to_reasoning(node_id: str) -> ReasoningStrategy:
    return NODE_TO_REASONING.get(node_id, ReasoningStrategy.SUPER_AGENT)
