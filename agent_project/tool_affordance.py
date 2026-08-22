"""
Lightweight tool affordance model for LV Agent.

Provides:
- Static metadata per tool: capabilities, input/output types, cost, reliability priors.
- Task -> tool relevance scoring using keyword overlap + heuristics.
- Simple reliability tracking from execution history.

No external model required. Can be upgraded to LLM-based scorer later.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import re

# Static affordance metadata. Extend as tools grow.
TOOL_META: Dict[str, Dict] = {
    "web_search": {
        "capabilities": ["search", "information_retrieval"],
        "input_keywords": ["搜索", "查找", "查一下", "搜一下", "网上", "最新", "资讯", "新闻", "价格", "官网"],
        "output_type": "text",
        "cost": 2,
        "reliability": 0.75,
    },
    "file_ops": {
        "capabilities": ["read", "write", "list", "grep"],
        "input_keywords": ["文件", "打开", "读取", "写入", "列表", "ls", "文档", "pdf", "md", "代码", "查看"],
        "output_type": "file",
        "cost": 1,
        "reliability": 0.9,
    },
    "python_exec": {
        "capabilities": ["compute", "code"],
        "input_keywords": ["计算", "算一下", "python", "代码", "脚本", "执行", "求"],
        "output_type": "text",
        "cost": 2,
        "reliability": 0.85,
    },
    "bash_exec": {
        "capabilities": ["shell", "system"],
        "input_keywords": ["命令", "bash", "shell", "执行命令", "终端", "!"],
        "output_type": "text",
        "cost": 2,
        "reliability": 0.8,
    },
    "calculator": {
        "capabilities": ["math"],
        "input_keywords": ["计算", "等于", "+", "-", "*", "/", "求值"],
        "output_type": "number",
        "cost": 0,
        "reliability": 0.99,
    },
    "pdf_tool": {
        "capabilities": ["pdf_generate"],
        "input_keywords": ["pdf", "生成pdf", "导出pdf", "报告pdf"],
        "output_type": "file",
        "cost": 2,
        "reliability": 0.85,
    },
    "github_search": {
        "capabilities": ["search", "code_search"],
        "input_keywords": ["github", "代码库", "仓库", "repo"],
        "output_type": "text",
        "cost": 2,
        "reliability": 0.7,
    },
}

# Simple reliability tracker in memory
_reliability_log: Dict[str, List[bool]] = {}

def record_tool_result(tool_name: str, success: bool) -> None:
    log = _reliability_log.setdefault(tool_name, [])
    log.append(success)
    # keep last 100
    if len(log) > 100:
        del log[0]

def empirical_reliability(tool_name: str) -> float:
    log = _reliability_log.get(tool_name)
    if not log:
        return TOOL_META.get(tool_name, {}).get("reliability", 0.5)
    return sum(log) / len(log)

def score_tools(task: str, candidate_tools: List[str]) -> List[Tuple[str, float, str]]:
    """
    Score tools for a given task string.
    Returns list of (tool_name, score, explanation) sorted descending.
    """
    task_lc = task.lower()
    scores = []
    for name in candidate_tools:
        meta = TOOL_META.get(name, {})
        kw_hits = [kw for kw in meta.get("input_keywords", []) if kw in task_lc]
        kw_score = min(1.0, len(kw_hits) / 3.0)
        cost = meta.get("cost", 1)
        reliability = empirical_reliability(name)
        score = 0.5 * kw_score + 0.3 * reliability - 0.1 * cost
        expl = f"kw_hits={kw_hits}, reliability={reliability:.2f}, cost={cost}"
        scores.append((name, max(0.0, score), expl))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def filter_tools_by_affordance(task: str, candidate_tools: List[str], top_k: int = 5, min_score: float = 0.15) -> List[str]:
    scored = score_tools(task, candidate_tools)
    # Log explainable selection
    for name, s, expl in scored[:top_k]:
        # In production, emit to logger
        pass
    filtered = [name for name, s, _ in scored if s >= min_score][:top_k]
    if re.search(r"文件|打开|读取|pdf|文档", task) and "file_ops" not in filtered:
        filtered.insert(0, "file_ops")
    return filtered
