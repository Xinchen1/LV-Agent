"""
Minimal evaluation harness for LV Agent.

Provides:
- Online metrics logging for strategies, tool usage, summary validation
- Simple decayed experience scoring
- Explainable tool selection
"""

from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List

_EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_log.jsonl"

def log_episode(strategy: str, tools_used: List[str], summary_ok: bool, latency_ms: int, tokens: int) -> None:
    rec = {
        "ts": int(time.time()),
        "strategy": strategy,
        "tools": tools_used,
        "summary_ok": bool(summary_ok),
        "latency_ms": latency_ms,
        "tokens": tokens,
    }
    try:
        with open(_EVAL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def summary_quality_check(text: str, required_sections: List[str]) -> bool:
    if not text:
        return False
    # Very simple heuristic: section headers present
    lowered = text.lower()
    hits = sum(1 for sec in required_sections if sec in text)
    return hits >= len(required_sections) * 0.6

def decay_score(base_score: float, age_days: float, half_life: float = 30.0) -> float:
    import math
    return base_score * (0.5 ** (age_days / half_life))
