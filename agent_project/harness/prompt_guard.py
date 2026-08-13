"""Light-weight prompt-injection and context-fencing guard.

This is a deterministic first line of defence, not a full adversarial
classifier. It looks for common jailbreak markers, system-prompt leakage
patterns, and fence-breaking newlines inside tool arguments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    triggered: bool = False
    matched_pattern: str = ""
    category: str = ""
    text_sample: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "matched_pattern": self.matched_pattern,
            "category": self.category,
            "text_sample": self.text_sample[:200],
        }


class PromptGuard:
    """Simple rule-based prompt injection detector."""

    # Patterns that attempt to override system instructions.
    _JAILBREAK_PATTERNS = [
        (r"ignore\s+(?:previous\s+)?instructions", "jailbreak"),
        (r"ignore\s+(?:the\s+)?system\s+prompt", "jailbreak"),
        (r"you\s+are\s+now\s+(?:DAN|dan)", "jailbreak"),
        (r"do\s+anything\s+now", "jailbreak"),
        (r"developer\s+mode", "jailbreak"),
        (r"jailbreak", "jailbreak"),
        (r"\"\s*from\s+now\s+on\s*\",?\s*you\s+are", "jailbreak"),
        (r"let\s+me\s+be\s+clear", "jailbreak"),
        (r"this\s+is\s+for\s+educational\s+purposes", "jailbreak"),
    ]

    # Patterns that try to close a memory/role fence from inside user content.
    _FENCE_BREAKERS = [
        (r"</\s*memory-context\s*>", "fence-break"),
        (r"</\s*system\s*>", "fence-break"),
        (r"\n\s*Tools:\s*", "fence-break"),
        (r"\n\s*System:\s*", "fence-break"),
        (r"\n\s*Assistant:\s*", "fence-break"),
    ]

    # Patterns inside tool arguments that look like injection attempts.
    _TOOL_ARG_PATTERNS = [
        (r";\s*rm\s+-rf", "tool-arg-injection"),
        (r"\|\s*(?:bash|sh|zsh)\s*", "tool-arg-injection"),
        (r"`\s*rm\s", "tool-arg-injection"),
        (r"\$\(\s*rm\s", "tool-arg-injection"),
    ]

    def __init__(self):
        self._patterns: List[tuple] = (
            self._JAILBREAK_PATTERNS
            + self._FENCE_BREAKERS
            + self._TOOL_ARG_PATTERNS
        )

    def scan(self, text: str) -> GuardResult:
        """Scan text; return GuardResult with the first match."""
        if not text:
            return GuardResult()
        lowered = text.lower()
        for raw_pattern, category in self._patterns:
            if re.search(raw_pattern, lowered, re.IGNORECASE):
                sample = text[max(0, lowered.find(re.search(raw_pattern, lowered).group(0)) - 40):][:120]
                return GuardResult(
                    triggered=True,
                    matched_pattern=raw_pattern,
                    category=category,
                    text_sample=sample,
                )
        return GuardResult()

    def scan_tool_args(self, tool_name: str, arguments: Dict[str, Any]) -> GuardResult:
        """Scan serialized tool arguments."""
        import json
        text = json.dumps(arguments, ensure_ascii=False, default=str)
        result = self.scan(text)
        if result.triggered:
            result.text_sample = f"[{tool_name}] {result.text_sample}"
        return result


def guard_input(text: str, source: str = "user") -> GuardResult:
    """Convenience guard for raw user input."""
    guard = PromptGuard()
    result = guard.scan(text)
    if result.triggered:
        logger.warning(
            "Prompt guard triggered for %s input: category=%s pattern=%r",
            source,
            result.category,
            result.matched_pattern,
        )
    return result
