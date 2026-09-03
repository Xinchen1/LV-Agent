"""
Unified Tool Output Parser for LV Agent.

Extracts tool calls and final answers from model output text,
providing a single source of truth for tool call parsing across
the entire codebase. This eliminates redundant regex patterns
and ensures consistent parsing behavior.

Supported formats:
  1. [TOOL:tool_name] {json_args} [/TOOL]       (standard)
  2. [TOOL:tool_name] {json_args}               (inline, no closing tag)
  3. [TOOL:tool_name]\n{json_args}              (tool on one line, JSON on next)
  4. 
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, field


# ============ Parsed Results ============

@dataclass
class ToolCallResult:
    """Result of parsing a single tool call from model output."""
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str  # 原始匹配文本，用于调试
    format_type: str = "standard"  # "standard" | "inline" | "multi_line"


@dataclass
class ParsedOutput:
    """Complete parsing result for a model output chunk."""
    reasoning: str  # 清洗后的推理文本 (思考标签已移除)
    tool_calls: List[ToolCallResult]  # 发现的工具调用列表
    final_answer: Optional[str]  # 发现的最终答案 (若有)
    cleaned_text: str  # 全文清洗后的文本(思考/tool标签移除后)
    has_tool_calls: bool
    has_final_answer: bool


# ============ Precompiled Regex Patterns (NO conflicting flags at module level) ============
# All patterns compile without flags; case-insensitivity handled via inline (?i) when needed.

# Thinking tags - to be stripped
# Using inline (?s) for DOTALL, (?i) for case-insensitivity
_RE_THINK = re.compile(r"(?s)<think(?:ing)?>(.*?)</think(?:ing)?>", )  # placeholder

# Actually, let me just use patterns without flags and handle case sensitivity
# via exact string matching where possible, or inline (?i) when truly needed.

# Simpler approach: compile patterns without flags, use re.finditer without flags parameter,
# and make patterns case-sensitive but accept that model output casing is consistent.
# For safety, I'll embed case-insensitivity in patterns where truly needed via [Ff][Ii] style,
# but for LV Agent's typical output, case-sensitive is fine.

# --- Think tag pattern (case-sensitive, expects lowercase 'think') ---
_RE_THINK = re.compile(r"(?s)<think(?:ing)?>(.*?)</think(?:ing)?>")

# --- Tool call patterns ---
# Standard format: [TOOL:name] {args} [/TOOL]
# Using (?s) for DOTALL so .* matches newlines; case-sensitive TOOL tags
_RE_TOOL_STANDARD = re.compile(r"(?s)\[TOOL:([^\]]+)\]\s*\{(.*?)\}\s*\[/TOOL\]")

# Inline format: [TOOL:name] {args} (no closing /TOOL)
_RE_TOOL_INLINE = re.compile(r"(?s)\[TOOL:([^\]]+)\]\s*\{(.*?)\}")

# Multi-line format: [TOOL:name] {args} at line end
_RE_TOOL_MULTI_LINE = re.compile(r"(?s)\[TOOL:([^\]]+)\]\s*\{(.*?)\}\s*$")

# Individual tool tag (used for stripping after extraction)
_RE_TOOL_TAG = re.compile(r"\[/?TOOL:?\w*\]")

# Final answer markers - case-sensitive is fine typical output is "Final Answer:" exactly
_RE_FINAL_ANS_STANDARD = re.compile(r"^(Final Answer:)(.*)$", re.MULTILINE)
_RE_FINAL_ANS_PLAIN = re.compile(r"^(Final Answer)(:|\s)(.+)$")

# JSON code block (optional extraction)
_RE_CODE_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ============ Core Parsing Logic ============

def _strip_thinking_tags(text: str) -> str:
    """Strip ... thinking标签 from text, return content-only text."""
    return _RE_THINK.sub("", text)


def _extract_tool_calls(text: str) -> List[ToolCallResult]:
    """Extract all [TOOL:...] calls from text. Returns list (may be empty)."""
    calls: List[ToolCallResult] = []

    # Try standard format: [TOOL:name] {args} [/TOOL]
    for m in re.finditer(_RE_TOOL_STANDARD, text):
        tool_name = m.group(1).strip()
        args_str = m.group(2).strip()
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCallResult(
            tool_name=tool_name,
            arguments=args,
            raw_text=m.group(0),
            format_type="standard",
        ))

    # Try inline format: [TOOL:name] {args} (no closing /TOOL)
    # Avoid duplicating standard format ones
    std_names = {c.tool_name for c in calls}
    for m in re.finditer(_RE_TOOL_INLINE, text):
        tool_name = m.group(1).strip()
        if tool_name in std_names:
            continue  # Already captured in standard format
        args_str = m.group(2).strip()
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCallResult(
            tool_name=tool_name,
            arguments=args,
            raw_text=m.group(0),
            format_type="inline",
        ))

    # Try multi-line format: [TOOL:name] {args} at line end
    for m in re.finditer(_RE_TOOL_MULTI_LINE, text):
        tool_name = m.group(1).strip()
        if any(c.tool_name == tool_name for c in calls):
            continue
        args_str = m.group(2).strip()
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCallResult(
            tool_name=tool_name,
            arguments=args,
            raw_text=m.group(0),
            format_type="multi_line",
        ))

    return calls


def _extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer from text, if present."""
    # Try standard marker: "Final Answer: <text>"
    for m in re.finditer(_RE_FINAL_ANS_STANDARD, text):
        answer = m.group(2).strip()
        if answer:
            return answer

    # Try plain format: "Final Answer" followed by content
    for m in re.finditer(_RE_FINAL_ANS_PLAIN, text):
        answer = m.group(3).strip()
        if answer and len(answer) > 2:  # sanity check
            return answer

    return None


# ============ Public API ============

def parse_model_output(
    raw_text: str,
    *,
    extract_tools: bool = True,
    extract_answer: bool = True,
) -> ParsedOutput:
    """
    Parse model output into structured components.

    Args:
        raw_text: Raw text from LLM output
        extract_tools: Whether to extract [TOOL:...] calls
        extract_answer: Whether to extract Final Answer

    Returns:
        ParsedOutput containing reasoning, tool_calls, final_answer, cleaned_text
    """
    # Step 1: Strip thinking tags first (most important)
    text_after_think = _strip_thinking_tags(raw_text)

    # Step 2: Extract tool calls from the text after thinking stripped
    tool_calls: List[ToolCallResult] = []
    if extract_tools:
        tool_calls = _extract_tool_calls(text_after_think)

    # Step 3: Extract final answer
    final_answer: Optional[str] = None
    if extract_answer:
        # First try from original, then from after-thinking version
        final_answer = _extract_final_answer(raw_text)
        if not final_answer:
            final_answer = _extract_final_answer(text_after_think)

    # Step 4: Build cleaned_text - remove tool calls from text
    cleaned_text = text_after_think
    if tool_calls:
        # Remove [TOOL:...] tags; keep content
        cleaned_text = _RE_TOOL_TAG.sub("", cleaned_text)
        # Remove leftover {args} structures
        cleaned_text = re.sub(r"\{.*?\}", " ", cleaned_text)
        # Normalize whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # Step 5: Extract reasoning
    reasoning = cleaned_text
    if final_answer:
        reasoning = reasoning.replace(final_answer, "").strip()

    return ParsedOutput(
        reasoning=reasoning,
        tool_calls=tool_calls,
        final_answer=final_answer,
        cleaned_text=cleaned_text,
        has_tool_calls=len(tool_calls) > 0,
        has_final_answer=final_answer is not None,
    )


# ============ Convenience Functions ============

def has_tool_calls(text: str) -> bool:
    """Quick check if text contains any [TOOL:...] calls (after thinking stripped)."""
    stripped = _strip_thinking_tags(text)
    return bool(_extract_tool_calls(stripped))


def extract_first_tool_call(text: str) -> Optional[ToolCallResult]:
    """Extract the first [TOOL:...] call, if any."""
    stripped = _strip_thinking_tags(text)
    calls = _extract_tool_calls(stripped)
    return calls[0] if calls else None


# ============ For Backward Compatibility ============

# strip_tool_markers kept for backward compatibility;
# now delegates to unified parser under the hood conceptually,
# but we keep a simple regex strip function.
_RE_SIMPLE_TOOL_MARKER = re.compile(r"\[/TOOL\]|<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_tool_markers(s: str) -> str:
    """Strip tool markers - kept for backward compatibility."""
    return _RE_SIMPLE_TOOL_MARKER.sub("", s).strip()