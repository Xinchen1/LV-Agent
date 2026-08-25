"""Tests for display-layer filtering: tool-result compression and leaked-English suppression."""

import json

from agent_project.response_filter import is_leaked_self_talk
from agent_project.stream_adapters import RichStreamAdapter, clean_content_text


def test_is_leaked_self_talk_markers():
    assert is_leaked_self_talk("We need to answer the user's question.")
    assert is_leaked_self_talk("The user asks for the capital of France.")
    assert is_leaked_self_talk("Let me think about this step by step.")
    assert is_leaked_self_talk("  According to the docs, this is wrong.")


def test_is_leaked_self_talk_keeps_chinese_and_natural_english():
    # Chinese content must never be flagged
    assert not is_leaked_self_talk("我目前没有独立的子agent来进行监督。")
    # Natural English answer lines (not meta-reasoning) should pass through
    assert not is_leaked_self_talk("I will help you set up the environment.")
    assert not is_leaked_self_talk("We provide a comprehensive guide below.")
    # Code / URL lines are not self-talk
    assert not is_leaked_self_talk("import os")
    assert not is_leaked_self_talk("https://example.com/some/path")


def test_clean_content_text_drops_leaked_line():
    raw = "We need to answer the user's question.\n我目前没有独立的子agent来进行监督。"
    out = clean_content_text(raw)
    assert "We need to answer" not in out
    assert "我目前没有独立的子agent来进行监督。" in out


def test_clean_content_text_keeps_partial_fragment():
    # A partial (no trailing newline) fragment is preserved until completed.
    out = clean_content_text("We need to answer")
    assert out == "We need to answer"


def test_clean_content_text_keeps_english_answer():
    # Pure-English answer (no following Chinese) must NOT be dropped as a leak.
    raw = (
        "To answer your question, quantum computing uses qubits.\n"
        "It is fundamentally different from classical computing."
    )
    out = clean_content_text(raw)
    assert "To answer your question, quantum computing uses qubits." in out
    assert "It is fundamentally different from classical computing." in out


def test_clean_content_text_drops_leak_only_before_chinese():
    # The leak line is dropped only because the NEXT line switches to Chinese.
    raw = "We need to answer the user's question.\n我目前没有独立的子agent。"
    out = clean_content_text(raw)
    assert "We need to answer" not in out
    assert "我目前没有独立的子agent。" in out
    # If the very next line is also English, it is NOT treated as a leak.
    raw2 = "We need to answer the user's question.\nHere is the result you asked for."
    out2 = clean_content_text(raw2)
    assert "We need to answer the user's question." in out2


def test_compact_tool_result_web_search():
    adapter = RichStreamAdapter(console=object())
    adapter._last_tool_name = "web_search"
    results = [
        {"title": "实在智能", "url": "https://www.example.com/" + "a" * 80, "snippet": "x"},
        {"title": "Another result", "url": "https://news.example.org/b"},
    ]
    text = json.dumps(results, ensure_ascii=False)
    out = adapter._compact_tool_result(text)
    assert out != text
    assert "实在智能" in out
    assert "Another result" in out
    # scheme stripped + truncated
    assert "https://" not in out
    assert "example.com" in out


def test_compact_tool_result_non_json_passthrough():
    adapter = RichStreamAdapter(console=object())
    adapter._last_tool_name = "bash_exec"
    text = "hello world\nsecond line"
    assert adapter._compact_tool_result(text) == text
