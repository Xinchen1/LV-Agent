#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research





from types import SimpleNamespace

from agent_project.harness.context import LLMCompactor, TruncatingCompactor, messages_tokens
from agent_project.harness.runner import run_task


def _msgs(n, width=300):
    base = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    for i in range(n):
        base.append({"role": "user", "content": f"[obs:grep] {'y' * width} {i}"})
    return base


# ---------- LLMCompactor ----------

def test_llm_compactor_uses_summary():
    seen = []

    def summarize(prompt):
        seen.append(prompt)
        return "read config; found bug in parser; todo: fix tests"

    compactor = LLMCompactor(summarize=summarize, keep_tail=4)
    rewritten, summary = compactor(_msgs(16), budget_tokens=400)
    assert seen, "summarizer was invoked"
    assert summary.startswith("read config")
    assert any("summary of earlier work" in str(m["content"]) for m in rewritten)
    assert rewritten[-1]["content"].endswith("15")


def test_llm_compactor_falls_back_on_exception():
    def boom(_prompt):
        raise RuntimeError("llm down")

    compactor = LLMCompactor(summarize=boom, keep_tail=4,
                             fallback=TruncatingCompactor(keep_tail=4))
    rewritten, summary = compactor(_msgs(16), budget_tokens=400)
    assert "compacted" in summary  # truncating stub, not an exception
    assert len(rewritten) < 18


def test_llm_compactor_falls_back_when_summary_too_fat():
    compactor = LLMCompactor(summarize=lambda p: "z" * 100_000, keep_tail=4,
                             max_summary_chars=100_000,
                             fallback=TruncatingCompactor(keep_tail=4))
    rewritten, summary = compactor(_msgs(16), budget_tokens=200)
    assert "compacted" in summary
    assert messages_tokens(rewritten) < messages_tokens(_msgs(16))


# ---------- run_task ----------

class _FakeBackend:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, prompt, n_loops=1, temperature=None, max_tokens=None,
                 tools=None, stream=False, stream_callback=None,
                 token_callback=None, **kwargs):
        if token_callback:
            token_callback(50)
        return self.replies.pop(0)


def test_run_task_end_to_end_result_shape(tmp_path):
    backend = _FakeBackend([
        'checking [TOOL:calculator] {"expression": "20+22"} [/TOOL]',
        "the answer is 42",
    ])
    stream, tokens = [], []
    result = run_task(
        "what is 20+22?",
        backend,
        harness_cfg=SimpleNamespace(policy="permissive", workspace_root=None),
        stream_callback=lambda kind, text: stream.append((kind, text)),
        token_callback=tokens.append,
        session_root=tmp_path,
    )
    assert result["final_answer"] == "the answer is 42"
    assert result["mode"] == "harness"
    assert result["metadata"]["status"] == "completed"
    assert result["metadata"]["tool_calls"] == 1
    assert result["session_token_usage"]["last_call_tokens"] > 0
    assert tokens  # token callback fired
    assert any(kind == "tool_call" for kind, _ in stream)
    assert any(kind == "content" for kind, _ in stream)
    assert (tmp_path / f"{result['metadata']['session_id']}.jsonl").exists()


def test_run_task_policy_denial_reaches_model(tmp_path):
    backend = _FakeBackend([
        '[TOOL:bash_exec] {"command": "sudo rm -rf /"} [/TOOL]',
        "policy refused; answering directly",
    ])
    result = run_task(
        "delete everything",
        backend,
        harness_cfg=SimpleNamespace(policy="safe", workspace_root=str(tmp_path)),
        session_root=tmp_path,
    )
    assert result["final_answer"] == "policy refused; answering directly"
    assert result["metadata"]["denials"] == 1


def test_run_task_failure_is_captured_not_raised(tmp_path):
    backend = _FakeBackend([])  # empty -> pop(0) raises IndexError
    result = run_task(
        "boom", backend,
        harness_cfg=SimpleNamespace(policy="permissive", workspace_root=None),
        session_root=tmp_path,
    )
    assert result["metadata"]["status"] == "failed"
    assert "harness run failed" in result["final_answer"]
