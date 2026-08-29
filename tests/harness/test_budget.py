#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research





import pytest

from agent_project.harness.budget import Ledger, Limits
from agent_project.harness.errors import BudgetExhaustedError, StagnationError


def test_token_limit_trips():
    ledger = Ledger(Limits(max_tokens=100))
    ledger.consume_tokens(60)
    with pytest.raises(BudgetExhaustedError):
        ledger.consume_tokens(50)
    assert ledger.remaining_tokens() == 0


def test_tool_call_and_turn_limits():
    ledger = Ledger(Limits(max_tool_calls=2, max_turns=1))
    ledger.consume_tool_call()
    ledger.consume_tool_call()
    with pytest.raises(BudgetExhaustedError):
        ledger.consume_tool_call()

    turns = Ledger(Limits(max_turns=1))
    turns.consume_turn()
    with pytest.raises(BudgetExhaustedError):
        turns.consume_turn()


def test_consecutive_error_breaker_and_reset():
    ledger = Ledger(Limits(max_consecutive_errors=3))
    ledger.record_error()
    ledger.record_error()
    ledger.record_success()          # resets the streak
    ledger.record_error()
    ledger.record_error()
    with pytest.raises(StagnationError):
        ledger.record_error()


def test_identical_effect_stagnation_detector():
    ledger = Ledger(Limits(max_identical_effects=3))
    ledger.record_effect("k1")
    ledger.record_effect("k2")
    ledger.record_effect("k2")
    with pytest.raises(StagnationError) as exc:
        ledger.record_effect("k2")
    assert "k2" in str(exc.value)


def test_distinct_effects_do_not_trip():
    ledger = Ledger(Limits(max_identical_effects=3))
    for key in ("a", "b", "c", "a", "b", "c"):
        ledger.record_effect(key)  # no raise


def test_unlimited_ledger_never_trips():
    ledger = Ledger(Limits())
    for _ in range(1000):
        ledger.consume_tokens(10)
        ledger.consume_tool_call()
    assert ledger.tokens == 10000


def test_snapshot_shape():
    snap = Ledger(Limits()).snapshot()
    assert set(snap) == {"tokens", "tool_calls", "cost_usd", "turns",
                         "elapsed_s", "consecutive_errors"}
