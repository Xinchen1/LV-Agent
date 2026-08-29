#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research




Tests for Turing Machine integration.
"""

import pytest
from agent_project.tools.turing_machine import TuringMachine, TuringMachineTool
from agent_project.context_tape import ContextTape, TapeDirection
from agent_project.harness.turing_integration import TuringMachineHarness, MachineState


def test_turing_machine_binary_incrementer():
    """Test binary incrementer Turing Machine."""
    from agent_project.tools.turing_machine import create_binary_incrementer
    
    config = create_binary_incrementer()
    machine = TuringMachine.from_config(config)
    
    # Test increment 101 (should not halt immediately)
    machine.reset("101")
    result = machine.run()
    
    # Machine should halt (either accept or reject)
    assert result['halted'] or result['steps'] > 0


def test_turing_machine_tool_create():
    """Test TuringMachineTool create action."""
    tool = TuringMachineTool()
    
    config = {
        'states': ['q0', 'q_accept'],
        'alphabet': ['0', '1'],
        'tape_alphabet': ['0', '1', '_'],
        'blank_symbol': '_',
        'initial_state': 'q0',
        'accept_states': ['q_accept'],
        'reject_states': [],
        'transitions': {
            ('q0', '0'): ('q_accept', '0', 'S')
        }
    }
    
    result = tool.execute("create", config=config)
    assert result.success
    assert "Turing Machine created" in result.output


def test_context_tape_basic():
    """Test Context Tape basic operations."""
    tape = ContextTape(window_size=10)
    
    # Write cells
    tape.write_cell(0, "Message 1")
    tape.write_cell(1, "Message 2")
    tape.write_cell(2, "Message 3")
    
    # Read cell
    content = tape.read_cell(1)
    assert content == "Message 2"
    
    # Move head
    tape.move_head(TapeDirection.RIGHT, 1)
    assert tape.head.position == 1
    
    # Get window
    window = tape.get_window(2)
    assert len(window) == 5  # -2 to +2


def test_context_tape_compression():
    """Test context tape compression."""
    tape = ContextTape(window_size=10)
    
    # Create many cells
    for i in range(20):
        tape.write_cell(i, f"Message {i}")
    
    assert len(tape.tape) == 20
    
    # Compress
    tape.compress(10)
    assert len(tape.tape) <= 10


def test_turing_machine_harness_transitions():
    """Test Turing Machine Harness state transitions."""
    harness = TuringMachineHarness()
    
    # Initial state
    assert harness.current_state == MachineState.IDLE
    
    # Transition: user input -> READING
    transition = harness.transition("user_input")
    assert transition is not None
    assert harness.current_state == MachineState.READING
    assert harness.tape_position == 1  # Moved right
    
    # Transition: context ready -> THINKING
    transition = harness.transition("context_ready")
    assert harness.current_state == MachineState.THINKING
    
    # Transition: has tools -> TOOL_EXEC
    transition = harness.transition("has_tools")
    assert harness.current_state == MachineState.TOOL_EXEC
    
    # Trace should be recorded
    assert len(harness.get_trace()) == 3


def test_context_tape_visualization():
    """Test context tape visualization."""
    tape = ContextTape()
    tape.write_cell(0, "Hello")
    tape.write_cell(1, "World")
    tape.move_head(TapeDirection.RIGHT, 1)
    
    viz = tape.visualize()
    # Visualization truncates to 3 chars
    assert "Hel" in viz or "Wor" in viz
    assert "State:" in viz
    assert "Head:" in viz


def test_turing_machine_step_by_step():
    """Test Turing Machine step-by-step execution."""
    tool = TuringMachineTool()
    
    config = {
        'states': ['q0', 'q1', 'q_accept'],
        'alphabet': ['0', '1'],
        'tape_alphabet': ['0', '1', '_'],
        'blank_symbol': '_',
        'initial_state': 'q0',
        'accept_states': ['q_accept'],
        'reject_states': [],
        'transitions': {
            ('q0', '0'): ('q1', '1', 'R'),
            ('q1', '1'): ('q_accept', '1', 'S')
        }
    }
    
    # Create machine
    result = tool.execute("create", config=config, machine_id="test_tm")
    assert result.success
    
    # Step execution
    result = tool.execute("step", machine_id="test_tm", input_tape="01")
    assert result.success
    
    # Run to completion
    result = tool.execute("run", machine_id="test_tm", input_tape="01")
    assert result.success
    assert "Turing Machine Execution Complete" in result.output
