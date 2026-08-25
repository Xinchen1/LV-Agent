"""
Turing Machine Simulator Tool

Implements a universal Turing Machine simulator as a demonstration
of LV Agent's Turing Machine-inspired architecture.

The simulator allows users to:
- Define a Turing Machine with states, symbols, transitions
- Execute the machine step-by-step or to completion
- Visualize the tape and state changes

Design inspiration: LV Agent's architecture is inspired by Turing Machine
with tape memory model → context window management,
state transition function → Agent state management,
universal computation → universal tool execution.
"""

import re
from typing import Dict, List, Optional, Tuple
from . import BaseTool, ToolResult, TOOLS_REGISTRY

# Turing Machine transition format:
# (current_state, read_symbol) -> (next_state, write_symbol, direction)
# direction: L (left), R (right), S (stay)
Transition = Tuple[str, str, str, str]

class TuringMachine:
    """Universal Turing Machine simulator."""
    
    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        tape_alphabet: List[str],
        blank_symbol: str,
        initial_state: str,
        accept_states: List[str],
        reject_states: List[str],
        transitions: Dict[Tuple[str, str], Tuple[str, str, str]]
    ):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.tape_alphabet = set(tape_alphabet)
        self.blank = blank_symbol
        self.initial_state = initial_state
        self.accept_states = set(accept_states)
        self.reject_states = set(reject_states)
        self.transitions = transitions
        
        self.reset()
    
    def reset(self, input_tape: str = ""):
        """Reset machine with new input."""
        self.tape = {0: c for i, c in enumerate(input_tape)} if input_tape else {}
        self.head = 0
        self.state = self.initial_state
        self.steps = 0
        self.history = []
    
    def _read(self) -> str:
        return self.tape.get(self.head, self.blank)
    
    def _write(self, symbol: str):
        if symbol == self.blank:
            self.tape.pop(self.head, None)
        else:
            self.tape[self.head] = symbol
    
    def step(self) -> bool:
        """Execute one step. Returns False if halted."""
        if self.state in self.accept_states or self.state in self.reject_states:
            return False
        
        symbol = self._read()
        key = (self.state, symbol)
        
        if key not in self.transitions:
            # No transition defined -> halt (reject)
            return False
        
        next_state, write_symbol, direction = self.transitions[key]
        
        # Record history
        self.history.append({
            'step': self.steps,
            'state': self.state,
            'head': self.head,
            'symbol': symbol,
            'tape_snapshot': self._get_tape_str()
        })
        
        # Execute transition
        self._write(write_symbol)
        self.state = next_state
        self.head += 1 if direction == 'R' else -1 if direction == 'L' else 0
        self.steps += 1
        
        return True
    
    def run(self, max_steps: int = 10000) -> Dict:
        """Run until halt or max_steps."""
        while self.steps < max_steps:
            if not self.step():
                break
        
        return {
            'halted': self.state in self.accept_states or self.state in self.reject_states,
            'accepted': self.state in self.accept_states,
            'rejected': self.state in self.reject_states,
            'steps': self.steps,
            'final_state': self.state,
            'tape': self._get_tape_str(),
            'head_position': self.head
        }
    
    def _get_tape_str(self, window: int = 20) -> str:
        """Get tape visualization around head."""
        if not self.tape:
            return self.blank
        
        min_pos = min(self.tape.keys())
        max_pos = max(self.tape.keys())
        
        start = max(min_pos, self.head - window)
        end = min(max_pos, self.head + window)
        
        result = []
        for i in range(start, end + 1):
            symbol = self.tape.get(i, self.blank)
            if i == self.head:
                result.append(f"[{symbol}]")
            else:
                result.append(f" {symbol} ")
        return "".join(result).strip()
    
    @staticmethod
    def parse_transition_table(table_str: str) -> Dict[Tuple[str, str], Tuple[str, str, str]]:
        """Parse transition table from text format."""
        transitions = {}
        lines = [l.strip() for l in table_str.strip().split('\n') if l.strip()]
        
        for line in lines:
            # Format: q0,0 -> q1,1,R or similar
            match = re.match(r'(\w+),\s*(\S+)\s*->\s*(\w+),\s*(\S+),?\s*([RLS])', line)
            if match:
                state, read_sym, next_state, write_sym, direction = match.groups()
                transitions[(state, read_sym)] = (next_state, write_sym, direction)
        
        return transitions
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TuringMachine':
        """Create machine from config dict."""
        return cls(
            states=config.get('states', []),
            alphabet=config.get('alphabet', []),
            tape_alphabet=config.get('tape_alphabet', []),
            blank_symbol=config.get('blank_symbol', '_'),
            initial_state=config.get('initial_state', 'q0'),
            accept_states=config.get('accept_states', []),
            reject_states=config.get('reject_states', []),
            transitions=config.get('transitions', {})
        )


class TuringMachineTool(BaseTool):
    """Turing Machine Simulator Tool."""
    
    name = "turing_machine"
    description = (
        "Simulate a universal Turing Machine. Define states, alphabet, transitions, "
        "and execute step-by-step. Inspired by LV Agent's Turing Machine architecture."
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "step", "run", "visualize"],
                "description": "Action to perform"
            },
            "machine_id": {
                "type": "string",
                "description": "Machine identifier for step/run operations"
            },
            "config": {
                "type": "object",
                "description": "Machine configuration with states, alphabet, transitions"
            },
            "input_tape": {
                "type": "string",
                "description": "Initial tape content"
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum steps to run",
                "default": 10000
            }
        },
        "required": ["action"]
    }
    
    def __init__(self):
        self.machines: Dict[str, TuringMachine] = {}
    
    def execute(self, action: str, machine_id: str = None, config: Dict = None,
                input_tape: str = "", max_steps: int = 10000) -> ToolResult:
        try:
            if action == "create":
                if not config:
                    return ToolResult(success=False, output="", error="Config required for create")
                
                machine = TuringMachine.from_config(config)
                if machine_id is None:
                    machine_id = f"tm_{len(self.machines)}"
                
                self.machines[machine_id] = machine
                return ToolResult(
                    success=True,
                    output=f"Turing Machine created with ID: {machine_id}\n"
                           f"States: {machine.states}\n"
                           f"Initial state: {machine.initial_state}"
                )
            
            elif action == "step":
                if machine_id not in self.machines:
                    return ToolResult(success=False, output="", error=f"Machine {machine_id} not found")
                
                machine = self.machines[machine_id]
                if input_tape:
                    machine.reset(input_tape)
                
                executed = machine.step()
                if not executed:
                    status = "ACCEPTED" if machine.state in machine.accept_states else "REJECTED"
                    return ToolResult(
                        success=True,
                        output=f"Machine halted. Status: {status}\n"
                               f"Final state: {machine.state}\n"
                               f"Steps: {machine.steps}\n"
                               f"Tape: {machine._get_tape_str()}"
                    )
                
                last = machine.history[-1]
                return ToolResult(
                    success=True,
                    output=f"Step {machine.steps} executed\n"
                           f"State: {machine.state}\n"
                           f"Head at: {machine.head}\n"
                           f"Tape: {machine._get_tape_str()}"
                )
            
            elif action == "run":
                if machine_id not in self.machines:
                    return ToolResult(success=False, output="", error=f"Machine {machine_id} not found")
                
                machine = self.machines[machine_id]
                machine.reset(input_tape)
                result = machine.run(max_steps)
                
                output = (
                    f"Turing Machine Execution Complete\n"
                    f"Steps: {result['steps']}\n"
                    f"Halted: {result['halted']}\n"
                    f"Accepted: {result['accepted']}\n"
                    f"Rejected: {result['rejected']}\n"
                    f"Final state: {result['final_state']}\n"
                    f"Head position: {result['head_position']}\n"
                    f"Final tape: {result['tape']}"
                )
                
                if result['steps'] >= max_steps:
                    output += "\n⚠️  Max steps reached - possible infinite loop"
                
                return ToolResult(success=True, output=output)
            
            elif action == "visualize":
                if machine_id not in self.machines:
                    return ToolResult(success=False, output="", error=f"Machine {machine_id} not found")
                
                machine = self.machines[machine_id]
                viz = []
                viz.append(f"States: {', '.join(machine.states)}")
                viz.append(f"Current state: {machine.state}")
                viz.append(f"Head position: {machine.head}")
                viz.append(f"Tape: {machine._get_tape_str(30)}")
                viz.append(f"Steps executed: {machine.steps}")
                
                if machine.history:
                    viz.append("\nHistory (last 5 steps):")
                    for entry in machine.history[-5:]:
                        viz.append(f"  Step {entry['step']}: state={entry['state']}, head={entry['head']}")
                
                return ToolResult(success=True, output="\n".join(viz))
            
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Turing Machine error: {str(e)}")


# Example machines for quick testing

def create_binary_incrementer() -> Dict:
    """Example: Binary incrementer Turing Machine."""
    return {
        'states': ['q0', 'q1', 'q2', 'q_accept'],
        'alphabet': ['0', '1'],
        'tape_alphabet': ['0', '1', '_'],
        'blank_symbol': '_',
        'initial_state': 'q0',
        'accept_states': ['q_accept'],
        'reject_states': [],
        'transitions': {
            ('q0', '1'): ('q0', '1', 'L'),  # Move left finding first 0
            ('q0', '0'): ('q1', '1', 'R'),  # Found 0, increment and accept
            ('q0', '_'): ('q2', '1', 'R'),  # All 1s, write 1 and accept
            ('q1', '0'): ('q1', '0', 'R'),  # Move right to end
            ('q1', '1'): ('q1', '1', 'R'),
            ('q1', '_'): ('q_accept', '_', 'S'),
            ('q2', '_'): ('q_accept', '_', 'S'),
        }
    }


TOOLS_REGISTRY.register(TuringMachineTool())
