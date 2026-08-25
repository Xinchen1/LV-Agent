"""
Context Tape Abstraction

Turing Machine-inspired context window management.

Inspired by LV Agent's Turing Machine architecture:
- Tape → Context window with infinite theoretical bound
- Read/Write head → Current working memory position
- Tape cells → Context chunks (messages/events)
- Movements → Context navigation and compression
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time


class TapeDirection(Enum):
    LEFT = "L"   # Move toward older context
    RIGHT = "R"  # Move toward newer context  
    STAY = "S"   # Keep current position


@dataclass
class TapeCell:
    """Single cell on the context tape."""
    position: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def access(self):
        self.access_count += 1
        return self.content


@dataclass
class TapeHead:
    """Read/write head position on context tape."""
    position: int = 0
    state: str = "idle"
    
    def move(self, direction: TapeDirection, steps: int = 1) -> int:
        """Move head and return new position."""
        if direction == TapeDirection.LEFT:
            self.position -= steps
        elif direction == TapeDirection.RIGHT:
            self.position += steps
        return self.position


class ContextTape:
    """
    Turing Machine-inspired context window management.
    
    Tape cells represent context chunks (messages, tool outputs, etc.)
    Head position represents current focus in context
    Movement allows navigation through historical context
    """
    
    def __init__(
        self,
        window_size: int = 100,
        compression_threshold: int = 80
    ):
        self.window_size = window_size
        self.compression_threshold = compression_threshold
        self.tape: Dict[int, TapeCell] = {}
        self.head = TapeHead()
        self.blank_symbol = ""
        self.step_count = 0
        
        # Turing Machine-like state registry
        self.states = {
            'idle': 'Waiting for input',
            'reading': 'Reading context',
            'writing': 'Writing new context',
            'compressing': 'Compressing tape',
            'moving': 'Moving head',
            'halting': 'Context exhausted'
        }
        
        self.current_state = 'idle'
        self.transition_log = []
    
    def write_cell(self, position: Optional[int] = None, content: str = "", metadata: Dict = None):
        """Write content to tape cell (like Turing Machine write operation)."""
        self._transition('writing')
        
        if position is None:
            position = self.head.position
        
        cell = TapeCell(
            position=position,
            content=content,
            metadata=metadata or {}
        )
        self.tape[position] = cell
        self.step_count += 1
        
        self._log_transition(f"WRITE {position} <- '{content[:50]}'")
    
    def read_cell(self, position: Optional[int] = None) -> Optional[str]:
        """Read content from tape cell (like Turing Machine read operation)."""
        self._transition('reading')
        
        if position is None:
            position = self.head.position
        
        cell = self.tape.get(position)
        if cell:
            cell.access()
            self._log_transition(f"READ {position} -> '{cell.content[:50]}'")
            return cell.content
        return self.blank_symbol
    
    def move_head(self, direction: TapeDirection, steps: int = 1):
        """Move read/write head (like Turing Machine head movement)."""
        self._transition('moving')
        
        old_pos = self.head.position
        new_pos = self.head.move(direction, steps)
        
        self._log_transition(f"MOVE {old_pos} -> {new_pos} ({direction.value})")
        return new_pos
    
    def _transition(self, new_state: str):
        """Record state transition."""
        old_state = self.current_state
        self.current_state = new_state
        self.transition_log.append({
            'step': self.step_count,
            'from': old_state,
            'to': new_state,
            'head_pos': self.head.position,
            'timestamp': time.time()
        })
    
    def _log_transition(self, action: str):
        """Log tape operation."""
        self.transition_log.append({
            'step': self.step_count,
            'action': action,
            'state': self.current_state,
            'head_pos': self.head.position
        })
    
    def get_window(self, size: int = None) -> List[Tuple[int, str]]:
        """Get tape window around head (like Turing Machine visible tape)."""
        if size is None:
            size = self.window_size // 2
        
        window = []
        for i in range(-size, size + 1):
            pos = self.head.position + i
            content = self.tape.get(pos, self.blank_symbol).content if pos in self.tape else self.blank_symbol
            window.append((pos, content))
        
        return window
    
    def compress(self, target_size: int):
        """Compress tape by removing old/less accessed cells."""
        self._transition('compressing')
        
        if len(self.tape) <= target_size:
            return
        
        # Sort by access count and age
        cells = sorted(
            self.tape.values(),
            key=lambda c: (c.access_count, time.time() - c.created_at)
        )
        
        # Keep most accessed recent cells
        cells_to_keep = cells[-target_size:]
        self.tape = {c.position: c for c in cells_to_keep}
        
        self._log_transition(f"COMPRESS {len(self.tape) + len(cells_to_keep) - target_size} cells")
    
    def snapshot(self) -> Dict[str, Any]:
        """Get current tape state snapshot."""
        return {
            'head_position': self.head.position,
            'current_state': self.current_state,
            'tape_size': len(self.tape),
            'steps': self.step_count,
            'window_sample': self.get_window(5),
            'transitions': len(self.transition_log)
        }
    
    def visualize(self, width: int = 40) -> str:
        """Visualize tape like Turing Machine tape display."""
        lines = []
        
        # State info
        lines.append(f"State: {self.current_state} | Head: {self.head.position} | Steps: {self.step_count}")
        lines.append("")
        
        # Tape visualization
        window = self.get_window(width // 2)
        
        # Tape cells
        cells_line = []
        for pos, content in window:
            if pos == self.head.position:
                cells_line.append(f"[{content[:3]}]")
            else:
                cells_line.append(f" {content[:3]} ")
        
        lines.append("".join(cells_line))
        
        # Position markers
        markers = []
        for pos, _ in window:
            if pos == self.head.position:
                markers.append(" ^ ")
            else:
                markers.append("   ")
        
        lines.append("".join(markers))
        
        # Position numbers
        num_line = []
        for pos, _ in window:
            num_str = str(pos)
            num_line.append(f"{num_str:>4}")
        
        lines.append("".join(num_line))
        
        return "\n".join(lines)


# Integration with AgentLoop

class ContextTapeAgent:
    """
    AgentLoop wrapper with Context Tape abstraction.
    
    Demonstrates integration of Turing Machine-inspired context management
    with existing AgentLoop architecture.
    """
    
    def __init__(self, tape: ContextTape = None):
        self.tape = tape or ContextTape()
        self.history = []
    
    def process_message(self, message: str, position: Optional[int] = None):
        """Process message as tape write operation."""
        if position is None:
            position = self.tape.head.position + 1
        
        self.tape.write_cell(position, message)
        self.tape.move_head(TapeDirection.RIGHT, 1)
        self.history.append(('write', position, message))
    
    def recall_context(self, steps_back: int = 1) -> List[str]:
        """Recall context by moving head left."""
        self.tape.move_head(TapeDirection.LEFT, steps_back)
        
        # Read window around head
        window = self.tape.get_window(3)
        return [content for _, content in window if content]
    
    def compress_old_context(self, target_size: int):
        """Compress context using tape compression."""
        self.tape.compress(target_size)
    
    def get_context_snapshot(self) -> Dict:
        """Get current context state."""
        return self.tape.snapshot()
    
    def visualize_context(self) -> str:
        """Visualize context tape."""
        return self.tape.visualize()
