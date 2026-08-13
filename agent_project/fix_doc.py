import re

with open("agent_project/agent.py", 'r') as f:
    lines = f.readlines()

# Find the broken method around line 1395
for i, line in enumerate(lines):
    if '_parse_output_for_action' in line and 'Optional[ToolCall]' in line:
        # Replace next 4 lines (the broken docstring)
        new_doc = [
            '        """Parse tool calls from model output.\n',
            '\n',
            '        Supports multiple formats (checked in order):\n',
            '        1. Anthropic-style tool_use XML blocks (preferred for native models)\n',
            '        2. [TOOL:name] key="value" [/TOOL] text format\n',
            '        3. Legacy XML function-call format\n',
            '        """\n',
        ]
        # Replace lines i+1 through i+4 (0-indexed)
        lines[i+1:i+5] = new_doc
        break

with open("agent_project/agent.py", 'w') as f:
    f.writelines(lines)

print(f"Fixed docstring at line {i+1}")
