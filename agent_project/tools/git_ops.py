"""
Git Operations Tool - Complete git workflow support
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool, ToolResult, TOOLS_REGISTRY


class GitTool(BaseTool):
    """
    Git operations tool for repository management
    Supports: clone, commit, push, pull, branch, diff, log, status
    """

    name = "git"
    description = "Perform git operations on local repositories. Supports cloning, committing, pushing, pulling, branching, and more."

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["clone", "init", "status", "add", "commit", "push", "pull", "branch", "checkout", "merge", "log", "diff", "fetch", "reset", "rebase"],
                "description": "Git command to execute"
            },
            "repository": {
                "type": "string",
                "description": "Repository URL (for clone) or local path"
            },
            "branch": {
                "type": "string",
                "description": "Branch name for various operations"
            },
            "message": {
                "type": "string",
                "description": "Commit message for commit command"
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of files to add (for add command)"
            },
            "options": {
                "type": "string",
                "description": "Additional git flags (e.g., '-m main' for init, '--force' for reset)"
            }
        },
        "required": ["command"]
    }

    def __init__(self, allowed_paths: List[str] = None, require_confirmation: bool = True):
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths] if allowed_paths else []
        self.require_confirmation = require_confirmation

        # Safety: block dangerous commands
        self.blocked_commands = ['clean', 'filter-branch', 'submodule', 'gc', 'fsck']
        self.logger = None

    def execute(
        self,
        command: Optional[str] = None,
        action: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        message: Optional[str] = None,
        files: Optional[List[str]] = None,
        options: Optional[str] = None
    ) -> ToolResult:
        """Execute git command. `action` is an alias for `command`."""
        command = action or command
        if not command:
            return ToolResult(success=False, output="", error="Git command/action is required")

        try:
            # Safety check
            if command in self.blocked_commands:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command '{command}' is blocked for safety"
                )

            # Build command
            cmd = ['git', command]

            if command == 'clone' and repository:
                cmd.append(repository)
                if branch:
                    cmd.extend(['-b', branch])
            elif command == 'init' and repository:
                cmd.append(repository)
            elif command == 'checkout' and branch:
                cmd.append(branch)
            elif command == 'commit' and message:
                cmd.extend(['-m', message])
            elif command == 'add' and files:
                cmd.extend(files)
            elif command == 'branch' and branch:
                if command == 'branch' and options and '--delete' in options:
                    cmd.extend(['-d', branch])
                else:
                    cmd.append(branch)
            elif command == 'push':
                if branch:
                    cmd.extend(['origin', branch])
                else:
                    cmd.append('origin')
                    if options:
                        cmd.append(options)
            elif command == 'pull':
                if branch:
                    cmd.extend(['origin', branch])
            elif options:
                cmd.append(options)

            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self._resolve_cwd(command, repository)
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            success = result.returncode == 0

            return ToolResult(
                success=success,
                output=output[:4000] if output else "(no output)",
                error="" if success else f"Git command failed with exit code {result.returncode}",
                metadata={
                    'command': command,
                    'returncode': result.returncode,
                    'cwd': str(self._resolve_cwd(command, repository))
                }
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Git command timed out after 30 seconds")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Git error: {str(e)}")

    def _resolve_cwd(self, command: str, repository: Optional[str]) -> Path:
        """Resolve working directory for command"""
        if command == 'clone':
            # clone runs in current dir
            return Path.cwd()
        elif repository:
            path = Path(repository).resolve()
            if path.is_dir():
                return path
            parent = path.parent
            if parent.exists():
                return parent
        return Path.cwd()


TOOLS_REGISTRY.register(GitTool())
