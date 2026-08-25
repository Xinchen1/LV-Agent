"""
Project Context Tool - 快速获取工程项目的整体上下文
像 Claude Code 一样：目录结构、git 状态、关键文件一览
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from . import BaseTool, ToolResult, TOOLS_REGISTRY


# Directories that are typically huge and not useful for project overview.
SKIP_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "ENV",
    "dist", "build", "target", "out", ".next", ".nuxt", ".cache", "site-packages",
    "Pods", "vendor", "ThirdParty", "third_party", "bin", "obj", ".gradle",
    ".tox", "htmlcov", "coverage", "*.egg-info", ".pytest_cache", ".mypy_cache",
}


# ANSI styling for de-emphasized (secondary) lines: faint + light gray so that
# "more lines" / limit hints read as small, very light text in the terminal.
# Respects the NO_COLOR standard (https://no-color.org).
_USE_COLOR = os.environ.get("NO_COLOR") is None


def _faint(text: str) -> str:
    """Wrap text in dim + light-gray ANSI so it reads as small, very-light text."""
    if not _USE_COLOR:
        return text
    return f"\033[2;90m{text}\033[0m"


class ProjectContextTool(BaseTool):
    name = "project_context"
    description = (
        "Get a concise overview of a project directory: tree structure, "
        "git status, recent changes, and key config/source files. "
        "Use this before coding tasks to understand the codebase."
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project root path (default: current working directory)"
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "description": "Maximum directory tree depth"
            },
            "max_files": {
                "type": "integer",
                "default": 100,
                "description": "Maximum files to list in tree"
            },
            "include_content": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["README*", "package.json", "pyproject.toml", "setup.py", "Cargo.toml", "requirements*.txt", "Makefile"],
                "description": "Glob patterns of files to preview content from"
            }
        },
        "required": []
    }

    def execute(
        self,
        path: str = ".",
        max_depth: int = 3,
        max_files: int = 100,
        include_content: Optional[List[str]] = None
    ) -> ToolResult:
        try:
            root = Path(path).expanduser().resolve()
            if not root.exists():
                return ToolResult(success=False, output="", error=f"Path does not exist: {root}")
            if not root.is_dir():
                return ToolResult(success=False, output="", error=f"Path is not a directory: {root}")

            include_content = include_content or [
                "README*", "package.json", "pyproject.toml", "setup.py",
                "Cargo.toml", "requirements*.txt", "Makefile"
            ]

            parts = []
            parts.append(f"Project context for: {root}")
            parts.append("=" * 60)

            # Git status
            git_info = self._get_git_status(root)
            parts.append(f"\n[Git]\n{git_info}")

            # Directory tree
            tree = self._build_tree(root, max_depth=max_depth, max_files=max_files)
            parts.append(f"\n[Directory Tree]\n{tree}")

            # Key file previews
            previews = self._preview_key_files(root, include_content, max_depth=max_depth)
            if previews:
                parts.append(f"\n[Key Files Preview]\n{previews}")

            parts.append(
                _faint(f"\n— project_context summary · {root.name} · depth {max_depth} · max {max_files} files —")
            )

            return ToolResult(success=True, output="\n".join(parts), metadata={
                "root": str(root),
                "git": git_info,
                "tree_lines": tree.count("\n") + 1
            })

        except Exception as e:
            return ToolResult(success=False, output="", error=f"project_context failed: {e}")

    def _get_git_status(self, root: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return "Not a git repository"
            status = result.stdout.strip()
            if not status:
                return "Working tree clean"
            return status
        except Exception as e:
            return f"Git status unavailable: {e}"

    def _build_tree(self, root: Path, max_depth: int, max_files: int, _depth: int = 0) -> str:
        lines = []
        count = [0]

        def walk(dir_path: Path, depth: int, prefix: str = ""):
            if depth > max_depth or count[0] >= max_files:
                return
            try:
                entries = sorted(
                    [e for e in dir_path.iterdir() if not e.name.startswith(".") and e.name not in SKIP_DIRS],
                    key=lambda e: (not e.is_dir(), e.name.lower())
                )
            except PermissionError:
                return

            for idx, entry in enumerate(entries):
                if count[0] >= max_files:
                    lines.append(f"{prefix}{_faint(f'... ({max_files} file limit reached)')}")
                    return
                is_last = idx == len(entries) - 1
                connector = "└── " if is_last else "├── "
                display = entry.name + ("/" if entry.is_dir() else "")
                lines.append(f"{prefix}{connector}{display}")
                count[0] += 1
                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    walk(entry, depth + 1, prefix + extension)

        lines.append(str(root))
        walk(root, 1)
        return "\n".join(lines)

    def _preview_key_files(self, root: Path, patterns: List[str], max_depth: int = 3) -> str:
        previews = []
        seen = set()
        max_preview_depth = max(1, min(max_depth, 3))

        def _collect_at_depth(dir_path: Path, depth: int):
            if depth > max_preview_depth:
                return
            try:
                for pattern in patterns:
                    for fp in dir_path.glob(pattern):
                        if fp.is_file() and not fp.name.startswith("."):
                            rel = str(fp.relative_to(root))
                            if rel not in seen and not any(part in SKIP_DIRS for part in Path(rel).parts):
                                seen.add(rel)
                                yield fp, rel
                if depth < max_preview_depth:
                    for entry in dir_path.iterdir():
                        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in SKIP_DIRS:
                            yield from _collect_at_depth(entry, depth + 1)
            except (PermissionError, OSError):
                return

        for fp, rel in _collect_at_depth(root, 1):
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                preview_lines = lines[:40]
                preview = "\n".join(preview_lines)
                if len(lines) > 40:
                    preview += f"\n{_faint(f'... ({len(lines) - 40} more lines)')}"
                previews.append(f"--- {rel} ---\n{preview}")
            except Exception as e:
                previews.append(f"--- {rel} ---\n(error reading: {e})")
            if len(previews) >= 10:
                break
        return "\n\n".join(previews)


TOOLS_REGISTRY.register(ProjectContextTool())
