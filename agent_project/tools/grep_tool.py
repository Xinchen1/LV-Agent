"""
Grep / Search Tool
Fast text search across files with regex support, respecting .gitignore and hidden files.
Prefers ripgrep (rg) when available, falls back to Python implementation.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from . import BaseTool, ToolResult, TOOLS_REGISTRY


# Directories that are typically huge and not useful for code search.
SKIP_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "ENV",
    "dist", "build", "target", "out", ".next", ".nuxt", ".cache", "site-packages",
    "Pods", "vendor", "ThirdParty", "third_party", "bin", "obj", ".gradle",
    ".tox", "htmlcov", "coverage", "*.egg-info", ".pytest_cache", ".mypy_cache",
}


class GrepTool(BaseTool):
    """Search for patterns across files in a directory tree."""

    name = "search_files"
    description = (
        "Search for text patterns across files using ripgrep (preferred) or Python fallback. "
        "Supports regex, case-insensitive search, file type filters, context lines, and exclusion of hidden files. "
        "Much faster than reading each file individually when looking for specific terms."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search pattern (regex or plain text). Case-insensitive by default.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search. Default: current directory.",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "File glob pattern to limit search, e.g. '*.py', '*.{ts,js}'.",
            },
            "glob_exclude": {
                "type": "string",
                "description": "Glob pattern to exclude files, e.g. '*.min.js', 'node_modules/**'.",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines around each match.",
                "default": 2,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 50,
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case-sensitive search.",
                "default": False,
            },
            "literal": {
                "type": "boolean",
                "description": "Treat query as literal string, not regex.",
                "default": False,
            },
        },
        "required": ["query"],
    }

    # Binary file extensions to skip
    BINARY_EXTS = {
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".tiff",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".sqlite", ".db", ".sqlite3",
        ".so", ".dylib", ".dll", ".exe", ".wasm", ".pyc", ".o",
        ".DS_Store",
    }

    def __init__(self):
        self._rg_available = shutil.which("rg") is not None

    def execute(
        self,
        query: str,
        path: str = ".",
        glob: str = "",
        glob_exclude: str = "",
        context_lines: int = 2,
        max_results: int = 50,
        case_sensitive: bool = False,
        literal: bool = False,
    ) -> ToolResult:
        """Execute search across files."""
        # LLM 可能生成空 query → 返回提示而非 TypeError
        if not query or not str(query).strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty search query: provide a pattern to search for.",
            )
        if not path or not str(path).strip():
            path = "."
        search_path = os.path.expanduser(str(path))
        if not os.path.exists(search_path):
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {search_path}",
            )

        if self._rg_available:
            return self._exec_rg(
                query, search_path, glob, glob_exclude,
                context_lines, max_results, case_sensitive, literal,
            )
        return self._exec_python(
            query, search_path, glob, glob_exclude,
            context_lines, max_results, case_sensitive, literal,
        )

    def _exec_rg(
        self, query, path, file_glob, glob_excl,
        context, max_res, case_sens, literal,
    ) -> ToolResult:
        """Use ripgrep for fast search."""
        args = ["rg", "--line-number", "--with-filename", "--smart-case"]
        if context > 0:
            args += ["-C", str(context)]
        else:
            args.append("--no-heading")
        if not case_sens:
            args.append("-i")
        if literal:
            args.append("-F")
        # Exclude huge directories by default.
        for d in SKIP_DIRS:
            args += ["-g", f"!{d}/**"]
        if file_glob:
            args += ["-g", file_glob]
        if glob_excl:
            args += ["-g", f"!{glob_excl}"]
        args += ["--max-columns", "500"]
        args.append(query)

        try:
            proc = subprocess.run(
                args,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = proc.stdout.strip()
            if proc.returncode == 1:
                output = "No matches found."
            elif proc.returncode > 1:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"rg error: {proc.stderr.strip()[:200]}",
                )

            results = output.splitlines()[:max_res]
            total_count = len(results)

            return ToolResult(
                success=True,
                output="\n".join(results) if results else "No matches found.",
                metadata={
                    "engine": "ripgrep",
                    "pattern": query,
                    "path": path,
                    "total_matches": total_count,
                    "truncated": len(output.splitlines()) > max_res,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Search timed out")
        except FileNotFoundError:
            return ToolResult(success=False, output="", error="ripgrep not found")

    def _exec_python(
        self, query, path, file_glob, glob_excl,
        context, max_res, case_sens, literal,
    ) -> ToolResult:
        """Pure Python fallback search with timeout and large-dir skipping."""
        flags = 0 if case_sens else re.IGNORECASE
        if literal:
            pattern = re.escape(query)
        else:
            try:
                re.compile(query)
                pattern = query
            except re.error:
                pattern = re.escape(query)

        compiled = re.compile(pattern, flags)
        results = []
        max_count = max_res
        start_time = time.time()
        time_limit = 25  # seconds

        search_root = Path(path)

        for filepath in self._iter_files(search_root, file_glob, glob_excl):
            if time.time() - start_time > time_limit:
                break
            if self._should_skip(filepath):
                continue
            try:
                # Skip very large files.
                if filepath.stat().st_size > 5 * 1024 * 1024:
                    continue
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if compiled.search(line):
                        results.append(f"{filepath.relative_to(search_root)}:{i}: {line[:200]}")
                        if len(results) >= max_count:
                            break
                if len(results) >= max_count:
                    break
            except Exception:
                continue

        truncated = len(results) >= max_count or (time.time() - start_time > time_limit)
        output = "\n".join(results) if results else "No matches found."
        if truncated and not results:
            output = "Search timed out before finding matches."

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "engine": "python",
                "pattern": query,
                "path": path,
                "total_matches": len(results),
                "truncated": truncated,
            },
        )

    def _iter_files(self, root: Path, file_glob: str, exclude: str):
        """Yield files matching glob filters, skipping huge/irrelevant dirs."""
        import fnmatch

        if root.is_file():
            yield root
            return

        for dirpath, dirnames, filenames in os.walk(root):
            # Remove skipped dirs from traversal.
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                f = Path(dirpath) / name
                if file_glob:
                    if not fnmatch.fnmatch(f.name, file_glob) and not fnmatch.fnmatch(str(f.relative_to(root)), file_glob):
                        continue
                if exclude:
                    exc_pattern = exclude.replace("**/", "").lstrip("/")
                    if fnmatch.fnmatch(f.name, exc_pattern):
                        continue
                    rel = str(f.relative_to(root))
                    if fnmatch.fnmatch(rel, exc_pattern):
                        continue
                yield f

    def _should_skip(self, path: Path) -> bool:
        """Skip binary files and hidden directories."""
        if any(part.startswith(".") for part in path.parts):
            return True
        if any(part in SKIP_DIRS for part in path.parts):
            return True
        ext = path.suffix.lower()
        if ext in self.BINARY_EXTS:
            return True
        return False


class GlobTool(BaseTool):
    """Find files matching glob patterns with respect for .gitignore."""

    name = "glob"
    description = (
        "Find files matching glob patterns like **/*.py, **/*.{ts,js}, or src/**/*.json. "
        "Recursively searches from the given path. "
        "Use when you know the FILE TYPE but not the exact filename. "
        "More powerful than ls for discovering project structure."
    )

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py', 'src/**/*.ts', '*.json'. ** matches any depth.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search from. Default: current directory.",
                "default": ".",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum files to return.",
                "default": 100,
            },
        },
        "required": ["pattern"],
    }

    def __init__(self):
        self._rg_available = shutil.which("rg") is not None

    def execute(
        self,
        pattern: str = "**",
        path: str = ".",
        max_results: int = 100,
    ) -> ToolResult:
        """Find files matching pattern."""
        # LLM 可能生成空 pattern/缺省参数 → 兜底为列当前目录所有文件
        if not pattern or not str(pattern).strip():
            pattern = "**"
        if not path or not str(path).strip():
            path = "."
        search_path = Path(os.path.expanduser(str(path)))
        if not search_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {search_path}",
            )

        try:
            files = []
            for f in search_path.rglob(pattern):
                if any(part in SKIP_DIRS or part.startswith(".") for part in f.relative_to(search_path).parts):
                    continue
                files.append(f)
                if len(files) >= max_results + 1:
                    break

            truncated = len(files) > max_results
            files = files[:max_results]
            results = []
            for f in files:
                rel = f.relative_to(search_path)
                size = f.stat().st_size if f.is_file() else 0
                kind = "/" if f.is_dir() else ""
                size_str = f" ({size:,} bytes)" if f.is_file() else ""
                results.append(f"{rel}{kind}{size_str}")

            output = f"Found {len(files)} file(s) matching '{pattern}' in {search_path}:\n"
            output += "\n".join(results) if results else "  (none)"

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "pattern": pattern,
                    "path": str(search_path),
                    "count": len(files),
                    "truncated": truncated,
                },
            )
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid glob pattern: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Glob search failed: {e}")


# Register tools
TOOLS_REGISTRY.register(GrepTool())
TOOLS_REGISTRY.register(GlobTool())
