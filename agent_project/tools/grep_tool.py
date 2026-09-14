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
import contextlib
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
        if os.path.isdir(path):
            args.append(".")
        else:
            args.append(os.path.basename(path))

        try:
            proc = subprocess.run(
                args,
                cwd=os.path.dirname(path) if not os.path.isdir(path) else path,
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
    """Find files matching glob patterns via the bash ``find`` command (fast, pruned)."""

    name = "glob"
    description = (
        "Find files matching glob patterns like **/*.py, **/*.{ts,js}, or src/**/*.json. "
        "Uses the system 'find' command, prunes huge/generated directories (node_modules, .git, "
        "target, ...) and hidden entries, and is timeout-bounded so it never hangs on big folders. "
        "Use when you know the FILE TYPE but not the exact filename."
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
        self._find = shutil.which("find")

    @staticmethod
    def _expand_braces(pattern: str) -> List[str]:
        """Expand a single '{a,b,c}' group into plain globs (bash-style)."""
        m = re.compile(r'\{([^{}]+)\}').search(pattern)
        if not m:
            return [pattern]
        opts = m.group(1).split(",")
        prefix, suffix = pattern[:m.start()], pattern[m.end():]
        return [prefix + o + suffix for o in opts]

    def execute(
        self,
        pattern: str = "**",
        path: str = ".",
        max_results: int = 100,
        timeout: float = 10.0,
    ) -> ToolResult:
        """Find files matching pattern via ``find`` (bounded: 最多 max_results 条或 timeout 秒)."""
        # LLM 可能生成空 pattern/缺省参数 → 兜底为列当前路径下所有文件
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

        deadline = time.monotonic() + max(float(timeout), 0.5)
        try:
            if search_path.is_file():
                name_part = pattern.rsplit("/", 1)[-1]
                name_patterns = self._expand_braces(name_part)
                import fnmatch
                matched = any(fnmatch.fnmatch(search_path.name, p) for p in name_patterns)
                rels = [search_path.name] if matched else []
                truncated = False
                timed_out = False
            elif self._find:
                rels, truncated, timed_out = self._run_find(search_path, pattern, max_results, deadline)
            else:
                rels, truncated, timed_out = self._walk_fallback(search_path, pattern, max_results, deadline)

            results = []
            for rel in rels:
                fp = search_path / rel
                is_dir = fp.is_dir()
                size = fp.stat().st_size if fp.is_file() else 0
                kind = "/" if is_dir else ""
                size_str = f" ({size:,} bytes)" if fp.is_file() else ""
                results.append(f"{rel}{kind}{size_str}")

            output = f"Found {len(results)} file(s) matching '{pattern}' in {search_path}:\n"
            output += "\n".join(results) if results else "  (none)"
            if timed_out:
                output += "\n(partial: 遍历超时, 仅为部分结果)"

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "pattern": pattern,
                    "path": str(search_path),
                    "count": len(results),
                    "truncated": truncated,
                    "timed_out": timed_out,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Glob search failed: {e}")

    def _run_find(
        self,
        search_path: Path,
        pattern: str,
        max_results: int,
        deadline: float,
    ) -> tuple:
        """Shell out to ``find`` with prune-skip + -name matching; stream & cap results.

        Bash glob semantics: `**` → recursive search; any other pattern (e.g. `*`,
        `*.py`) matches CURRENT level only, exactly like `ls path/*.py`.
        """
        # pattern 含 `**` 才递归; 否则仅当前层(与 bash glob 一致, 避免 * 全盘倾倒)
        recursive = "**" in pattern
        name_part = pattern.rsplit("/", 1)[-1]
        if name_part in ("", "*", "**"):
            name_part = "*"
        name_patterns = [p for p in self._expand_braces(name_part) if p] or ["*"]

        # find 默认自带 nestable 剪枝; hidden + 巨型目录不递归
        prune_names = list(SKIP_DIRS) + [".*"]
        args = ["find", str(search_path)]
        if not recursive:
            args.append("-maxdepth")
            args.append("1")
        args += ["("]
        for i, dn in enumerate(prune_names):
            if i:
                args.append("-o")
            args += ["-name", dn]
        args += [")", "-prune", "-o", "("]
        for i, np_ in enumerate(name_patterns):
            if i:
                args.append("-o")
            args += ["-name", np_]
        args += [")", "-print"]

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
        )
        rels: List[str] = []
        truncated = False
        timed_out = False
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                if time.monotonic() > deadline:
                    timed_out = True
                    truncated = True
                    break
                line = raw.rstrip("\n")
                if not line:
                    continue
                try:
                    rel = os.path.relpath(line, str(search_path))
                except ValueError:
                    rel = line.rsplit("/", 1)[-1]
                if rel in (".", os.sep) or rel.startswith(".."):
                    continue
                rels.append(rel)
                if len(rels) >= max_results + 1:
                    truncated = True
                    break
        finally:
            with contextlib.suppress(Exception):
                proc.kill()
        return rels[:max_results], truncated, timed_out

    def _walk_fallback(
        self,
        search_path: Path,
        pattern: str,
        max_results: int,
        deadline: float,
    ) -> tuple:
        """Pure-Python os.walk fallback with the same prune semantics when find is missing."""
        import fnmatch

        recursive = "**" in pattern
        name_part = pattern.rsplit("/", 1)[-1]
        if name_part in ("", "*", "**"):
            name_part = "*"
        name_patterns = [p for p in self._expand_braces(name_part) if p] or ["*"]

        rels: List[str] = []
        timed_out = False
        if not recursive:
            dirnames: List[str] = []
            filenames: List[str] = []
            try:
                for e in os.scandir(search_path):
                    if e.is_dir(follow_symlinks=True):
                        dirnames.append(e.name)
                    else:
                        filenames.append(e.name)
            except OSError:
                dirnames = filenames = []
            for name in sorted(dirnames + filenames):
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                if name in SKIP_DIRS or name.startswith("."):
                    continue
                if not any(fnmatch.fnmatch(name, p) for p in name_patterns):
                    continue
                rels.append(name)
                if len(rels) >= max_results + 1:
                    break
            return rels[:max_results], len(rels) > max_results, timed_out

        for root, dirnames, filenames in os.walk(search_path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            if time.monotonic() > deadline:
                timed_out = True
                break
            for names, is_dir in ((dirnames, True), (filenames, False)):
                for name in names:
                    if time.monotonic() > deadline:
                        timed_out = True
                        break
                    if is_dir and (name in SKIP_DIRS or name.startswith(".")):
                        continue
                    if not any(fnmatch.fnmatch(name, p) for p in name_patterns):
                        continue
                    rel = os.path.relpath(str(Path(root) / name), str(search_path))
                    rels.append(rel)
                    if len(rels) >= max_results + 1:
                        break
                if len(rels) >= max_results + 1 or timed_out:
                    break
            if len(rels) >= max_results + 1 or timed_out:
                break
        return rels[:max_results], len(rels) > max_results, timed_out


# Register tools
TOOLS_REGISTRY.register(GrepTool())
TOOLS_REGISTRY.register(GlobTool())
