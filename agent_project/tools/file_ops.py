"""
File Operations Tool - Python wrapper around rust_file_ops.
All actual file I/O is delegated to the Rust binary for speed and safety.

Enhancements:
- Persistent Rust process pool (--loop mode) eliminates per-call spawn overhead.
- Core actions only: read/write/list/apply_diff/delete/move/copy/mkdir/exists.
- Default line numbers for read action.
- Plain text status labels (DELETED/NEW/MODIFIED).
"""

import os
import re
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from ..checkpoint import CheckpointManager
from ..terminal import style as _style


def _rust_binary_path() -> str:
    """Locate the rust_file_ops binary relative to this module."""
    candidate = Path(__file__).resolve().parents[2] / "rust_file_ops" / "target" / "release" / "rust_file_ops"
    if candidate.exists():
        return str(candidate)
    for path in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(path) / "rust_file_ops"
        if p.exists():
            return str(p)
    return str(candidate)


RUST_BINARY = _rust_binary_path()


def _check_rust_binary() -> bool:
    """Check if the Rust binary is runnable on this platform.

    Returns False when the binary exists but has a wrong CPU architecture
    (e.g. x86_64 binary on arm64), which is the classic copy-to-new-machine
    failure mode. Also returns False if the binary is missing entirely.
    """
    import platform
    binary = Path(RUST_BINARY)
    if not binary.exists():
        return False
    try:
        # 用真实 JSON 请求探测(one-shot 模式 read_to_string(stdin), TTY 下继承终端
        # stdin 会永远阻塞到超时 -> 误判"不可用"掉 Python fallback)。
        # input= 写完即关写端, 子进程立刻拿到 EOF, 不挂 stdin。
        proc = subprocess.run(
            [str(binary)],
            input='{"action":"exists","path":"/tmp"}',
            capture_output=True, text=True, timeout=3,
        )
        return proc.returncode == 0 and '"success"' in (proc.stdout or "").lower()
    except Exception:
        # "Bad CPU type in executable" lands here
        return False


RUST_AVAILABLE = _check_rust_binary()
if not RUST_AVAILABLE and Path(RUST_BINARY).exists():
    import platform
    print(_style(f"  warning: rust_file_ops binary incompatible with {platform.machine()} — "
          f"using Python fallback for file_ops", "33"))


class _RustProcess:
    """A single persistent rust_file_ops --loop worker."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        try:
            self._proc = subprocess.Popen(
                [RUST_BINARY, "--loop"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._proc = None
            raise RuntimeError(f"Failed to start rust_file_ops --loop: {exc}")

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def call(self, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
        with self._lock:
            if not self.is_alive():
                self._start()
            proc = self._proc
            try:
                line = json.dumps(payload, ensure_ascii=False) + "\n"
                proc.stdin.write(line)
                proc.stdin.flush()
                response = proc.stdout.readline()
                if not response:
                    raise RuntimeError("rust_file_ops closed stdout")
                return json.loads(response)
            except Exception as exc:
                # If the worker dies mid-call, restart once and retry.
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._start()
                line = json.dumps(payload, ensure_ascii=False) + "\n"
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
                response = self._proc.stdout.readline()
                if not response:
                    raise RuntimeError("rust_file_ops closed stdout after restart")
                return json.loads(response)

    def close(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=2)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass


class _RustProcessPool:
    """Pool of persistent rust_file_ops workers with round-robin dispatch."""

    def __init__(self, size: int = 4):
        self.size = size
        self._workers: List[_RustProcess] = []
        self._idx = 0
        self._idx_lock = threading.Lock()
        self._available = False
        try:
            for _ in range(size):
                self._workers.append(_RustProcess())
            self._available = True
        except Exception as exc:
            # Fallback to one-shot mode if pool cannot start.
            self._workers = []

    def _next_worker(self) -> Optional[_RustProcess]:
        if not self._workers:
            return None
        with self._idx_lock:
            worker = self._workers[self._idx % len(self._workers)]
            self._idx += 1
            return worker

    def call(self, payload: Dict[str, Any], timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        worker = self._next_worker()
        if worker is None:
            return None
        try:
            return worker.call(payload, timeout=timeout)
        except Exception as exc:
            # Last resort: one-shot fallback.
            try:
                proc = subprocess.run(
                    [RUST_BINARY],
                    input=json.dumps(payload, ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    return None
                return json.loads(proc.stdout)
            except Exception:
                return None

    def close(self):
        for worker in self._workers:
            worker.close()
        self._workers = []
        self._available = False


class FileOpsTool(BaseTool):
    name = "file_ops"

    description = (
        "Core file operations: read, write, list, exists, apply_diff, mkdir, delete, move, copy. "
        "Use search_files for content search, glob for filename search, and bash_exec for system "
        "commands (open/stat/tree/du/diff/backup). Native Rust backend with a persistent process pool."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list", "exists", "apply_diff", "mkdir", "delete", "move", "copy"],
                "description": "File operation to perform"
            },
            "path": {
                "type": "string",
                "description": "File or directory path (supports Chinese aliases like 桌面/文档/下载)"
            },
            "content": {
                "type": "string",
                "description": "Content to write (required for write action)"
            },
            "offset": {
                "type": "integer",
                "description": "Line offset for partial reads"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read"
            },
            "diff": {
                "type": "string",
                "description": "Search/replace diff text for apply_diff action. Format: <<<<<<< SEARCH\\nold text\\n=======\\nnew text\\n>>>>>>> REPLACE"
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "File encoding hint (Rust backend auto-detects UTF-8/GBK/Latin-1)"
            },
            "line_numbers": {
                "type": "boolean",
                "default": True,
                "description": "Whether to prepend line numbers to each line when reading a file"
            },
            "destination": {
                "type": "string",
                "description": "Target path (required for move / copy actions)"
            },
        },
        "required": ["action", "path"]
    }

    def __init__(self, allowed_dirs: List[str] = None, max_file_size: int = 10485760, unrestricted: bool = True):
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or ["."])]
        self.max_file_size = max_file_size
        self.unrestricted = unrestricted
        self._checkpoint = CheckpointManager()
        self._pool = _RustProcessPool(size=4)
        self._prewarm_pool()

    _PATH_ALIASES = {
        "文档": "Documents",
        "文档文件夹": "Documents",
        "文档目录": "Documents",
        "桌面": "Desktop",
        "桌面文件夹": "Desktop",
        "下载": "Downloads",
        "下载文件夹": "Downloads",
        "图片": "Pictures",
        "图片文件夹": "Pictures",
        "音乐": "Music",
        "音乐文件夹": "Music",
        "视频": "Movies",
        "视频文件夹": "Movies",
        "应用": "Applications",
        "应用程序": "Applications",
        " home": "~",
        "家目录": "~",
        "主目录": "~",
    }

    def _prewarm_pool(self) -> None:
        """后台预热 rust 进程池: 消除首次 file_ops 调用 ~20ms 的进程拉起冷启动."""
        import threading as _th
        def _warm():
            try:
                self._pool.call({"action": "exists", "path": "/tmp"}, timeout=5)
            except Exception as e:
                import logging as _l
                _l.getLogger("lv.file_ops").debug(f"pool prewarm failed: {e}")
        _th.Thread(target=_warm, daemon=True).start()

    def __del__(self):
        try:
            self._pool.close()
        except Exception:
            pass

    def _resolve_path_smart(self, path: str) -> Path:
        """Resolve path with micro-tweaks: strip quotes, env vars, relative dirs, case-insensitive fallback, Chinese aliases."""
        cleaned = path.strip().strip("'\"`")
        cleaned = os.path.expandvars(os.path.expanduser(cleaned))

        for alias, target in self._PATH_ALIASES.items():
            if cleaned == alias or cleaned.lower() == alias.lower():
                cleaned = os.path.expanduser(f"~/{target}") if target != "~" else os.path.expanduser("~")
                break
            if cleaned.startswith(f"{alias}/") or cleaned.startswith(f"{alias}\\"):
                rest = cleaned[len(alias) + 1:]
                cleaned = os.path.join(os.path.expanduser(f"~/{target}"), rest) if target != "~" else os.path.join(os.path.expanduser("~"), rest)
                break

        p = Path(cleaned)

        if p.is_absolute() and p.exists():
            return p.resolve()

        resolved = Path.cwd() / p
        if resolved.exists():
            return resolved.resolve()

        for base in self.allowed_dirs:
            candidate = base / p
            if candidate.exists():
                return candidate.resolve()

        # Fallback: relative path not found in CWD or allowed_dirs -> search common locations
        # This handles cases where the agent refers to a folder by name only (e.g. "./claude-code-main")
        # but the actual folder lives in Downloads/Desktop/Documents/Home.
        if not p.is_absolute():
            common_roots = [Path.home()]
            for sub in ["Downloads", "Desktop", "Documents"]:
                d = Path.home() / sub
                if d.exists():
                    common_roots.append(d)
            common_roots.extend(self.allowed_dirs)
            seen_roots = set()
            for root in common_roots:
                try:
                    r = root.resolve()
                    if r in seen_roots:
                        continue
                    seen_roots.add(r)
                except Exception:
                    continue
                # direct match
                candidate = root / p
                if candidate.exists():
                    return candidate.resolve()
                # case-insensitive match for the first component
                if p.parts and p.parts[0] not in ('/', '\\', '.', '..'):
                    first_lower = p.parts[0].lower()
                    for child in root.iterdir():
                        if child.name.lower() == first_lower:
                            if len(p.parts) == 1:
                                if child.exists():
                                    return child.resolve()
                            else:
                                candidate = child.joinpath(*p.parts[1:])
                                if candidate.exists():
                                    return candidate.resolve()
                            break

        def _case_insensitive_lookup(root: Path, parts: List[str]) -> Optional[Path]:
            current = root
            for part in parts:
                if part == '/' or part == os.sep:
                    continue
                if not current.is_dir():
                    return None
                found = None
                lower = part.lower()
                for child in current.iterdir():
                    if child.name.lower() == lower:
                        found = child
                        break
                if found is None:
                    return None
                current = found
            return current

        parts = list(p.parts)
        for root in [Path('/'), Path.cwd()] + self.allowed_dirs:
            try:
                candidate = _case_insensitive_lookup(root, parts)
                if candidate and candidate.exists():
                    return candidate.resolve()
            except Exception:
                pass

        return p.resolve()

    # 受保护的系统路径前缀: 写入/删除类操作绝不应当触碰(防 ../../etc/passwd 等穿越)
    _PROTECTED_PREFIXES = (
        "/etc/", "/usr/", "/System/", "/Library/", "/bin/", "/sbin/",
        "/private/etc/", "/private/var/db/", "/boot/", "/root/",
    )

    def _assert_contained(self, resolved: Path) -> Optional[str]:
        """Return an error reason if ``resolved`` targets a protected system path; else None.

        防路径穿越(../../etc/passwd、绝对路径 /etc/...): 这些写入/删除会被拒绝。
        家目录、工作目录(allowed_dirs)以及其它普通路径(如 /tmp、/Volumes、用户自选目录)放行,
        以免破坏正常文件操作。只读操作(read/open)不在此受限。
        """
        try:
            resolved = resolved.resolve()
        except Exception:
            return f"cannot resolve path: {resolved}"
        rstr = str(resolved) + "/"
        for pfx in self._PROTECTED_PREFIXES:
            if rstr.startswith(pfx) or rstr.rstrip("/") == pfx.rstrip("/"):
                return f"refusing to write/delete into protected system path: {resolved}"
        # 家目录与 allowed_dirs 永远放行(用户自己的文件)
        try:
            home = Path.home().resolve()
            if resolved == home or str(resolved).startswith(str(home) + "/"):
                return None
        except Exception:
            pass
        for base in self.allowed_dirs:
            try:
                base_r = base.resolve()
            except Exception:
                continue
            if resolved == base_r or str(resolved).startswith(str(base_r) + "/"):
                return None
        return None

    def _call_rust(self, payload: Dict[str, Any], timeout: float = 120.0) -> ToolResult:
        """One-shot fallback when the persistent pool is unavailable."""
        try:
            proc = subprocess.run(
                [RUST_BINARY],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return ToolResult(success=False, output="", error=f"rust_file_ops binary not found at {RUST_BINARY}. Please build it with `cargo build --release`.")
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="rust_file_ops timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to run rust_file_ops: {e}")

        if proc.returncode != 0:
            return ToolResult(success=False, output="", error=f"rust_file_ops exited with code {proc.returncode}: {proc.stderr}")

        try:
            data = json.loads(proc.stdout)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Invalid JSON from rust_file_ops: {e}\n{proc.stdout}")

        return self._json_to_result(data)

    _fallback_notified = False  # 类级: 本进程内只提示一次 fallback 模式

    def _notify_fallback_once(self, reason: str) -> None:
        """首次进入 Python fallback 时打印一行可见提示, 避免刷屏."""
        if FileOpsTool._fallback_notified:
            return
        FileOpsTool._fallback_notified = True
        try:
            print(_style(f"  file_ops: Python fallback mode ({reason}) — 部分操作稍慢但功能完整", "33"), flush=True)
        except Exception:
            pass

    def _call_rust_fast(self, payload: Dict[str, Any], timeout: float = 120.0) -> ToolResult:
        """Use persistent process pool when available, else fall back to one-shot,
        and finally to a pure-Python fallback if the Rust binary is incompatible."""
        # Skip Rust entirely when the binary has a wrong CPU architecture.
        if not RUST_AVAILABLE:
            self._notify_fallback_once("rust binary unavailable")
            return self._python_fallback(payload)
        if self._pool._available:
            data = self._pool.call(payload, timeout=timeout)
            if data is not None:
                return self._json_to_result(data)
            self._notify_fallback_once("rust process pool failed")
        result = self._call_rust(payload, timeout=timeout)
        # Last resort: Python fallback if Rust failed at runtime.
        if not result.success:
            fb = self._python_fallback(payload)
            if fb.success:
                self._notify_fallback_once("rust runtime failure")
                return fb
        return result

    def _list_summary(self, path: str) -> ToolResult:
        """返回真正对用户友好的目录摘要，而不是裸 ls -la。"""
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")
        if not p.is_dir():
            return ToolResult(success=False, output="", error=f"Path is not a directory: {path}")
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        lines = []
        lines.append(f"Directories({len(dirs)}):")
        for e in dirs[:20]:
            lines.append(f"  {e.name}")
        if len(dirs) > 20:
            lines.append(f"  ... (+{len(dirs)-20} more)")
        lines.append(f"Files({len(files)}):")
        for e in files[:20]:
            try:
                size = e.stat().st_size
            except Exception:
                size = 0
            lines.append(f"  {e.name} ({size:d} bytes)")
        if len(files) > 20:
            lines.append(f"  ... (+{len(files)-20} more)")
        return ToolResult(success=True, output="\n".join(lines), metadata={"count": len(entries), "dirs": len(dirs), "files": len(files)})

    def _python_fallback(self, payload: Dict[str, Any]) -> ToolResult:
        """Pure-Python fallback for basic file ops when the Rust binary is
        unavailable (wrong architecture / missing). Supports the core actions:
        read, write, list, exists, apply_diff."""
        action = payload.get("action", "")
        path = payload.get("path", "")
        try:
            p = Path(path)
            if action == "read":
                if not p.exists():
                    return ToolResult(success=False, output="", error=f"File not found: {path}")
                if p.is_dir():
                    return ToolResult(success=False, output="", error=f"Path is a directory: {path}")
                enc = payload.get("encoding", "utf-8")
                try:
                    text = p.read_text(encoding=enc)
                except UnicodeDecodeError:
                    text = p.read_text(encoding="latin-1")
                lines = text.splitlines()
                offset = payload.get("offset") or 0
                limit = payload.get("limit")
                if limit:
                    lines = lines[offset:offset + limit]
                elif offset:
                    lines = lines[offset:]
                if payload.get("line_numbers", True):
                    out = "\n".join(f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(lines))
                else:
                    out = "\n".join(lines)
                return ToolResult(success=True, output=out, metadata={"lines": len(lines), "fallback": "python"})
            if action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(payload.get("content", ""), encoding="utf-8")
                return ToolResult(success=True, output=f"Wrote {len(p.read_bytes())} bytes to {path}", metadata={"fallback": "python"})
            if action == "list":
                if not p.exists():
                    return ToolResult(success=False, output="", error=f"Path not found: {path}")
                entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                lines = []
                dirs = [e for e in entries if e.is_dir()]
                files = [e for e in entries if e.is_file()]
                lines.append(f"Directories({len(dirs)}):")
                for e in dirs[:30]:
                    lines.append(f"  - {e.name}")
                if len(dirs) > 30:
                    lines.append(f"  … 还有 {len(dirs) - 30} 项")
                lines.append(f"Files({len(files)}):")
                for e in files[:30]:
                    size = e.stat().st_size
                    lines.append(f"  - {e.name} ({size:d} bytes)")
                if len(files) > 30:
                    lines.append(f"  … 还有 {len(files) - 30} 项")
                return ToolResult(success=True, output="\n".join(lines) or "(empty)", metadata={"count": len(entries), "dirs": len(dirs), "files": len(files), "fallback": "python"})
            if action == "exists":
                return ToolResult(success=True, output=str(p.exists()), metadata={"exists": p.exists(), "fallback": "python"})
            if action == "apply_diff":
                return self._python_apply_diff(p, payload)
            return ToolResult(
                success=False, output="",
                error=f"Action '{action}' requires the Rust binary (incompatible/missing on this machine). "
                      f"Install Rust and run `cargo build --release` in the rust_file_ops/ directory, "
                      f"or use a supported action: read, write, list, exists, apply_diff."
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Python fallback error: {e}")


    def _python_apply_diff(self, p: Path, payload: Dict[str, Any]) -> ToolResult:
        """纯 Python 的 apply_diff: 支持 SEARCH/REPLACE 块, 精确+行级匹配, 失败给最近行提示."""
        import shutil
        diff = payload.get("diff", "")
        blocks = re.findall(
            r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
            diff,
            flags=re.DOTALL,
        )
        if not blocks:
            return ToolResult(success=False, output="", error="No valid diff blocks found")
        if not p.exists():
            return ToolResult(success=False, output="", error=f"File not found: {p}")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read {p}: {e}")

        def apply_block(current: str, search: str, replace: str) -> Optional[str]:
            if search in current:
                return current.replace(search, replace, 1)
            # 行级匹配(忽略行尾空白差异)
            cur_lines = current.split("\n")
            s_lines = search.split("\n")
            for i in range(len(cur_lines) - len(s_lines) + 1):
                if all(cur_lines[i + j].rstrip() == s_lines[j].rstrip() for j in range(len(s_lines))):
                    return "\n".join(cur_lines[:i] + replace.split("\n") + cur_lines[i + len(s_lines):])
            return None

        backup = None
        try:
            backup = str(p) + f".{int(time.time())}.bak"
            shutil.copy2(p, backup)
        except Exception:
            backup = None

        current = content
        applied = 0
        for idx, (search, replace) in enumerate(blocks, 1):
            next_content = apply_block(current, search, replace)
            if next_content is None:
                # 最近行定位
                hint = None
                for line in search.split("\n"):
                    for li, cl in enumerate(current.split("\n"), 1):
                        if line.strip() and (line.strip() in cl or cl.strip() == line.strip()):
                            hint = li
                            break
                    if hint:
                        break
                detail = f"Block {idx}/{len(blocks)} not found"
                if hint:
                    detail += f" (nearest match around line {hint})"
                detail += ". Check whitespace/indentation and that SEARCH exactly matches file content."
                return ToolResult(success=False, output="", error=detail)
            current = next_content
            applied += 1

        try:
            p.write_text(current, encoding="utf-8")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to write {p}: {e}")
        out = f"Applied {applied} diff block(s) to {p}"
        if backup:
            out += f"\nBackup saved to {backup}"
        return ToolResult(success=True, output=out, metadata={"fallback": "python"})

    def _json_to_result(self, data: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            metadata=data.get("metadata") or {},
        )

    _REMOVED_ACTIONS = {
        "multi_read": "read each file once (calls can run in parallel)",
        "fast_read": "read + offset/limit",
        "verify": "bash_exec: python -m py_compile <path>",
        "open": "bash_exec: open <path> (macOS) or xdg-open <path>",
        "stat": "bash_exec: stat <path> or ls -l <path>",
        "tree": "bash_exec: find <path> -maxdepth N",
        "du": "bash_exec: du -sh <path>",
        "grep": "search_files",
        "find": "glob",
        "analyze": "bash_exec: ls -la <path>",
        "backup": "bash_exec: cp -a <path> <path>.bak",
        "diff": "bash_exec: diff <fileA> <fileB>",
    }

    def execute(self, action: str, path: str = "", content: Optional[str] = None,
                destination: Optional[str] = None,
                offset: Optional[int] = None, limit: Optional[int] = None,
                diff: Optional[str] = None,
                encoding: str = "utf-8", line_numbers: bool = True,
                options: Optional[dict] = None, **kwargs) -> ToolResult:
        action = str(action).strip().lower()
        hint = self._REMOVED_ACTIONS.get(action)
        if hint:
            return ToolResult(
                success=False, output="",
                error=f"file_ops action '{action}' is removed. Use {hint}.",
            )
        # LLM 常用嵌套 options={'offset':…,'limit':…} 形式, 展开到顶层参数
        if options and isinstance(options, dict):
            for k, v in options.items():
                if k == "offset" and offset is None:
                    offset = v
                elif k == "limit" and limit is None:
                    limit = v
                elif k == "diff" and diff is None:
                    diff = v
                elif k == "encoding" and encoding == "utf-8":
                    encoding = v
                elif k == "line_numbers":
                    line_numbers = bool(v)
        try:
            resolved = str(self._resolve_path_smart(path))

            # 路径穿越防护: 写入/删除类操作必须落在 allowed_dirs 之内
            if action in ("write", "apply_diff", "delete"):
                containment = self._assert_contained(Path(resolved))
                if containment:
                    return ToolResult(success=False, output="", error=containment)

            if action == "list":
                return self._list_summary(resolved)

            # 仅 mkdir/delete/move/copy 由 Python 原生实现, 带路径防护
            if action in ("mkdir", "delete", "move", "copy"):
                import shutil
                p = Path(resolved)
                if action == "mkdir":
                    p.mkdir(parents=True, exist_ok=True)
                    return ToolResult(success=True, output=f"mkdir -p {resolved}")
                if action == "delete":
                    containment = self._assert_contained(p)
                    if containment:
                        return ToolResult(success=False, output="", error=containment)
                    if not p.exists():
                        return ToolResult(success=False, output="", error=f"Path not found: {resolved}")
                    if p.is_dir():
                        shutil.rmtree(p)
                        return ToolResult(success=True, output=f"Removed directory {resolved}")
                    p.unlink()
                    return ToolResult(success=True, output=f"Deleted {resolved}")
                dest = destination or kwargs.get("destination") or (options or {}).get("destination")
                if not dest:
                    return ToolResult(success=False, output="", error=f"'destination' required for {action}")
                dest_p = Path(str(self._resolve_path_smart(str(dest))))
                containment = self._assert_contained(dest_p)
                if containment:
                    return ToolResult(success=False, output="", error=containment)
                if action == "move":
                    shutil.move(str(p), str(dest_p))
                    return ToolResult(success=True, output=f"Moved {resolved} -> {dest_p}")
                if action == "copy":
                    if p.is_dir():
                        shutil.copytree(str(p), str(dest_p), dirs_exist_ok=True)
                    else:
                        dest_p.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(p), str(dest_p))
                    return ToolResult(success=True, output=f"Copied {resolved} -> {dest_p}")

            payload: Dict[str, Any] = {"action": action, "path": resolved}
            if content is not None:
                payload["content"] = content
            if offset is not None:
                payload["offset"] = offset
            if limit is not None:
                payload["limit"] = limit
            if diff is not None:
                payload["diff"] = diff
            if encoding is not None:
                payload["encoding"] = encoding
            if line_numbers:
                payload["line_numbers"] = line_numbers

            cp_meta = None
            if action in ("write", "apply_diff"):
                cp_meta = self._checkpoint.snapshot(resolved, tag=action)

            result = self._call_rust_fast(payload)

            if cp_meta is not None:
                rust_backup = CheckpointManager.parse_rust_backup_path(result.output or "")
                self._checkpoint.record_result(
                    cp_meta.get("id", ""),
                    success=result.success,
                    error=result.error,
                    rust_backup_path=rust_backup,
                )

            return result

        except ValueError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"File operation failed: {e}")


TOOLS_REGISTRY.register(FileOpsTool())
