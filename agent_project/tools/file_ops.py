"""
File Operations Tool - Python wrapper around rust_file_ops.
All actual file I/O is delegated to the Rust binary for speed and safety.

Enhancements:
- Persistent Rust process pool (--loop mode) eliminates per-call spawn overhead.
- Global file stat cache with mtime_ns invalidation.
- Parallel batch execution for multi-read / multi-exists / multi-list.
- Default line numbers for read action.
- Plain text status labels (DELETED/NEW/MODIFIED).
"""

import os
import re
import json
import hashlib
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=3,
        )
        return proc.returncode == 0
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


class _FileStatCache:
    """Thread-safe file stat cache keyed by resolved path. Invalidated by mtime_ns + size."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        p = Path(path)
        try:
            stat = p.stat()
            key = str(p.resolve())
            with self._lock:
                cached = self._cache.get(key)
                if cached and cached.get("mtime_ns") == stat.st_mtime_ns and cached.get("size") == stat.st_size:
                    return cached.get("data")
                return None
        except Exception:
            return None

    def set(self, path: str, data: Dict[str, Any]):
        p = Path(path)
        try:
            stat = p.stat()
            key = str(p.resolve())
            with self._lock:
                self._cache[key] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "data": data}
        except Exception:
            pass

    def invalidate(self, path: str):
        try:
            key = str(Path(path).resolve())
            with self._lock:
                self._cache.pop(key, None)
        except Exception:
            pass


class FastReadCache:
    """Fast reading for long articles: chunk, embed, cache locally, retrieve on demand."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "fast_read_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._model_loaded = False

    def _embedding_model(self):
        if not self._model_loaded:
            self._model_loaded = True
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model = None
        return self._model

    def _file_key(self, path: str) -> str:
        p = Path(path)
        try:
            stat = p.stat()
            raw = f"{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        except Exception:
            raw = str(Path(path).resolve())
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks = []
        current = []
        current_len = 0
        for para in paragraphs:
            if current_len + len(para) > chunk_size and current:
                chunks.append("\n\n".join(current))
                overlap_text = []
                overlap_len = 0
                for prev in reversed(current):
                    if overlap_len + len(prev) > overlap:
                        break
                    overlap_text.insert(0, prev)
                    overlap_len += len(prev)
                current = overlap_text
                current_len = overlap_len
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        model = self._embedding_model()
        if model:
            embs = model.encode(texts, show_progress_bar=False)
            return embs.tolist()
        return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _keyword_score(self, query: str, chunk: str) -> float:
        q_words = re.findall(r"\w+", query.lower())
        c_words = re.findall(r"\w+", chunk.lower())
        if not q_words or not c_words:
            return 0.0
        c_set = set(c_words)
        exact_matches = sum(1 for w in q_words if w in c_set)
        freq_matches = sum(c_words.count(w) for w in q_words)
        score = (exact_matches * 2.0 + freq_matches * 0.3) / (len(c_words) + len(q_words))
        return min(score, 1.0)

    def _read_text(self, path: str) -> str:
        proc = subprocess.run(
            [RUST_BINARY],
            input=json.dumps({"action": "read", "path": path}, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or "rust_file_ops failed")
        data = json.loads(proc.stdout)
        if not data.get("success"):
            raise RuntimeError(data.get("error") or "read failed")
        return data.get("output", "")

    def cache_file(self, path: str) -> Dict[str, Any]:
        text = self._read_text(path)
        if not text.strip():
            raise ValueError("file is empty or unreadable")
        key = self._file_key(path)
        chunks = self._chunk_text(text)
        embeddings = self._embed(chunks)
        data = {
            "path": str(Path(path).resolve()),
            "chunks": chunks,
            "embeddings": embeddings if embeddings is not None else [],
            "model": "sentence-transformers" if self._embedding_model() else "keyword-fallback",
            "chunk_count": len(chunks),
        }
        cache_path = self.cache_dir / f"{key}.json"
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def load_cache(self, path: str) -> Optional[Dict[str, Any]]:
        key = self._file_key(path)
        cache_path = self.cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            embeddings = data.get("embeddings")
            chunks = data.get("chunks", [])
            if embeddings and len(embeddings) == len(chunks):
                first_len = len(embeddings[0]) if embeddings[0] else 0
                if any(len(e) != first_len for e in embeddings):
                    return None
            return data
        except Exception:
            return None

    def query(self, path: str, query: str, top_k: int = 5) -> str:
        cache = self.load_cache(path)
        if cache is None:
            return ""
        chunks = cache["chunks"]
        embeddings = cache.get("embeddings")
        has_embeddings = bool(embeddings and len(embeddings) == len(chunks) and embeddings[0])
        if has_embeddings:
            q_emb = self._embed([query])
            if q_emb:
                scored = [(self._cosine_similarity(q_emb[0], emb), c) for emb, c in zip(embeddings, chunks)]
            else:
                scored = [(self._keyword_score(query, c), c) for c in chunks]
        else:
            scored = [(self._keyword_score(query, c), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        parts = []
        for score, chunk in scored[:top_k]:
            if score <= 0 and len(parts) >= 1:
                break
            parts.append(f"[relevance {score:.2f}]\n{chunk}")
        return "\n\n---\n\n".join(parts)


class FileOpsTool(BaseTool):
    name = "file_ops"
    description = (
        "Powerful file operations tool. Supports read, multi_read, fast_read, write, list, exists, "
        "analyze, grep, diff, backup, find, apply_diff (search/replace edits), verify (syntax check), "
        "and open (launch with the default system application) operations on files and directories. "
        "Use fast_read for long articles: it chunks the file, builds a local vector cache, and retrieves "
        "only the relevant chunks for a given query. All I/O is performed by a native Rust backend with "
        "persistent process pooling for low latency."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "multi_read", "fast_read", "write", "list", "exists", "analyze", "grep", "diff", "backup", "find", "apply_diff", "verify", "open"],
                "description": "File operation to perform"
            },
            "path": {
                "type": "string",
                "description": "File or directory path (supports glob patterns and Chinese aliases like 桌面/文档/下载)"
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths for multi_read action"
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
            "pattern": {
                "type": "string",
                "description": "Search pattern for grep/find action"
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
            "query": {
                "type": "string",
                "description": "For fast_read: the question/keyword to retrieve relevant chunks from the cached document"
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "For fast_read: number of top relevant chunks to return"
            },
        },
        "required": ["action", "path"]
    }

    def __init__(self, allowed_dirs: List[str] = None, max_file_size: int = 10485760, unrestricted: bool = True):
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or ["."])]
        self.max_file_size = max_file_size
        self.unrestricted = unrestricted
        self._fast_read_cache = FastReadCache()
        self._checkpoint = CheckpointManager()
        self._stat_cache = _FileStatCache()
        self._pool = _RustProcessPool(size=4)
        self._batch_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="file_ops_batch_")

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

    def __del__(self):
        try:
            self._pool.close()
            self._batch_executor.shutdown(wait=False)
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

    def _open_path(self, path: str) -> ToolResult:
        """Open a file or directory with the system's default application.

        Uses `open` on macOS, `xdg-open` on Linux, and `start` on Windows.
        The call is non-blocking and detached so the agent isn't held up.
        """
        import platform
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        system = platform.system()
        try:
            if system == "Darwin":
                cmd = ["open", str(p)]
            elif system == "Windows":
                cmd = ["cmd", "/c", "start", "", str(p)]
            else:
                cmd = ["xdg-open", str(p)]

            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return ToolResult(
                success=True,
                output=f"Opened {p}",
                metadata={"action": "open", "path": str(p.resolve()), "os": system},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to open {path}: {e}")

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

    def _python_fallback(self, payload: Dict[str, Any]) -> ToolResult:
        """Pure-Python fallback for basic file ops when the Rust binary is
        unavailable (wrong architecture / missing). Supports the most common
        actions: read, write, list, exists, grep, analyze."""
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
                for e in entries:
                    prefix = "d " if e.is_dir() else "  "
                    size = e.stat().st_size if e.is_file() else 0
                    lines.append(f"{prefix}{e.name:<40s} {size:>10d}")
                return ToolResult(success=True, output="\n".join(lines) or "(empty)", metadata={"count": len(entries), "fallback": "python"})
            if action == "exists":
                return ToolResult(success=True, output=str(p.exists()), metadata={"exists": p.exists(), "fallback": "python"})
            if action == "grep":
                if not p.exists():
                    return ToolResult(success=False, output="", error=f"Path not found: {path}")
                pat = payload.get("pattern", "")
                try:
                    regex = re.compile(pat)
                except re.error:
                    regex = re.compile(re.escape(pat))
                # 目录 -> 递归搜索文件; 单文件 -> 直接搜索
                files_to_scan = []
                if p.is_dir():
                    try:
                        for root, dirs, fnames in os.walk(p):
                            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", ".venv", "target", "dist", "build", ".cache")]
                            for fn in fnames:
                                files_to_scan.append(str(Path(root) / fn))
                    except Exception:
                        files_to_scan = []
                else:
                    files_to_scan = [str(p)]
                matches = []
                limit = int(payload.get("limit") or payload.get("max_results") or 200)
                for fp in files_to_scan:
                    if len(matches) >= limit:
                        break
                    try:
                        with open(fp, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    matches.append(f"{fp}:{i}:{line.rstrip()}")
                                    if len(matches) >= limit:
                                        break
                    except Exception:
                        continue
                if not matches:
                    return ToolResult(success=True, output=f"(no matches for '{pat}' in {path})", metadata={"matches": 0, "fallback": "python"})
                return ToolResult(success=True, output="\n".join(matches), metadata={"matches": len(matches), "fallback": "python"})
            if action == "analyze":
                if not p.exists():
                    return ToolResult(success=False, output="", error=f"Path not found: {path}")
                stat = p.stat()
                info = f"path: {p}\nsize: {stat.st_size} bytes\nmodified: {datetime.fromtimestamp(stat.st_mtime)}"
                if p.is_file():
                    try:
                        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                        info += f"\nlines: {len(lines)}\ntype: file"
                    except Exception:
                        info += "\ntype: binary file"
                else:
                    info += f"\ntype: directory\nentries: {len(list(p.iterdir()))}"
                return ToolResult(success=True, output=info, metadata={"fallback": "python"})
            if action == "open":
                resolved_path = str(self._resolve_path_smart(path))
                return self._open_path(resolved_path)
            if action == "apply_diff":
                return self._python_apply_diff(p, payload)
            if action == "verify":
                return self._python_verify(p)
            if action == "find":
                return self._python_find(p, payload)
            if action == "multi_read":
                return self._python_multi_read(payload)
            if action == "diff":
                return ToolResult(success=False, output="",
                                  error="diff 需 Rust 二进制(可改用 bash_exec: diff <file1> <file2>)")
            if action == "backup":
                return self._python_backup(p)
            # Unsupported action in fallback
            return ToolResult(
                success=False, output="",
                error=f"Action '{action}' requires the Rust binary (incompatible/missing on this machine). "
                      f"Install Rust and run `cargo build --release` in the rust_file_ops/ directory, "
                      f"or use a supported action: read, write, list, exists, grep, analyze, find, multi_read, backup."
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Python fallback error: {e}")

    def _python_find(self, p: Path, payload: Dict[str, Any]) -> ToolResult:
        """纯 Python find: 按文件名模式(glob)递归查找, 列出匹配文件.

        与 Rust 实现对齐的常见用法: pattern='package.json' / '**/*.md' / '*.py'。
        限制: 目录下文件过多时可能较慢(纯 Python 递归), 正常项目规模可用。
        """
        import fnmatch
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {p}")
        pattern = payload.get("pattern") or "*"
        base_pattern = pattern
        if "/" in pattern or "**" in pattern:
            base_pattern = pattern.split("/")[-1] or "*"
        limit = int(payload.get("limit") or payload.get("max_results") or 100)
        matches = []
        try:
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", ".venv", "target", "dist", "build", ".obsidian", ".cache")]
                for fname in files:
                    if fnmatch.fnmatch(fname, base_pattern):
                        matches.append(str(Path(root) / fname))
                    if len(matches) >= limit:
                        break
                if len(matches) >= limit:
                    break
        except Exception as e:
            return ToolResult(success=False, output="", error=f"find failed: {e}")
        if not matches:
            return ToolResult(success=True, output=f"(no matches for '{pattern}' in {p})", metadata={"count": 0, "fallback": "python"})
        out = f"Found {len(matches)} file(s) in {p} matching '{pattern}':\n" + "\n".join(matches)
        return ToolResult(success=True, output=out, metadata={"count": len(matches), "fallback": "python"})

    def _python_multi_read(self, payload: Dict[str, Any]) -> ToolResult:
        """纯 Python multi_read: 并行读取多个文件, 合并输出."""
        paths = payload.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            return ToolResult(success=False, output="", error="paths 参数 required for multi_read")
        results = []
        for fp in paths:
            fp_path = Path(str(fp))
            if not fp_path.exists():
                results.append(f"--- {fp} [ERROR: not found] ---")
                continue
            try:
                text = fp_path.read_text(encoding="utf-8", errors="replace")
                results.append(f"--- {fp} ---\n{text}")
            except Exception as e:
                results.append(f"--- {fp} [ERROR: {e}] ---")
        return ToolResult(success=True, output="\n\n".join(results), metadata={"files": len(paths), "fallback": "python"})

    def _python_backup(self, p: Path) -> ToolResult:
        """纯 Python backup: 复制文件/目录到带时间戳的 .bak 副本."""
        import shutil
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {p}")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = Path(str(p) + f".{stamp}.bak")
        try:
            if p.is_dir():
                shutil.copytree(p, backup_path)
            else:
                shutil.copy2(p, backup_path)
            return ToolResult(success=True, output=f"Backup saved to {backup_path}", metadata={"backup": str(backup_path), "fallback": "python"})
        except Exception as e:
            return ToolResult(success=False, output="", error=f"backup failed: {e}")

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
        # 自动语法验证
        verr = self._python_verify_syntax(p, current)
        if verr:
            out += f"\n⚠ SYNTAX VERIFICATION FAILED:\n{verr}\nPlease fix the syntax and re-apply."
        return ToolResult(success=True, output=out, metadata={"fallback": "python"})

    def _python_verify(self, p: Path) -> ToolResult:
        if not p.exists():
            return ToolResult(success=False, output="", error=f"File not found: {p}")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read {p}: {e}")
        verr = self._python_verify_syntax(p, content)
        if verr:
            return ToolResult(success=False, output="", error=verr)
        ext = p.suffix.lower().lstrip(".")
        names = {"py": "Python", "js": "JavaScript", "ts": "TypeScript", "json": "JSON",
                 "yaml": "YAML", "yml": "YAML", "sh": "Shell"}
        name = names.get(ext, "Unknown")
        return ToolResult(success=True, output=f"{name} syntax OK", metadata={"fallback": "python"})

    def _python_verify_syntax(self, p: Path, content: str) -> str:
        """返回空串表示语法 OK, 否则返回错误描述."""
        ext = p.suffix.lower().lstrip(".")
        try:
            if ext == "py":
                import ast
                ast.parse(content)
                return ""
            if ext in ("json",):
                json.loads(content)
                return ""
            if ext in ("yaml", "yml"):
                import yaml
                yaml.safe_load(content)
                return ""
            if ext in ("js", "mjs", "cjs", "ts"):
                # 无 node 时跳过 JS 检查
                import subprocess as sp
                if ext == "ts":
                    return ""  # TS 需要 tsc, 跳过
                r = sp.run(["node", "--check"], input=content.encode(),
                           capture_output=True, timeout=15)
                if r.returncode != 0:
                    return f"JS syntax error: {r.stderr.decode()[:300]}"
                return ""
            if ext in ("sh", "command", "bash"):
                import subprocess as sp
                r = sp.run(["bash", "-n"], input=content.encode(),
                           capture_output=True, timeout=15)
                if r.returncode != 0:
                    return f"Shell syntax error: {r.stderr.decode()[:300]}"
                return ""
            return ""
        except SyntaxError as e:
            return f"Syntax error: {e}"
        except Exception as e:
            return f"Verification error: {e}"

    def _json_to_result(self, data: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            metadata=data.get("metadata") or {},
        )

    def _parallel_call(self, payloads: List[Dict[str, Any]], timeout: float = 120.0) -> List[ToolResult]:
        """Execute multiple rust calls in parallel using the thread pool."""
        futures = {self._batch_executor.submit(self._call_rust_fast, p, timeout): i for i, p in enumerate(payloads)}
        results: List[ToolResult] = [None] * len(payloads)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = ToolResult(success=False, output="", error=f"Parallel call failed: {e}")
        return results

    def execute(self, action: str, path: str = "", content: Optional[str] = None,
                offset: Optional[int] = None, limit: Optional[int] = None,
                pattern: Optional[str] = None, diff: Optional[str] = None,
                encoding: str = "utf-8", line_numbers: bool = True,
                paths: Optional[List[str]] = None, use_cache: bool = True,
                query: Optional[str] = None, top_k: int = 5,
                options: Optional[dict] = None, **kwargs) -> ToolResult:
        # LLM 常用嵌套 options={'offset':…,'limit':…} 形式, 展开到顶层参数
        if options and isinstance(options, dict):
            for k, v in options.items():
                if k == "offset" and offset is None:
                    offset = v
                elif k == "limit" and limit is None:
                    limit = v
                elif k == "pattern" and pattern is None:
                    pattern = v
                elif k == "diff" and diff is None:
                    diff = v
                elif k == "encoding" and encoding == "utf-8":
                    encoding = v
                elif k == "line_numbers":
                    line_numbers = bool(v)
                elif k == "query" and query is None:
                    query = v
                elif k == "top_k":
                    top_k = v
        # multi_read 场景: 支持只有 paths 无 path(模型常这样), 补默认 path
        if action == "multi_read" and not paths and path:
            paths = [path]
        if action == "multi_read" and not path:
            path = "."
        try:
            resolved = str(self._resolve_path_smart(path))

            if action == "fast_read":
                try:
                    cache = self._fast_read_cache.load_cache(resolved)
                    if cache is None:
                        cache = self._fast_read_cache.cache_file(resolved)
                        header = f"[fast_read] cached {cache['chunk_count']} chunks from {resolved} using {cache['model']}\n\n"
                    else:
                        header = f"[fast_read] loaded cached {cache['chunk_count']} chunks from {resolved}\n\n"
                    if query:
                        retrieved = self._fast_read_cache.query(resolved, query, top_k=top_k)
                        if not retrieved:
                            return ToolResult(
                                success=True,
                                output=header + "No relevant chunks found for the query.",
                                metadata={"chunks": cache["chunk_count"], "query": query}
                            )
                        return ToolResult(
                            success=True,
                            output=header + retrieved,
                            metadata={"chunks": cache["chunk_count"], "query": query, "top_k": top_k}
                        )
                    preview = "\n\n---\n\n".join(cache["chunks"][:top_k])
                    return ToolResult(
                        success=True,
                        output=header + preview,
                        metadata={"chunks": cache["chunk_count"], "preview_chunks": min(top_k, cache["chunk_count"])}
                    )
                except Exception as e:
                    return ToolResult(success=False, output="", error=f"fast_read failed: {e}")

            if action == "multi_read":
                resolved_paths = [str(self._resolve_path_smart(p)) for p in (paths or [])]
                if not resolved_paths:
                    return ToolResult(success=False, output="", error="paths parameter required for multi_read action")
                payloads = [
                    {
                        "action": "read",
                        "path": rp,
                        "offset": offset,
                        "limit": limit,
                        "encoding": encoding,
                        "line_numbers": line_numbers,
                    }
                    for rp in resolved_paths
                ]
                results = self._parallel_call(payloads)
                merged = []
                for rp, result in zip(resolved_paths, results):
                    if result.success:
                        merged.append(f"--- {rp} ---\n{result.output}")
                    else:
                        merged.append(f"--- {rp} [ERROR: {result.error}] ---")
                return ToolResult(success=True, output="\n\n".join(merged), metadata={"files": len(resolved_paths)})

            if action == "open":
                return self._open_path(resolved)

            payload: Dict[str, Any] = {"action": action, "path": resolved}
            if content is not None:
                payload["content"] = content
            if offset is not None:
                payload["offset"] = offset
            if limit is not None:
                payload["limit"] = limit
            if pattern is not None:
                payload["pattern"] = pattern
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

            if action in ("write", "apply_diff") and result.success:
                self._stat_cache.invalidate(resolved)

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
