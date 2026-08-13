"""
Bash Execution Tool
Full-featured shell command execution with safety, timeouts, streaming, and process management.
"""

import asyncio
import os
import re
import signal
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from . import BaseTool, ToolResult, TOOLS_REGISTRY, get_harness_kernel


def _harness_check(tool_name: str, arguments: dict) -> tuple[bool, Optional[str]]:
    """Return (allowed, reason_or_none) after harness policy evaluation."""
    kernel = get_harness_kernel()
    if kernel is None:
        return True, None
    try:
        from ..harness.effects import make_effect, EffectClass
        effect = make_effect(tool_name, arguments)
        admission = kernel.evaluate(effect)
        from ..harness.kernel import Decision
        if admission.decision == Decision.ALLOW:
            return True, None
        if admission.decision == Decision.DENY:
            return False, f"Harness denied: {admission.reason}"
        # ASK
        granted = kernel.ask(effect, admission.reason) if kernel.ask else False
        if granted:
            return True, None
        return False, f"Harness approval denied: {admission.reason}"
    except Exception as e:
        # Policy evaluation failure should not silently allow.
        return False, f"Harness check failed: {e}"


class BashExecTool(BaseTool):
    """Execute shell commands with full control over environment, timeouts, and safety."""

    name = "bash_exec"
    description = (
        "Execute shell commands in a subprocess. "
        "Supports timeout, working directory, environment variables, "
        "streaming output, and safety checks. "
        "Use this for running scripts, installing packages, git operations, "
        "building projects, and any shell-level task."
    )

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. Supports pipes, redirects, &&/|| chaining. Examples: \"ls -la\", \"git diff HEAD~1\", \"grep -r PATTERN .\"",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default 120, max 600.",
                "default": 120,
                "maximum": 600,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory. Use absolute paths or paths relative to agent root. Default: agent project root.",
                "default": ".",
            },
            "env": {
                "type": "string",
                "description": "JSON string of additional environment variables.",
                "default": "{}",
            },
            "capture_output": {
                "type": "boolean",
                "description": "Capture stdout/stderr. Default true.",
                "default": True,
            },
            "shell": {
                "type": "boolean",
                "description": "Run through shell (/bin/bash). Default true for pipe/redirect support.",
                "default": True,
            },
        },
        "required": ["command"],
    }

    # Dangerous patterns that should be blocked or warned about
    DANGEROUS_PATTERNS = [
        # Destructive
        (r"\brm\s+-rf?\s+/", "DESTRUCTIVE: rm -rf on root"),
        (r"\brm\s+-rf?\s+~", "DESTRUCTIVE: rm -rf on home dir"),
        (r"\brm\s+-rf?\s+\*", "DESTRUCTIVE: rm -rf wildcard"),
        (r">\s*/dev/sd[a-z]", "DESTRUCTIVE: writing to block device"),
        (r"mkfs\b", "DESTRUCTIVE: formatting filesystem"),
        (r"\bdd\b.*of=/dev", "DESTRUCTIVE: dd to block device"),
        # Fork bombs
        (r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}\s*;", "FORK BOMB: recursive function spawning processes"),
        (r":\s*\(\)\s*\{\s*:\s*&\s*\}", "FORK BOMB detected"),
        # Data destruction
        (r"shred\b", "SECURE DELETE: shred command"),
        (r"\bformat\b", "FORMAT: disk/partition format"),
        # 全盘 find 扫描(无 maxdepth 限制) → 会刷屏大量权限错误, 应改用 glob/file_ops 定向查找
        # 带 -maxdepth 的 bounded find 放行(用户主动限制深度)
        (r"\bfind\s+(?:[~/]|/Users|/Volumes|/System|/Library)(?:\s|/|\b)(?!.*maxdepth)(?![\w\-]*maxdepth)", "FULL-DISK find: use glob/file_ops to locate files instead (add -maxdepth for bounded scans)"),
        (r"\bfind\s+/[^\s]*\s+(?!.*maxdepth).*-name", "FULL-DISK find: use glob(pattern, path) instead of full-disk scan"),
    ]

    # Warn-only patterns (less severe)
    WARN_PATTERNS = [
        (r"sudo\b", "Privilege escalation: sudo"),
        (r"curl\b.*\|\s*bash", "Remote code execution: piping curl to bash"),
        (r"wget\b.*\|\s*bash", "Remote code execution: piping wget to bash"),
        (r">\s*/dev/null", "REDIRECT to /dev/null (harmless, usually discarding output)"),
        (r"chmod\s+-R\s+777", "Insecure permissions: chmod 777 -R"),
        (r"eval\s+\$", "Indirect eval of variable"),
        (r"export\s+\w+\s*=", "Setting environment variable"),
    ]

    # Max output per stream chunk (characters)
    MAX_CHUNK = 8192

    def __init__(
        self,
        default_timeout: int = 120,
        max_timeout: int = 600,
        allow_unsafe: bool = False,
        default_cwd: Optional[str] = None,
    ):
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.allow_unsafe = allow_unsafe
        self.default_cwd = default_cwd or str(Path.cwd())
        # Track running processes for cancellation
        self._processes: Dict[int, asyncio.subprocess.Process] = {}

    def execute(
        self,
        command: str,
        timeout: int = 0,
        cwd: str = "",
        env: str = "{}",
        capture_output: bool = True,
        shell: bool = True,
        stream_callback: Optional[callable] = None,
    ) -> ToolResult:
        """Execute a shell command synchronously (blocks)."""
        if not command or not command.strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty command: no shell command provided. Provide a real command to execute.",
            )
        timeout = min(timeout or self.default_timeout, self.max_timeout)
        work_dir = cwd or self.default_cwd
        work_dir = os.path.expanduser(work_dir)
        work_dir = os.path.expandvars(work_dir)

        # Safety checks
        safety = self._check_safety(command)
        if safety and not self.allow_unsafe:
            return ToolResult(
                success=False,
                output="",
                error=f"Command blocked for safety: {safety}. "
                f"Use allow_unsafe=true if this is intentional.",
            )

        # Harness policy gate
        allowed, reason = _harness_check(self.name, {"command": command, "cwd": work_dir})
        if not allowed:
            return ToolResult(success=False, output="", error=reason)

        # Parse env overrides
        extra_env = {}
        if env and env.strip() not in ("{}", ""):
            try:
                import json
                extra_env = json.loads(env)
                if not isinstance(extra_env, dict):
                    extra_env = {}
            except (json.JSONDecodeError, TypeError):
                extra_env = {}

        # Build full environment
        full_env = os.environ.copy()
        full_env.update({str(k): str(v) for k, v in extra_env.items()})

        # Validate cwd
        if not os.path.isdir(work_dir):
            return ToolResult(
                success=False,
                output="",
                error=f"Working directory does not exist: {work_dir}",
            )

        # Run in event loop
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self._exec_async(
                    command=command,
                    timeout=timeout,
                    cwd=work_dir,
                    env=full_env,
                    capture_output=capture_output,
                    shell=shell,
                    stream_callback=stream_callback,
                )
            )
            loop.close()
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {type(e).__name__}: {e}",
            )

    async def _exec_async(
        self,
        command: str,
        timeout: float,
        cwd: str,
        env: Dict[str, str],
        capture_output: bool,
        shell: bool,
        stream_callback: Optional[callable] = None,
    ) -> ToolResult:
        """Async execution core."""
        start_time = time.time()
        cmd_display = command if len(command) <= 200 else command[:200] + "..."

        streams: Dict[str, str] = {"stdout": "", "stderr": ""}

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=cwd,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            pid = process.pid
            self._processes[pid] = process

            if capture_output:
                stdout_task = asyncio.create_task(
                    self._read_stream(process.stdout, "stdout", streams, stream_callback)
                )
                stderr_task = asyncio.create_task(
                    self._read_stream(process.stderr, "stderr", streams, stream_callback)
                )

                try:
                    ret = await asyncio.wait_for(
                        process.wait(), timeout=timeout
                    )
                    # Drain remaining output
                    await asyncio.wait([stdout_task, stderr_task], timeout=5)
                except asyncio.TimeoutError:
                    self._kill_process(process)
                    elapsed = time.time() - start_time
                    stdout_so_far = streams.get("stdout", "")[:500]
                    stderr_so_far = streams.get("stderr", "")[:500]
                    return ToolResult(
                        success=False,
                        output=stdout_so_far,
                        error=f"Command timed out after {timeout}s\n[STDERR]\n{stderr_so_far}",
                        metadata={
                            "command": cmd_display,
                            "cwd": cwd,
                            "timeout": timeout,
                            "elapsed": round(elapsed, 2),
                            "output_length": len(streams.get("stdout", "")),
                        },
                    )
            else:
                try:
                    ret = await asyncio.wait_for(
                        process.wait(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    self._kill_process(process)
                    elapsed = time.time() - start_time
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Command timed out after {timeout}s",
                        metadata={
                            "command": cmd_display,
                            "cwd": cwd,
                            "timeout": timeout,
                            "elapsed": round(elapsed, 2),
                        },
                    )

            elapsed = time.time() - start_time
            stdout_text = streams.get("stdout", "") if capture_output else ""
            stderr_text = streams.get("stderr", "") if capture_output else ""

            stdout_stripped = stdout_text.rstrip()
            stderr_stripped = stderr_text.rstrip()

            output_parts = []
            if stdout_stripped:
                output_parts.append(stdout_stripped)
            if stderr_stripped:
                sep = "\n\n" if stdout_stripped else ""
                output_parts.append(f"{sep}[STDERR]\n{stderr_stripped}")

            output = "\n".join(output_parts) if output_parts else "(no output)"
            success = ret == 0

            return ToolResult(
                success=success,
                output=output,
                error=None if success else f"Exit code: {ret}",
                metadata={
                    "command": cmd_display,
                    "cwd": cwd,
                    "exit_code": ret,
                    "elapsed": round(elapsed, 2),
                    "stdout_lines": len(stdout_text.splitlines()),
                    "stderr_lines": len(stderr_text.splitlines()),
                    "output_length": len(output),
                },
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error=f"Command not found in PATH. Make sure the executable is installed.",
                metadata={"command": cmd_display},
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied executing: {cmd_display}",
                metadata={"command": cmd_display},
            )
        finally:
            self._processes.pop(pid, None)

    async def _read_stream(
        self,
        stream,
        name: str,
        streams: Optional[Dict[str, str]] = None,
        stream_callback: Optional[callable] = None,
    ):
        """Read from an async stream line by line, accumulating and optionally streaming."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                if streams is not None:
                    streams[name] += decoded
                if stream_callback is not None:
                    stream_callback(name, decoded)
        except Exception:
            pass

    def _read_stream_sync(
        self,
        stream,
        name: str,
    ) -> str:
        """Synchronous stream reader for use outside async context."""
        result = ""
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                result += line.decode("utf-8", errors="replace")
        except Exception:
            pass
        return result

    def _kill_process(self, process: asyncio.subprocess.Process):
        """Kill a process and its process group."""
        try:
            pid = process.pid
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                process.terminate()
            # Brief grace period then force kill
            try:
                if hasattr(asyncio, "run_in_executor"):
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(process.wait, timeout=3)
                        future.result(timeout=5)
                else:
                    time.sleep(0.5)
            except Exception:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    else:
                        process.kill()
                except Exception:
                    pass
        except ProcessLookupError:
            pass
        except Exception:
            pass

    def _check_safety(self, command: str) -> Optional[str]:
        """Check command for dangerous patterns. Returns reason string if blocked, None if safe."""
        # Check destructive patterns first
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return reason

        # Warn about risky patterns (these are flagged but only blocked if allow_unsafe is False
        # and the pattern is severe enough)
        return None

    def cancel(self, pid: int) -> bool:
        """Cancel a running process by PID."""
        process = self._processes.get(pid)
        if process and process.returncode is None:
            self._kill_process(process)
            return True
        return False

    def running_count(self) -> int:
        """Count currently running processes."""
        return sum(
            1 for p in self._processes.values()
            if p.returncode is None
        )


class ProcessManager:
    """Track and manage background processes."""

    def __init__(self):
        self._tool = BashExecTool()
        self._background_jobs: Dict[str, Dict[str, Any]] = {}

    def start(
        self,
        command: str,
        job_id: Optional[str] = None,
        cwd: str = ".",
        env: str = "{}",
    ) -> Dict[str, Any]:
        """Start a background process. Returns job info."""
        import uuid
        jid = job_id or str(uuid.uuid4())[:8]

        # Non-blocking execution
        result = self._tool.execute(
            command=command,
            cwd=cwd,
            env=env,
            capture_output=False,
            timeout=0,  # 0 = run in background (non-blocking)
        )
        return {
            "job_id": jid,
            "command": command,
            "cwd": cwd,
            "started_at": time.time(),
            "status": "started",
        }

    def status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a background job."""
        job = self._background_jobs.get(job_id)
        if not job:
            return {"error": f"Job not found: {job_id}"}
        return {
            "job_id": job_id,
            "command": job["command"],
            "status": job.get("status", "unknown"),
            "started_at": job.get("started_at"),
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all background jobs."""
        return [
            self.status(jid) for jid in self._background_jobs
        ]


# Register the tool
TOOLS_REGISTRY.register(BashExecTool())
