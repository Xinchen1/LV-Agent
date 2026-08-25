"""
Process Manager
Background process tracking with start/stop/status/list capabilities.
"""

import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from . import BaseTool, ToolResult, TOOLS_REGISTRY


class BackgroundProcess:
    """Represents a running or completed background process."""

    def __init__(self, pid: int, command: str, cwd: str):
        self.job_id = str(uuid.uuid4())[:8]
        self.pid = pid
        self.command = command
        self.cwd = cwd
        self.status = "running"
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.stdout: str = ""
        self.stderr: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pid": self.pid,
            "command": self.command,
            "cwd": self.cwd,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "elapsed": round(
                (self.finished_at or time.time()) - self.started_at, 2
            ),
        }


class ProcessManagerTool(BaseTool):
    """Manage long-running background processes."""

    name = "process_manager"
    description = (
        "Start, stop, and monitor background processes. "
        "Useful for running dev servers, daemons, or any long-lived command. "
        "Each background process gets a job_id for later control."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "status", "list", "logs"],
                "description": "Action to perform on processes.",
            },
            "command": {
                "type": "string",
                "description": "Shell command to run (required for 'start').",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the process.",
                "default": ".",
            },
            "job_id": {
                "type": "string",
                "description": "Job ID from a previous 'start' action.",
            },
            "stdout_lines": {
                "type": "integer",
                "description": "Number of recent stdout lines to return (for 'logs').",
                "default": 50,
            },
            "stderr_lines": {
                "type": "integer",
                "description": "Number of recent stderr lines to return (for 'logs').",
                "default": 50,
            },
        },
        "required": ["action"],
    }

    STATE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "bg_processes.json"

    def __init__(self):
        self._processes: Dict[str, BackgroundProcess] = {}
        self._subprocess_refs: Dict[str, Any] = {}
        self._load_state()

    def execute(
        self,
        action: str,
        command: str = "",
        cwd: str = ".",
        job_id: str = "",
        stdout_lines: int = 50,
        stderr_lines: int = 50,
    ) -> ToolResult:
        """Execute a process management action."""
        cwd = os.path.expanduser(cwd)

        if action == "start":
            if not command:
                return ToolResult(success=False, output="", error="command required for start")
            return self._start(command, cwd)
        elif action == "stop":
            if not job_id:
                return ToolResult(success=False, output="", error="job_id required for stop")
            return self._stop(job_id)
        elif action == "status":
            if not job_id:
                return self._list_all()
            return self._status(job_id)
        elif action == "list":
            return self._list_all()
        elif action == "logs":
            if not job_id:
                return ToolResult(success=False, output="", error="job_id required for logs")
            return self._logs(job_id, stdout_lines, stderr_lines)
        else:
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")

    def _start(self, command: str, cwd: str) -> ToolResult:
        """Start a background process."""
        from agent_project.tools.bash_exec import BashExecTool
        import subprocess

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            bg = BackgroundProcess(pid=proc.pid, command=command, cwd=cwd)
            self._processes[bg.job_id] = bg
            self._subprocess_refs[bg.job_id] = proc
            self._save_state()

            return ToolResult(
                success=True,
                output=f"Started: {bg.job_id} (PID {proc.pid})",
                metadata=bg.to_dict(),
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to start: {e}")

    def _stop(self, job_id: str) -> ToolResult:
        """Stop a running process."""
        bg = self._processes.get(job_id)
        if not bg:
            return ToolResult(success=False, output="", error=f"Job not found: {job_id}")

        if bg.status in ("stopped", "finished"):
            return ToolResult(
                success=True,
                output=f"Process {job_id} already {bg.status}",
                metadata=bg.to_dict(),
            )

        proc = self._subprocess_refs.get(job_id)
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(0.5)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                return ToolResult(success=False, output="", error=f"Failed to stop: {e}")

        bg.status = "stopped"
        bg.finished_at = time.time()
        bg.exit_code = -1
        self._save_state()

        return ToolResult(
            success=True,
            output=f"Stopped: {job_id} (PID {bg.pid})",
            metadata=bg.to_dict(),
        )

    def _status(self, job_id: str) -> ToolResult:
        """Check process status."""
        bg = self._processes.get(job_id)
        if not bg:
            return ToolResult(success=False, output="", error=f"Job not found: {job_id}")

        proc = self._subprocess_refs.get(job_id)
        if proc and bg.status == "running":
            ret = proc.poll()
            if ret is not None:
                bg.status = "finished"
                bg.exit_code = ret
                bg.finished_at = time.time()
                try:
                    bg.stdout = proc.stdout.read().decode("utf-8", errors="replace")[-4096:]
                    bg.stderr = proc.stderr.read().decode("utf-8", errors="replace")[-4096:]
                except Exception:
                    pass
                self._save_state()

        return ToolResult(
            success=True,
            output=str(bg.to_dict()),
            metadata=bg.to_dict(),
        )

    def _list_all(self) -> ToolResult:
        """List all tracked processes."""
        lines = []
        for bg in self._processes.values():
            state = bg.to_dict()
            lines.append(
                f"{bg.job_id} | PID {bg.pid} | {bg.status} | {state['elapsed']}s | {bg.command[:60]}"
            )
        output = "\n".join(lines) if lines else "No background processes"
        return ToolResult(success=True, output=output, metadata={
            "processes": [bg.to_dict() for bg in self._processes.values()]
        })

    def _logs(
        self, job_id: str, stdout_n: int, stderr_n: int
    ) -> ToolResult:
        """Get recent logs from a process."""
        bg = self._processes.get(job_id)
        if not bg:
            return ToolResult(success=False, output="", error=f"Job not found: {job_id}")

        proc = self._subprocess_refs.get(job_id)
        if proc and bg.status == "running":
            ret = proc.poll()
            if ret is not None:
                bg.status = "finished"
                bg.exit_code = ret
                bg.finished_at = time.time()
                try:
                    bg.stdout = proc.stdout.read().decode("utf-8", errors="replace")
                    bg.stderr = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass

        out_lines = bg.stdout.strip().splitlines()[-stdout_n:] if bg.stdout else []
        err_lines = bg.stderr.strip().splitlines()[-stderr_n:] if bg.stderr else []

        parts = []
        if out_lines:
            parts.append("[STDOUT]\n" + "\n".join(out_lines))
        if err_lines:
            parts.append("[STDERR]\n" + "\n".join(err_lines))
        output = "\n\n".join(parts) if parts else "(no output captured)"

        return ToolResult(success=True, output=output, metadata=bg.to_dict())

    def _save_state(self):
        """Persist process list to disk."""
        try:
            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            import json
            data = {
                jid: {
                    "pid": bg.pid,
                    "command": bg.command,
                    "cwd": bg.cwd,
                    "status": bg.status,
                    "started_at": bg.started_at,
                    "finished_at": bg.finished_at,
                    "exit_code": bg.exit_code,
                    "job_id": bg.job_id,
                }
                for jid, bg in self._processes.items()
            }
            self.STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_state(self):
        """Load persisted process list (no subprocess refs on restart)."""
        try:
            if not self.STATE_FILE.exists():
                return
            import json
            data = json.loads(self.STATE_FILE.read_text())
            for jid, info in data.items():
                bg = BackgroundProcess(
                    pid=info["pid"],
                    command=info["command"],
                    cwd=info.get("cwd", "."),
                )
                bg.job_id = info.get("job_id", jid)
                bg.status = info.get("status", "finished")
                bg.started_at = info.get("started_at", time.time())
                bg.finished_at = info.get("finished_at")
                bg.exit_code = info.get("exit_code")
                self._processes[jid] = bg
        except Exception:
            pass


# Register
TOOLS_REGISTRY.register(ProcessManagerTool())
