"""LV Agent web server — FastAPI + WebSocket bridge to OpenMythosAgent.

Run:  python web/server.py  (or: uvicorn web.server:app --host 0.0.0.0 --port 8787)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agent_project.config import load_config
from agent_project.agent import OpenMythosAgent
from agent_project.tools import TOOLS_REGISTRY, ToolResult

WORKSPACE_ROOT = ROOT / "data" / "web_workspaces"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LV Agent Web")

DEFAULT_CONFIG: Dict[str, Any] = {
    "backend": os.environ.get("LV_BACKEND", "openai"),
    "base_url": os.environ.get("LV_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"),
    "model": os.environ.get("LV_MODEL", "DeepSeek-V4-Flash"),
    "api_key": os.environ.get("LV_API_KEY", ""),
    "temperature": 0.7,
    "max_tokens": 4096,
}


class RemoteFileOpsTool:
    """代理 file_ops：把调用转发到前端用 File System Access API 在用户本地执行。"""
    name = "file_ops"

    def __init__(self, local_tool):
        self._local = local_tool
        self._ws: Optional[WebSocket] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending: Dict[int, list] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def bind(self, ws: WebSocket, loop: asyncio.AbstractEventLoop) -> None:
        self._ws = ws
        self._loop = loop

    @property
    def description(self):
        return self._local.description

    @property
    def parameters(self):
        return self._local.parameters

    def execute(self, **kwargs) -> ToolResult:
        if not self._ws or not self._loop:
            return self._local.execute(**kwargs)

        with self._lock:
            call_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            self._pending[call_id] = [event, None]

        msg = json.dumps({"type": "tool_call", "call_id": call_id, "tool": "file_ops", "args": kwargs})
        asyncio.run_coroutine_threadsafe(self._ws.send_text(msg), self._loop)

        if not event.wait(timeout=120):
            with self._lock:
                self._pending.pop(call_id, None)
            return ToolResult(success=False, output="", error="本地文件操作超时(120s)")

        with self._lock:
            r = self._pending.pop(call_id, [None, None])[1]

        if r is None:
            return ToolResult(success=False, output="", error="无结果")
        return ToolResult(
            success=r.get("success", False),
            output=r.get("output", ""),
            error=r.get("error", ""),
            metadata=r.get("metadata", {}),
        )

    def handle_result(self, call_id: int, result: dict) -> None:
        with self._lock:
            if call_id in self._pending:
                self._pending[call_id][1] = result
                self._pending[call_id][0].set()


class Session:
    def __init__(self) -> None:
        self.agent: Optional[OpenMythosAgent] = None
        self.workspace: Optional[Path] = None
        self.config_info: Dict[str, Any] = {}
        self.remote_file_ops: Optional[RemoteFileOpsTool] = None
        self._saved_file_ops = None
        self.artifacts: list[str] = []

    def build_agent(self, overrides: Dict[str, Any]) -> None:
        cfg = load_config()
        backend = (overrides.get("backend") or cfg.backend or "openai").strip()
        if backend not in {"openai", "deepseek", "openrouter", "anthropic", "openmythos"}:
            raise ValueError(f"unknown backend: {backend}")
        cfg.backend = backend
        seg: Dict[str, Any] = getattr(cfg, backend) if hasattr(cfg, backend) else {}
        if overrides.get("api_key"):
            seg["api_key"] = overrides["api_key"]
        if overrides.get("model"):
            seg["model"] = overrides["model"]
        if overrides.get("base_url"):
            seg["base_url"] = overrides["base_url"]
        if overrides.get("temperature") is not None:
            seg["temperature"] = float(overrides["temperature"])
        if overrides.get("max_tokens"):
            seg["max_tokens"] = int(overrides["max_tokens"])

        ws_dir = WORKSPACE_ROOT / uuid.uuid4().hex
        ws_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = ws_dir

        try:
            cfg.tools.file_ops.enabled = True
            cfg.tools.file_ops.allowed_dirs = [str(ws_dir)]
        except Exception:
            pass
        try:
            cfg.tools.code_exec.enabled = True
            cfg.tools.code_exec.timeout = 30
        except Exception:
            pass
        try:
            cfg.harness.workspace_root = str(ws_dir)
            cfg.harness.policy = "safe"
            cfg.harness.enabled = True
        except Exception:
            pass

        self.agent = OpenMythosAgent(cfg)
        self.config_info = {
            "backend": backend,
            "model": seg.get("model"),
            "base_url": seg.get("base_url"),
            "workspace": str(ws_dir),
        }

    async def run_task(self, task: str, ws: WebSocket) -> None:
        if self.agent is None:
            await ws.send_json({"type": "error", "message": "请先在设置面板配置模型与 API key"})
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        self.artifacts = []
        tool_output_text: list[str] = []

        def stream_cb(kind: str, token: str) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, token))
                if kind == "tool_result":
                    tool_output_text.append(token)
            except Exception:
                pass

        async def pump() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, token = item
                await ws.send_json({"type": "stream", "kind": kind, "token": token})

        pump_task = asyncio.create_task(pump())
        await ws.send_json({"type": "start", "task": task})
        try:
            result = await asyncio.to_thread(
                self.agent.run, task, stream_callback=stream_cb
            )
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"{e}", "trace": traceback.format_exc()})
            queue.put_nowait(None)
            await pump_task
            return
        finally:
            queue.put_nowait(None)
        await pump_task

        await self._collect_and_push_artifacts(result, ws, tool_output_text)

        payload = {
            "type": "done",
            "success": bool(result.get("success")),
            "final_answer": result.get("final_answer", ""),
            "metadata": result.get("metadata", {}),
            "outer_loops": result.get("outer_loops", 0),
            "thinking_steps": result.get("thinking_steps", 0),
            "artifacts": self.artifacts,
        }
        await ws.send_json(payload)

    async def _collect_and_push_artifacts(self, result: dict, ws: WebSocket, tool_outputs: list[str] = None) -> None:
        """从 agent 结果、工具输出和 workspace 中收集产物文件, base64 推送给前端."""
        import base64
        import re

        tool_outputs = tool_outputs or []
        all_text = (result.get("final_answer", "") or "") + "\n" + "\n".join(tool_outputs)
        await ws.send_json({"type": "stream", "kind": "status", "token": f"[debug] tool_outputs={len(tool_outputs)}, all_text_len={len(all_text)}"})

        for m in re.finditer(r"(?:PDF 已生成|文件已生成|已生成文件|文件位置)[:\s`]*([^\n`]+?\.\w+)", all_text):
            p = m.group(1).strip().strip("`").strip("'").strip('"')
            await ws.send_json({"type": "stream", "kind": "status", "token": f"[debug] regex matched: {p}, isfile={os.path.isfile(p)}"})
            if os.path.isfile(p) and p not in self.artifacts:
                self.artifacts.append(p)

        scan_dirs = [self.workspace, Path("/app"), Path.cwd()]
        for d in scan_dirs:
            if not d or not d.exists():
                continue
            count = 0
            for f in d.rglob("*"):
                try:
                    if f.is_file() and f.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".csv", ".json", ".txt", ".html", ".md"):
                        p = str(f)
                        if p not in self.artifacts and f.stat().st_size > 0:
                            self.artifacts.append(p)
                            count += 1
                except Exception:
                    continue
            await ws.send_json({"type": "stream", "kind": "status", "token": f"[debug] scanned {d}: +{count} artifacts"})

        await ws.send_json({"type": "stream", "kind": "status", "token": f"[debug] total artifacts: {len(self.artifacts)}"})

        for p in self.artifacts:
            try:
                size = os.path.getsize(p)
                if size > 10 * 1024 * 1024:
                    continue
                with open(p, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode("ascii")
                await ws.send_json({
                    "type": "artifact",
                    "filename": os.path.basename(p),
                    "path": p,
                    "size": size,
                    "data": data,
                })
            except Exception:
                pass


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "workspaces": str(WORKSPACE_ROOT)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session()
    loop = asyncio.get_running_loop()
    try:
        try:
            session.build_agent(DEFAULT_CONFIG)
            await ws.send_json({"type": "ready", "config": session.config_info, "default": True})
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"default agent init failed: {e}",
                                "trace": traceback.format_exc()})
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid json"})
                continue
            mtype = msg.get("type")
            if mtype == "config":
                try:
                    session.build_agent(msg.get("config", {}))
                    await ws.send_json({"type": "ready", "config": session.config_info})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"agent init failed: {e}",
                                        "trace": traceback.format_exc()})
            elif mtype == "fs_ready":
                try:
                    local = TOOLS_REGISTRY.get("file_ops")
                    if local and not session.remote_file_ops:
                        session._saved_file_ops = local
                        session.remote_file_ops = RemoteFileOpsTool(local)
                        session.remote_file_ops.bind(ws, loop)
                        TOOLS_REGISTRY._tools["file_ops"] = session.remote_file_ops
                    await ws.send_json({"type": "fs_status", "enabled": True, "folder": msg.get("folder", "")})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"fs bind failed: {e}"})
            elif mtype == "fs_off":
                if session._saved_file_ops:
                    TOOLS_REGISTRY._tools["file_ops"] = session._saved_file_ops
                    session.remote_file_ops = None
                    session._saved_file_ops = None
                await ws.send_json({"type": "fs_status", "enabled": False})
            elif mtype == "tool_result":
                if session.remote_file_ops:
                    session.remote_file_ops.handle_result(msg.get("call_id", -1), msg.get("result", {}))
            elif mtype == "message":
                task = (msg.get("task") or "").strip()
                if not task:
                    await ws.send_json({"type": "error", "message": "empty task"})
                    continue
                await session.run_task(task, ws)
            elif mtype == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": f"{e}"})
        except Exception:
            pass


frontend_dir = Path(__file__).resolve().parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


async def _keepalive_loop():
    """Render 免费版 15min 无外部访问会休眠; 每 14min ping 自己保持活跃.

    Render 自动设置 RENDER_EXTERNAL_URL 环境变量。服务被唤醒一次后,
    后台任务持续运行, 不依赖本地 cron 或 GitHub Actions。
    """
    external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not external_url:
        return
    health_url = f"{external_url.rstrip('/')}/api/health"
    await asyncio.sleep(60)
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "-m", "10", health_url,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=15)
        except Exception:
            pass
        await asyncio.sleep(14 * 60)


@app.on_event("startup")
async def _start_keepalive():
    asyncio.create_task(_keepalive_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8787")))