"""LV Agent web server — FastAPI + WebSocket bridge to OpenMythosAgent.

Run:  python web/server.py  (or: uvicorn web.server:app --host 0.0.0.0 --port 8787)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
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


class Session:
    def __init__(self) -> None:
        self.agent: Optional[OpenMythosAgent] = None
        self.workspace: Optional[Path] = None
        self.config_info: Dict[str, Any] = {}

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
            cfg.tools.file_ops.allowed_dirs = [str(ws_dir)]
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

        def stream_cb(kind: str, token: str) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, token))
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

        payload = {
            "type": "done",
            "success": bool(result.get("success")),
            "final_answer": result.get("final_answer", ""),
            "metadata": result.get("metadata", {}),
            "outer_loops": result.get("outer_loops", 0),
            "thinking_steps": result.get("thinking_steps", 0),
        }
        await ws.send_json(payload)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "workspaces": str(WORKSPACE_ROOT)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8787")))