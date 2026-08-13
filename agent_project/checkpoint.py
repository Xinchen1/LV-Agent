"""
CheckpointManager - lightweight project snapshots for the agent.

Captures file state before modifications and reasoning state at key moments,
allowing rollback and post-mortem analysis when tasks fail.
"""

import hashlib
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckpointManager:
    """Manage file + state snapshots for a single agent session.

    注意: 无参 `CheckpointManager()` 复用进程内共享的默认会话目录,
    这样 file_ops 写入前的 snapshot 与 reasoning/execution 失败时的
    rollback_latest 才能落在同一个会话里(否则每次 new 新实例都会
    得到空的 session 目录, 回滚永远 no-op).
    """

    _DEFAULT_SESSION_ID: Optional[str] = None

    def __init__(self, session_id: Optional[str] = None, root_dir: Optional[Path] = None):
        if session_id is None:
            if CheckpointManager._DEFAULT_SESSION_ID is None:
                CheckpointManager._DEFAULT_SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_id = CheckpointManager._DEFAULT_SESSION_ID
        self.session_id = session_id
        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[1] / "data" / "checkpoints"
        self.root_dir = Path(root_dir)
        self.session_dir = self.root_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.session_dir / f"checkpoint_{checkpoint_id}.json"

    def _copy_path(self, checkpoint_id: str, original: Path) -> Path:
        safe_name = original.name.replace("/", "_").replace("\\", "_")
        return self.session_dir / f"{checkpoint_id}_{safe_name}"

    @staticmethod
    def _file_hash(path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except Exception:
            return ""
        return h.hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def parse_rust_backup_path(output: str) -> Optional[str]:
        if not output:
            return None
        marker = "Backup saved to "
        idx = output.find(marker)
        if idx == -1:
            return None
        path = output[idx + len(marker):].split("\n", 1)[0].strip()
        return path or None

    def snapshot(self, path: str, tag: str = "auto") -> Optional[Dict[str, Any]]:
        """Record the current state of a file before it is modified."""
        original = Path(path).resolve()
        checkpoint_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        copy_path: Optional[str] = None
        file_hash = ""

        if original.exists() and original.is_file():
            file_hash = self._file_hash(original)
            try:
                dst = self._copy_path(checkpoint_id, original)
                shutil.copy2(original, dst)
                copy_path = str(dst)
            except Exception as e:
                return {
                    "id": checkpoint_id,
                    "path": str(original),
                    "tag": tag,
                    "exists_before": True,
                    "error": f"copy failed: {e}",
                    "created_at": self._now(),
                }

        meta = {
            "id": checkpoint_id,
            "path": str(original),
            "tag": tag,
            "exists_before": original.exists(),
            "hash_before": file_hash,
            "copy_path": copy_path,
            "rust_backup_path": None,
            "success": None,
            "error": None,
            "created_at": self._now(),
        }
        try:
            self._checkpoint_path(checkpoint_id).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            return {**meta, "error": f"metadata write failed: {e}"}
        return meta

    def record_result(
        self,
        checkpoint_id: str,
        success: bool,
        error: Optional[str] = None,
        rust_backup_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a checkpoint after the operation finishes."""
        cp_path = self._checkpoint_path(checkpoint_id)
        if not cp_path.exists():
            return None
        try:
            meta = json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        meta["success"] = success
        meta["error"] = error
        meta["rust_backup_path"] = rust_backup_path
        meta["finished_at"] = self._now()
        try:
            cp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            meta["error"] = f"{meta.get('error') or ''}; result save failed: {e}".strip("; ")
        return meta

    def list_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent checkpoints in this session."""
        checkpoints: List[Dict[str, Any]] = []
        if not self.session_dir.exists():
            return checkpoints
        for p in sorted(self.session_dir.glob("checkpoint_*.json"), reverse=True):
            try:
                checkpoints.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
            if len(checkpoints) >= n:
                break
        return checkpoints

    def rollback(self, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Restore a file to the state captured in the given checkpoint.

        当 checkpoint_id 为 None 时, 只回滚"未成功完成"的最新快照
        (success 不是 True 的), 避免把一次成功写入的文件错误还原。
        """
        if checkpoint_id is None:
            recent = self.list_recent(n=100)
            # 跳过已成功完成的快照, 只考虑未完成/失败/未知结果的
            recent = [c for c in recent if c.get("success") is not True]
            if not recent:
                return None
            meta = recent[0]
        else:
            cp_path = self._checkpoint_path(checkpoint_id)
            if not cp_path.exists():
                return None
            try:
                meta = json.loads(cp_path.read_text(encoding="utf-8"))
            except Exception:
                return None

        original = Path(meta["path"])
        copy_path = meta.get("copy_path")
        restored = False
        error = None

        try:
            if copy_path and Path(copy_path).exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(copy_path, original)
                restored = True
            elif not meta.get("exists_before", True):
                if original.exists():
                    original.unlink()
                restored = True
            else:
                error = "no snapshot copy available and original was expected to exist"
        except Exception as e:
            error = f"restore failed: {e}\n{traceback.format_exc()}"

        return {
            "checkpoint_id": meta.get("id"),
            "path": str(original),
            "restored": restored,
            "error": error,
            "rolled_back_at": self._now(),
        }

    def rollback_latest(self) -> Optional[Dict[str, Any]]:
        """Convenience wrapper: rollback the most recent checkpoint."""
        return self.rollback(None)

    def cleanup(self, keep: int = 50) -> int:
        """Remove old checkpoints, keeping the most recent `keep`."""
        removed = 0
        if not self.session_dir.exists():
            return removed
        all_paths = sorted(self.session_dir.glob("checkpoint_*.json"))
        for p in all_paths[:-keep] if keep > 0 else all_paths:
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                copy_path = meta.get("copy_path")
                if copy_path:
                    Path(copy_path).unlink(missing_ok=True)
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        return removed
