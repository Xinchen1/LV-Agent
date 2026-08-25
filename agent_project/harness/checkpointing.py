"""Checkpoint middleware -- every mutation leaves an undo trail.

Wraps an executor: before a mutating effect runs, each declared target path
is snapshotted via the existing :class:`CheckpointManager`; after it runs,
the outcome is recorded. Snapshots are reported through a callback so the
loop can journal CheckpointWritten -- the journal then doubles as an undo
index: every write in the log names the checkpoint that undoes it.

Read/pure/net effects pass through untouched; snapshots only happen where
the world actually changes.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..checkpoint import CheckpointManager
from .effects import Effect
from .kernel import Executor


class CheckpointMiddleware:
    """Executor wrapper adding snapshot-before-mutate semantics."""

    def __init__(
        self,
        manager: CheckpointManager,
        on_checkpoint: Optional[Callable[[str, str], None]] = None,
    ):
        self.manager = manager
        self.on_checkpoint = on_checkpoint or (lambda _cid, _desc: None)

    def wrap(self, executor: Executor) -> Executor:
        def _run(effect: Effect) -> str:
            if not effect.is_mutating or not effect.target_paths:
                return executor(effect)

            metas = []
            for path in effect.target_paths:
                meta = self.manager.snapshot(path, tag=effect.tool_name)
                metas.append(meta)
                if meta and meta.get("id"):
                    self.on_checkpoint(meta["id"], f"{effect.tool_name}:{meta['path']}")

            try:
                output = executor(effect)
            except Exception as exc:
                self._record_all(metas, False, str(exc))
                raise
            self._record_all(metas, True, None)
            return output

        return _run

    def _record_all(
        self, metas: List[dict], success: bool, error: Optional[str]
    ) -> None:
        for meta in metas:
            if meta and meta.get("id"):
                self.manager.record_result(meta["id"], success, error)
