"""Session facade -- one object wiring journal, loop, budget and bus.

Also owns **forks**: because state is a fold over events, forking a session
at event N is literally copying the first N events into a child journal and
resuming. No snapshot formats, no serializers to keep in sync.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .budget import Ledger, Limits
from .checkpointing import CheckpointMiddleware
from .context import ContextAssembler
from .events import CheckpointWritten, Event, SessionState, fold
from .journal import ForkedJournal, Journal
from .kernel import Kernel
from .loop import AgentLoop, Sampler
from .scheduler import Scheduler
from .stream import EventBus


class Session:
    """A runnable, resumable, forkable agent session."""

    def __init__(
        self,
        root: Path | str,
        sampler: Sampler,
        kernel: Optional[Kernel] = None,
        limits: Optional[Limits] = None,
        system_prompt: str = "You are a capable agent. Use tools when useful.",
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        pause_check: Optional[Callable[[], bool]] = None,
        context_budget_tokens: Optional[int] = None,
        checkpoint_manager: Optional[Any] = None,
        verify_final: bool = True,
        max_verification_rounds: int = 2,
        converge_on_stable: bool = True,
        max_model_retries: int = 3,
    ):
        self.session_id = session_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.root = Path(root)
        self.journal = Journal(self.root / f"{self.session_id}.jsonl")
        self.bus = EventBus()
        self.ledger = Ledger(limits or Limits())
        self.scheduler = Scheduler()
        self.kernel = kernel or Kernel()
        self.checkpoints = checkpoint_manager

        context = (
            ContextAssembler(system_prompt, budget_tokens=context_budget_tokens)
            if context_budget_tokens
            else None
        )

        if checkpoint_manager is not None:
            def _on_checkpoint(cid: str, desc: str) -> None:
                stamped = self.journal.append(
                    CheckpointWritten(checkpoint_id=cid, description=desc)
                )
                self.bus.emit_event(stamped)

            middleware = CheckpointMiddleware(checkpoint_manager, on_checkpoint=_on_checkpoint)
            self.kernel.executor = middleware.wrap(self.kernel.executor)

        self._loop = AgentLoop(
            kernel=self.kernel,
            scheduler=self.scheduler,
            ledger=self.ledger,
            journal=self.journal,
            bus=self.bus,
            sampler=sampler,
            system_prompt=system_prompt,
            tools_schema=tools_schema or [],
            pause_check=pause_check,
            context=context,
            verify_final=verify_final,
            max_verification_rounds=max_verification_rounds,
            converge_on_stable=converge_on_stable,
            max_model_retries=max_model_retries,
        )

    # ---------- running ----------

    async def run(self, task: str) -> str:
        return await self._loop.run(task)

    async def resume(self) -> str:
        return await self._loop.resume()

    # ---------- introspection ----------

    @property
    def state(self) -> SessionState:
        return fold(self.journal.read_all())

    def events(self) -> List[Event]:
        return self.journal.read_all()

    # ---------- forking ----------

    def fork(self, at_seq: int, session_id: Optional[str] = None) -> "Session":
        """Create a child session containing events [0..at_seq], resumable."""

        child = Session(
            root=self.root,
            sampler=self._loop.sampler,
            kernel=self.kernel,
            limits=self.ledger.limits,
            system_prompt=self._loop.system_prompt,
            tools_schema=self._loop.tools_schema,
            session_id=session_id,
            pause_check=self._loop.pause_check,
            context_budget_tokens=(
                self._loop.context.budget_tokens if self._loop.context else None
            ),
        )
        child.journal = ForkedJournal.from_parent(self.journal, at_seq, child.journal.path)
        return child

    # ---------- undo ----------

    def rollback(self, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Restore workspace state via the checkpoint manager (if configured)."""

        if self.checkpoints is None:
            return None
        return self.checkpoints.rollback(checkpoint_id)
