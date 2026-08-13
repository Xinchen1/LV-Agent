"""OpenMythos Harness -- an event-sourced agent runtime.

Layers (bottom-up):

* ``errors``    -- typed failure taxonomy (kind x retriable).
* ``events``    -- immutable events; session state is a fold over them.
* ``journal``   -- append-only JSONL log with crash-safe replay and forks.
* ``effects``   -- tools declare intents (WHAT), never execute directly.
* ``kernel``    -- capability kernel: policy-as-data decides IF/HOW.
* ``budget``    -- token/time/call/cost ledger + failure-dynamic breakers.
* ``scheduler`` -- lanes: reads parallel, writes serialize per-path, dedup.
* ``stream``    -- typed event bus; frontends are plain subscribers.
* ``loop``      -- continuation-based turn loop (pause/resume/fork).
* ``session``   -- facade wiring it all together.
* ``bridge``    -- adapters running the legacy agent stack on the harness.
"""

from .budget import Ledger, Limits
from .bridge import BackendSampler, build_kernel_from_registry, parse_tool_blocks
from .checkpointing import CheckpointMiddleware
from .context import ContextAssembler, TruncatingCompactor
from .effects import Effect, EffectClass, make_effect
from .errors import ErrorKind, ErrorRecord, HarnessError
from .events import Event, SessionState, fold
from .journal import ForkedJournal, Journal
from .kernel import Decision, Kernel, Rule, safe_default_policy
from .loop import AgentLoop, SampleResult, rebuild_messages
from .scheduler import Outcome, Scheduler
from .session import Session
from .stream import Delta, EventBus

__version__ = "0.1.0"

__all__ = [
    "AgentLoop",
    "BackendSampler",
    "CheckpointMiddleware",
    "ContextAssembler",
    "Decision",
    "Delta",
    "Effect",
    "EffectClass",
    "ErrorKind",
    "ErrorRecord",
    "Event",
    "EventBus",
    "ForkedJournal",
    "HarnessError",
    "Journal",
    "Kernel",
    "Ledger",
    "Limits",
    "Outcome",
    "Rule",
    "SampleResult",
    "Scheduler",
    "Session",
    "SessionState",
    "TruncatingCompactor",
    "build_kernel_from_registry",
    "fold",
    "make_effect",
    "parse_tool_blocks",
    "rebuild_messages",
    "safe_default_policy",
]
