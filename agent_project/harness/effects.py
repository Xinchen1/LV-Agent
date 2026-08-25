"""Effect intents -- tools declare *what*, the kernel decides *if/how*.

A tool never touches the filesystem, network or process table directly.
It returns (or is wrapped into) an :class:`Effect`: a pure-data description
of the desired world-interaction. The capability kernel then:

1. classifies the effect (read / write / exec / net),
2. evaluates policy (allow / deny / ask),
3. assigns a scheduler lane (writes serialize per path, reads parallelize),
4. executes and audits the outcome.

This is the microkernel boundary: policy lives in one place, tools stay dumb
and testable, and new restrictions never require editing tool code.
"""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field


class EffectClass(str, enum.Enum):
    READ = "read"        # observes world state; no mutation
    WRITE = "write"      # mutates files/state; serializes per path
    EXEC = "exec"        # runs a subprocess; highest scrutiny
    NET = "net"          # network egress
    PURE = "pure"        # no world interaction (calculator, parser)


class Effect(BaseModel):
    """A single intended world-interaction, before policy/execution."""

    model_config = {"frozen": True}

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    effect_class: EffectClass = EffectClass.PURE

    #: Paths the effect reads/writes, for lane assignment and policy.
    #: Write effects MUST declare at least one path (or be class EXEC/NET).
    target_paths: Tuple[str, ...] = ()

    #: Optional capability the caller asserts it holds; the kernel verifies
    #: against the active policy profile instead of trusting the caller.
    capability: Optional[str] = None

    timeout_s: Optional[float] = None

    @property
    def idempotency_key(self) -> str:
        """Stable content hash: identical intents deduplicate in a turn."""

        blob = json.dumps(
            {"t": self.tool_name, "a": self.arguments, "c": self.effect_class.value},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    @property
    def is_mutating(self) -> bool:
        return self.effect_class in (EffectClass.WRITE, EffectClass.EXEC)


# ---------- classification helpers ----------

#: Conservative defaults; tools may override via ``tool.effect_spec()``.
_TOOL_CLASSES: Dict[str, EffectClass] = {
    "web_search": EffectClass.NET,
    "api_call": EffectClass.NET,
    "playwright_browser": EffectClass.NET,
    "telegram_bot": EffectClass.NET,
    "github_search": EffectClass.NET,
    "weather": EffectClass.NET,
    "calculator": EffectClass.PURE,
    "python_exec": EffectClass.EXEC,
    "bash_exec": EffectClass.EXEC,
    "process_manager": EffectClass.EXEC,
    "mcp_setup": EffectClass.NET,
    "file_ops": EffectClass.WRITE,
    "git_ops": EffectClass.WRITE,
    "git": EffectClass.WRITE,
    "database": EffectClass.WRITE,
    "grep": EffectClass.READ,
    "search_files": EffectClass.READ,
    "glob": EffectClass.READ,
    "project_context": EffectClass.READ,
    "browser": EffectClass.NET,
    "mcp_client": EffectClass.NET,
}

_PATH_ARGS = ("path", "file", "file_path", "filepath", "filename", "target", "output_path", "repo_path", "cwd")


def classify(tool_name: str, arguments: Dict[str, Any]) -> Tuple[EffectClass, Tuple[str, ...]]:
    """Infer effect class + target paths from a tool name and arguments."""

    cls = _TOOL_CLASSES.get(tool_name, EffectClass.EXEC)
    paths = tuple(
        str(v) for k, v in arguments.items() if k in _PATH_ARGS and isinstance(v, (str, bytes))
    )
    # file_ops is read OR write depending on its operation argument.
    if tool_name == "file_ops":
        op = str(
            arguments.get("operation") or arguments.get("action") or arguments.get("op") or ""
        ).lower()
        read_ops = {
            "read", "multi_read", "fast_read", "list", "exists", "stat", "info",
            "tail", "head", "grep", "find", "analyze", "diff", "verify",
        }
        if op in read_ops:
            cls = EffectClass.READ
        else:
            cls = EffectClass.WRITE
    if tool_name == "git_ops":
        op = str(
            arguments.get("operation") or arguments.get("action") or arguments.get("op") or ""
        ).lower()
        if op in ("status", "log", "diff", "show", "blame", "branch"):
            cls = EffectClass.READ
    if tool_name == "git":
        op = str(
            arguments.get("command") or arguments.get("cmd") or arguments.get("action") or ""
        ).lower()
        if op.split()[0] in ("status", "log", "diff", "show", "blame", "branch", "remote", "ls-files"):
            cls = EffectClass.READ
    return cls, paths


def make_effect(tool_name: str, arguments: Dict[str, Any], **kw: Any) -> Effect:
    """Build an :class:`Effect` from a legacy (tool_name, args) call."""

    cls, paths = classify(tool_name, arguments)
    return Effect(
        tool_name=tool_name,
        arguments=arguments,
        effect_class=cls,
        target_paths=kw.pop("target_paths", paths),
        **kw,
    )
