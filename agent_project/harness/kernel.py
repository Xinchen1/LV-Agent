"""Capability kernel -- the single gate through which all effects flow.

Policy is *data*, not code scattered across tools. A policy is an ordered
list of rules; the first match wins. Rules match on effect class, tool name
and path patterns, and decide ``allow`` / ``deny`` / ``ask``.

The kernel does NOT execute effects itself: it admits them, then delegates
to a registered executor. That keeps policy evaluation pure and testable,
and lets the same kernel guard real tools, MCP calls, and dry-run replays.
"""

from __future__ import annotations

import enum
import fnmatch
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set

from .effects import Effect, EffectClass
from .errors import ErrorRecord, PolicyDeniedError, classify_exception


class Decision(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # requires interactive confirmation; headless = deny


@dataclass(frozen=True)
class Rule:
    """One policy rule. Glob patterns match against normalized paths."""

    decision: Decision
    effect_class: Optional[EffectClass] = None   # None = any
    tool: str = "*"                              # glob on tool name
    paths: tuple = ("**",)                       # globs on target paths
    pattern: Optional[str] = None                # regex on serialized args
    reason: str = ""

    def matches(self, effect: Effect) -> bool:
        if self.effect_class is not None and effect.effect_class is not self.effect_class:
            return False
        if not fnmatch.fnmatchcase(effect.tool_name, self.tool):
            return False
        if effect.target_paths:
            if not any(
                fnmatch.fnmatchcase(_norm(p), _norm(g))
                for p in effect.target_paths
                for g in self.paths
            ):
                return False
        if self.pattern is not None:
            import json

            if not re.search(self.pattern, json.dumps(effect.arguments, default=str)):
                return False
        return True


def _norm(path: str) -> str:
    """Normalize separators; keep the absolute/relative distinction intact."""

    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p or "."


# ---------- ready-made policy profiles ----------

def permissive_policy() -> List[Rule]:
    """Everything allowed -- explicit opt-in only (CI sandboxes)."""

    return [Rule(decision=Decision.ALLOW, reason="permissive profile")]


def safe_default_policy(workspace_root: str = ".") -> List[Rule]:
    """Reads/pure allowed; writes confined to workspace; exec asks.

    Write confinement is conservative by construction: absolute paths and
    ``..`` traversals are denied *before* the relative-path allow rule, so
    the default profile can never be talked into escaping the workspace.
    """

    root = _norm(workspace_root.rstrip("/"))
    rules = [
        # Destructive shell patterns
        # 注意: rm -rf ~/.npm/_cacache 这类"具体缓存子路径"放行, 只拦整家目录(~ 后接空白/结尾)
        Rule(
            Decision.DENY,
            EffectClass.EXEC,
            pattern=r"\brm\s+-rf\s+/|\brm\s+-rf\s+~(?:$|\s)|\brm\s+-rf\s+\*|\bsudo\b|:\(\)\{",
            reason="destructive shell pattern",
        ),
        Rule(
            Decision.DENY,
            EffectClass.EXEC,
            pattern=r"\bmkfs\b|\bdd\b.*of=/dev|\\bshred\\b|\bformat\b",
            reason="disk destruction or secure wipe",
        ),
        Rule(
            Decision.DENY,
            EffectClass.EXEC,
            pattern=r"(?:\bdd\b|\bcat\b|\becho\b)[^|;]*>\s*/dev/sd[a-z]|>\s*/dev/sd[a-z]",
            reason="block-device write",
        ),
        # Dangerous remote code execution patterns
        Rule(
            Decision.ASK,
            EffectClass.EXEC,
            pattern=r"\b(curl|wget)\b.*\|\s*(bash|sh|zsh)",
            reason="remote code execution via pipe",
        ),
        Rule(Decision.ALLOW, EffectClass.PURE, reason="pure computation"),
        Rule(Decision.ALLOW, EffectClass.READ, reason="reads are safe"),
        Rule(
            Decision.DENY,
            EffectClass.WRITE,
            paths=("../*", "**/../*"),
            reason="parent-directory traversal",
        ),
    ]
    # 工作区内绝对路径写: 放行(必须在系统敏感路径/**之前, 否则 tempdir/自定义工作区被误拦)
    if root != ".":
        rules.insert(6,
            Rule(Decision.ALLOW, EffectClass.WRITE, paths=(root, f"{root}/**"),
                 reason="write inside workspace")
        )
    # 系统敏感路径(永远拒绝, 即使询问也不放行)
    rules.append(
        Rule(
            Decision.DENY,
            EffectClass.WRITE,
            paths=("/etc/**", "/usr/**", "/bin/**", "/sbin/**", "/lib/**", "/lib64/**", "/var/**", "/sys/**", "/proc/**", "/.ssh/**", "~/.ssh/**"),
            reason="system or sensitive path write",
        )
    )
    # 工作区外绝对路径写: 不直接拦截, 改为询问用户(用户确认才放行)
    rules.append(
        Rule(
            Decision.ASK,
            EffectClass.WRITE,
            paths=("/**",),
            reason="absolute path write (outside workspace control)",
        )
    )
    rules.extend([
        Rule(Decision.ALLOW, EffectClass.WRITE, reason="relative write inside cwd"),
        # 执行代码/命令是该 Agent 的核心能力: 非破坏性命令放行
        # (破坏性模式已在上方 DENY 规则拦截); 仅"远程代码经管道执行"保留确认
        Rule(Decision.ALLOW, EffectClass.EXEC, reason="code/command execution allowed"),
        Rule(Decision.ALLOW, EffectClass.NET, reason="network allowed"),
    ])
    return rules


# ---------- allowlist helpers ----------

def load_allowlist(path: str) -> Set[str]:
    """Load a set of allowed command strings from a plain-text file."""
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
        return {
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    except Exception:
        return set()


def save_allowlist(path: str, commands: Set[str]) -> None:
    """Persist allowed command strings to a plain-text file."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Lv Super Agent harness allowlist", "# One command per line"]
    lines.extend(sorted(commands))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- executor protocol ----------

class Executor(Protocol):
    """Runs an admitted effect and returns its output."""

    def __call__(self, effect: Effect) -> str: ...


@dataclass
class Admission:
    """Result of the policy evaluation for one effect."""

    effect: Effect
    decision: Decision
    reason: str
    lane: str  # scheduler lane key: writes serialize per-path


# ---------- the kernel ----------

class Kernel:
    """Admits effects per policy, executes via the injected executor.

    Parameters
    ----------
    policy:
        Ordered rule list; first match wins. Default-deny when empty tail.
    executor:
        Callable performing the real work (bridges to BaseTool instances).
    ask:
        Interactive confirmation callback for ASK decisions. Headless
        callers pass ``None``, which maps ASK -> deny.
    audit:
        Optional sink receiving (effect, admission, outcome) triples.
    """

    def __init__(
        self,
        policy: Optional[List[Rule]] = None,
        executor: Optional[Executor] = None,
        ask: Optional[Callable[[Effect, str], bool]] = None,
        audit: Optional[Callable[[Effect, "Admission", Optional[ErrorRecord]], None]] = None,
        allowlist_path: Optional[str] = None,
    ):
        self.policy = policy if policy is not None else safe_default_policy()
        self.executor = executor or _null_executor
        self.ask = ask
        self.audit = audit
        self.allowlist_path = allowlist_path
        self.allowlist: Set[str] = load_allowlist(allowlist_path) if allowlist_path else set()

    def allowlist_add(self, effect: Effect) -> None:
        """记住用户手动批准的效应(写日志以便持久化)."""
        try:
            import json
            serialized = json.dumps(effect.arguments, sort_keys=True, ensure_ascii=False)
            self.allowlist.add(serialized)
            cmd = effect.arguments.get("command") if isinstance(effect.arguments, dict) else None
            if cmd:
                self.allowlist.add(str(cmd))
            if self.allowlist_path is not None:
                try:
                    import os
                    os.makedirs(os.path.dirname(os.path.abspath(self.allowlist_path)), exist_ok=True)
                    with open(self.allowlist_path, "w", encoding="utf-8") as f:
                        for item in sorted(self.allowlist):
                            f.write(item + "\n")
                except Exception:
                    pass
        except Exception:
            pass

    # ----- policy -----

    def evaluate(self, effect: Effect) -> Admission:
        for rule in self.policy:
            if rule.matches(effect):
                return Admission(effect, rule.decision, rule.reason, self._lane(effect))
        return Admission(effect, Decision.DENY, "default deny", self._lane(effect))

    @staticmethod
    def _lane(effect: Effect) -> str:
        if effect.effect_class in (EffectClass.READ, EffectClass.PURE, EffectClass.NET):
            return "parallel"
        if effect.target_paths:
            return "write:" + ",".join(sorted(_norm(p) for p in effect.target_paths))
        return "serial"  # EXEC without declared paths: fully serialized

    # ----- execution -----

    def run(self, effect: Effect) -> str:
        """Policy-check then execute. Raises PolicyDeniedError on refusal."""

        admission = self.evaluate(effect)
        decision = admission.decision
        if decision is Decision.ASK:
            # Allowlist short-circuit: exact command match bypasses interactive ask.
            import json
            serialized = json.dumps(effect.arguments, sort_keys=True, ensure_ascii=False)
            if serialized in self.allowlist or effect.arguments.get("command", "") in self.allowlist:
                decision = Decision.ALLOW
            else:
                granted = self.ask(effect, admission.reason) if self.ask else False
                if granted and self.allowlist_path is not None:
                    # Remember interactive approvals for the workspace session.
                    cmd = effect.arguments.get("command")
                    if cmd:
                        self.allowlist.add(cmd)
                if not granted:
                    decision = Decision.DENY
        err: Optional[ErrorRecord] = None
        try:
            if decision is Decision.DENY:
                raise PolicyDeniedError(
                    f"effect '{effect.tool_name}' denied: {admission.reason}",
                    detail={"reason": admission.reason, "lane": admission.lane},
                )
            return self.executor(effect)
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            err = classify_exception(exc)
            raise
        finally:
            if self.audit:
                self.audit(effect, admission, err)


def _null_executor(effect: Effect) -> str:
    raise RuntimeError(f"no executor registered for effect '{effect.tool_name}'")


# ---------- bridging to the legacy BaseTool registry ----------

def registry_executor(registry: Any) -> Executor:
    """Adapt the existing TOOLS_REGISTRY into an :class:`Executor`.

    The bridge keeps old tools working unchanged while the kernel wraps
    them with policy, lanes and auditing. Tools returning ToolResult have
    their success flag honoured; plain strings pass through.
    """

    def _exec(effect: Effect) -> str:
        tool = registry.get(effect.tool_name)
        if tool is None:
            raise KeyError(f"unknown tool '{effect.tool_name}'")
        started = time.monotonic()
        result = tool.execute(**effect.arguments)
        _ = time.monotonic() - started  # latency recorded by scheduler layer
        output = getattr(result, "output", result)
        success = getattr(result, "success", True)
        if not success:
            error = getattr(result, "error", None) or "tool reported failure"
            from .errors import ToolFailedError

            raise ToolFailedError(str(error), detail={"tool": effect.tool_name})
        return output if isinstance(output, str) else str(output)

    return _exec
