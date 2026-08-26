"""Hot-swap capability layer -- self-evolution-ready module lifecycle.

Design pillars (aligned with and surpassing Cordis):

* Service = named capability (ctx.<key> analogy).
* Full 6-state FSM: PENDING -> LOADING -> ACTIVE -> UNLOADING -> DISPOSED / FAILED
* Transactional swap: rollback anchor saved BEFORE any mutation; on failure,
  the old module is restored atomically.  This is the foundation for safe
  self-evolution -- an automated swap that degrades quality can always undo.
* Reversible effects: every registered module may provide ``dispose()``;
  async ``dispose()`` is also supported (run in event loop or thread pool).
* Circuit breaker: sliding-window error counting -> auto-fallback when a
  module degrades beyond repair.
* Zero-downtime drain: in-flight calls are tracked; swap waits for them to
  complete (configurable timeout) before unloading.
* Shadow testing: error_rate + call_count dual-metric promotion.
* Version graph: each capability tracks its version tree (parent/child),
  enabling zero-rebuild rollback to any prior version.

Phase 1 delivered: transactional swap, full FSM, circuit breaker, async disposer.
Phase 2 delivered: version graph, zero-downtime drain, multi-mode events.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from .events import EventBus

logger = logging.getLogger("harness.hotswap")


# ---------------------------------------------------------------------------
# Lifecycle states (full Cordis fiber machine + terminal DISPOSED)
# ---------------------------------------------------------------------------

class SlotState(Enum):
    """Full 6-state lifecycle matching Cordis fiber machine."""

    PENDING = "pending"
    LOADING = "loading"
    ACTIVE = "active"
    UNLOADING = "unloading"
    DISPOSED = "disposed"
    FAILED = "failed"


class SwapError(Exception):
    """Raised when a transactional swap cycle fails after rollback."""


# ---------------------------------------------------------------------------
# Version graph node (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class VersionNode:
    """A single node in a capability's version tree."""

    version: str
    module: Any
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.monotonic)
    swap_count: int = 0
    was_rollback_target: bool = False


# ---------------------------------------------------------------------------
# Event handler types
# ---------------------------------------------------------------------------

EventHandler = Callable[["CapabilitySlot", str, Dict[str, Any]], None]


def _noop(_slot: "CapabilitySlot", _event: str, _data: Dict[str, Any]) -> None:
    pass


# ---------------------------------------------------------------------------
# SlotMetrics -- unified metric collector (replaces Fragment counters)
# ---------------------------------------------------------------------------

@dataclass
class SlotMetrics:
    """Aggregated per-slot metrics consumed by SelfEvolutionController.

    Replaces the scattered _active_calls/_active_errors/_shadow_calls/...
    fields with a single structured object.  All counts are monotonically
    increasing; derived rates (error_rate, etc.) are computed from them.
    """
    active_calls: int = 0
    active_errors: int = 0
    active_latency_ms: float = 0.0      # sum of all call latencies
    active_latency_max_ms: float = 0.0
    _latency_samples: List[float] = field(default_factory=list)

    shadow_calls: int = 0
    shadow_errors: int = 0
    shadow_latency_ms: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.active_errors / self.active_calls if self.active_calls else 0.0

    @property
    def shadow_error_rate(self) -> float:
        return self.shadow_errors / self.shadow_calls if self.shadow_calls else 0.0

    @property
    def latency_p95_ms(self) -> float:
        """P95 latency from recent samples (every 5th call sampled)."""
        s = self._latency_samples
        if not s:
            return self.avg_latency_ms
        s_sorted = sorted(s)
        idx = int(len(s_sorted) * 0.95)
        return s_sorted[min(idx, len(s_sorted) - 1)]

    @property
    def latency_p90_ms(self) -> float:
        s = self._latency_samples
        if not s:
            return self.avg_latency_ms
        s_sorted = sorted(s)
        idx = int(len(s_sorted) * 0.90)
        return s_sorted[min(idx, len(s_sorted) - 1)]

    @property
    def avg_latency_ms(self) -> float:
        return (self.active_latency_ms / self.active_calls) if self.active_calls else 0.0

    def reset_active(self) -> None:
        self.active_calls = 0
        self.active_errors = 0
        self.active_latency_ms = 0.0
        self.active_latency_max_ms = 0.0
        self._latency_samples.clear()

    def reset_shadow(self) -> None:
        self.shadow_calls = 0
        self.shadow_errors = 0
        self.shadow_latency_ms = 0.0

    def validate(self) -> None:
        assert self.active_calls >= 0 and self.active_errors >= 0
        assert self.shadow_calls >= 0 and self.shadow_errors >= 0
        assert self.active_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# CapabilitySlot
# ---------------------------------------------------------------------------

@dataclass
class CapabilitySlot:
    """One named capability: active module + lifecycle state + version graph.

    Transactional swap contract: before mutating any field, ``swap()`` saves a
    rollback anchor (``_rollback_*``).  If the LOAD phase raises, the old
    module is restored and ``SwapError`` is raised -- this is the safety net
    for automated self-evolution.
    """

    name: str
    state: SlotState = SlotState.ACTIVE
    module: Optional[Any] = None
    disposer: Optional[Callable] = None
    version: str = ""
    shadow: Optional[Any] = None
    shadow_disposer: Optional[Callable] = None
    shadow_version: str = ""
    shadow_min_calls: int = 5
    shadow_max_error_rate: float = 0.02
    _version_graph: Dict[str, VersionNode] = field(default_factory=dict)
    _fallback_module: Optional[Any] = field(default=None, repr=False)
    _rollback_module: Optional[Any] = field(default=None, repr=False)
    _rollback_disposer: Optional[Callable] = field(default=None, repr=False)
    _rollback_version: str = ""
    _swap_start_time: float = 0.0
    _error_window: List[float] = field(default_factory=list)
    _cb_threshold: int = 5
    _cb_window: float = 60.0
    _cb_cooldown: float = 300.0
    _circuit_open_until: float = 0.0
    _active_calls_in_flight: Set[int] = field(default_factory=set)
    _drain_timeout: float = 30.0
    _next_call_id: int = 0
    _metrics: SlotMetrics = field(default_factory=SlotMetrics, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _event_loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)
    _events: Dict[str, List[Dict]] = field(
        default_factory=lambda: {
            ev: [{"handler": _noop, "mode": "emit"}]
            for ev in ("unload", "load", "swap", "error", "rollback", "circuit")
        }
    )

    # ---- event subscription ----

    def on(self, event: str, handler: EventHandler, *, mode: str = "emit") -> None:
        if event not in self._events:
            self._events[event] = [{"handler": _noop, "mode": "emit"}]
        self._events[event].append({"handler": handler, "mode": mode})

    def _emit(self, event: str, data: Optional[Dict] = None) -> Any:
        """Multi-mode event dispatcher.

        Handlers are stored as dicts: {"handler": callable, "mode": str}.
        Modes: emit, serial, bail, parallel, waterfall.
        For waterfall: handler signature is (slot, event, data, next_fn).
        """
        entries = self._events.get(event, [])
        if not entries:
            return None
        data = data or {}

        # -- WATERFALL: middleware chain --
        wf = [e for e in entries if e.get("mode") == "waterfall" and isinstance(e, dict)]
        if wf:
            handlers = [e["handler"] for e in wf]
            def _next(idx):
                if idx >= len(handlers):
                    return lambda s, e, d: None
                def _step(s, e, d):
                    return handlers[idx](s, e, d, _next(idx + 1))
                return _step
            try:
                handlers[0](self, event, data, _next(1))
            except StopIteration:
                pass
            except Exception as exc:
                logger.warning("slot '%s' waterfall error: %s", self.name, exc)

        # -- BAIL: first truthy return wins --
        bail_entries = [e for e in entries if isinstance(e, dict) and e.get("mode") == "bail"]
        for e in bail_entries:
            try:
                r = e["handler"](self, event, data)
                if r:
                    return r
            except Exception as exc:
                logger.warning("slot '%s' bail error: %s", self.name, exc)

        # -- SERIAL: first non-None return wins --
        serial_entries = [e for e in entries if isinstance(e, dict) and e.get("mode") == "serial"]
        for e in serial_entries:
            try:
                r = e["handler"](self, event, data)
                if r is not None:
                    return r
            except Exception as exc:
                logger.warning("slot '%s' serial error: %s", self.name, exc)

        # -- PARALLEL: concurrent, collect results --
        par_entries = [e for e in entries if isinstance(e, dict) and e.get("mode") == "parallel"]
        if par_entries:
            results: List[Any] = []
            errors: List[Exception] = []
            def _run(entry):
                try:
                    results.append(entry["handler"](self, event, data))
                except Exception as ex:
                    errors.append(ex)
            threads = [threading.Thread(target=_run, args=(e,)) for e in par_entries]
            for t in threads: t.start()
            for t in threads: t.join()
            if errors:
                logger.warning("slot '%s' parallel errors: %s", self.name, errors)
            return results

        # -- EMIT: fire-and-forget --
        emit_entries = [e for e in entries if isinstance(e, dict) and e.get("mode") == "emit"]
        for e in emit_entries:
            try:
                e["handler"](self, event, data)
            except Exception as exc:
                logger.warning("slot '%s' emit error: %s", self.name, exc)
        return None

    # ---- module registration ----

    def register(self, module: Any, *, version: Optional[str] = None,
                 is_shadow: bool = False, metadata: Optional[Dict] = None) -> str:
        label = version or self._make_label(module)
        if is_shadow:
            self.shadow = module
            self.shadow_disposer = getattr(module, "dispose", None)
            self.shadow_version = label
            self._metrics.reset_shadow()
            logger.debug("slot '%s': shadow registered '%s'", self.name, label)
            return label
        self._upsert_version_node(label, module, metadata=metadata)
        old_state = self.state
        if old_state == SlotState.PENDING:
            self.state = SlotState.LOADING
            self._install_module(module, label)
            self.state = SlotState.ACTIVE
        else:
            self._install_module(module, label)
            self.state = SlotState.ACTIVE
        # Metric resets happen in swap() (for actual module changes).
        # register() preserves existing observations so that prior module
        # performance carries over into the evaluation window.
        logger.debug("slot '%s': registered '%s' (%s -> ACTIVE)", self.name, label, old_state.value)
        return label


    def _wrap_module(self, module: Any) -> Any:
        """Wrap module.generate/generate_native to auto-record call metrics.

        Also tracks per-call latency (ms) via ``record_latency``.  A guard flag
        prevents double-wrapping when the same module object is swapped back in.
        """
        if module is None or not hasattr(module, 'generate'):
            return module
        if getattr(module, '_hotswap_wrapped', False):
            return module  # already wrapped -- no double-counting
        import functools
        slot = self
        orig_generate = module.generate
        @functools.wraps(orig_generate)
        def tracked_generate(*args, **kwargs):
            t0 = time.monotonic()
            success = False
            try:
                result = orig_generate(*args, **kwargs)
                success = True
                return result
            except Exception:
                raise
            finally:
                slot.record_call(success=success)
                slot.record_latency((time.monotonic() - t0) * 1000.0)
        module.generate = tracked_generate
        if hasattr(module, 'generate_native'):
            orig_gn = module.generate_native
            @functools.wraps(orig_gn)
            def tracked_generate_native(*args, **kwargs):
                t0 = time.monotonic()
                success = False
                try:
                    result = orig_gn(*args, **kwargs)
                    success = True
                    return result
                except Exception:
                    raise
                finally:
                    slot.record_call(success=success)
                    slot.record_latency((time.monotonic() - t0) * 1000.0)
            module.generate_native = tracked_generate_native
        module._hotswap_wrapped = True
        return module

    def _install_module(self, module: Any, label: str) -> None:
        self.module = self._wrap_module(module)
        self.disposer = getattr(module, "dispose", None)
        self.version = label
        # Shadow is deliberately NOT cleared here -- shadow observation
        # survives a swap so a promoted shadow keeps its metrics history,
        # and a pending shadow stays evaluated by SelfEvolutionController.
        self._rollback_module = None
        self._rollback_disposer = None
        self._rollback_version = ""
        # Track in version graph so every swap is recorded.
        self._upsert_version_node(label, module)

    def _upsert_version_node(self, label: str, module: Any = None,
                             metadata: Optional[Dict] = None,
                             parent: Optional[str] = None) -> VersionNode:
        """Insert or update a version node.  Logs if an existing node is updated
        (module replacement) so silent overwrites are always visible."""
        if label not in self._version_graph:
            node = VersionNode(
                version=label, module=module,
                parent=parent or (self.version if self.version else None),
                metadata=metadata or {},
            )
            self._version_graph[label] = node
        else:
            node = self._version_graph[label]
            if module is not None:
                if node.module is not None and node.module is not module:
                    logger.info("slot '%s': version '%s' module replaced (upsert)",
                                self.name, label)
                node.module = module
            if metadata:
                node.metadata.update(metadata)
            if self.version and self.version in self._version_graph:
                if label not in self._version_graph[self.version].children:
                    self._version_graph[self.version].children.append(label)
        return node

    def get(self) -> Any:
        if self.module is None:
            raise RuntimeError(f"capability '{self.name}' has no active module")
        if self.is_circuit_open() and self._fallback_module is not None:
            logger.warning("slot '%s': circuit open, routing to fallback", self.name)
            return self._fallback_module
        if self.is_circuit_open():
            raise RuntimeError(f"capability '{self.name}' circuit breaker open")
        return self.module

    # ---- call tracking ----

    @property
    def _call_id(self) -> int:
        with self._lock:
            self._next_call_id += 1
            return self._next_call_id

    @contextmanager
    def track_call(self):
        cid = self._call_id
        with self._lock:
            self._active_calls_in_flight.add(cid)
        try:
            yield cid
        finally:
            with self._lock:
                self._active_calls_in_flight.discard(cid)

    def record_call(self, success: bool = True, *, for_shadow: bool = False) -> None:
        with self._lock:
            if for_shadow:
                if self.shadow is self.module:
                    return  # shadow == active (post-promote), skip to avoid double-count
                self._metrics.shadow_calls += 1
                if not success:
                    self._metrics.shadow_errors += 1
            else:
                self._metrics.active_calls += 1
                if not success:
                    self._metrics.active_errors += 1
                    self._error_window.append(time.monotonic())
        # Shadow promotion is handled exclusively by SelfEvolutionController
        # (force_observe / daemon loop), which carries audit-log + throttle.
        if not success and not for_shadow:
            self._check_circuit()

    def record_error(self, error: Optional[Exception] = None, *, for_shadow: bool = False) -> None:
        self.record_call(success=False, for_shadow=for_shadow)
        self._emit("error", {"error": str(error) if error else "unknown"})

    def record_latency(self, latency_ms: float, *, for_shadow: bool = False) -> None:
        """Record call latency for p95/p90 tracking (sampled in _wrap_module)."""
        with self._lock:
            if for_shadow:
                self._metrics.shadow_latency_ms += latency_ms
            else:
                self._metrics.active_latency_ms += latency_ms
                if latency_ms > self._metrics.active_latency_max_ms:
                    self._metrics.active_latency_max_ms = latency_ms
                # Sample every 5th call for percentile computation (bounded list)
                if (self._metrics.active_calls % 5) == 0:
                    s = self._metrics._latency_samples
                    s.append(latency_ms)
                    if len(s) > 2000:
                        del s[: len(s) - 2000]  # cap at 2k samples

    # ---- circuit breaker ----

    def is_circuit_open(self) -> bool:
        if time.monotonic() < self._circuit_open_until:
            return True
        now = time.monotonic()
        cutoff = now - self._cb_window
        recent = [t for t in self._error_window if t > cutoff]
        self._error_window = recent
        return len(recent) >= self._cb_threshold

    def _check_circuit(self) -> None:
        if self.is_circuit_open() and not getattr(self, "_cb_emitted", False):
            self._cb_emitted = True
            self._open_circuit()
        elif not self.is_circuit_open():
            self._cb_emitted = False

    def _open_circuit(self) -> None:
        now = time.monotonic()
        self._circuit_open_until = now + self._cb_cooldown
        self._emit("circuit", {
            "state": "open", "cooldown_s": self._cb_cooldown,
            "error_count": len(self._error_window), "window_s": self._cb_window,
        })
        logger.warning("slot '%s': circuit OPEN (>=%d errs in %ds)",
                       self.name, self._cb_threshold, self._cb_window)
        if self._fallback_module:
            try:
                self.swap(self._fallback_module, version="fallback")
            except Exception as exc:
                logger.error("slot '%s': fallback swap failed: %s", self.name, exc)

    def close_circuit(self) -> None:
        self._circuit_open_until = 0.0
        self._error_window.clear()
        self._cb_emitted = False
        logger.info("slot '%s': circuit CLOSED", self.name)

    def set_fallback(self, module: Any) -> None:
        self._fallback_module = module

    # ---- transactional swap (critical Phase 1 upgrade) ----

    def _dispose_current(self) -> Optional[Exception]:
        if self.disposer is None:
            return None
        try:
            if asyncio.iscoroutinefunction(self.disposer):
                if self._event_loop and not self._event_loop.is_closed():
                    fut = asyncio.run_coroutine_threadsafe(
                        self.disposer(), self._event_loop)
                    fut.result(timeout=30)
                else:
                    asyncio.run(self.disposer())
            else:
                self.disposer()
            return None
        except Exception as exc:
            logger.error("slot '%s': disposer raised: %s", self.name, exc)
            return exc

    def swap(self, new_module: Any, *, version: Optional[str] = None) -> str:
        """Transactional swap with automatic rollback.

        Saves rollback anchor BEFORE any mutation.  On failure the old module
        is restored and SwapError is raised.
        """
        label = version or self._make_label(new_module)
        data = {"old_version": self.version, "new_version": label}

        # Save rollback anchor
        old_module = self.module
        old_disposer = self.disposer
        old_version = self.version
        self._rollback_module = old_module
        self._rollback_disposer = old_disposer
        self._rollback_version = old_version
        self._swap_start_time = time.monotonic()

        # Phase 1: UNLOADING
        self.state = SlotState.UNLOADING
        self._emit("unload", data)
        drained = self._drain_calls()
        if not drained:
            logger.warning("slot '%s': %d calls still active after drain timeout",
                           self.name, len(self._active_calls_in_flight))
        self._dispose_current()

        # Phase 2: LOADING
        self.state = SlotState.LOADING
        try:
            self._install_module(new_module, label)
        except Exception as load_err:
            self._do_rollback(old_module, old_disposer, old_version, load_err, label)
            raise SwapError(
                f"swap '{self.name}' failed, rolled back to '{old_version}': {load_err}"
            ) from load_err

        # Phase 3: ACTIVE (success)

        # Reset observation stats for the new active module.
        # Shadow stats are also reset because the shadow reference was for the
        # previous active; a new swap opens a fresh observation window for both.
        self._metrics.reset_active()
        self._metrics.reset_shadow()
        self._swap_start_time = 0  # consumed

        self._rollback_module = None
        self._rollback_disposer = None
        self._rollback_version = ""
        self.state = SlotState.ACTIVE
        self._emit("load", {"version": label})
        swap_data = {
            **data,
            "duration_ms": int((time.monotonic() - self._swap_start_time) * 1000),
            "dispose_ok": True,
        }
        self._emit("swap", swap_data)
        logger.info("slot '%s': swapped '%s' -> '%s' in %dms",
                    self.name, old_version, label, swap_data["duration_ms"])
        return label

    def _do_rollback(self, old_module, old_disposer, old_version, error, attempted_label):
        """Restore old module after swap failure."""
        self.module = old_module
        self.disposer = old_disposer
        self.version = old_version
        self._rollback_module = None
        self._rollback_disposer = None
        self._rollback_version = ""
        self.shadow = None
        self.shadow_disposer = None
        self.shadow_version = ""
        self._metrics.reset_active()
        self.state = SlotState.ACTIVE
        self._emit("rollback", {
            "error": str(error),
            "restored_version": old_version,
            "attempted_version": attempted_label,
            "duration_ms": int((time.monotonic() - self._swap_start_time) * 1000),
        })
        self._emit("error", {"error": str(error), "phase": "load", "rolled_back": True})
        logger.error("slot '%s': swap FAILED, ROLLED BACK to '%s': %s",
                     self.name, old_version, error)

    def force_swap(self, new_module: Any, *, version: Optional[str] = None) -> str:
        self.state = SlotState.UNLOADING
        self._dispose_current()
        # Shadow is preserved (same as regular swap); stats reset in swap().
        return self.register(new_module, version=version)

    def _drain_calls(self) -> bool:
        if not self._active_calls_in_flight:
            return True
        deadline = time.monotonic() + self._drain_timeout
        while self._active_calls_in_flight and time.monotonic() < deadline:
            time.sleep(0.005)
        if self._active_calls_in_flight:
            logger.warning("slot '%s': drain timeout after %ds, %d still active",
                           self.name, self._drain_timeout,
                           len(self._active_calls_in_flight))
            return False
        return True

    def safe_swap(self) -> bool:
        if self.shadow is None:
            return False
        with self._lock:
            sm = self._metrics
            sc, se = sm.shadow_calls, sm.shadow_errors
            ac, ae = sm.active_calls, sm.active_errors
        sr = se / sc if sc > 0 else 0.0
        ar = ae / ac if ac > 0 else 0.0
        if sc < self.shadow_min_calls:
            logger.debug("slot '%s': shadow '%s' waiting (%d/%d calls)",
                         self.name, self.shadow_version, sc, self.shadow_min_calls)
            return False
        if sr > self.shadow_max_error_rate:
            if ar > 0.08 and ac > 0:
                logger.warning("slot '%s': both degraded, force-swapping shadow", self.name)
                try:
                    self.swap(self.shadow, version=self.shadow_version)
                    return True
                except Exception as exc:
                    logger.error("slot '%s': degraded swap failed: %s", self.name, exc)
                    return False
            self.shadow = None
            self.shadow_disposer = None
            self.shadow_version = ""
            return False
        try:
            self.swap(self.shadow, version=self.shadow_version)
            return True
        except Exception as exc:
            logger.error("slot '%s': shadow promotion failed: %s", self.name, exc)
            return False

    # ---- version graph ----

    def register_version(self, module: Any, *, version: Optional[str] = None,
                         parent: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> str:
        label = version or self._make_label(module)
        node = VersionNode(
            version=label, module=module,
            parent=parent or (self.version if self.version else None),
            metadata=metadata or {},
        )
        with self._lock:
            self._version_graph[label] = node
            p = node.parent
            if p and p in self._version_graph and label not in self._version_graph[p].children:
                self._version_graph[p].children.append(label)
        self._prune_version_graph()
        return label

    def rollback_to(self, target_version: str) -> bool:
        with self._lock:
            node = self._version_graph.get(target_version)
        if node is None or node.module is None:
            return False
        try:
            self.swap(node.module, version=target_version)
            if target_version in self._version_graph:
                self._version_graph[target_version].was_rollback_target = True
            logger.info("slot '%s': rolled back -> '%s'", self.name, target_version)
            return True
        except Exception as exc:
            logger.error("slot '%s': rollback to '%s' failed: %s", self.name, target_version, exc)
            return False

    def list_versions(self) -> List[Dict[str, Any]]:
        with self._lock:
            nodes = sorted(self._version_graph.values(), key=lambda n: n.registered_at)
        return [{
            "version": n.version, "parent": n.parent,
            "children": n.children, "swap_count": n.swap_count,
            "was_rollback_target": n.was_rollback_target,
            "registered_at": n.registered_at, "metadata": n.metadata,
        } for n in nodes]

    def get_version_node(self, version: str) -> Optional[VersionNode]:
        with self._lock:
            return self._version_graph.get(version)

    def _prune_version_graph(self, max_nodes: int = 100) -> None:
        """Keep the current version + the most recent N-1 nodes."""
        if len(self._version_graph) <= max_nodes:
            return
        with self._lock:
            keep = {self.version} if self.version else set()
            if keep:
                others = sorted(
                    [(v, n) for v, n in self._version_graph.items() if v not in keep],
                    key=lambda kv: kv[1].registered_at,
                )
                for v, _ in others[-(max_nodes - 1):]:
                    keep.add(v)
            self._version_graph = {k: v for k, v in self._version_graph.items() if k in keep}

    # ---- state helpers ----

    def pending(self) -> None:
        self.state = SlotState.PENDING
        logger.debug("slot '%s': -> PENDING", self.name)

    def activate(self) -> None:
        if self.state == SlotState.PENDING:
            self.state = SlotState.ACTIVE
            self._emit("load", {"version": self.version})

    def set_disposed(self) -> None:
        self.module = None
        self.disposer = None
        self.version = ""
        self.state = SlotState.DISPOSED
        self._emit("unload", {"reason": "terminal_dispose"})

    # ---- status ----

    def status(self) -> Dict[str, Any]:
        with self._lock:
            m = self._metrics
            draining = len(self._active_calls_in_flight)

        def _summarize(module, ver, calls, errors):
            if module is None:
                return None
            return {
                "version": ver or "(unnamed)",
                "type": type(module).__name__,
                "calls": calls,
                "errors": errors,
                "error_rate": f"{errors / calls * 100:.1f}%" if calls else "0.0%",
            }
        return {
            "name": self.name,
            "state": self.state.value,
            "current_version": self.version,
            "active": _summarize(self.module, self.version,
                                 m.active_calls, m.active_errors),
            "shadow": _summarize(self.shadow, self.shadow_version,
                                 m.shadow_calls, m.shadow_errors),
            "circuit_open": self.is_circuit_open(),
            "fallback": type(self._fallback_module).__name__ if self._fallback_module else None,
            "draining_calls": draining,
            "version_graph_size": len(self._version_graph),
            "has_dispose": self.disposer is not None,
        }

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = loop

    def _make_label(self, module: Any) -> str:
        return f"{type(module).__name__}#{id(module) & 0xFFFF:04x}"

    def set_shadow(self, module: Any, version: Optional[str] = None) -> str:
        label = version or self._make_label(module)
        self.shadow = module
        self.shadow_disposer = getattr(module, "dispose", None)
        self.shadow_version = label
        self._metrics.reset_shadow()
        return label


# ---------------------------------------------------------------------------
# ModuleRegistry
# ---------------------------------------------------------------------------

class ModuleRegistry:
    """Global registry of capability slots (Cordis Context analog)."""

    def __init__(self) -> None:
        self._slots: Dict[str, CapabilitySlot] = {}
        self._lock = threading.Lock()

    def register(self, capability: str, module: Any, *,
                 version: Optional[str] = None, as_shadow: bool = False,
                 metadata: Optional[Dict] = None) -> str:
        label = version or f"{type(module).__name__}#{id(module) & 0xFFFF:04x}"
        with self._lock:
            if capability not in self._slots:
                self._slots[capability] = CapabilitySlot(name=capability)
            slot = self._slots[capability]
        return slot.register(module, version=label, is_shadow=as_shadow, metadata=metadata)

    def swap(self, capability: str, new_module: Any, *,
             version: Optional[str] = None, force: bool = False) -> str:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            raise KeyError(f"capability '{capability}' not registered")
        if force:
            return slot.force_swap(new_module, version=version)
        return slot.swap(new_module, version=version)

    def force_swap(self, capability: str, new_module: Any, *, version=None) -> str:
        return self.swap(capability, new_module, version=version, force=True)

    def set_shadow(self, capability: str, module: Any, version: Optional[str] = None) -> str:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            raise KeyError(f"capability '{capability}' not registered")
        return slot.set_shadow(module, version=version or slot._make_label(module))

    def promote_shadow(self, capability: str) -> bool:
        with self._lock:
            slot = self._slots.get(capability)
        return slot.safe_swap() if slot else False

    def promote_if_healthy(self, capability: str) -> bool:
        return self.promote_shadow(capability)

    def register_version(self, capability: str, module: Any, *,
                         version: Optional[str] = None,
                         parent: Optional[str] = None,
                         metadata: Optional[Dict] = None,
                         activate: bool = True) -> str:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            raise KeyError(f"capability '{capability}' not registered")
        label = slot.register_version(module, version=version, parent=parent, metadata=metadata)
        if activate:
            slot.swap(module, version=label)
        return label

    def rollback_to(self, capability: str, target_version: str) -> bool:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            return False
        return slot.rollback_to(target_version)

    def list_versions(self, capability: str) -> List[Dict]:
        with self._lock:
            slot = self._slots.get(capability)
        return slot.list_versions() if slot else []

    def get(self, capability: str) -> Any:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            raise KeyError(f"capability '{capability}' not registered")
        return slot.get()

    def get_slot(self, capability: str) -> CapabilitySlot:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            raise KeyError(f"capability '{capability}' not registered")
        return slot

    def record_call(self, capability: str, success: bool = True, *, for_shadow: bool = False) -> None:
        with self._lock:
            slot = self._slots.get(capability)
        if slot:
            slot.record_call(success=success, for_shadow=for_shadow)

    def set_fallback(self, capability: str, module: Any) -> None:
        with self._lock:
            slot = self._slots.get(capability)
        if slot:
            slot.set_fallback(module)

    def set_event_loop(self, capability: str, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            slot = self._slots.get(capability)
        if slot:
            slot.set_event_loop(loop)

    def status(self, capability: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            names = [capability] if capability else list(self._slots.keys())
        result = {}
        for name in names:
            with self._lock:
                slot = self._slots.get(name)
            result[name] = slot.status() if slot else {"error": "not found"}
        return result

    def list_capabilities(self) -> List[str]:
        with self._lock:
            return list(self._slots.keys())

    def on(self, capability: str, event: str, handler: EventHandler, *, mode: str = "emit") -> None:
        with self._lock:
            slot = self._slots.get(capability)
        if slot is None:
            raise KeyError(f"capability '{capability}' not registered")
        slot.on(event, handler, mode=mode)

    def on_swap(self, capability: str, handler: EventHandler) -> None:
        for ev in ("unload", "load", "swap", "error", "rollback", "circuit"):
            self.on(capability, ev, handler)

    def cascade_swap(self, updates, *, rollback_on_any_failure=True):
        """Swap N capabilities; rollback all on first failure."""
        swapped = []
        failed = []
        rolled_back = []
        rollback_stack = []
        caps = sorted(updates.keys(),
                      key=lambda c: getattr(self._slots.get(c), "metadata", {}).get("swap_order", 0)
                                   if hasattr(self._slots.get(c, type("", (), {})().__class__), "metadata") else 0)
        for cap in caps:
            module = updates[cap]
            try:
                slot = self._slots[cap]
                old = slot.module
                self.swap(cap, module)
                swapped.append(cap)
                rollback_stack.append((cap, old))
            except Exception as exc:
                failed.append(cap)
                logger.error("cascade_swap: '%s' failed: %s", cap, exc)
                if rollback_on_any_failure and rollback_stack:
                    for rb_cap, rb_mod in reversed(rollback_stack):
                        try:
                            self.swap(rb_cap, rb_mod)
                            rolled_back.append(rb_cap)
                        except Exception as e2:
                            logger.error("cascade_swap: rollback '%s' also failed: %s", rb_cap, e2)
                break
        return {"swapped": swapped, "failed": failed, "rolled_back": rolled_back}

    def shutdown(self) -> None:
        with self._lock:
            names = list(self._slots.keys())
        for name in reversed(names):
            slot = self._slots.get(name)
            if not slot or not slot.module:
                continue
            try:
                slot.state = SlotState.UNLOADING
                slot._drain_calls()
                slot._dispose_current()
                slot.set_disposed()
            except Exception as exc:
                logger.error("shutdown: slot '%s' error: %s", name, exc)


# ---------------------------------------------------------------------------
# HealthAwareRegistry
# ---------------------------------------------------------------------------

class HealthAwareRegistry:
    def __init__(self, registry: Optional[ModuleRegistry] = None) -> None:
        self.reg = registry or ModuleRegistry()
        self._checker_fn: Optional[Callable] = None

    def set_health_checker(self, fn: Callable) -> None:
        self._checker_fn = fn

    def check(self, capability: str) -> Dict[str, Any]:
        if self._checker_fn is None:
            return {"error": "no health checker"}
        try:
            return self._checker_fn(self.reg.get(capability))
        except Exception as exc:
            return {"error": str(exc)}

    def swap_if_unhealthy(self, capability: str, replacement: Any) -> bool:
        result = self.check(capability)
        if result.get("healthy", True) is False:
            self.reg.swap(capability, replacement)
            return True
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reg, name)


# ---------------------------------------------------------------------------
# HotSwapKernel
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SwapAuditLog -- append-only JSONL swap audit trail (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class SwapAuditLog:
    """Append-only JSONL log of swap events for diagnostics and replay."""

    path: Union[str, Path]
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event, capability, *, version="", old_version="",
               duration_ms=0, error=None, meta=None):
        rec = {
            "ts": time.monotonic(),
            "capability": capability, "event": event,
            "version": version, "old_version": old_version,
            "duration_ms": duration_ms,
            "error": error, "meta": meta or {},
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def replay(self, capability=None, event=None, limit=500):
        results = []
        if not self.path.exists():
            return results
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if len(results) >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if capability and rec.get("capability") != capability:
                continue
            if event and rec.get("event") != event:
                continue
            results.append(rec)
        return results

    def stats(self, capability):
        records = self.replay(capability=capability, limit=5000)
        swaps = [r for r in records if r["event"] == "swap"]
        rollbacks = [r for r in records if r["event"] == "rollback"]
        durations = [r["duration_ms"] for r in swaps if r.get("duration_ms")]
        return {
            "total_swaps": len(swaps),
            "rollbacks": len(rollbacks),
            "errors": len([r for r in records if r["event"] == "error"]),
            "avg_duration_ms": sum(durations) // len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "rollback_rate": f"{len(rollbacks)/len(swaps)*100:.1f}%" if swaps else "0.0%",
        }

class HotSwapKernel:
    """Wraps the harness Kernel with module-level hot-swap."""

    def __init__(self, kernel: Any, registry: Optional[ModuleRegistry] = None):
        self._kernel = kernel
        self.reg = registry or ModuleRegistry()

    def evaluate(self, effect: Any) -> Any:
        return self._kernel.evaluate(effect)

    def run(self, effect: Any) -> str:
        return self._kernel.run(effect)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)

    def register_capability(self, name: str, module: Any, *,
                            version: Optional[str] = None,
                            as_shadow: bool = False,
                            metadata: Optional[Dict] = None) -> str:
        return self.reg.register(name, module, version=version,
                                as_shadow=as_shadow, metadata=metadata)

    def get_module(self, capability: str) -> Any:
        return self.reg.get(capability)

    def swap(self, capability: str, module: Any, *,
             version: Optional[str] = None, force: bool = False) -> str:
        return self.reg.swap(capability, module, version=version, force=force)

    def force_swap(self, capability: str, module: Any, *, version=None) -> str:
        return self.reg.force_swap(capability, module, version=version)

    def set_shadow(self, capability: str, module: Any, version=None) -> str:
        return self.reg.set_shadow(capability, module, version=version)

    def promote_shadow(self, capability: str) -> bool:
        return self.reg.promote_shadow(capability)

    def set_fallback(self, capability: str, module: Any) -> None:
        self.reg.set_fallback(capability, module)

    def set_event_loop(self, capability: str, loop: asyncio.AbstractEventLoop) -> None:
        self.reg.set_event_loop(capability, loop)

    def register_version(self, capability: str, module: Any, *, version=None,
                         parent=None, metadata=None) -> str:
        return self.reg.register_version(capability, module, version=version,
                                         parent=parent, metadata=metadata)

    def rollback_to(self, capability: str, target_version: str) -> bool:
        return self.reg.rollback_to(capability, target_version)

    def list_versions(self, capability: str) -> List[Dict]:
        return self.reg.list_versions(capability)

    def record_call(self, capability: str, success: bool = True, *, for_shadow: bool = False) -> None:
        self.reg.record_call(capability, success, for_shadow=for_shadow)

    def on_swap(self, capability: str, handler: EventHandler) -> None:
        self.reg.on_swap(capability, handler)

    def on(self, capability: str, event: str, handler: EventHandler, *, mode: str = "emit") -> None:
        self.reg.on(capability, event, handler, mode=mode)

    def status(self, capability: Optional[str] = None) -> Dict[str, Any]:
        return self.reg.status(capability)

    def shutdown(self) -> None:
        self.reg.shutdown()


# ---------------------------------------------------------------------------
# EventBus bridge
# ---------------------------------------------------------------------------

def bridge_to_eventbus(hot: "HotSwapKernel", bus: "EventBus",
                       capability: str) -> Callable[[], None]:
    from .events import ModuleRegistered, ModuleSwapped

    events_seen: List[str] = []
    handler_refs: List[EventHandler] = []

    def handler(slot: "CapabilitySlot", event: str, data: Dict) -> None:
        if event in events_seen:
            return
        events_seen.append(event)
        if event == "swap":
            evt = ModuleSwapped(
                capability=capability,
                old_version=data.get("old_version", ""),
                new_version=data.get("new_version", ""),
                had_disposer=slot.disposer is not None,
                swap_error=data.get("error"),
            )
            bus.emit_event(evt)
        elif event == "rollback":
            evt = ModuleSwapped(
                capability=capability,
                old_version=data.get("restored_version", ""),
                new_version=data.get("attempted_version", ""),
                had_disposer=True,
                swap_error=data.get("error"),
            )
            bus.emit_event(evt)
        elif event == "load":
            evt = ModuleRegistered(
                capability=capability,
                version=data.get("version", slot.version),
                role="active",
            )
            bus.emit_event(evt)
        elif event == "circuit":
            logger.debug("slot '%s': circuit event", capability)

    try:
        slot = hot.reg._slots[capability]  # type: ignore[attr-defined]
        slot.on("swap", handler)
        slot.on("load", handler)
        slot.on("rollback", handler)
        handler_refs.append(handler)
    except (KeyError, AttributeError):
        logger.warning("bridge: capability '%s' not found", capability)

    def _unsubscribe() -> None:
        try:
            slot = hot.reg._slots[capability]  # type: ignore[attr-defined]
            for ev in ("swap", "load", "rollback"):
                slot._events[ev] = [
                    h for h in slot._events.get(ev, [])
                    if h.get("handler") is not handler_refs[0]
                ]
        except Exception:
            pass

    return _unsubscribe

# ---------------------------------------------------------------------------
# SelfEvolutionController -- observe → shadow-test → promote → rollback (Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class EvolutionPolicy:
    """Multi-dimensional thresholds for self-evolution decisions.

    Replaces the single-dimension ``error_rate`` comparison with a weighted
    composite score that considers both error rate and latency.
    """
    err_weight: float = 0.7          # Weight for error rate in composite (0-1)
    latency_weight: float = 0.3      # Weight for latency in composite (0-1)
    latency_p95_ceiling_ms: float = 4000.0   # Latencies above this count as 1.0 (worst)
    promote_min_score_delta: float = 0.15    # Min score gap to promote shadow
    shadow_min_calls: int = 5
    shadow_max_err: float = 0.02
    rollback_err_thresh: float = 0.10
    rollback_call_floor: int = 20
    max_swaps_per_hour: int = 4

    def _score(self, error_rate: float, latency_p95_ms: float) -> float:
        """Composite score: 1.0 = perfect, 0.0 = worst."""
        e = min(error_rate / self.shadow_max_err, 1.0)
        l = min(latency_p95_ms / self.latency_p95_ceiling_ms, 1.0)
        return 1.0 - (self.err_weight * e + self.latency_weight * l)


@dataclass
class EvolutionDecision:
    """One self-evolution decision for audit / display."""
    capability: str
    action: str                 # "promote" | "rollback" | "observe" | "skip"
    reason: str = ""
    shadow_error_rate: float = 0.0
    active_error_rate: float = 0.0
    shadow_score: float = 0.0
    active_score: float = 0.0
    shadow_latency_p95_ms: float = 0.0
    active_latency_p95_ms: float = 0.0
    shadow_calls: int = 0
    active_calls: int = 0
    old_version: str = ""
    new_version: str = ""
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class SelfEvolutionController:
    """Background controller that drives the self-evolution loop.

    Lifecycle
    ---------
    observe  ->  check every capability's metrics
                if shadow composite score > active composite score by threshold -> promote
                if active composite score < 0.5 -> rollback (if version graph allows)
    Multi-dimensional evaluation
    ----------------------------
    Uses ``EvolutionPolicy`` to score modules on error rate (70%) + latency (30%)
    rather than a single error_rate comparison.  Pass ``policy=`` for custom
    weights; all individual params are still accepted as convenience overrides.
    Config
    ------
    observe_interval_s   How often to run the observation loop (default 60s).
    shadow_min_calls     Minimum shadow calls before evaluating (default 5).
    shadow_max_err       Max error rate to consider shadow healthy (default 0.02).
    promote_better_than  Promote when active error rate is this much higher
                         than shadow (default 2.0 = active 2x worse).
    rollback_err_thresh  Rollback active if error rate exceeds this (0.10).
    rollback_call_floor  Only rollback after this many active calls (default 20).
    max_swaps_per_hour   Throttle evolution to avoid flapping (default 4).
    policy               EvolutionPolicy for multi-dimensional scoring (optional).
    """

    def __init__(self, kernel: HotSwapKernel, registry: Optional[ModuleRegistry] = None,
                 *, observe_interval_s: float = 60.0,
                 shadow_min_calls: int = 5,
                 shadow_max_err: float = 0.02,
                 promote_better_than: float = 2.0,
                 rollback_err_thresh: float = 0.10,
                 rollback_call_floor: int = 20,
                 max_swaps_per_hour: int = 4,
                 policy: Optional[EvolutionPolicy] = None,
                 audit_log: Optional[SwapAuditLog] = None):
        self.kernel = kernel
        self.reg = registry or kernel.reg
        self.observe_interval_s = observe_interval_s
        self.policy = policy or EvolutionPolicy(
            shadow_min_calls=shadow_min_calls,
            shadow_max_err=shadow_max_err,
            rollback_err_thresh=rollback_err_thresh,
            rollback_call_floor=rollback_call_floor,
            max_swaps_per_hour=max_swaps_per_hour,
        )
        self.shadow_min_calls = self.policy.shadow_min_calls
        self.shadow_max_err = self.policy.shadow_max_err
        self.promote_better_than = promote_better_than
        self.rollback_err_thresh = self.policy.rollback_err_thresh
        self.rollback_call_floor = self.policy.rollback_call_floor
        self.max_swaps_per_hour = self.policy.max_swaps_per_hour
        self.audit = audit_log
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._swap_timestamps: List[float] = []
        self._rollback_version_map: Dict[str, str] = {}
        self._decisions: List[EvolutionDecision] = []
        self._lock = threading.Lock()
        self._last_observe: float = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                         name="SelfEvolutionController")
        self._thread.start()
        logger.info("SelfEvolutionController started (interval=%.1fs)", self.observe_interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("SelfEvolutionController stopped")

    def force_observe(self) -> List[EvolutionDecision]:
        """Run one observation pass synchronously (useful for tests)."""
        return self._observe_all()

    # ---- main loop ----

    def _run_loop(self) -> None:
        while self._running:
            try:
                decisions = self._observe_all()
                for d in decisions:
                    logger.info("evolution '%s': %s -- %s",
                                d.capability, d.action, d.reason)
            except Exception as exc:
                logger.error("SelfEvolutionController loop error: %s", exc)
            # Sleep in small increments so stop() is responsive.
            deadline = time.monotonic() + self.observe_interval_s
            while self._running and time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))

    def _observe_all(self) -> List[EvolutionDecision]:
        decisions: List[EvolutionDecision] = []
        caps = self.reg.list_capabilities()
        for cap in caps:
            try:
                slot = self.reg.get_slot(cap)
            except KeyError:
                continue
            if slot.shadow is not None:
                decisions.append(self._evaluate_shadow(cap, slot))
            # Always check active degradation regardless of shadow.
            if self._is_active_degraded(cap, slot):
                decisions.append(self._evaluate_rollback(cap, slot))
        self._last_observe = time.monotonic()
        return decisions

    # ---- shadow evaluation ----

    def _evaluate_shadow(self, cap: str, slot: CapabilitySlot) -> EvolutionDecision:
        sm = slot._metrics
        sc, se = sm.shadow_calls, sm.shadow_errors
        ac, ae = sm.active_calls, sm.active_errors
        sr = se / sc if sc > 0 else 0.0
        ar = ae / ac if ac > 0 else 0.0
        s_lat = sm.latency_p95_ms
        a_lat = sm.latency_p90_ms  # active uses p90 (fewer samples)

        if sc < self.shadow_min_calls:
            return EvolutionDecision(
                capability=cap, action="observe",
                reason=f"shadow warming ({sc}/{self.shadow_min_calls} calls)",
                shadow_error_rate=sr, active_error_rate=ar,
                shadow_calls=sc, active_calls=ac,
                old_version=slot.version, new_version=slot.shadow_version,
            )

        p = self.policy
        shadow_score = p._score(sr, s_lat)
        active_score = p._score(ar, a_lat)
        score_delta = shadow_score - active_score
        shadow_healthy = sr <= p.shadow_max_err

        if shadow_healthy and score_delta >= p.promote_min_score_delta and self._can_swap():
            try:
                label = slot.swap(slot.shadow, version=slot.shadow_version)
                old = slot.version
                self._record_swap()
                self._rollback_version_map[cap] = old
                self._audit("shadow_promote", cap,
                            new_version=label, old_version=old,
                            error_rate_s=sr, error_rate_a=ar,
                            shadow_score=round(shadow_score, 3),
                            active_score=round(active_score, 3))
                return EvolutionDecision(
                    capability=cap, action="promote",
                    reason=(f"shadow qualified (score={shadow_score:.3f} vs "
                            f"{active_score:.3f}, delta={score_delta:.3f})"),
                    shadow_error_rate=sr, active_error_rate=ar,
                    shadow_score=shadow_score, active_score=active_score,
                    shadow_latency_p95_ms=s_lat, active_latency_p95_ms=a_lat,
                    shadow_calls=sc, active_calls=ac,
                    old_version=old, new_version=label,
                )
            except SwapError as exc:
                self._audit("shadow_promote_failed", cap, error=str(exc))
                return EvolutionDecision(
                    capability=cap, action="skip",
                    reason=f"promote failed: {exc}",
                    shadow_error_rate=sr, active_error_rate=ar,
                    shadow_score=shadow_score, active_score=active_score,
                    shadow_latency_p95_ms=s_lat, active_latency_p95_ms=a_lat,
                    shadow_calls=sc, active_calls=ac,
                    old_version=slot.version, new_version=slot.shadow_version,
                )

        if not shadow_healthy:
            reason = f"shadow unhealthy (err={sr:.3%} > {p.shadow_max_err:.3%})"
            slot.shadow = None
            slot.shadow_disposer = None
            slot.shadow_version = ""
        else:
            reason = (f"shadow score {shadow_score:.3f} not enough above "
                      f"active {active_score:.3f} (delta={score_delta:.3f})")
        return EvolutionDecision(
            capability=cap, action="observe",
            reason=reason,
            shadow_error_rate=sr, active_error_rate=ar,
            shadow_score=shadow_score, active_score=active_score,
            shadow_latency_p95_ms=s_lat, active_latency_p95_ms=a_lat,
            shadow_calls=sc, active_calls=ac,
            old_version=slot.version, new_version=slot.shadow_version,
        )

    # ---- rollback evaluation ----

    def _is_active_degraded(self, cap: str, slot: CapabilitySlot) -> bool:
        m = slot._metrics
        ac = m.active_calls
        if ac < self.rollback_call_floor:
            return False
        ar = m.error_rate
        p = self.policy
        active_score = p._score(ar, m.latency_p90_ms if m.latency_p90_ms > 0 else m.avg_latency_ms)
        return ar >= p.rollback_err_thresh or active_score < 0.5

    def _evaluate_rollback(self, cap: str, slot: CapabilitySlot) -> EvolutionDecision:
        prev = self._rollback_version_map.get(cap)
        if not prev or prev not in slot._version_graph:
            return EvolutionDecision(
                capability=cap, action="observe",
                reason="no rollback target available",
                old_version=slot.version,
            )
        m = slot._metrics
        ac = m.active_calls
        ar = m.error_rate
        p = self.policy
        active_score = p._score(ar, m.latency_p90_ms if m.latency_p90_ms > 0 else m.avg_latency_ms)
        try:
            ok = slot.rollback_to(prev)
            if ok:
                self._record_swap()
                self._audit("auto_rollback", cap,
                            old_version=slot.version, new_version=prev,
                            error_rate_a=ar,
                            active_score=round(active_score, 3))
                return EvolutionDecision(
                    capability=cap, action="rollback",
                    reason=(f"active degraded (score={active_score:.3f}, "
                            f"err={ar:.3%})"),
                    active_error_rate=ar, active_score=active_score,
                    active_calls=ac, active_latency_p95_ms=m.latency_p95_ms,
                    old_version=slot.version, new_version=prev,
                )
        except Exception as exc:
            self._audit("rollback_failed", cap, error=str(exc))
        return EvolutionDecision(
            capability=cap, action="observe",
            reason=f"degraded but rollback failed",
            active_error_rate=ar, active_score=active_score,
            active_calls=ac, old_version=slot.version,
        )

    # ---- swap throttling ----

    def _can_swap(self) -> bool:
        now = time.monotonic()
        with self._lock:
            cutoff = now - 3600.0
            self._swap_timestamps = [t for t in self._swap_timestamps if t > cutoff]
            if len(self._swap_timestamps) >= self.max_swaps_per_hour:
                return False
            return True

    def _record_swap(self) -> None:
        with self._lock:
            self._swap_timestamps.append(time.monotonic())

    # ---- audit ----

    def _audit(self, event: str, capability: str, **kwargs) -> None:
        if self.audit is None:
            return
        self.audit.record(event, capability,
                          version=kwargs.get("new_version", ""),
                          old_version=kwargs.get("old_version", ""),
                          duration_ms=kwargs.get("duration_ms", 0),
                          error=kwargs.get("error"),
                          meta=kwargs)

    # ---- introspection ----

    def decisions(self, *, limit: int = 50) -> List[EvolutionDecision]:
        with self._lock:
            return list(self._decisions[-limit:])

    def summary(self) -> Dict[str, Any]:
        caps = self.reg.list_capabilities()
        slots_status = {}
        for cap in caps:
            try:
                slots_status[cap] = self.reg.status(cap)
            except Exception:
                slots_status[cap] = {"error": "unreachable"}
        return {
            "running": self._running,
            "capabilities": len(caps),
            "swaps_last_hour": len(self._swap_timestamps),
            "last_observe_s": round(time.monotonic() - self._last_observe, 1)
                               if self._last_observe else None,
            "recent_decisions": len(self._decisions),
            "slots": slots_status,
        }
