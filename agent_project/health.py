"""
Module health checker for OpenMythos Agent.

Provides structured visibility into whether advanced modules are ready,
degraded, disabled or failed, plus actionable install hints when a heavy
optional dependency is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModuleStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass
class ModuleHealth:
    name: str
    status: ModuleStatus
    dependency: Optional[str] = None
    fallback: Optional[str] = None
    error: Optional[str] = None
    install_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "dependency": self.dependency,
            "fallback": self.fallback,
            "error": self.error,
            "install_hint": self.install_hint,
        }


@dataclass
class HealthReport:
    modules: List[ModuleHealth] = field(default_factory=list)

    def by_name(self, name: str) -> Optional[ModuleHealth]:
        for m in self.modules:
            if m.name == name:
                return m
        return None

    def status_ok(self) -> bool:
        return all(m.status in (ModuleStatus.READY, ModuleStatus.DISABLED) for m in self.modules)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.status_ok(),
            "modules": [m.to_dict() for m in self.modules],
        }


class ModuleHealthChecker:
    """Check the health of every advanced module without importing heavy deps eagerly."""

    def __init__(self, config: Any):
        self.config = config

    def check_all(self) -> HealthReport:
        report = HealthReport()
        report.modules.append(self._check_experience())
        report.modules.append(self._check_strategy())
        report.modules.append(self._check_reflection())
        report.modules.append(self._check_planning())
        report.modules.append(self._check_reasoning())
        report.modules.append(self._check_memory())
        report.modules.append(self._check_context())
        report.modules.append(self._check_self_correction())
        report.modules.append(self._check_memskill())
        report.modules.append(self._check_file_memory())
        report.modules.append(self._check_sqlite_memory())
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vector_deps(self) -> "tuple[bool, bool]":
        """向量检索依赖可用性 (chromadb + sentence-transformers). 供多个模块复用."""
        return self._importable("chromadb"), self._importable("sentence_transformers")

    @staticmethod
    def _importable(module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except Exception:
            return False

    def _enabled(self, key: str) -> bool:
        cfg = getattr(self.config, key, None)
        if cfg is None:
            return False
        return bool(getattr(cfg, "enabled", False))

    def _check_experience(self) -> ModuleHealth:
        if not (self._enabled("memory") or self._enabled("reflection")):
            return ModuleHealth("experience", ModuleStatus.DISABLED)
        has_chroma, has_st = self._vector_deps()
        if has_chroma and has_st:
            return ModuleHealth("experience", ModuleStatus.READY, dependency="chromadb, sentence-transformers")
        return ModuleHealth(
            "experience",
            ModuleStatus.DEGRADED,
            dependency="chromadb, sentence-transformers",
            fallback="JSON file store",
            install_hint="pip install chromadb sentence-transformers",
        )

    def _check_strategy(self) -> ModuleHealth:
        try:
            from .strategies import StrategyDatabase
            return ModuleHealth("strategy", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("strategy", ModuleStatus.DEGRADED, fallback="rule-based", error=str(e))

    def _check_reflection(self) -> ModuleHealth:
        if not self._enabled("reflection"):
            return ModuleHealth("reflection", ModuleStatus.DISABLED)
        try:
            from .reflection import ReflectionModule
            return ModuleHealth("reflection", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("reflection", ModuleStatus.DEGRADED, fallback="disabled", error=str(e))

    def _check_planning(self) -> ModuleHealth:
        if not self._enabled("planning"):
            return ModuleHealth("planning", ModuleStatus.DISABLED)
        try:
            from .planning import Planner
            return ModuleHealth("planning", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("planning", ModuleStatus.DEGRADED, fallback="rule-based", error=str(e))

    def _check_reasoning(self) -> ModuleHealth:
        if not self._enabled("reasoning"):
            return ModuleHealth("reasoning", ModuleStatus.DISABLED)
        try:
            from .reasoning import ReasoningEngine
            return ModuleHealth("reasoning", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("reasoning", ModuleStatus.DEGRADED, fallback="DirectPolicy", error=str(e))

    def _check_memory(self) -> ModuleHealth:
        if not self._enabled("memory"):
            return ModuleHealth("memory", ModuleStatus.DISABLED)
        has_chroma, has_st = self._vector_deps()
        try:
            from .memory import create_memory_manager
            from .wiki_memory import LLMWikiManager
            return ModuleHealth(
                "memory",
                ModuleStatus.READY if (has_chroma and has_st) else ModuleStatus.DEGRADED,
                dependency="chromadb, sentence-transformers" if not (has_chroma and has_st) else None,
                fallback="keyword + JSON" if not (has_chroma and has_st) else None,
                install_hint="pip install chromadb sentence-transformers" if not (has_chroma and has_st) else None,
            )
        except Exception as e:
            return ModuleHealth("memory", ModuleStatus.DEGRADED, fallback="keyword + JSON", error=str(e))

    def _check_context(self) -> ModuleHealth:
        if not self._enabled("memory"):
            return ModuleHealth("context", ModuleStatus.DISABLED)
        try:
            from .context_engine import ContextEngine
            return ModuleHealth("context", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("context", ModuleStatus.DEGRADED, fallback="no context injection", error=str(e))

    def _check_self_correction(self) -> ModuleHealth:
        if not self._enabled("self_correction"):
            return ModuleHealth("self_correction", ModuleStatus.DISABLED)
        try:
            from .self_correction import SelfCorrectionModule
            return ModuleHealth("self_correction", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("self_correction", ModuleStatus.DEGRADED, fallback="disabled", error=str(e))

    def _check_memskill(self) -> ModuleHealth:
        if not self._enabled("memory"):
            return ModuleHealth("memskill", ModuleStatus.DISABLED)
        has_st = self._importable("sentence_transformers")
        try:
            from .memskill import MemSkillEngine
            return ModuleHealth(
                "memskill",
                ModuleStatus.READY if has_st else ModuleStatus.DEGRADED,
                dependency="sentence-transformers" if not has_st else None,
                fallback="keyword matching" if not has_st else None,
                install_hint="pip install sentence-transformers" if not has_st else None,
            )
        except Exception as e:
            return ModuleHealth("memskill", ModuleStatus.DEGRADED, fallback="keyword matching", error=str(e))

    def _check_file_memory(self) -> ModuleHealth:
        if not self._enabled("memory"):
            return ModuleHealth("file_memory", ModuleStatus.DISABLED)
        try:
            from .file_memory import FileMemoryManager
            mgr = FileMemoryManager()
            return ModuleHealth("file_memory", ModuleStatus.READY)
        except Exception as e:
            return ModuleHealth("file_memory", ModuleStatus.DEGRADED, fallback="no file memory", error=str(e))

    def _check_sqlite_memory(self) -> ModuleHealth:
        if not self._enabled("memory"):
            return ModuleHealth("sqlite_memory", ModuleStatus.DISABLED)
        try:
            from .sqlite_memory import SQLiteSessionStore
            store = SQLiteSessionStore()
            has_fts5 = getattr(store, "_fts5_available", False)
            return ModuleHealth(
                "sqlite_memory",
                ModuleStatus.READY if has_fts5 else ModuleStatus.DEGRADED,
                dependency="sqlite3 FTS5" if not has_fts5 else None,
                fallback="LIKE keyword search" if not has_fts5 else None,
            )
        except Exception as e:
            return ModuleHealth("sqlite_memory", ModuleStatus.DEGRADED, fallback="no session memory", error=str(e))
