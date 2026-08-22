"""
MemSkill-style Memory Skill Evolution
=====================================
把静态的记忆操作（INSERT/UPDATE/DELETE/SKIP）升级为可学习、可进化的记忆技能库。

核心组件：
- MemorySkill: 单个记忆技能定义（YAML frontmatter + Markdown body）
- SkillBank: 技能库文件管理
- SkillController: 根据当前上下文选择 Top-K 技能
- SkillExecutor: 根据选中技能生成结构化记忆操作
- SkillDesigner: 从失败/困难案例中学习，进化技能库
- MemSkillEngine: 统筹以上组件，对接现有 MemoryManager/LLMWikiManager
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from pydantic import BaseModel, Field

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

logger = logging.getLogger("memskill")


# =============================================================================
# 0. 数据模型
# =============================================================================

class MemoryOperation(BaseModel):
    """Executor 输出的结构化记忆操作。"""
    op: str = Field(..., description="操作类型: insert | update | delete | skip")
    content: str = Field("", description="记忆内容")
    page_title: str = Field("", description="用于 wiki memory 的页面标题")
    section: str = Field("", description="用于 wiki memory 的章节名")
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    target_id: str = Field("", description="update/delete 时指向已有记忆 ID")
    reason: str = Field("", description="为什么执行这个操作")
    confidence: float = Field(0.7, ge=0.0, le=1.0)


class MemorySkill(BaseModel):
    """记忆技能定义。"""
    name: str
    description: str = ""
    version: str = "0.1.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    author: str = "LvAgent"
    tags: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    when_to_use: str = ""
    how_to_apply: str = ""
    constraints: str = ""
    examples: str = ""
    body: str = ""  # 完整 markdown 正文，用于嵌入
    source: str = "builtin"  # builtin | learned | evolved
    lineage: List[str] = Field(default_factory=list)  # 进化血缘
    embedding: Optional[List[float]] = None

    def to_markdown(self) -> str:
        """序列化为 SKILL.md 格式。"""
        frontmatter = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "tags": self.tags,
            "platforms": self.platforms,
            "source": self.source,
            "lineage": self.lineage,
        }
        parts = ["---", yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip(), "---"]
        if self.when_to_use:
            parts.extend(["", "# When to Use", self.when_to_use])
        if self.how_to_apply:
            parts.extend(["", "# How to Apply", self.how_to_apply])
        if self.constraints:
            parts.extend(["", "# Constraints", self.constraints])
        if self.examples:
            parts.extend(["", "# Examples", self.examples])
        if self.body:
            parts.extend(["", self.body])
        return "\n".join(parts) + "\n"

    def embedding_text(self) -> str:
        """用于计算嵌入的文本。"""
        return "\n".join([
            self.name,
            self.description,
            self.when_to_use,
            self.how_to_apply,
            " ".join(self.tags),
        ]).strip()


# =============================================================================
# 1. SkillBank
# =============================================================================

class SkillBank:
    """管理记忆技能文件的加载、保存、版本。"""

    DEFAULT_SKILLS: List[Dict[str, Any]] = [
        {
            "name": "insert",
            "description": "Insert new factual information into memory.",
            "when_to_use": "Use when the interaction contains new facts, preferences, procedures, or insights not already in memory.",
            "how_to_apply": "1. Identify key entities and relationships.\n2. Summarize the insight concisely.\n3. Store as a memory entry with linked entities.",
            "constraints": "- Do not duplicate existing memories.\n- Keep content under 300 characters.\n- Only store durable facts, not transient details.",
            "examples": "Input: User prefers dark mode.\nOutput: {\"op\": \"insert\", \"content\": \"User prefers dark mode UI\", \"entities\": [\"dark mode\"]}",
            "tags": ["memory", "core"],
        },
        {
            "name": "update",
            "description": "Update existing memory when new information contradicts or refines it.",
            "when_to_use": "Use when the user corrects a previous fact, changes a preference, or provides a newer version of known information.",
            "how_to_apply": "1. Retrieve the most relevant existing memory.\n2. Merge old and new information.\n3. Store the updated version and link to the old one.",
            "constraints": "- Only update when confidence is high.\n- Preserve historical nuance; do not overwrite causally.",
            "examples": "Input: User said they now use Python 3.12, previously 3.10.\nOutput: {\"op\": \"update\", \"target_id\": \"...\", \"content\": \"User uses Python 3.12\"}",
            "tags": ["memory", "core"],
        },
        {
            "name": "delete",
            "description": "Remove outdated or incorrect memories.",
            "when_to_use": "Use when a memory is explicitly invalidated, obsolete, or harmful.",
            "how_to_apply": "1. Identify the target memory by ID or title.\n2. Confirm it should be removed.\n3. Issue delete operation.",
            "constraints": "- Only delete when explicitly requested or strongly invalidated.\n- Prefer update over delete for partial changes.",
            "examples": "Input: Delete my old API key memory.\nOutput: {\"op\": \"delete\", \"target_id\": \"...\", \"reason\": \"User requested removal\"}",
            "tags": ["memory", "core"],
        },
        {
            "name": "skip",
            "description": "Decide that nothing durable should be stored.",
            "when_to_use": "Use for greetings, chitchat, transient debugging output, or redundant information already well covered.",
            "how_to_apply": "Return skip with a short reason.",
            "constraints": "- Do not skip user preferences, constraints, or recurring workflows.\n- Err toward storing if uncertain.",
            "examples": "Input: hello.\nOutput: {\"op\": \"skip\", \"reason\": \"Greeting, no durable information\"}",
            "tags": ["memory", "core"],
        },
        {
            "name": "summarize_error",
            "description": "Extract failure patterns and fixes into memory.",
            "when_to_use": "Use when a tool fails, code throws an error, or a multi-step plan goes wrong and a workaround is found.",
            "how_to_apply": "1. Capture the error signature.\n2. Capture the root cause.\n3. Capture the fix or workaround.\n4. Store as a reusable lesson.",
            "constraints": "- Strip sensitive paths/credentials.\n- Generalize the pattern so it helps future similar cases.",
            "examples": "Input: bash_exec blocked redirect to /dev/null.\nOutput: {\"op\": \"insert\", \"content\": \"bash_exec safety blocks /dev/null redirects; avoid 2>/dev/null\", \"tags\": [\"bash_exec\", \"safety\"]}",
            "tags": ["memory", "error", "learning"],
        },
        {
            "name": "extract_preference",
            "description": "Capture user preferences and constraints.",
            "when_to_use": "Use when the user states a personal preference, design choice, workflow constraint, or communication style.",
            "how_to_apply": "1. Identify the preference type (UI, output format, process, model).\n2. Phrase as a stable fact.\n3. Tag appropriately.",
            "constraints": "- Distinguish tentative preferences from firm constraints.\n- Update rather than duplicate if preference changes.",
            "examples": "Input: I prefer concise output without emojis.\nOutput: {\"op\": \"insert\", \"content\": \"User prefers concise output without emojis\", \"tags\": [\"preference\"]}",
            "tags": ["memory", "preference"],
        },
        {
            "name": "capture_workflow",
            "description": "Remember successful multi-step workflows as reusable procedures.",
            "when_to_use": "Use when the agent completes a non-trivial task and the steps could be reused later.",
            "how_to_apply": "1. Abstract the goal.\n2. List the key steps.\n3. Name any tools/config used.\n4. Store as a procedure memory.",
            "constraints": "- Keep steps general, not tied to one file path.\n- Include verification step if possible.",
            "examples": "Input: Successfully restarted Lv Agent after code changes.\nOutput: {\"op\": \"insert\", \"content\": \"Restart flow: kill opencode, clear port 8080, run SuperAgent.command\", \"tags\": [\"workflow\"]}",
            "tags": ["memory", "workflow"],
        },
    ]

    def __init__(self, skills_dir: str = "./data/memory_skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: Dict[str, MemorySkill] = {}
        self._lock = threading.RLock()
        self._stats_path = self.skills_dir / ".skill_stats.json"
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._load_stats()
        self._ensure_defaults()
        self.load_all()

    def _load_stats(self):
        """加载技能使用统计。"""
        if self._stats_path.exists():
            try:
                self._stats = json.loads(self._stats_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load skill stats: {e}")
                self._stats = {}

    def _save_stats(self):
        """保存技能使用统计。"""
        try:
            self._stats_path.write_text(json.dumps(self._stats, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save skill stats: {e}")

    def record_skill_usage(self, name: str, success: bool):
        """记录一次技能使用结果。"""
        with self._lock:
            stat = self._stats.setdefault(name, {"used": 0, "success": 0, "failed": 0, "last_used": ""})
            stat["used"] += 1
            stat["success"] += 1 if success else 0
            stat["failed"] += 0 if success else 1
            stat["last_used"] = datetime.now().isoformat()
            self._save_stats()

    def get_skill_score(self, name: str) -> float:
        """计算技能得分：成功率 * ln(使用次数+1)。"""
        stat = self._stats.get(name, {"used": 0, "success": 0})
        used = max(1, stat.get("used", 0))
        success = stat.get("success", 0)
        success_rate = success / used
        # 使用对数折扣，避免低使用次数的极端成功率
        usage_factor = math.log(used + 1)
        return success_rate * usage_factor

    def evaluate_and_prune(self, min_score: float = 0.3, min_usage: int = 3) -> List[str]:
        """评估 evolved 技能，低分技能自动回滚/删除。返回被删除的技能名。"""
        pruned = []
        with self._lock:
            for name, skill in list(self._skills.items()):
                if skill.source != "evolved":
                    continue
                stat = self._stats.get(name, {"used": 0})
                if stat.get("used", 0) < min_usage:
                    continue
                score = self.get_skill_score(name)
                if score < min_score:
                    # 尝试回滚到血缘中的上一个版本
                    if skill.lineage and self._rollback_skill(skill):
                        pruned.append(f"{name} (rolled back)")
                    else:
                        self.delete(name)
                        pruned.append(f"{name} (deleted)")
                    self._stats.pop(name, None)
            self._save_stats()
        return pruned

    def _rollback_skill(self, skill: MemorySkill) -> bool:
        """尝试回滚 evolved 技能到上一个版本。"""
        # lineage 中找上一个版本号，如 evolved_from_0.1.0
        prev_version = None
        for entry in reversed(skill.lineage):
            m = re.search(r"evolved_from_([\d.]+)", entry)
            if m:
                prev_version = m.group(1)
                break
        if not prev_version:
            return False
        # 从备份目录找对应版本快照
        backup_dir = self.backup_dir()
        candidates = sorted(backup_dir.glob("*"), reverse=True)
        for cand in candidates:
            path = cand / f"{skill.name}.md"
            if not path.exists():
                continue
            try:
                old_skill = self._parse_file(path)
                if old_skill.version == prev_version:
                    self.save(old_skill)
                    logger.info(f"Rolled back skill {skill.name} to version {prev_version}")
                    return True
            except Exception:
                continue
        return False

    def _ensure_defaults(self):
        """首次运行时写入默认技能。"""
        if any(self.skills_dir.glob("*.md")):
            return
        for spec in self.DEFAULT_SKILLS:
            skill = MemorySkill(**spec)
            self.save(skill)
        logger.info("Initialized default memory skills.")

    def save(self, skill: MemorySkill):
        """保存技能到文件。"""
        if not _HAS_YAML:
            raise RuntimeError("pyyaml is required for skill persistence")
        path = self.skills_dir / f"{skill.name}.md"
        path.write_text(skill.to_markdown(), encoding="utf-8")
        with self._lock:
            self._skills[skill.name] = skill

    def load_all(self) -> Dict[str, MemorySkill]:
        """加载所有技能。"""
        if not _HAS_YAML:
            logger.warning("pyyaml not available; using empty skill bank")
            return {}
        skills: Dict[str, MemorySkill] = {}
        for path in sorted(self.skills_dir.glob("*.md")):
            try:
                skill = self._parse_file(path)
                skills[skill.name] = skill
            except Exception as e:
                logger.warning(f"Failed to load skill {path}: {e}")
        with self._lock:
            self._skills = skills
        return skills

    def _parse_file(self, path: Path) -> MemorySkill:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError(f"Missing frontmatter in {path}")
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid frontmatter in {path}")
        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()

        # 解析 markdown 章节
        sections = self._split_markdown_sections(body)
        frontmatter.setdefault("when_to_use", sections.get("When to Use", ""))
        frontmatter.setdefault("how_to_apply", sections.get("How to Apply", ""))
        frontmatter.setdefault("constraints", sections.get("Constraints", ""))
        frontmatter.setdefault("examples", sections.get("Examples", ""))
        frontmatter["body"] = body
        frontmatter.setdefault("name", path.stem)
        return MemorySkill(**frontmatter)

    @staticmethod
    def _split_markdown_sections(text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        current = []
        current_title = ""
        for line in text.splitlines():
            m = re.match(r"^#\s+(.+)$", line)
            if m:
                if current_title:
                    sections[current_title] = "\n".join(current).strip()
                current_title = m.group(1).strip()
                current = []
            else:
                current.append(line)
        if current_title:
            sections[current_title] = "\n".join(current).strip()
        return sections

    def get(self, name: str) -> Optional[MemorySkill]:
        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> List[MemorySkill]:
        with self._lock:
            return list(self._skills.values())

    def delete(self, name: str) -> bool:
        path = self.skills_dir / f"{name}.md"
        if path.exists():
            path.unlink()
        with self._lock:
            return self._skills.pop(name, None) is not None

    def backup_dir(self) -> Path:
        """返回用于版本回滚的备份目录。"""
        backup = self.skills_dir / ".backups"
        backup.mkdir(parents=True, exist_ok=True)
        return backup

    def snapshot(self, tag: str = "") -> Path:
        """对当前技能库做快照。"""
        tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.backup_dir() / tag
        backup.mkdir(parents=True, exist_ok=True)
        for path in self.skills_dir.glob("*.md"):
            if path.parent == backup:
                continue
            (backup / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup

    def restore(self, backup: Path):
        """从快照恢复技能库。"""
        if not backup.exists():
            raise FileNotFoundError(backup)
        for path in self.skills_dir.glob("*.md"):
            path.unlink()
        for path in backup.glob("*.md"):
            (self.skills_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        self.load_all()


# =============================================================================
# 2. SkillController
# =============================================================================

class SkillController:
    """根据当前上下文选择最相关的 Top-K 记忆技能。"""

    def __init__(self, bank: SkillBank, top_k: int = 3):
        self.bank = bank
        self.top_k = top_k

    def select(self, context: str) -> List[Tuple[MemorySkill, float]]:
        """返回 (skill, score) 列表，按相关性降序。"""
        skills = self.bank.list_skills()
        if not skills:
            return []
        return self._keyword_select(skills, context)

    def _keyword_select(self, skills: List[MemorySkill], context: str) -> List[Tuple[MemorySkill, float]]:
        ctx_words = set(re.findall(r"\w+", context.lower()))
        scored = []
        for skill in skills:
            skill_words = set(re.findall(r"\w+", skill.embedding_text().lower()))
            if not skill_words:
                continue
            overlap = len(ctx_words & skill_words)
            score = overlap / max(1, len(skill_words))
            if score > 0.05:
                scored.append((skill, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]


# =============================================================================
# 3. SkillExecutor
# =============================================================================

class SkillExecutor:
    """根据选中的技能，让 LLM 生成结构化记忆操作。"""

    def __init__(self, llm_call: Callable[[str, float, int], str]):
        """
        llm_call: (prompt, temperature, max_tokens) -> str
        """
        self.llm_call = llm_call

    def execute(
        self,
        task: str,
        trajectory: Dict[str, Any],
        outcome: str,
        success: bool,
        selected_skills: List[Tuple[MemorySkill, float]],
        existing_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> List[MemoryOperation]:
        """生成记忆操作列表。"""
        if not selected_skills:
            return []

        prompt = self._build_prompt(task, trajectory, outcome, success, selected_skills, existing_memories)
        try:
            raw = self.llm_call(prompt, 0.2, 1024)
            return self._parse_operations(raw)
        except Exception as e:
            logger.warning(f"SkillExecutor failed: {e}")
            return []

    def _build_prompt(
        self,
        task: str,
        trajectory: Dict[str, Any],
        outcome: str,
        success: bool,
        selected_skills: List[Tuple[MemorySkill, float]],
        existing_memories: Optional[List[Dict[str, Any]]],
    ) -> str:
        skill_texts = []
        for skill, score in selected_skills:
            skill_texts.append(
                f"### {skill.name} (relevance: {score:.2f})\n"
                f"Description: {skill.description}\n"
                f"When to use: {skill.when_to_use}\n"
                f"How to apply: {skill.how_to_apply}\n"
                f"Constraints: {skill.constraints}\n"
                f"Examples:\n{skill.examples}"
            )

        existing = ""
        if existing_memories:
            existing = "\n".join(
                f"- [{m.get('id', '?')}] {m.get('content', m.get('summary', ''))[:200]}"
                for m in existing_memories
            )

        trajectory_text = json.dumps(trajectory, ensure_ascii=False, indent=2)[:1500]

        return (
            "You are a memory curator for an AI agent. Given the task, execution trace, outcome, "
            "and a set of memory skills, decide what memory operations to perform.\n\n"
            "Rules:\n"
            "- Only store durable, reusable information.\n"
            "- Prefer UPDATE over INSERT if a similar memory already exists.\n"
            "- For failures, capture the error pattern and fix as a reusable lesson.\n"
            "- Output ONLY a JSON array of operations. No markdown fences, no preamble.\n\n"
            "JSON schema for each operation:\n"
            '{"op": "insert|update|delete|skip", '
            '"content": "...", '
            '"page_title": "short title for wiki memory", '
            '"section": "optional section name", '
            '"tags": ["tag1"], '
            '"importance": 0.0-1.0, '
            '"entities": ["entity1"], '
            '"target_id": "existing memory id for update/delete", '
            '"reason": "why this operation", '
            '"confidence": 0.0-1.0}\n\n'
            f"Selected memory skills:\n{'\n\n'.join(skill_texts)}\n\n"
            f"Task: {task}\n"
            f"Outcome: {outcome}\n"
            f"Success: {success}\n\n"
            f"Existing similar memories:\n{existing or '(none)'}\n\n"
            f"Execution trace (truncated):\n{trajectory_text}\n\n"
            "Memory operations (JSON array):"
        )

    def _parse_operations(self, raw: str) -> List[MemoryOperation]:
        raw = raw.strip()
        # 去掉可能的 markdown fence
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # 尝试提取 JSON 数组
        candidates = []
        if raw.startswith("["):
            candidates.append(raw)
        else:
            # 找第一个 [ 和最后一个 ]
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidates.append(raw[start:end + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    ops = []
                    for item in data:
                        if isinstance(item, dict):
                            ops.append(MemoryOperation(**item))
                    return ops
            except Exception:
                continue

        return []


# =============================================================================
# 4. SkillDesigner
# =============================================================================

@dataclass
class HardCase:
    """用于技能进化的困难案例。"""
    id: str
    task: str
    outcome: str
    success: bool
    trajectory_summary: str
    stored_memory_summary: str
    skills_used: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SkillDesigner:
    """定期复盘困难案例，进化技能库。"""

    def __init__(
        self,
        bank: SkillBank,
        llm_call: Callable[[str, float, int], str],
        hard_case_buffer_size: int = 50,
        evolution_interval: int = 5,
    ):
        self.bank = bank
        self.llm_call = llm_call
        self.hard_case_buffer: deque[HardCase] = deque(maxlen=hard_case_buffer_size)
        self.evolution_interval = evolution_interval
        self._evolution_count = 0

    def add_case(
        self,
        task: str,
        outcome: str,
        success: bool,
        trajectory_summary: str,
        stored_memory_summary: str,
        skills_used: List[str],
    ):
        """添加一个困难案例。失败案例自动加入；成功案例只加入高价值/复杂案例。"""
        if success and len(trajectory_summary) < 200:
            return
        case = HardCase(
            id=str(uuid.uuid4())[:8],
            task=task,
            outcome=outcome,
            success=success,
            trajectory_summary=trajectory_summary,
            stored_memory_summary=stored_memory_summary,
            skills_used=skills_used,
        )
        self.hard_case_buffer.append(case)
        logger.debug(f"Added hard case {case.id} (success={success})")

    def maybe_evolve(self) -> List[str]:
        """如果缓存足够，尝试进化技能。返回产生的技能名列表。"""
        if len(self.hard_case_buffer) < self.evolution_interval:
            return []

        failures = [c for c in self.hard_case_buffer if not c.success]
        if len(failures) < self.evolution_interval:
            return []

        return self.evolve(failures)

    def evolve(self, cases: Optional[List[HardCase]] = None) -> List[str]:
        """执行一次技能进化。"""
        cases = cases or list(self.hard_case_buffer)
        if not cases:
            return []

        # 先对案例聚类
        clusters = self._cluster_cases(cases)
        new_skills: List[str] = []

        for cluster in clusters:
            skill = self._propose_skill(cluster)
            if skill and self._accept_skill(skill, cluster):
                self.bank.save(skill)
                new_skills.append(skill.name)
                logger.info(f"Evolved new skill: {skill.name}")

        self._evolution_count += 1
        # 仅在成功产出技能时清空已消费的失败案例; 否则保留, 供后续批次继续尝试学习
        if new_skills:
            self.hard_case_buffer.clear()
        return new_skills

    def _cluster_cases(self, cases: List[HardCase], min_cluster_size: int = 2) -> List[List[HardCase]]:
        """Simple clustering: group all cases into one batch since no embeddings."""
        if len(cases) < min_cluster_size:
            return []
        return [list(cases)]

    def _propose_skill(self, cluster: List[HardCase]) -> Optional[MemorySkill]:
        """让 LLM 从聚类中提出新技能或技能改进。"""
        cases_text = "\n\n".join(
            f"Case {i+1}:\nTask: {c.task}\nOutcome: {c.outcome}\n"
            f"Trajectory: {c.trajectory_summary[:400]}\n"
            f"Memory stored: {c.stored_memory_summary}\n"
            f"Skills used: {', '.join(c.skills_used)}"
            for i, c in enumerate(cluster)
        )

        prompt = (
            "You are a skill designer for an AI agent's memory system. "
            "Analyze the following failure cases and propose ONE new memory skill or a refinement to an existing skill.\n\n"
            "A memory skill should describe: when to use it, how to apply it, and what constraints to follow.\n"
            "If the cases reveal a missing skill, create a new one. "
            "If they reveal an existing skill is incomplete, propose a refined version.\n\n"
            f"{cases_text}\n\n"
            "Output ONLY a JSON object with this schema:\n"
            '{"name": "lowercase-hyphenated-name", '
            '"description": "one sentence <= 60 chars", '
            '"when_to_use": "...", '
            '"how_to_apply": "...", '
            '"constraints": "...", '
            '"examples": "...", '
            '"tags": ["memory", "..."], '
            '"is_refinement": false, '
            '"refines_skill": ""}\n\n'
            "Skill JSON:"
        )

        try:
            raw = self.llm_call(prompt, 0.3, 1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw.strip())
            if not isinstance(data, dict):
                return None

            name = data.get("name", "")
            if not name:
                return None

            # 如果是改进，继承原技能并升级版本
            refines = data.get("refines_skill", "")
            if refines:
                old = self.bank.get(refines)
                if old:
                    new_data = old.dict()
                    new_data["version"] = _bump_version(old.version)
                    new_data["updated_at"] = datetime.now().isoformat()
                    new_data["lineage"] = old.lineage + [f"evolved_from_{old.version}"]
                    for key in ["description", "when_to_use", "how_to_apply", "constraints", "examples"]:
                        if data.get(key):
                            new_data[key] = data[key]
                    return MemorySkill(**new_data)

            return MemorySkill(
                name=name,
                description=data.get("description", ""),
                when_to_use=data.get("when_to_use", ""),
                how_to_apply=data.get("how_to_apply", ""),
                constraints=data.get("constraints", ""),
                examples=data.get("examples", ""),
                tags=data.get("tags", ["memory"]),
                source="evolved",
            )
        except Exception as e:
            logger.warning(f"Skill proposal failed: {e}")
            return None

    def _accept_skill(self, skill: MemorySkill, cluster: List[HardCase]) -> bool:
        """简单验收：名称合法、描述非空、不与现有完全相同。

        注意: 内置技能名(如 capture_workflow / summarize_error)使用下划线,
        校验必须允许 `_`, 否则对它们的改进会全部被拒。
        """
        if not re.match(r"^[a-z0-9_-]+$", skill.name):
            return False
        if len(skill.description) == 0:
            return False
        existing = self.bank.get(skill.name)
        if existing:
            # 简单对比核心字段
            if (existing.when_to_use == skill.when_to_use and
                existing.how_to_apply == skill.how_to_apply):
                return False
        return True

    @staticmethod
    def _bump_version(version: str) -> str:
        """简单版本号 +1，例如 0.1.0 -> 0.2.0。"""
        parts = version.split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except Exception:
            parts = ["0", "1", "0"]
        return ".".join(parts[:3])


# =============================================================================
# 5. MemSkillEngine
# =============================================================================

class MemSkillEngine:
    """
    MemSkill 总控引擎。
    对接现有 memory backend：MemoryManager 或 LLMWikiManager。
    """

    def __init__(
        self,
        llm_call: Callable[[str, float, int], str],
        skills_dir: str = "./data/memory_skills",
        top_k: int = 3,
        hard_case_buffer_size: int = 50,
        evolution_interval: int = 5,
    ):
        self.bank = SkillBank(skills_dir=skills_dir)
        self.controller = SkillController(self.bank, top_k=top_k)
        self.executor = SkillExecutor(llm_call=llm_call)
        self.designer = SkillDesigner(
            self.bank,
            llm_call=llm_call,
            hard_case_buffer_size=hard_case_buffer_size,
            evolution_interval=evolution_interval,
        )
        self._llm_call = llm_call

    def learn_from_interaction(
        self,
        task: str,
        trajectory: Dict[str, Any],
        outcome: str,
        success: bool,
        memory_backend: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        一次交互结束后，用 MemSkill 生成并执行记忆操作。
        返回 {operations, skills_used, evolved_skills}。
        """
        # 1. 构建上下文
        context = f"Task: {task}\nOutcome: {outcome}\nSuccess: {success}"

        # 2. 选择技能
        selected = self.controller.select(context)
        skill_names = [s.name for s, _ in selected]

        # 3. 检索相关已有记忆，避免重复
        existing_memories = []
        if memory_backend is not None:
            try:
                existing_memories = self._retrieve_similar_memories(memory_backend, task, k=3)
            except Exception as e:
                logger.debug(f"Failed to retrieve existing memories: {e}")

        # 4. 生成记忆操作
        operations = self.executor.execute(
            task=task,
            trajectory=trajectory,
            outcome=outcome,
            success=success,
            selected_skills=selected,
            existing_memories=existing_memories,
        )

        # 5. 执行记忆操作
        applied = []
        for op in operations:
            try:
                self._apply_operation(memory_backend, op)
                applied.append(op)
            except Exception as e:
                logger.warning(f"Failed to apply memory operation {op.op}: {e}")

        # 5.5 记录技能使用统计(供 get_skill_score / evaluate_and_prune 打分与回滚)
        # 之前无调用点导致 .skill_stats.json 恒空, 剪枝/回滚闭环休眠
        for name in skill_names:
            try:
                self.bank.record_skill_usage(name, success=bool(applied))
            except Exception as e:
                logger.debug(f"record_skill_usage failed for {name}: {e}")

        # 6. 把这次学习作为困难案例记录，并尝试进化
        stored_summary = "; ".join(
            f"{op.op}:{op.content[:60]}" for op in applied
        ) or "no-op"
        self.designer.add_case(
            task=task,
            outcome=outcome,
            success=success,
            trajectory_summary=self._summarize_trajectory(trajectory),
            stored_memory_summary=stored_summary,
            skills_used=skill_names,
        )
        evolved = self.designer.maybe_evolve()

        # 6.5 定期剪枝低分 evolved 技能: 使用不足或低分者回滚到上一版本
        try:
            self.bank.evaluate_and_prune()
        except Exception as e:
            logger.debug(f"evaluate_and_prune failed: {e}")

        return {
            "operations": [op.dict() for op in applied],
            "skills_used": skill_names,
            "evolved_skills": evolved,
        }

    def force_evolve(self) -> List[str]:
        """手动触发一次技能进化。"""
        return self.designer.evolve()

    def list_skills(self) -> List[Dict[str, Any]]:
        return [s.dict(exclude={"body", "embedding"}) for s in self.bank.list_skills()]

    def snapshot_skills(self, tag: str = "") -> Path:
        return self.bank.snapshot(tag=tag)

    def restore_skills(self, backup: Path):
        self.bank.restore(backup)

    def _retrieve_similar_memories(self, backend: Any, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """兼容 MemoryManager / LLMWikiManager 的检索。"""
        memories: List[Dict[str, Any]] = []
        try:
            if hasattr(backend, "get_relevant_context"):
                ctx = backend.get_relevant_context(query, max_episodes=k, max_nodes=k)
                for ep in ctx.get("episodes", []):
                    memories.append({"id": ep.get("id", ""), "content": ep.get("summary", "")})
                for node in ctx.get("knowledge_nodes", []):
                    props = node.get("properties", {})
                    memories.append({"id": node.get("id", ""), "content": props.get("name", "")})
            elif hasattr(backend, "retrieve_similar"):
                # EpisodicMemory
                for ep, score in backend.retrieve_similar(query, k=k):
                    memories.append({"id": getattr(ep, "id", ""), "content": getattr(ep, "summary", "")})
            elif hasattr(backend, "search"):
                # LLMWikiManager
                results = backend.search(query, top_k=k)
                for r in results:
                    memories.append({"id": r.get("id", ""), "content": r.get("content", "")[:200]})
        except Exception as e:
            logger.debug(f"Memory retrieval compatibility failed: {e}")
        return memories

    def _apply_operation(self, backend: Any, op: MemoryOperation):
        """把操作应用到具体 memory backend。"""
        if backend is None:
            return

        if op.op == "skip":
            return

        # 尝试 LLMWikiManager / WikiGraph 接口
        if op.op == "insert":
            if hasattr(backend, "quick_store"):
                backend.quick_store(
                    content=op.content,
                    title=op.page_title or self._auto_title(op.content),
                    page_type="episode",
                    tags=op.tags,
                )
                return
            if hasattr(backend, "store_interaction"):
                backend.store_interaction(
                    task=op.page_title or self._auto_title(op.content),
                    trajectory={"content": op.content, "entities": op.entities},
                    outcome=op.content,
                    success=True,
                    entities=op.entities,
                )
                return
            if hasattr(backend, "remember"):
                backend.remember(op.content, metadata={"tags": op.tags, "importance": op.importance})
                return

        if op.op in ("update", "delete"):
            logger.debug(f"{op.op} operation requires explicit backend support; skipping apply")

    def _auto_title(self, content: str) -> str:
        """从内容生成一个简短标题。"""
        content = content.strip()
        if len(content) <= 40:
            return content
        return content[:40].rsplit(" ", 1)[0] + "..."

    def _summarize_trajectory(self, trajectory: Dict[str, Any]) -> str:
        """生成 trajectory 的简短摘要，用于 Designer。"""
        steps = trajectory.get("steps", [])
        tools = []
        for step in steps:
            if isinstance(step, dict):
                tool = step.get("tool") or step.get("action")
                if tool:
                    tools.append(str(tool))
        return f"steps={len(steps)}, tools={','.join(tools[:5])}"


def _bump_version(version: str) -> str:
    """简单版本号 +1，例如 0.1.0 -> 0.2.0。"""
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
    except Exception:
        parts = ["0", "1", "0"]
    return ".".join(parts[:3])
