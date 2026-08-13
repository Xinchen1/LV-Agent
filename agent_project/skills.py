"""
Lv Super Agent - Skill Engine
=============================
Lightweight, self-evolving skill system for the agent.

A Skill is a reusable mini-agent: a prompt template plus metadata that tells
Lv Super Agent how to behave for a specific class of tasks.  Skills are stored
as Markdown files with YAML frontmatter and can be loaded, created, and invoked
at runtime via the `/skill` slash command.

Integration:
- Agent initialization registers a global SkillEngine.
- The engine searches the skills directory on startup and registers every
  `.skill.md` / `.md` file as a Skill.
- When the user types `/skill <name>` the matching skill is loaded into the
  system prompt for the next turn.
- Skills can declare preferred tools; the agent uses this as a hint when
  choosing tools.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger("skill_engine")


# =============================================================================
# 0. Data model
# =============================================================================

class Skill(BaseModel):
    """A reusable agent skill."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    author: str = "LvAgent"
    tags: List[str] = Field(default_factory=list)
    trigger_keywords: List[str] = Field(default_factory=list)
    preferred_tools: List[str] = Field(default_factory=list)
    prompt_template: str = ""
    examples: List[Dict[str, str]] = Field(default_factory=list)
    source: str = "builtin"  # builtin | user | evolved
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    success_rate: float = 0.0
    embedding: Optional[List[float]] = None

    def to_markdown(self) -> str:
        """Serialize to Markdown with YAML frontmatter."""
        frontmatter = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "trigger_keywords": self.trigger_keywords,
            "preferred_tools": self.preferred_tools,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
        }
        parts = [
            "---",
            yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip() if _HAS_YAML else json.dumps(frontmatter, ensure_ascii=False),
            "---",
        ]
        if self.prompt_template:
            parts.extend(["", "# Prompt Template", self.prompt_template])
        if self.examples:
            parts.extend(["", "# Examples"])
            for ex in self.examples:
                parts.append(f"## Input: {ex.get('input', '')}")
                parts.append(f"## Output: {ex.get('output', '')}")
                parts.append("")
        return "\n".join(parts) + "\n"

    def embedding_text(self) -> str:
        """Text used for semantic skill retrieval."""
        return "\n".join([
            self.name,
            self.description,
            " ".join(self.tags),
            " ".join(self.trigger_keywords),
            self.prompt_template[:500],
        ]).strip()

    def render(self, task: str, context: str = "") -> str:
        """Render the skill prompt for a concrete task."""
        text = self.prompt_template
        text = text.replace("{{task}}", task)
        text = text.replace("{{context}}", context)
        text = text.replace("{{skill_name}}", self.name)
        return text


# =============================================================================
# 1. Built-in skills
# =============================================================================

BUILTIN_SKILLS: List[Dict[str, Any]] = [
    {
        "name": "deep_research",
        "description": "Conduct iterative deep research and synthesize a comprehensive report.",
        "tags": ["research", "analysis"],
        "trigger_keywords": ["深度研究", "deep research", "调研报告", "research report", "industry analysis"],
        "preferred_tools": ["web_search", "web_fetch", "file_ops"],
        "prompt_template": (
            "You are in deep-research mode. Your goal is to thoroughly investigate: {{task}}\n\n"
            "Follow this protocol:\n"
            "1. Expand the topic into multiple search queries (angles: overview, latest news, "
            "technical details, market/opinion, risks/challenges).\n"
            "2. Search widely; aggregate and deduplicate results.\n"
            "3. Fetch high-quality sources and extract key evidence.\n"
            "4. Synthesize findings into a structured report with citations.\n"
            "5. Identify gaps and perform follow-up searches if needed.\n\n"
            "Always cite sources with URLs. Output in Chinese unless asked otherwise."
        ),
        "examples": [
            {"input": "深度研究 实在智能", "output": "生成包含公司背景、产品技术、融资、最新动态的调研报告"},
        ],
    },
    {
        "name": "code_assistant",
        "description": "Write, review, debug, and refactor code with best practices.",
        "tags": ["code", "programming"],
        "trigger_keywords": ["写代码", "code", "编程", "debug", "refactor", "review code"],
        "preferred_tools": ["python_exec", "file_ops", "bash_exec", "web_search"],
        "prompt_template": (
            "You are an expert software engineer. Task: {{task}}\n\n"
            "Protocol:\n"
            "1. Understand requirements and ask clarifying questions if ambiguous.\n"
            "2. Produce clean, idiomatic, well-commented code.\n"
            "3. Include brief tests or usage examples when applicable.\n"
            "4. Explain trade-offs and key decisions concisely.\n"
            "5. If fixing bugs, show the root cause and the fix."
        ),
        "examples": [
            {"input": "写个 Python 函数算斐波那契", "output": "提供递归和迭代实现并解释复杂度"},
        ],
    },
    {
        "name": "file_manager",
        "description": "Read, write, list, and organize local files and folders.",
        "tags": ["files", "filesystem"],
        "trigger_keywords": ["文件", "folder", "打开文件", "read file", "list files", "整理文件"],
        "preferred_tools": ["file_ops", "bash_exec", "grep_tool", "glob_tool"],
        "prompt_template": (
            "You are a file-system assistant. Task: {{task}}\n\n"
            "Protocol:\n"
            "1. Resolve paths safely; verify existence before destructive operations.\n"
            "2. Prefer `file_ops` for read/write/list and `bash_exec` for bulk operations.\n"
            "3. Summarize what you found or changed.\n"
            "4. If the user asks to open a file, use `file_ops(action='open', path=...)` directly."
        ),
        "examples": [
            {"input": "打开最近的报告", "output": "调用 file_ops action=open 打开 last_report_path"},
        ],
    },
    {
        "name": "security_guard",
        "description": "Review commands, code, or requests for safety and policy compliance.",
        "tags": ["security", "safety"],
        "trigger_keywords": ["安全", "危险", "safety", "security", "prompt injection", "jailbreak"],
        "preferred_tools": ["bash_exec"],
        "prompt_template": (
            "You are a security reviewer. Task: {{task}}\n\n"
            "Protocol:\n"
            "1. Identify dangerous or suspicious patterns.\n"
            "2. Reference the harness safety policy if relevant.\n"
            "3. Provide a clear verdict and mitigations.\n"
            "Never execute destructive commands yourself."
        ),
        "examples": [],
    },
]


# =============================================================================
# 2. Skill Bank
# =============================================================================

class SkillBank:
    """Loads, saves, and indexes skill files."""

    def __init__(
        self,
        skills_dir: str = "./data/skills",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.RLock()
        self._embedding_model: Optional[Any] = None
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self._embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.warning(f"SkillBank embedding model unavailable: {e}")

    def _ensure_builtins(self):
        """Write built-in skills to disk if they do not exist."""
        for data in BUILTIN_SKILLS:
            path = self.skills_dir / f"{data['name']}.skill.md"
            if not path.exists():
                skill = Skill(**data)
                path.write_text(skill.to_markdown(), encoding="utf-8")

    def load_all(self) -> Dict[str, Skill]:
        """Reload all skills from disk."""
        with self._lock:
            self._ensure_builtins()
            self._skills.clear()
            for path in sorted(self.skills_dir.glob("*.md")):
                skill = self._load_file(path)
                if skill:
                    self._skills[skill.name] = skill
            return dict(self._skills)

    def _load_file(self, path: Path) -> Optional[Skill]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cannot read skill {path}: {e}")
            return None

        # Parse YAML frontmatter between --- fences.
        frontmatter: Dict[str, Any] = {}
        prompt_template = ""
        examples: List[Dict[str, str]] = []

        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
        if m:
            fm_text = m.group(1)
            body = m.group(2)
            if _HAS_YAML:
                try:
                    frontmatter = yaml.safe_load(fm_text) or {}
                except Exception:
                    frontmatter = {}
            else:
                try:
                    frontmatter = json.loads(fm_text)
                except Exception:
                    frontmatter = {}

            # Split body into prompt template and examples.
            if "# Examples" in body:
                prompt_part, examples_part = body.split("# Examples", 1)
                prompt_template = prompt_part.strip()
                examples = self._parse_examples(examples_part)
            else:
                prompt_template = body.strip()
        else:
            prompt_template = text.strip()

        if not frontmatter.get("name"):
            frontmatter["name"] = path.stem.replace(".skill", "")
        if "prompt_template" not in frontmatter or not frontmatter.get("prompt_template"):
            frontmatter["prompt_template"] = prompt_template
        if "examples" not in frontmatter and examples:
            frontmatter["examples"] = examples

        try:
            skill = Skill(**frontmatter)
        except Exception as e:
            logger.warning(f"Invalid skill {path}: {e}")
            return None

        # Compute embedding if model is ready.
        if self._embedding_model is not None:
            try:
                skill.embedding = self._embedding_model.encode(skill.embedding_text()).tolist()
            except Exception as e:
                logger.debug(f"skill embedding failed: {e}")
        return skill

    @staticmethod
    def _parse_examples(text: str) -> List[Dict[str, str]]:
        examples: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("## Input:"):
                if current:
                    examples.append(current)
                current = {"input": line[len("## Input:"):].strip()}
            elif line.startswith("## Output:"):
                current["output"] = line[len("## Output:"):].strip()
            elif current:
                key = "input" if "input" in current and "output" not in current else "output"
                current[key] = current.get(key, "") + "\n" + line
        if current:
            examples.append(current)
        return examples

    def save(self, skill: Skill) -> Path:
        """Persist a skill to disk."""
        skill.updated_at = datetime.now().isoformat()
        path = self.skills_dir / f"{skill.name}.skill.md"
        path.write_text(skill.to_markdown(), encoding="utf-8")
        with self._lock:
            self._skills[skill.name] = skill
        return path

    def get(self, name: str) -> Optional[Skill]:
        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        with self._lock:
            return list(self._skills.values())

    def delete(self, name: str) -> bool:
        with self._lock:
            skill = self._skills.pop(name, None)
        if skill is None:
            return False
        path = self.skills_dir / f"{name}.skill.md"
        try:
            path.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Skill, float]]:
        """Keyword + semantic search over skills."""
        query_lower = query.lower()
        scored: List[Tuple[Skill, float]] = []
        with self._lock:
            skills = list(self._skills.values())

        # Keyword score
        for skill in skills:
            score = 0.0
            texts = [
                skill.name.lower(),
                skill.description.lower(),
                " ".join(skill.tags).lower(),
                " ".join(skill.trigger_keywords).lower(),
            ]
            for text in texts:
                if query_lower in text:
                    score += 1.0
            if score:
                scored.append((skill, score))

        # Semantic score
        if self._embedding_model is not None and query.strip():
            try:
                q_emb = self._embedding_model.encode(query).tolist()
                for skill in skills:
                    if skill.embedding:
                        sim = _cosine_similarity(q_emb, skill.embedding)
                        scored.append((skill, sim))
            except Exception as e:
                logger.debug(f"semantic skill search failed: {e}")

        # Merge and rank
        merged: Dict[str, Tuple[Skill, float]] = {}
        for skill, score in scored:
            if skill.name in merged:
                merged[skill.name] = (skill, max(merged[skill.name][1], score))
            else:
                merged[skill.name] = (skill, score)
        ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# =============================================================================
# 3. Skill Engine
# =============================================================================

class SkillEngine:
    """Runtime skill management for Lv Super Agent."""

    def __init__(
        self,
        skills_dir: str = "./data/skills",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_call: Optional[Callable[[str, float, int], str]] = None,
    ):
        self.bank = SkillBank(skills_dir=skills_dir, embedding_model=embedding_model)
        self.llm_call = llm_call
        self.active_skill: Optional[Skill] = None
        self.logger = logging.getLogger("SkillEngine")
        self.bank.load_all()

    # ---- runtime API ----

    def list(self) -> List[Skill]:
        return self.bank.list_skills()

    def load(self, name: str) -> Optional[Skill]:
        """Activate a skill by exact name."""
        skill = self.bank.get(name)
        if skill:
            self.active_skill = skill
            skill.usage_count += 1
            self.bank.save(skill)
        return skill

    def unload(self):
        """Deactivate the current skill."""
        self.active_skill = None

    def suggest(self, task: str, top_k: int = 3) -> List[Tuple[Skill, float]]:
        """Suggest skills for a task."""
        return self.bank.search(task, top_k=top_k)

    def create(
        self,
        name: str,
        description: str,
        prompt_template: str,
        trigger_keywords: Optional[List[str]] = None,
        preferred_tools: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> Skill:
        """Create and persist a new user skill."""
        skill = Skill(
            name=name,
            description=description,
            prompt_template=prompt_template,
            trigger_keywords=trigger_keywords or [],
            preferred_tools=preferred_tools or [],
            tags=tags or [],
            examples=examples or [],
            source="user",
        )
        self.bank.save(skill)
        self.logger.info(f"created skill '{name}'")
        return skill

    def delete(self, name: str) -> bool:
        if name in {s["name"] for s in BUILTIN_SKILLS}:
            return False
        return self.bank.delete(name)

    def render_active(self, task: str, context: str = "") -> str:
        """Render the currently active skill prompt, if any."""
        if self.active_skill is None:
            return ""
        return self.active_skill.render(task, context)

    def get_active_tools_hint(self) -> List[str]:
        if self.active_skill is None:
            return []
        return list(self.active_skill.preferred_tools)

    def report_outcome(self, success: bool):
        """Update active skill success rate after a turn."""
        skill = self.active_skill
        if skill is None:
            return
        # Incremental moving average.
        n = max(skill.usage_count, 1)
        skill.success_rate = (skill.success_rate * (n - 1) + (1.0 if success else 0.0)) / n
        self.bank.save(skill)

    # ---- slash command parser ----

    def handle_slash(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a `/skill ...` command.

        Returns (remaining_task, result_dict).  If the command is purely about
        skill management, remaining_task will be empty and result_dict contains
        the response to show the user.
        """
        text = text.strip()
        if not text.lower().startswith("/skill"):
            return text, {}

        rest = text[len("/skill"):].strip()
        if not rest:
            skills = self.list()
            lines = [f"Loaded skills ({len(skills)}):", ""]
            for s in skills:
                active = " (active)" if self.active_skill and self.active_skill.name == s.name else ""
                lines.append(f"  - {s.name}: {s.description}{active}")
            lines.append("")
            lines.append("Usage: /skill <name> | /skill create <name> | /skill delete <name> | /skill list")
            return "", {"type": "info", "output": "\n".join(lines)}

        tokens = rest.split(None, 1)
        sub = tokens[0].lower()
        arg = tokens[1] if len(tokens) > 1 else ""

        if sub == "list":
            return self.handle_slash("/skill")

        if sub == "unload":
            self.unload()
            return "", {"type": "info", "output": "Unloaded active skill."}

        if sub == "create":
            return "", self._cmd_create(arg)

        if sub == "delete":
            if not arg:
                return "", {"type": "error", "output": "Please provide a skill name to delete."}
            ok = self.delete(arg)
            if ok:
                return "", {"type": "info", "output": f"Deleted skill '{arg}'."}
            return "", {"type": "error", "output": f"Could not delete '{arg}' (not found or built-in)."}

        if sub == "search":
            if not arg:
                return "", {"type": "error", "output": "Please provide a search query."}
            results = self.suggest(arg, top_k=5)
            lines = [f"Skills matching '{arg}':", ""]
            for skill, score in results:
                lines.append(f"  - {skill.name} (score {score:.2f}): {skill.description}")
            return "", {"type": "info", "output": "\n".join(lines)}

        # Default: treat sub as a skill name to load.
        skill = self.load(sub)
        if skill:
            return arg, {"type": "skill_loaded", "skill": skill.name, "output": f"Activated skill '{skill.name}'. {skill.description}"}

        # Fuzzy match
        results = self.suggest(sub, top_k=1)
        if results:
            best, score = results[0]
            if score >= 0.5:
                self.load(best.name)
                return arg, {"type": "skill_loaded", "skill": best.name, "output": f"Activated skill '{best.name}' (fuzzy match). {best.description}"}

        return "", {"type": "error", "output": f"Unknown skill or command: '{sub}'. Use /skill list to see available skills."}

    def _cmd_create(self, arg: str) -> Dict[str, Any]:
        """Handle `/skill create <name> [description]`."""
        parts = arg.split(None, 1)
        if not parts:
            return {"type": "error", "output": "Usage: /skill create <name> [description]"}
        name = parts[0]
        description = parts[1] if len(parts) > 1 else ""
        if self.bank.get(name):
            return {"type": "error", "output": f"Skill '{name}' already exists."}

        # If we have an LLM helper, auto-generate a reasonable template.
        prompt_template = (
            "You are an expert assistant focused on the following task:\n"
            "{{task}}\n\n"
            "Provide a thorough, accurate, and concise response."
        )
        if self.llm_call and description:
            try:
                gen_prompt = (
                    f"Write a short system-prompt template for a skill named '{name}' described as: {description}.\n"
                    "Use {{task}} as the placeholder for the user's request. "
                    "Return ONLY the prompt template text."
                )
                generated = self.llm_call(gen_prompt, temperature=0.3, max_tokens=512)
                if generated and "{{task}}" in generated:
                    prompt_template = generated.strip()
            except Exception as e:
                self.logger.debug(f"auto-generate skill prompt failed: {e}")

        skill = self.create(
            name=name,
            description=description or f"User-created skill: {name}",
            prompt_template=prompt_template,
            trigger_keywords=[name.lower()],
            tags=["user"],
        )
        return {"type": "info", "output": f"Created skill '{skill.name}'. Use `/skill {skill.name}` to activate it."}


# =============================================================================
# 4. Helpers
# =============================================================================

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def format_skills_for_prompt(skills: List[Skill], max_chars: int = 2000) -> str:
    """Format a list of skills as a compact system-prompt appendix."""
    lines = ["## Available Skills"]
    total = 0
    for skill in skills:
        entry = f"- /skill {skill.name}: {skill.description}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry) + 1
    return "\n".join(lines)
