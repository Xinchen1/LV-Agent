"""File-level memory: human-readable MEMORY.md and USER.md storage.

This layer keeps long-term facts in plain Markdown so they are diffable,
searchable, and editable by the user outside the agent.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FileMemoryManager:
    """Manage MEMORY.md (project memory) and USER.md (user preferences)."""

    def __init__(
        self,
        memory_path: str = "./data/memory.md",
        user_path: str = "./data/user.md",
        project_root: Optional[str] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.memory_path = self._resolve_path(memory_path)
        self.user_path = self._resolve_path(user_path)
        self._ensure_files()

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return p.expanduser().resolve()

    def _ensure_files(self) -> None:
        for p in (self.memory_path, self.user_path):
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text(
                        f"# {p.stem.upper()}\n\n"
                        f"<!-- Auto-managed by Lv Super Agent -->\n"
                        f"<!-- Last updated: {datetime.now().isoformat()} -->\n\n",
                        encoding="utf-8",
                    )
            except Exception as e:
                logger.warning(f"Could not ensure file memory {p}: {e}")

    def read(self, topic: str = "", source: str = "both") -> str:
        """Read memory files, optionally filtering by topic heading.

        Args:
            topic: Optional heading/title to locate (case-insensitive).
            source: "memory", "user", or "both".
        """
        files = self._select_files(source)
        parts = []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read {p}: {e}")
                continue
            if topic:
                section = self._extract_section(text, topic)
                if section:
                    parts.append(f"--- {p.name} ---\n{section}")
            else:
                parts.append(f"--- {p.name} ---\n{text}")
        return "\n\n".join(parts)

    def write(
        self,
        topic: str,
        content: str,
        source: str = "memory",
        append: bool = True,
    ) -> bool:
        """Write or append a topic section to the chosen memory file."""
        p = self.user_path if source == "user" else self.memory_path
        try:
            self._ensure_files()
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            new_section = self._format_section(topic, content)

            if append and topic:
                existing = self._extract_section(text, topic)
                if existing:
                    text = self._replace_section(text, topic, existing + "\n\n" + content)
                else:
                    text = text.rstrip() + "\n\n" + new_section + "\n"
            else:
                text = text.rstrip() + "\n\n" + new_section + "\n"

            p.write_text(text, encoding="utf-8")
            return True
        except Exception as e:
            logger.warning(f"Could not write file memory {p}: {e}")
            return False

    def search(self, keyword: str, source: str = "both", k: int = 10) -> List[Tuple[str, str]]:
        """Keyword search across memory files. Returns list of (file_name, snippet)."""
        files = self._select_files(source)
        results = []
        kw_lower = keyword.lower()
        for p in files:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            for para in text.split("\n\n"):
                if kw_lower in para.lower():
                    results.append((p.name, para.strip()))
                    if len(results) >= k:
                        return results
        return results

    def _select_files(self, source: str) -> List[Path]:
        if source == "memory":
            return [self.memory_path]
        if source == "user":
            return [self.user_path]
        return [self.memory_path, self.user_path]

    @staticmethod
    def _format_section(topic: str, content: str) -> str:
        timestamp = datetime.now().isoformat()
        return f"## {topic}\n<!-- {timestamp} -->\n{content.strip()}"

    @staticmethod
    def _extract_section(text: str, topic: str) -> Optional[str]:
        pattern = re.compile(rf"^##\s+{re.escape(topic)}\s*$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(text)
        if not match:
            return None
        start = match.end()
        next_heading = re.search(r"^##\s+", text[start:], re.MULTILINE)
        end = start + next_heading.start() if next_heading else len(text)
        return text[start:end].strip()

    @staticmethod
    def _replace_section(text: str, topic: str, new_body: str) -> str:
        pattern = re.compile(rf"^##\s+{re.escape(topic)}\s*$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(text)
        if not match:
            return text
        start = match.start()
        next_heading = re.search(r"^##\s+", text[match.end():], re.MULTILINE)
        end = match.end() + next_heading.start() if next_heading else len(text)
        return text[:start] + f"## {topic}\n<!-- {datetime.now().isoformat()} -->\n{new_body.strip()}\n" + text[end:]


def create_file_memory(
    memory_path: str = "./data/memory.md",
    user_path: str = "./data/user.md",
    project_root: Optional[str] = None,
) -> FileMemoryManager:
    return FileMemoryManager(memory_path=memory_path, user_path=user_path, project_root=project_root)
