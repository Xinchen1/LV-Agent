"""MemoryProvider ABC + 薄适配器(对标 Hermes memory_provider ABC).

现状三套存储形状各异, 强行统一接口是削足适履:
- FileMemoryManager : read/write/search (markdown 文件)
- SQLiteSessionStore: 会话轮次 (按 session 检索)
- LLMWikiManager    : remember(text)/search (实体图谱)

本模块只定最小契约 remember/recall + 给 file/wiki 写适配器,
调用方从此只认 MemoryProvider, 不直连具体类. sqlite 会话流暂不纳入
(语义不同: 按轮次而非按主题), 后续需要再加 SessionAdapter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """长期记忆最小契约: 存文本, 按查询取回文本列表."""

    @abstractmethod
    def remember(self, text: str, context: str = "", **kwargs: Any) -> List[str]:
        """存入并返回 id/页标识列表(至少非空异常时抛, 不静默)."""

    @abstractmethod
    def recall(self, query: str, k: int = 5) -> List[str]:
        """取回相关文本片段(无命中返回空列表, 不抛)."""


class WikiMemoryAdapter(MemoryProvider):
    """LLMWikiManager 适配器(原生 remember/search, 只做形状归一)."""

    def __init__(self, wiki: Any, auto_link: bool = True):
        self.wiki = wiki
        self.auto_link = auto_link

    def remember(self, text: str, context: str = "", **kwargs: Any) -> List[str]:
        ids = self.wiki.remember(text, context=context or None,
                                 auto_link=kwargs.get("auto_link", self.auto_link))
        return [str(i) for i in (ids or [])]

    def recall(self, query: str, k: int = 5) -> List[str]:
        try:
            hits = self.wiki.search(query, k=k) or []
        except Exception:
            return []
        out = []
        for h in hits:
            if isinstance(h, dict):
                out.append(h.get("content") or h.get("text") or h.get("title") or "")
            else:
                out.append(str(h))
        return [t for t in out if t]


class FileMemoryAdapter(MemoryProvider):
    """FileMemoryManager 适配器: remember=追加写入, recall=关键词检索."""

    def __init__(self, store: Any):
        self.store = store

    def remember(self, text: str, context: str = "", **kwargs: Any) -> List[str]:
        topic = context.strip()[:60] or "general"
        ok = self.store.write(topic=topic, content=text,
                              source=kwargs.get("source", "memory"), append=True)
        if not ok:
            raise IOError(f"file memory write failed: topic={topic!r}")
        return [topic]

    def recall(self, query: str, k: int = 5) -> List[str]:
        try:
            hits = self.store.search(query, k=k) or []
        except Exception:
            return []
        return [f"{t}: {s}" for t, s in hits[:k]]


def fanout_recall(providers: List[MemoryProvider], query: str,
                  k: int = 5) -> List[str]:
    """多 provider 扇出召回并去重(保序). 单个失败不影响其他."""
    seen, out = set(), []
    for p in providers:
        try:
            hits = p.recall(query, k=k) or []
        except Exception as e:
            logger.debug("fanout recall 跳过故障 provider %s: %s",
                         type(p).__name__, e)
            continue
        for h in hits:
            if h and h not in seen:
                seen.add(h)
                out.append(h)
            if len(out) >= k:
                return out
    return out
