"""
Search result quality scoring.

Scores each result on:
- Information richness (title + snippet length)
- Domain trust (configured domain whitelist)
- Query relevance (keyword overlap)
- URL authenticity (real URL vs redirect/ad placeholder)
- Freshness signals in title/snippet
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class ScoredResult:
    title: str
    snippet: str
    url: str
    score: float
    factors: Dict[str, float]


class SearchScorer:
    """Score and rank individual search results."""

    def __init__(
        self,
        domain_trust: Optional[Dict[str, float]] = None,
        query: str = "",
        quality_threshold: float = 0.3,
    ):
        self.domain_trust = domain_trust or {}
        self.query = query.lower()
        self.quality_threshold = quality_threshold
        self.query_terms = self._tokenize(self.query)

    def score(self, result: Dict[str, Any]) -> ScoredResult:
        title = (result.get("title") or "").strip()
        snippet = (result.get("snippet") or "").strip()
        url = (result.get("url") or "").strip()

        factors = {
            "richness": self._richness_score(title, snippet),
            "domain": self._domain_score(url),
            "relevance": self._relevance_score(title, snippet),
            "authenticity": self._authenticity_score(url, title, snippet),
            "freshness": self._freshness_score(title, snippet),
        }

        # Weighted combination
        weights = {
            "richness": 0.25,
            "domain": 0.20,
            "relevance": 0.25,
            "authenticity": 0.20,
            "freshness": 0.10,
        }
        score = sum(factors[k] * weights[k] for k in factors)
        return ScoredResult(title=title, snippet=snippet, url=url, score=score, factors=factors)

    def filter_and_sort(
        self,
        results: List[Dict[str, Any]],
        min_score: Optional[float] = None,
    ) -> List[ScoredResult]:
        min_score = min_score if min_score is not None else self.quality_threshold
        scored = [self.score(r) for r in results]
        scored = [s for s in scored if s.score >= min_score]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def _richness_score(self, title: str, snippet: str) -> float:
        title_len = len(title)
        snippet_len = len(snippet)
        if title_len < 5 and snippet_len < 20:
            return 0.0
        score = 0.0
        if title_len >= 8:
            score += 0.3
        if snippet_len >= 40:
            score += 0.4
        if snippet_len >= 120:
            score += 0.3
        return min(score, 1.0)

    def _domain_score(self, url: str) -> float:
        if not url:
            return 0.0
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            return 0.0
        if not netloc:
            return 0.0
        # Exact match
        if netloc in self.domain_trust:
            return self.domain_trust[netloc]
        # Suffix match
        for domain, trust in sorted(self.domain_trust.items(), key=lambda x: -len(x[0])):
            if domain in netloc or netloc.endswith("." + domain.lstrip(".")):
                return trust
        return 0.5

    def _relevance_score(self, title: str, snippet: str) -> float:
        if not self.query_terms:
            return 0.7
        text = f"{title} {snippet}".lower()
        text_terms = set(self._tokenize(text))
        matches = sum(1 for t in self.query_terms if t in text_terms)
        ratio = matches / max(1, len(self.query_terms))

        # 中文查询: 整词(相邻中文串)命中比单字命中权重更高
        # 例: 查询 "nous research 融资 商业模式" → 结果同时含 "融资"+"商业模式" 优于只含单个字
        cjk_words = re.findall(r"[\u4e00-\u9fff]{2,}", self.query)
        if cjk_words:
            text_flat = re.sub(r"\s+", "", text)
            word_hits = sum(1 for w in cjk_words if w in text_flat)
            # 单词命中与单字命中各占一半权重, 避免"盈/利"等单字噪声刷分
            ratio = 0.5 * ratio + 0.5 * (word_hits / max(1, len(cjk_words)))

        # 英文专有实体(如 nous research / claude code)必须是强相关信号:
        # 中文结果若不含这些实体, 即使匹配到"盈利/2025"也不该拿高分
        en_entities = re.findall(r"[a-z][a-z0-9]+(?:\s+[a-z][a-z0-9]+){0,2}", self.query)
        en_entities = [e for e in en_entities if len(e) >= 4]
        if en_entities:
            text_low = text
            entity_hits = sum(1 for e in en_entities if e in text_low)
            if entity_hits == 0:
                # 核心英文实体完全未命中 → 强降权
                ratio *= 0.25
            else:
                ratio = 0.6 * ratio + 0.4 * (entity_hits / max(1, len(en_entities)))

        return min(1.0, ratio)

    def _authenticity_score(self, url: str, title: str, snippet: str) -> float:
        score = 0.5
        if not url:
            score -= 0.3
        else:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                score += 0.3
            # Penalize known redirect/aggregator domains
            redirect_domains = ("so.com/link", "news.so.com", "jump.", "redirect.", "click.link")
            if any(d in url for d in redirect_domains):
                score -= 0.4
        # Penalize placeholder titles
        placeholder = ("no results", "empty result", "没有找到", "抱歉")
        if any(p in (title + snippet).lower() for p in placeholder):
            score -= 0.4
        return max(0.0, min(1.0, score))

    def _freshness_score(self, title: str, snippet: str) -> float:
        text = (title + " " + snippet).lower()
        fresh_markers = [
            "2024", "2025", "2026", "最新", "最近", "now", "today",
            "刚刚", "今日", "今年", "new", "updated",
        ]
        return 0.3 + 0.7 * min(1.0, sum(1 for m in fresh_markers if m in text) / 2.0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Simple CJK / latin tokenization
        # 中文: 除了单字, 也切出 2-4 字滑动窗口词组, 便于匹配"融资/商业模式"这类完整词
        tokens = re.findall(r"[a-z0-9]+|\u4e00-\u9fff", text.lower())
        # 中文片段里再切滑动双字词
        cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text.lower())
        for run in cjk_runs:
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]
