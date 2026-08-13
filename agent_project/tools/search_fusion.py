"""
Multi-provider search result fusion.

Pipeline:
  1. Resolve every URL to its real landing URL.
  2. Deduplicate by canonical URL (ignore scheme/www).
  3. Score each result.
  4. Re-rank by score, keeping complementary sources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .search_scorer import ScoredResult, SearchScorer
from .url_unshorten import URLUnshortener


class SearchFusion:
    """Fuse results from multiple search providers into one ranked list."""

    def __init__(
        self,
        domain_trust: Optional[Dict[str, float]] = None,
        quality_threshold: float = 0.3,
    ):
        self.unshortener = URLUnshortener()
        self.scorer = SearchScorer(domain_trust=domain_trust, quality_threshold=quality_threshold)
        self.quality_threshold = quality_threshold

    def fuse(
        self,
        provider_results: Dict[str, List[Dict[str, Any]]],
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            provider_results: {provider_name: [result_dict, ...]}
            query: original search query
            max_results: number of results to return

        Returns:
            List of result dicts with real URLs and quality scores.
        """
        all_results: List[Dict[str, Any]] = []
        for provider, results in provider_results.items():
            for r in results:
                r = dict(r)
                r["_provider"] = provider
                all_results.append(r)

        # Resolve URLs
        urls_to_resolve = [r.get("url", "") for r in all_results if r.get("url")]
        resolved = self.unshortener.resolve_many(urls_to_resolve)
        for r in all_results:
            original = r.get("url", "")
            r["url"] = resolved.get(original, original)
            r["_original_url"] = original

        # Score
        self.scorer.query = query.lower()
        self.scorer.query_terms = self.scorer._tokenize(self.scorer.query)
        scored = self.scorer.filter_and_sort(all_results, min_score=self.quality_threshold)

        # Deduplicate by canonical URL, keeping best score per canonical URL
        seen: Dict[str, ScoredResult] = {}
        for s in scored:
            canonical = self._canonical_url(s.url)
            if canonical not in seen or seen[canonical].score < s.score:
                seen[canonical] = s

        unique = sorted(seen.values(), key=lambda x: x.score, reverse=True)

        # Promote source diversity: if we have many results, don't let one domain dominate
        diversified = self._diversify(unique, max_results)

        return [
            {
                "title": s.title,
                "snippet": s.snippet,
                "url": s.url,
                "score": round(s.score, 3),
                "score_factors": s.factors,
            }
            for s in diversified[:max_results]
        ]

    @staticmethod
    def _canonical_url(url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            path = parsed.path.rstrip("/")
            return f"{netloc}{path}"
        except Exception:
            return url.lower().rstrip("/")

    def _diversify(self, scored: List[ScoredResult], max_results: int) -> List[ScoredResult]:
        """Keep top results but cap same-domain results so complementary sources appear."""
        domain_counts: Dict[str, int] = {}
        max_per_domain = max(2, max_results // 3)
        out = []
        for s in scored:
            try:
                domain = urlparse(s.url).netloc.lower()
            except Exception:
                domain = ""
            if not domain:
                out.append(s)
                continue
            count = domain_counts.get(domain, 0)
            if count < max_per_domain:
                out.append(s)
                domain_counts[domain] = count + 1
            if len(out) >= max_results * 2:
                break
        return out[:max_results]
