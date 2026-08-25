"""Web Search Tool - multi-provider search with quality enhancement.

Pipeline:
  query → cache check → concurrent provider calls → URL unshortening
  → quality scoring → cross-provider fusion → snippet enrichment
  → cache write → ranked results.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

# Optional async HTTP
try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

# Optional fast parser
try:
    from bs4 import BeautifulSoup  # noqa: F401
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    import lxml  # noqa: F401
    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False
from . import BaseTool, ToolResult, TOOLS_REGISTRY
from .search_cache import SearchCache
from .search_fusion import SearchFusion
from .search_scorer import SearchScorer
from .url_unshorten import URLUnshortener
from .web_fetcher import WebContentFetcher


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current information. "
        "Use this when you need recent facts, docs, or sources."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Be specific and concise.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 10).",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        self.cfg = config or {}

        # Resolve effective provider list - default to only DuckDuckGo for speed
        self.providers = self.cfg.get("providers", ["duckduckgo", "bing", "google"])

        # News-specific providers (auto-enabled when query contains news keywords)
        self.news_providers = self.cfg.get("news_providers", ["bing-news", "google-news"])

        self.max_results_default = self.cfg.get("max_results", 3)
        self.quality_threshold = self.cfg.get("quality_threshold", 0.6)
        self.use_playwright = self.cfg.get("use_playwright", False)
        self.sequential_fallback = self.cfg.get("sequential_fallback", False)
        self.max_fetch_urls = self.cfg.get("max_fetch_urls", 0)
        self.cache_ttl = self.cfg.get("cache_ttl", 300)
        self.domain_trust = self.cfg.get("domain_trust", {})

        # 共享 Session + 大连接池: 并发搜索时避免 urllib3 全局池满丢连接
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50, pool_maxsize=50, max_retries=0,
            pool_block=False,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self.cache = SearchCache(ttl=self.cache_ttl, disk_path="./data/search_cache")
        self.unshortener = URLUnshortener()
        self.scorer = SearchScorer(domain_trust=self.domain_trust, quality_threshold=self.quality_threshold)
        self.fusion = SearchFusion(domain_trust=self.domain_trust, quality_threshold=self.quality_threshold)
        self.fetcher = WebContentFetcher(use_playwright=self.use_playwright)
        self._recent_sigs = []

        # 内存级缓存：热查询零网络/磁盘/解析
        self._mem_cache = {}

    @staticmethod
    def _detect_search_need(query: str) -> Dict[str, Any]:
        """意图感知: 分析 query 特征, 返回搜索策略调整建议.

        Returns:
            {"lang": "zh"|"en"|"mixed", "type": "news"|"tech"|"product"|"general",
             "freshness": bool(是否要最新信息), "domain_hint": 可选权威域提示}
        """
        q = (query or "").lower()
        has_zh = any('\u4e00' <= ch <= '\u9fff' for ch in q)
        has_en = bool(re.search(r'[a-zA-Z]{2,}', q))
        lang = "mixed" if has_zh and has_en else ("zh" if has_zh else "en")

        # 内容类型判断
        qtype = "general"
        if any(k in q for k in ("新闻", "最新", "资讯", "breaking", "news", "今天", "今日")):
            qtype = "news"
        elif any(k in q for k in ("技术", "原理", "架构", "实现", "tech", "how", "api", "代码", "开源")):
            qtype = "tech"
        elif any(k in q for k in ("产品", "价格", "购买", "评测", "product", "price", "buy", "review")):
            qtype = "product"

        # 时效性: 请求最新/今天/2025等 → 需要新鲜结果
        freshness = any(k in q for k in ("最新", "今天", "今日", "今年", "近期", "最近", "20", "breaking", "news", "now"))

        # 权威域提示: 技术/文档类优先官方与知名源
        domain_hint = ""
        if qtype == "tech" and any(k in q for k in ("github", "文档", "doc", "官方", "官网", "api")):
            domain_hint = "github.com,stackoverflow.com,developer.*"

        return {"lang": lang, "type": qtype, "freshness": freshness, "domain_hint": domain_hint}

    @staticmethod
    def _smart_query(query: str) -> str:
        """智能 Query 优化: 在发送前清理/增强搜索词, 提高结果相关性.

        - 去噪: 去掉动作词残留/标点/无意义后缀
        - 中英混合修正: "Nous Research 的融资" → "Nous Research 融资"
        - 时间补全: 含"最新/今年"且无年份时补当前年份(增强时效性)
        """
        if not query:
            return query
        q = query.strip()

        # 1) 去噪: 去掉常见引导词/无意义后缀(覆盖"X一下"组合)
        q = re.sub(r'^(请|帮我|麻烦)?\s*(搜索一下|查找一下|查一下|查下|搜一下|研究一下|调研一下|了解一下|搜索|查找|查|搜|找一下|找)\s*[:：]?\s*', '', q)
        q = re.sub(r'的(最新|最新情况|进展|新闻|情况|信息|资料|内容)$', r' \1', q)
        # 中文"的"去噪: "Nous Research 的融资" → "Nous Research 融资"
        q = re.sub(r'([A-Za-z\u4e00-\u9fff])\s*的\s*', r'\1 ', q)

        # 2) 时间补全: 请求含"最新/今年/现在"但无年份 → 补当前年份
        import datetime
        this_year = str(datetime.datetime.now().year)
        # 同时检测完整日期格式 2026-08-23 / 2026/08/23 / 2026.08.23
        has_full_date = bool(re.search(r'20\d{2}[-/.]\d{2}[-/.]\d{2}', q))
        if not re.search(r'(20\d{2})', q) and not has_full_date:
            if re.search(r'(最新|今年|现在|近期|最近|当下)', q):
                q = f"{q} {this_year}"

        # 3) 压缩多余空白
        q = re.sub(r'\s{2,}', ' ', q).strip()
        return q[:120]

    def execute(
        self,
        query: str,
        max_results: int = 5,
        deep_max_results: Optional[int] = None,
    ) -> ToolResult:
        """Run the full search-enhancement pipeline.

        Args:
            query: search query.
            max_results: normal mode maximum results (capped at 10 for the public API).
            deep_max_results: if provided, enables deep-research mode with a much
                higher result cap and more concurrent providers.
        """
        try:
            is_deep = bool(deep_max_results and int(deep_max_results) > 10)
            # 0. 重复查询去重
            sig = f"{query}|{max_results}"
            if sig in self._recent_sigs:
                return ToolResult(success=True, output="[]", metadata={"cached": True, "dedup": True})
            self._recent_sigs.append(sig)
            if len(self._recent_sigs) > 20:
                self._recent_sigs.pop(0)
            # 0. 模型先行分析: 只在查询较复杂且不含日期时触发，避免额外延迟
            # 检测是否包含日期格式: 2026-08-23 / 2026/08/23 / 2026.08.23 / 2026 / 2025 等
            has_date = bool(re.search(r'20\d{2}[-/.]\d{2}[-/.]\d{2}|20\d{2}', query))
            if len(query) > 20 and not has_date and any(k in query for k in ['最新','新闻','产品','价格']):
                try:
                    from .model_backends import get_backend
                    backend = get_backend()
                    analysis_prompt = f"""用户原始查询：{query}
请抽取核心实体和意图，输出一条准确、可直接用于搜索引擎的查询关键词，5-15字。只输出关键词，不要解释。"""
                    refined = backend.generate(analysis_prompt, max_tokens=64, temperature=0.0).strip()
                    if refined and len(refined) > 2 and refined != query:
                        query = refined
                except Exception:
                    # 熔断：模型失败不影响搜索
                    pass
            # 智能 Query 优化(清理/增强), deep 模式保留原始深度词但同样优化
            if query:
                smart = self._smart_query(query)
                if smart and smart != query:
                    query = smart
            if is_deep:
                effective_max_results = max(1, int(deep_max_results))
                max_workers = min(len(self.providers), 8)
                fusion_max = max(effective_max_results * 2, 8)
                fetch_max = max(self.max_fetch_urls, 20)
            else:
                effective_max_results = max(1, min(int(max_results), 10))
                max_workers = min(len(self.providers), 4)
                fusion_max = max(effective_max_results * 2, 8)
                fetch_max = self.max_fetch_urls

            # Memory cache key
            mem_key = (query, max_results, deep_max_results)
            if mem_key in self._mem_cache:
                return self._mem_cache[mem_key]

            # Disk cache check
            cached = self.cache.get(query, threshold=0.5)
            if cached:
                # Determine effective_max_results for slicing
                if is_deep:
                    effective_max_results = max(1, int(deep_max_results))
                else:
                    effective_max_results = max(1, min(int(max_results), 10))
                result = ToolResult(
                    success=True,
                    output=json.dumps(cached[:effective_max_results], indent=2, ensure_ascii=False),
                    metadata={
                        "query": query,
                        "provider": "cache",
                        "num_results": len(cached[:effective_max_results]),
                        "cached": True,
                    },
                )
                self._mem_cache[mem_key] = result
                return result

            # Live search via _search_and_fuse
            result = self._search_and_fuse(query, max_results, deep_max_results)
            self._mem_cache[mem_key] = result
            # Simple LRU eviction
            if len(self._mem_cache) > 100:
                oldest = next(iter(self._mem_cache))
                del self._mem_cache[oldest]
            return result
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _search_and_fuse(self, query: str, max_results: int, deep_max_results: Optional[int] = None) -> ToolResult:
        """Core search pipeline without cache checks."""
        is_deep = bool(deep_max_results and int(deep_max_results) > 10)
        # Compute parameters
        if is_deep:
            effective_max_results = max(1, int(deep_max_results))
            max_workers = min(len(self.providers), 8)
            fusion_max = max(effective_max_results * 2, 8)
            fetch_max = max(self.max_fetch_urls, 20)
        else:
            effective_max_results = max(1, min(int(max_results), 10))
            max_workers = min(len(self.providers), 4)
            fusion_max = max(effective_max_results * 2, 8)
            fetch_max = self.max_fetch_urls

        # Intent detection
        need = self._detect_search_need(query)
        effective_providers = list(self.providers)
        if need["lang"] == "zh" and "360" in effective_providers:
            effective_providers = ["360"] + [p for p in effective_providers if p != "360"]

        # Auto-enable news providers for news-like queries
        is_news_query = any(k in query.lower() for k in ['新闻', '最新', 'news', 'latest', '今日', '今日头条', 'breaking', '头条'])
        if is_news_query:
            for np in self.news_providers:
                if np not in effective_providers:
                    effective_providers.append(np)

        errors: List[str] = []
        # Use async collection if aiohttp is available, else fall back to sync
        if _HAS_AIOHTTP:
            provider_results, errors = self._run_async(
                self._async_collect_provider_results(
                    query, effective_max_results, max_workers=max_workers,
                    providers=effective_providers, is_deep=is_deep
                )
            )
        else:
            provider_results, errors = self._collect_provider_results(
                query, effective_max_results, max_workers=max_workers, providers=effective_providers
            )

        # Fallback
        if not provider_results:
            relaxed = self._relax_query(query)
            if relaxed and relaxed != query:
                if _HAS_AIOHTTP:
                    extra_results, extra_errors = self._run_async(
                        self._async_collect_provider_results(
                            relaxed, effective_max_results, max_workers=max_workers,
                            is_deep=is_deep
                        )
                    )
                else:
                    extra_results, extra_errors = self._collect_provider_results(
                        relaxed, effective_max_results, max_workers=max_workers
                    )
                provider_results.update(extra_results)
                errors.extend(extra_errors)

        if not provider_results:
            return ToolResult(
                success=False,
                output="",
                error="; ".join(errors) if errors else "No search results returned",
            )

        # Fuse
        fused = self.fusion.fuse(provider_results, query, max_results=fusion_max)

        # Domain priority
        if fused and need.get("domain_hint"):
            pref_domains = [d.strip() for d in need["domain_hint"].split(",")]
            def _domain_of(url: str) -> str:
                from urllib.parse import urlparse
                try:
                    return (urlparse(url).netloc or "").lower()
                except Exception:
                    return ""
            fused.sort(key=lambda r: not any(
                pref in _domain_of(r.get("url", "")) or pref.rstrip("*.") in r.get("url", "")
                for pref in pref_domains
            ))

        # Enrich snippets
        if fused:
            avg_snippet_len = sum(len((r.get("snippet") or "").strip()) for r in fused) / len(fused)
            weak_snippets = avg_snippet_len < 80
            has_real_urls = any(self._looks_like_url(r.get("url", "")) for r in fused[:3])
            if weak_snippets or has_real_urls:
                self.fetcher.enrich_results(fused, max_urls=fetch_max)

        # Entity filter
        if fused and ('ai' in query.lower() or '人工智能' in query):
            keywords = ['ai','人工智能','人工智','机器人','大模型','模型','ChatGPT','GPT','LLM']
            fused = [r for r in fused if any(k in (r.get('title') or '').lower() or k in (r.get('snippet') or '') for k in keywords)]

        # Time-word filter for news queries: require time markers in title/snippet
        if fused and is_news_query:
            time_keywords = ['2026','2025','2024','今日','今日头条','昨日','刚刚','刚发','刚刚发布','刚刚更新',
                            '小时前','分钟前','分钟前发布','hours ago','minutes ago','mins ago','hours前','mins前',
                            'breaking','breaking news','breaking:', '速报','快讯','速递','实时','实时更新',
                            '刚刚发布','最新消息','最新报道','最新进展']
            fused = [r for r in fused if any(
                tk in ((r.get('title') or '') + (r.get('snippet') or '')).lower() 
                for tk in time_keywords
            )]

        final = fused[:effective_max_results]

        # Disk cache
        if final:
            self.cache.set(query, final)

        return ToolResult(
            success=True,
            output=json.dumps(final, indent=2, ensure_ascii=False),
            metadata={
                "query": query,
                "providers": list(provider_results.keys()),
                "num_results": len(final),
                "cached": False,
                "deep_mode": is_deep,
                "errors": errors if errors else None,
            },
        )

    # ------------------------------------------------------------------
    # Provider collection
    # ------------------------------------------------------------------

    def _collect_provider_results(
        self,
        query: str,
        max_results: int,
        max_workers: int = 4,
        providers: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        """Call all configured providers and return {provider: results} plus errors."""
        provider_results: Dict[str, List[Dict[str, Any]]] = {}
        errors: List[str] = []
        provider_list = providers or self.providers

        if self.sequential_fallback:
            for provider in provider_list:
                results, error = self._call_provider(provider, query, max_results)
                if error:
                    errors.append(error)
                if self._is_real_results(results):
                    provider_results[provider] = results
                    break
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {
                    executor.submit(self._call_provider, provider, query, max_results): provider
                    for provider in provider_list
                }
                # Wait for futures as they complete, with a total timeout of 8 seconds.
                # Stop early once we have at least one real result.
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=8):
                        provider = futures[future]
                        try:
                            results, error = future.result()
                            if error:
                                errors.append(f"{provider}: {error}")
                            if self._is_real_results(results):
                                provider_results[provider] = results
                                # Early exit: one good provider is enough for normal mode
                                break
                        except Exception as e:
                            errors.append(f"{provider}: {e}")
                except concurrent.futures.TimeoutError:
                    pass
                # Cancel any unfinished futures to free threads quickly
                for future in futures:
                    if not future.done():
                        future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        return provider_results, errors

    def _call_provider(
        self,
        provider: str,
        query: str,
        max_results: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Call a single provider and return (results, error_message)."""
        try:
            if provider == "duckduckgo":
                return self._search_duckduckgo(query, max_results), None
            if provider == "360":
                return self._search_360(query, max_results), None
            if provider == "bing":
                return self._search_bing(query, max_results), None
            if provider == "google":
                return self._search_google(query, max_results), None
            if provider == "serpapi":
                if not self.api_key:
                    return [], "SerpAPI key not configured"
                return self._search_serpapi(query, max_results), None
            return [], f"Unknown provider: {provider}"
        except Exception as e:
            return [], str(e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_real_results(results: list) -> bool:
        """Return True if the provider returned at least one genuine result."""
        if not results:
            return False
        for r in results:
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            if not title and not snippet:
                continue
            if title.lower().startswith("no results for query"):
                continue
            placeholder_prefixes = (
                "duckduckgo returned an empty",
                "360 returned an empty",
                "bing returned an empty",
                "google returned an empty",
            )
            if snippet.lower().startswith(placeholder_prefixes):
                continue
            return True
        return False

    @staticmethod
    def _looks_like_url(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        return s.startswith("http://") or s.startswith("https://") or (
            "." in s and not s.startswith("/") and " " not in s
        )

    @staticmethod
    def _relax_query(query: str) -> str:
        """Strip quotes, brackets, and excessive punctuation for a broader search."""
        relaxed = re.sub(r'["\']', " ", query)
        relaxed = re.sub(r"[\(\)\[\]\{\}<>]", " ", relaxed)
        relaxed = re.sub(r"\s{2,}", " ", relaxed).strip()
        return relaxed

    # ------------------------------------------------------------------
    # Provider implementations (parsing kept from previous version)
    # ------------------------------------------------------------------

    def _get_bs(self):
        """Return a BeautifulSoup constructor using lxml if available.

        Falls back to the stdlib html.parser at call time if lxml cannot be
        used (e.g. not installed in the active environment), so search never
        breaks on a missing optional dependency.
        """
        if _HAS_LXML:
            def _bs(html):
                try:
                    return BeautifulSoup(html, "lxml")
                except Exception:
                    return BeautifulSoup(html, "html.parser")
            return _bs
        return lambda html: BeautifulSoup(html, "html.parser")

    def _search_duckduckgo(self, query: str, max_results: int) -> List[dict]:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            "Referer": "https://duckduckgo.com/",
        }
        last_exception = None

        for attempt in range(2):
            try:
                response = self._session.post(
                    url,
                    data={"q": query},
                    headers=headers,
                    timeout=4,
                )
                response.raise_for_status()
                return self._parse_duckduckgo(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.2 * (attempt + 1))

        raise last_exception or RuntimeError("DuckDuckGo search failed")

    def _parse_duckduckgo(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html)
        results: List[dict] = []
        for result in soup.select(".result"):
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            url_el = result.select_one(".result__url")

            title = title_el.get_text(strip=True) if title_el else ""
            url = url_el.get("href", "") if url_el else (
                title_el.get("href", "") if title_el else ""
            )
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            if not title and not snippet:
                continue

            results.append(
                {
                    "title": title[:160],
                    "snippet": snippet[:240],
                    "url": url,
                }
            )

        if not results:
            return [
                {
                    "title": "No results for query",
                    "snippet": "DuckDuckGo returned an empty result page.",
                    "url": "https://duckduckgo.com",
                }
            ]

        return results[: max(max_results, 1)]

    def _search_360(self, query: str, max_results: int) -> List[dict]:
        """360搜索（so.com）作为DuckDuckGo的国内fallback。"""
        url = "https://www.so.com/s"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        last_exception = None

        for attempt in range(2):
            try:
                response = self._session.get(
                    url,
                    params={"q": query},
                    headers=headers,
                    timeout=5,
                )
                response.raise_for_status()
                return self._parse_360(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.8 * (attempt + 1))

        raise last_exception or RuntimeError("360 search failed")

    def _parse_360(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html)
        results: List[dict] = []

        # 新版360(so.com)结构:
        #  - 顶层容器: li.res-list (多个结果集)
        #  - 每个结果集里含:
        #     * h3.g-title → 主标题
        #     * div.g-card-layout / div.g-figure-layout-h → 新闻条目
        #     * a.mh-news-title → 单条新闻标题
        #     * p.mh-news-desc  → 单条新闻摘要
        #  老版结构兼容: h3 a + .res-desc / .summary

        # A) 优先:单个新闻条目级容器(保证 1 条 = 1 个标题 + 1 个对应的摘要,不会跨条目错配)
        single_item_selectors = [
            'div.mh-news-no-image',
            'div.mh-news-has-image',
            'div.mh-news-item',
            'li.mh-news-item',
        ]
        for sel in single_item_selectors:
            for item in soup.select(sel):
                title_el = item.select_one('a.mh-news-title') or item.select_one('.g-title-inner a') or item.select_one('a[href]')
                title = title_el.get_text(strip=True) if title_el else ""

                snippet_text = ""
                desc_el = item.select_one('p.mh-news-desc')
                if desc_el:
                    snippet_text = desc_el.get_text(' ', strip=True)
                if not snippet_text:
                    for ssel in ['div.g-figure-caption', '.g-card-text', '.g-desc', '.desc', 'p']:
                        node = item.select_one(ssel)
                        if node:
                            t = node.get_text(' ', strip=True)
                            if len(t) > len(snippet_text):
                                snippet_text = t

                url = ""
                if title_el and title_el.get('href'):
                    url = title_el['href'].strip()
                if not url:
                    a = item.select_one('a[href]')
                    if a:
                        url = (a.get('href') or '').strip()

                if not title and not snippet_text:
                    continue
                results.append({
                    "title": title[:160],
                    "snippet": snippet_text[:240],
                    "url": url,
                })
                if len(results) >= max(max_results * 2, 12):
                    break
            if len(results) >= max_results:
                break

        # A-2) 再用卡片级(跨条目的聚合)作为补充
        if len(results) < max_results:
            aggregate_selectors = [
                'div.g-card-layout',
                'div.g-mohe-item',
                'div.g-card',
                '.g-figure-caption',
            ]
            for sel in aggregate_selectors:
                for item in soup.select(sel):
                    title_el = (
                        item.select_one('.g-title a')
                        or item.select_one('h3 a')
                        or item.select_one('a.mh-news-title')
                        or item.select_one('.g-figure-caption a')
                        or item.select_one('a[href]')
                    )
                    snippet_text = ""
                    for ssel in [
                        'p.mh-news-desc', '.g-figure-caption', '.res-desc', '.summary',
                        '.g-card-text', '.g-desc', '.desc', '.abstract', 'p'
                    ]:
                        node = item.select_one(ssel)
                        if node:
                            t = node.get_text(' ', strip=True)
                            if len(t) > len(snippet_text):
                                snippet_text = t
                    title = title_el.get_text(strip=True) if title_el else ""
                    url = ""
                    if title_el and title_el.get('href'):
                        url = title_el['href']
                    elif not title:
                        a = item.select_one('a[href]')
                        if a:
                            url = a.get('href', '') or ''

                    if not title and not snippet_text:
                        full = item.get_text(' ', strip=True)
                        if len(full) < 20:
                            continue
                        title = full[:80]
                        snippet_text = full[80:320] if len(full) > 80 else ""
                    results.append({
                        "title": title[:160],
                        "snippet": snippet_text[:240],
                        "url": url,
                    })
                if len(results) >= max_results:
                    break

        # B) 老版/通用结果列表(单条结果一个 res-list 容器)
        if len(results) < max_results:
            for result in soup.select('.res-list'):
                title_el = (
                    result.select_one('h3 a')
                    or result.select_one('.res-title a')
                    or result.select_one('.g-title a')
                    or result.select_one('h3.g-title')
                )
                snippet_text = ""
                for ssel in [
                    '.res-desc', '.summary', '.g-desc', 'p.mh-news-desc',
                    '.g-card-text', '.g-figure-caption', 'p', '.desc', '.abstract'
                ]:
                    node = result.select_one(ssel)
                    if node:
                        t = node.get_text(' ', strip=True)
                        if len(t) > len(snippet_text):
                            snippet_el = node
                            snippet_text = t

                title = title_el.get_text(strip=True) if title_el else ""
                url_el_src = ""
                url_el = result.select_one('cite, .res-site, .g-source')
                if url_el:
                    url_el_src = url_el.get_text(strip=True)
                if title_el and title_el.get('href'):
                    url = title_el.get('href', '')
                elif url_el_src:
                    url = url_el_src
                else:
                    a = result.select_one('a[href]')
                    if a:
                        url = a.get('href', '')
                    else:
                        url = ""

                if not snippet_text and (title or url):
                    full = result.get_text(' ', strip=True)
                    if len(full) > len(title) + 20:
                        start = full.find(title) if title in full else 0
                        snippet_text = full[start + len(title):start + len(title) + 260].strip()

                if title or snippet_text:
                    dup = any(r.get('title') == title and r.get('url') == url for r in results)
                    if not dup:
                        results.append({
                            "title": title[:160],
                            "snippet": snippet_text[:240],
                            "url": url,
                        })
                if len(results) >= max_results:
                    break

        # C) 老版 placeholder
        if not results:
            return [
                {
                    "title": "No results for query",
                    "snippet": "360 returned an empty result page.",
                    "url": "https://www.so.com",
                }
            ]

        return results[: max(max_results, 1)]

    def _search_bing(self, query: str, max_results: int) -> List[dict]:
        """Bing HTML search fallback."""
        url = "https://www.bing.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        last_exception = None
        for attempt in range(2):
            try:
                response = self._session.get(
                    url, params={"q": query}, headers=headers, timeout=5
                )
                response.raise_for_status()
                return self._parse_bing(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.8 * (attempt + 1))
        raise last_exception or RuntimeError("Bing search failed")

    def _parse_bing(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html)
        results = []
        for li in soup.select(".b_algo"):
            title_el = li.select_one("h2 a")
            snippet_el = li.select_one("p")
            url_el = li.select_one("cite")
            title = title_el.get_text(strip=True) if title_el else ""
            url = url_el.get_text(strip=True) if url_el else (
                title_el.get("href", "") if title_el else ""
            )
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if not title:
                continue
            results.append({"title": title[:160], "snippet": snippet[:240], "url": url})
        return results[:max(max_results, 1)]

    def _search_google(self, query: str, max_results: int) -> List[dict]:
        """Google HTML search fallback (often blocked; used last)."""
        url = "https://www.google.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        last_exception = None
        for attempt in range(2):
            try:
                response = self._session.get(
                    url, params={"q": query, "num": max_results}, headers=headers, timeout=5
                )
                response.raise_for_status()
                return self._parse_google(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.8 * (attempt + 1))
        raise last_exception or RuntimeError("Google search failed")

    def _parse_google(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html)
        results = []
        for g in soup.select("div.g"):
            title_el = g.select_one("h3")
            link_el = g.select_one("a[href]")
            snippet_el = g.select_one("div[data-sncf], .VwiC3b, span.st")
            title = title_el.get_text(strip=True) if title_el else ""
            url = link_el.get("href", "") if link_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if not title:
                continue
            results.append({"title": title[:160], "snippet": snippet[:240], "url": url})
        return results[:max(max_results, 1)]

    # ------------------------------------------------------------------
    # Async provider implementations (used when aiohttp is available)
    # ------------------------------------------------------------------
    async def _async_search_duckduckgo(self, query: str, max_results: int) -> List[dict]:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            "Referer": "https://duckduckgo.com/",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                    html = await resp.text()
                    return self._parse_duckduckgo(html, max_results)
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo async search failed: {e}") from e

    async def _async_search_360(self, query: str, max_results: int) -> List[dict]:
        url = "https://www.so.com/s"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    html = await resp.text()
                    return self._parse_360(html, max_results)
        except Exception as e:
            raise RuntimeError(f"360 async search failed: {e}") from e

    async def _async_search_serpapi(self, query: str, max_results: int) -> List[dict]:
        if not self.api_key:
            return []
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": max_results,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search", params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json()
                    results = []
                    for item in data.get("organic_results", [])[:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                        })
                    return results
        except Exception as e:
            raise RuntimeError(f"SerpAPI async search failed: {e}") from e

    async def _async_search_bing(self, query: str, max_results: int) -> List[dict]:
        """Bing HTML search fallback."""
        url = "https://www.bing.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    html = await resp.text()
                    return self._parse_bing(html, max_results)
        except Exception as e:
            raise RuntimeError(f"Bing async search failed: {e}") from e

    async def _async_search_google(self, query: str, max_results: int) -> List[dict]:
        """Google HTML search fallback (often blocked; used last)."""
        url = "https://www.google.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query, "num": max_results}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    html = await resp.text()
                    return self._parse_google(html, max_results)
        except Exception as e:
            raise RuntimeError(f"Google async search failed: {e}") from e

    async def _async_search_google_news(self, query: str, max_results: int) -> List[dict]:
        """Google News search."""
        url = "https://news.google.com/rss/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    xml = await resp.text()
                    return self._parse_google_news(xml, max_results)
        except Exception as e:
            raise RuntimeError(f"Google News async search failed: {e}") from e

    async def _async_search_bing(self, query: str, max_results: int) -> List[dict]:
        """Bing HTML search fallback."""
        url = "https://www.bing.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    html = await resp.text()
                    return self._parse_bing(html, max_results)
        except Exception as e:
            raise RuntimeError(f"Bing async search failed: {e}") from e

    async def _async_search_google(self, query: str, max_results: int) -> List[dict]:
        """Google HTML search fallback (often blocked; used last)."""
        url = "https://www.google.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query, "num": max_results}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    html = await resp.text()
                    return self._parse_google(html, max_results)
        except Exception as e:
            raise RuntimeError(f"Google async search failed: {e}") from e

    async def _async_search_bing_news(self, query: str, max_results: int) -> List[dict]:
        """Bing News search."""
        """Bing News search."""
        url = "https://www.bing.com/news/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    html = await resp.text()
                    return self._parse_bing_news(html, max_results)
        except Exception as e:
            raise RuntimeError(f"Bing News async search failed: {e}") from e

    async def _async_search_google_news(self, query: str, max_results: int) -> List[dict]:
        """Google News search."""
        url = "https://news.google.com/rss/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    xml = await resp.text()
                    return self._parse_google_news(xml, max_results)
        except Exception as e:
            raise RuntimeError(f"Google News async search failed: {e}") from e

    def _parse_bing_news(self, html: str, max_results: int) -> List[dict]:
        """Parse Bing News HTML results."""
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html)
        results: List[dict] = []
        for article in soup.select(".news-card, .newsitem, .news-card-body"):
            title_el = article.select_one(".title a, h3 a, .news-title a")
            snippet_el = article.select_one(".snippet, .description, .news-snippet")
            url_el = title_el if title_el else article.select_one("a[href]")
            
            title = title_el.get_text(strip=True) if title_el else ""
            url = title_el.get("href", "") if title_el else (url_el.get("href", "") if url_el else "")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            
            if not title:
                continue
                
            # Ensure absolute URL
            if url.startswith("/"):
                url = "https://www.bing.com" + url
                
            results.append({
                "title": title[:160],
                "snippet": snippet[:240],
                "url": url,
            })
            if len(results) >= max_results:
                break
        
        if not results:
            # Fallback to generic parsing
            for item in soup.select(".news-card, .newsitem"):
                title_el = item.select_one("h3 a, .title a")
                snippet_el = item.select_one(".snippet, .description")
                if title_el:
                    title = title_el.get_text(strip=True)
                    url = title_el.get("href", "")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    if title:
                        results.append({"title": title[:160], "snippet": snippet[:240], "url": url})
        
        return results[:max_results]

    def _parse_google_news(self, xml: str, max_results: int) -> List[dict]:
        """Parse Google News RSS results."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml)
            results: List[dict] = []
            for item in root.findall(".//item")[:max_results]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                description = item.findtext("description", "").strip()
                # Clean HTML from description
                import re
                description = re.sub(r'<[^>]+>', '', description)
                if title:
                    results.append({
                        "title": title[:160],
                        "snippet": description[:240],
                        "url": link,
                    })
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Async collection
    # ------------------------------------------------------------------
    async def _async_collect_provider_results(
        self,
        query: str,
        max_results: int,
        max_workers: int = 4,
        providers: Optional[List[str]] = None,
        is_deep: bool = False,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        """Async version of _collect_provider_results with early exit."""
        provider_results: Dict[str, List[Dict[str, Any]]] = {}
        errors: List[str] = []
        provider_list = providers or self.providers

        # Map provider name to async method
        async_methods = {
            "duckduckgo": self._async_search_duckduckgo,
            "360": self._async_search_360,
            "bing": self._async_search_bing,
            "google": self._async_search_google,
            "bing-news": self._async_search_bing_news,
            "google-news": self._async_search_google_news,
            "serpapi": self._async_search_serpapi,
        }

        # Create tasks mapping task -> provider
        task_to_provider = {}
        for provider in provider_list:
            method = async_methods.get(provider)
            if method:
                task = asyncio.create_task(method(query, max_results))
                task_to_provider[task] = provider
            else:
                errors.append(f"Unknown provider: {provider}")

        # Wait for tasks with early exit on first real result (unless deep mode)
        try:
            for coro in asyncio.as_completed(task_to_provider.keys(), timeout=8):
                task = coro
                provider = task_to_provider[task]
                try:
                    results = await task
                    if error:
                        errors.append(f"{provider}: {error}")
                    if self._is_real_results(results):
                        provider_results[provider] = results
                        # Early exit: one good provider is enough for normal mode
                        if not is_deep:
                            # Cancel remaining tasks
                            for t in task_to_provider:
                                if not t.done():
                                    t.cancel()
                            break
                except Exception as e:
                    errors.append(f"{provider}: {e}")
        except asyncio.TimeoutError:
            # Timeout: collect any completed tasks
            for task, provider in task_to_provider.items():
                if task.done() and provider not in provider_results:
                    try:
                        results = task.result()
                        if self._is_real_results(results):
                            provider_results[provider] = results
                    except Exception as e:
                        errors.append(f"{provider}: {e}")
            # Cancel pending
            for task in task_to_provider:
                if not task.done():
                    task.cancel()

        return provider_results, errors

    # Helper to run async code from synchronous context
    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # If there's already a running loop, we cannot use asyncio.run.
            # We'll run in a new thread with a new event loop.
            import threading
            result_container = []
            exc_container = []
            def runner():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_container.append(new_loop.run_until_complete(coro))
                except Exception as e:
                    exc_container.append(e)
                finally:
                    new_loop.close()
            t = threading.Thread(target=runner)
            t.start()
            t.join()
            if exc_container:
                raise exc_container[0]
            return result_container[0]
        else:
            return asyncio.run(coro)


if not TOOLS_REGISTRY.get("web_search"):
    TOOLS_REGISTRY.register(WebSearchTool())
