"""Web Search Tool - multi-provider search with quality enhancement.

Pipeline:
  query → cache check → concurrent provider calls → URL unshortening
  → quality scoring → cross-provider fusion → snippet enrichment
  → cache write → ranked results.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
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

        # Resolve effective provider list
        self.providers = self.cfg.get("providers", [provider])
        if provider in ("duckduckgo", "360") and set(self.providers) <= {"duckduckgo", "360"}:
            # Legacy / default free-provider set
            self.providers = ["duckduckgo", "360", "bing", "google"]

        self.max_results_default = self.cfg.get("max_results", 5)
        self.quality_threshold = self.cfg.get("quality_threshold", 0.6)
        self.use_playwright = self.cfg.get("use_playwright", False)
        self.sequential_fallback = self.cfg.get("sequential_fallback", False)
        self.max_fetch_urls = self.cfg.get("max_fetch_urls", 3)
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
        if not re.search(r'(20\d{2})', q):
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

            # 1. Cache check (deep mode also consults cache; only full cache hits
            #    from a near-identical query bypass the search to save tokens/time.
            #    Deep queries still fall through to live search when there is no hit.)
            cached = self.cache.get(query, threshold=0.5)
            if cached:
                return ToolResult(
                    success=True,
                    output=json.dumps(cached[:effective_max_results], indent=2, ensure_ascii=False),
                    metadata={
                        "query": query,
                        "provider": "cache",
                        "num_results": len(cached[:effective_max_results]),
                        "cached": True,
                    },
                )

            # 2. Gather results from providers
            # 意图感知: 中文查询优先国内源, 英文查询优先国际源(提高相关性)
            need = self._detect_search_need(query)
            effective_providers = list(self.providers)
            if need["lang"] == "zh" and "360" in effective_providers:
                effective_providers = ["360"] + [p for p in effective_providers if p != "360"]
            elif need["lang"] == "en" and "bing" in effective_providers:
                effective_providers = ["bing"] + [p for p in effective_providers if p != "bing"]
            provider_results, errors = self._collect_provider_results(
                query, effective_max_results, max_workers=max_workers, providers=effective_providers
            )

            # 3. Fallback: if nothing usable, try a relaxed query
            if not provider_results:
                relaxed = self._relax_query(query)
                if relaxed and relaxed != query:
                    provider_results, extra_errors = self._collect_provider_results(
                        relaxed, effective_max_results, max_workers=max_workers
                    )
                    errors.extend(extra_errors)

            if not provider_results:
                return ToolResult(
                    success=False,
                    output="",
                    error="; ".join(errors) if errors else "No search results returned",
                )

            # 4. Fuse, score, dedupe, diversify
            fused = self.fusion.fuse(provider_results, query, max_results=fusion_max)

            # 4b. 意图感知的权威域优先: tech 类 query 时把技术域结果提前
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

            # 5. Enrich low-quality snippets with real page content
            if fused:
                avg_snippet_len = sum(len((r.get("snippet") or "").strip()) for r in fused) / len(fused)
                weak_snippets = avg_snippet_len < 80
                has_real_urls = any(self._looks_like_url(r.get("url", "")) for r in fused[:3])
                if weak_snippets or has_real_urls:
                    self.fetcher.enrich_results(fused, max_urls=fetch_max)

            final = fused[:effective_max_results]

            # 6. Cache final results (deep results cached separately by query)
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
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

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
        provider_list = providers or provider_list

        if self.sequential_fallback:
            for provider in provider_list:
                results, error = self._call_provider(provider, query, max_results)
                if error:
                    errors.append(error)
                if self._is_real_results(results):
                    provider_results[provider] = results
                    # In sequential fallback we stop at the first real provider
                    break
        else:
            # 并发收集, 动态等待: 每 2s 检查一次, 已有足够结果即提前返回, 不等慢 provider
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {
                    executor.submit(self._call_provider, provider, query, max_results): provider
                    for provider in provider_list
                }
                deadline = time.monotonic() + 10
                poll_interval = 2
                done_set = set()
                while time.monotonic() < deadline and len(done_set) < len(futures):
                    newly_done, _ = concurrent.futures.wait(
                        futures, timeout=poll_interval, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    done_set.update(newly_done)
                    # 检查是否已拿到足够真结果, 无需再等
                    any_real = False
                    for future in done_set:
                        provider = futures[future]
                        try:
                            results, error = future.result()
                            if error:
                                errors.append(f"{provider}: {error}")
                            if self._is_real_results(results):
                                provider_results[provider] = results
                                any_real = True
                        except Exception as e:
                            errors.append(f"{provider}: {e}")
                    if any_real:
                        break
                # 收尾: 处理剩余已完成 future(避免漏掉并发完成的)
                remaining = set(futures.keys()) - done_set
                for future in remaining:
                    if future.done():
                        done_set.add(future)
                for future in done_set:
                    if future in futures and futures[future] not in provider_results:
                        provider = futures[future]
                        try:
                            results, error = future.result()
                            if error:
                                errors.append(f"{provider}: {error}")
                            if self._is_real_results(results) and provider not in provider_results:
                                provider_results[provider] = results
                        except Exception as e:
                            errors.append(f"{provider}: {e}")
                # 未完成 future 不阻塞, 直接取消后台线程
                not_done = set(futures.keys()) - done_set
                for future in not_done:
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
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for real web search parsing. "
                "Install it with: pip install beautifulsoup4"
            ) from exc
        return BeautifulSoup

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
        soup = BeautifulSoup(html, "html.parser")
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
                    timeout=10,
                )
                response.raise_for_status()
                return self._parse_360(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.8 * (attempt + 1))

        raise last_exception or RuntimeError("360 search failed")

    def _parse_360(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html, "html.parser")
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
                    url, params={"q": query}, headers=headers, timeout=10
                )
                response.raise_for_status()
                return self._parse_bing(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.8 * (attempt + 1))
        raise last_exception or RuntimeError("Bing search failed")

    def _parse_bing(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html, "html.parser")
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
                    url, params={"q": query, "num": max_results}, headers=headers, timeout=10
                )
                response.raise_for_status()
                return self._parse_google(response.text, max_results)
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(0.8 * (attempt + 1))
        raise last_exception or RuntimeError("Google search failed")

    def _parse_google(self, html: str, max_results: int) -> List[dict]:
        BeautifulSoup = self._get_bs()
        soup = BeautifulSoup(html, "html.parser")
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

    def _search_serpapi(self, query: str, max_results: int) -> List[dict]:
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": max_results,
        }
        response = requests.get(
            "https://serpapi.com/search", params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                }
            )
        return results


if not TOOLS_REGISTRY.get("web_search"):
    TOOLS_REGISTRY.register(WebSearchTool())
