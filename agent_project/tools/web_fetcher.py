"""
Web content fetcher for search result enrichment.

Fetches real webpage content with requests first; if the page looks like it
needs JS rendering and the caller allows it, falls back to Playwright.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import requests

from . import BaseTool
from . import ToolResult as _ToolResult


class WebContentFetcher:
    """Fetch and clean webpage content for LLM consumption."""

    def __init__(
        self,
        timeout: int = 6,
        min_text_length: int = 120,
        max_text_length: int = 1200,
        use_playwright: bool = False,
        polite_delay: float = 0.15,
    ):
        self.timeout = timeout
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.use_playwright = use_playwright
        self.polite_delay = polite_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
        )

    def fetch(self, url: str) -> Optional[str]:
        """Return cleaned text from *url*, or None if unsuitable."""
        if not self._looks_fetchable(url):
            return None

        text = self._fetch_with_requests(url)
        if text and len(text) >= self.min_text_length:
            return text[: self.max_text_length]

        if self.use_playwright and (not text or len(text) < self.min_text_length):
            text = self._fetch_with_playwright(url)

        if text and len(text) >= self.min_text_length:
            return text[: self.max_text_length]
        return None

    def enrich_results(
        self,
        results: List[Dict[str, Any]],
        max_urls: int = 3,
    ) -> None:
        """In-place enrichment of search results with fetched body text."""
        candidates = []
        for r in results:
            url = (r.get("url") or "").strip()
            if not url or not self._looks_fetchable(url):
                continue
            current_snippet = (r.get("snippet") or "").strip()
            # Skip if snippet is already substantial
            if len(current_snippet) >= self.min_text_length * 1.5:
                continue
            candidates.append(r)

        # 并行抓取正文(深研模式 max_urls 可达 20+, 串行会严重拖慢)
        chosen = candidates[:max_urls]
        if not chosen:
            return
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(len(chosen), 8)) as executor:
                future_map = {executor.submit(self.fetch, r.get("url", "")): r for r in chosen}
                for future in as_completed(future_map, timeout=max(8, self.timeout * 2)):
                    r = future_map[future]
                    try:
                        text = future.result()
                    except Exception:
                        continue
                    if not text or len(text) < self.min_text_length:
                        continue
                    current_snippet = (r.get("snippet") or "").strip()
                    if len(current_snippet) < len(text) * 0.6 or not current_snippet:
                        r["snippet"] = text
                        r["_enriched_from"] = r.get("url", "")
        except Exception:
            pass

    def _fetch_with_requests(self, url: str) -> Optional[str]:
        try:
            # 用独立连接请求, 避免共享 Session 的连接池锁导致线程并行退化为串行
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers=self._session.headers,
                stream=False,
            )
            if resp.status_code != 200:
                return None
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct.lower() and "text" not in ct.lower():
                return None
            try:
                if resp.encoding:
                    resp.encoding = resp.apparent_encoding or resp.encoding
            except Exception:
                pass
            return self._extract_text(resp.text)
        except Exception:
            return None
        finally:
            time.sleep(self.polite_delay)

    def _fetch_with_playwright(self, url: str) -> Optional[str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")
                # Wait a short moment for JS hydration
                page.wait_for_timeout(1500)
                html = page.content()
                browser.close()
                return self._extract_text(html)
        except Exception:
            return None
        finally:
            time.sleep(self.polite_delay)

    def _extract_text(self, html: str) -> Optional[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select("script, style, noscript, iframe, svg, header, footer, nav, aside"):
            try:
                tag.decompose()
            except Exception:
                pass

        body_el = (
            soup.select_one("article")
            or soup.select_one("main")
            or soup.select_one('[id*="content" i]')
            or soup.select_one('[class*="content" i]')
            or soup.select_one('[class*="article" i]')
            or soup.select_one("body")
        )
        text = body_el.get_text("\n", strip=True) if body_el else ""
        if len(text) < self.min_text_length:
            for meta in [
                soup.find("meta", attrs={"name": "description"}),
                soup.find("meta", attrs={"property": "og:description"}),
            ]:
                if meta and meta.get("content"):
                    text = (meta.get("content") or "").strip()
                    break

        text = re.sub(r"\s{2,}", " ", text).strip()
        return text if len(text) >= self.min_text_length else None

    @staticmethod
    def _looks_fetchable(url: str) -> bool:
        if not url:
            return False
        if not url.startswith(("http://", "https://")):
            return False
        blocked = (
            "news.so.com", "so.com/link", "jump.", "redirect.", "click.link",
            "youtube.com", "bilibili.com/video", "vimeo.com",
        )
        return not any(d in url for d in blocked)


class ReadWebTool(BaseTool):
    """read_web(url) → 打开网页提取正文(供 LLM 消费).

    fast path 的 system prompt 教模型用 read_web 打开网页; 这里让它成为真实注册的工具,
    包装 WebContentFetcher(轻量 requests 抓取 + 文本清理), 不依赖 Playwright。
    """

    name = "read_web"
    description = "Fetch a webpage and extract its readable text content. Use for reading articles/docs/pages."

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (http/https).",
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        from . import TOOLS_REGISTRY, ToolResult
        self._fetcher = WebContentFetcher()

    def execute(self, url: str) -> Any:
        from . import ToolResult
        if not url or not isinstance(url, str):
            return ToolResult(success=False, output="", error="url 参数 required for read_web")
        try:
            text = self._fetcher.fetch(url)
            if not text:
                return ToolResult(
                    success=False, output="",
                    error=f"无法提取 {url} 的正文(可能需 JS 渲染或页面无文本)",
                )
            return ToolResult(success=True, output=text, metadata={"url": url})
        except Exception as e:
            return ToolResult(success=False, output="", error=f"read_web failed: {e}")
