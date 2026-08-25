"""
URL unshortener / redirect resolver for web search results.

Resolves obfuscated URLs such as 360's `so.com/link?m=...&url=...` into the
real landing URL without fetching the full page body (HEAD / short timeout).
"""

from __future__ import annotations

import re
import time
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests


class URLUnshortener:
    """Resolve redirect chains and extract real URLs from search engine wrappers."""

    def __init__(self, timeout: int = 6, max_redirects: int = 5, delay: float = 0.15):
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.delay = delay
        self._session = requests.Session()
        # 连接池加大: resolve_many 会并行对多个 URL 发 HEAD, 默认池(10)易满导致丢连接
        adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
        )
        self._cache: dict[str, str] = {}

    @staticmethod
    def _looks_direct(url: str) -> bool:
        """快速路径: 非搜索引擎包裹的普通直链, 无需网络解析."""
        if not url.startswith(("http://", "https://")):
            return False
        parsed = urlparse(url)
        # 无 query 包裹参数、host 非 so.com/link 等 → 大概率是直链
        if "so.com/link" in url or "baidu.com/link" in url or "google.com/url" in url or "bing.com/ck" in url:
            return False
        if parsed.query and any(k in parsed.query.lower() for k in ("url=", "u=", "target=", "dest=")):
            return False
        return True

    def resolve(self, url: str) -> str:
        """Return the real landing URL for *url*.

        Handles:
        - 360 so.com/link?m=...&url=... wrappers
        - HTTP 3xx redirects
        - JS/meta refresh redirects (lightweight regex extraction)
        """
        if not url or not isinstance(url, str):
            return url

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                return url
            else:
                return url

        # 缓存命中
        if url in self._cache:
            return self._cache[url]

        # Fast path: extract target from known search-engine query parameters
        real = self._extract_from_query(url)
        if real and real != url:
            self._cache[url] = real
            return real

        # 普通直链: 不做网络请求(搜索引擎返回的真实落地页通常无需再解析)
        if self._looks_direct(url):
            self._cache[url] = url
            return url

        # Network path: follow redirects
        try:
            resp = self._session.head(
                url,
                timeout=self.timeout,
                allow_redirects=False,
                headers={"Referer": "https://www.so.com/"},
            )
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location") or resp.headers.get("location")
                if location:
                    resolved = self._normalize_location(url, location)
                    self._cache[url] = resolved
                    return resolved
        except Exception:
            pass

        # Fallback: GET the page and look for meta refresh / JS location
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=False)
            body = resp.text[:12000]
            real = self._extract_meta_refresh(body) or self._extract_js_redirect(body)
            if real:
                resolved = self._normalize_location(url, real)
                self._cache[url] = resolved
                return resolved
        except Exception:
            pass
        finally:
            time.sleep(self.delay)

        self._cache[url] = url
        return url

    def resolve_many(self, urls: list[str]) -> dict[str, str]:
        """Resolve multiple URLs in parallel (with shared cache)."""
        out: dict[str, str] = {}
        pending = []
        for url in urls:
            if url in self._cache:
                out[url] = self._cache[url]
            else:
                pending.append(url)
        if not pending:
            return out
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(pending), 8)) as executor:
                futures = {executor.submit(self.resolve, u): u for u in pending}
                for future, u in futures.items():
                    try:
                        out[u] = future.result()
                    except Exception:
                        out[u] = u
        except Exception:
            for u in pending:
                out[u] = self.resolve(u)
        return out

    def _extract_from_query(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        for key in ("url", "u", "target", "link", "href", "dest", "destination"):
            if key in query and query[key]:
                candidate = unquote(query[key][0]).strip()
                if candidate:
                    return self._normalize(candidate)
        return url

    @staticmethod
    def _normalize(url: str) -> str:
        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith(("http://", "https://")):
            return url
        # Strip trailing whitespace and common tracking fragments
        url = re.sub(r"[\s\x00-\x1f]+", "", url)
        return url

    @staticmethod
    def _normalize_location(base: str, location: str) -> str:
        if not location:
            return base
        location = location.strip()
        if location.startswith("//"):
            return "https:" + location
        if location.startswith(("http://", "https://")):
            return location
        if location.startswith("/"):
            parsed = urlparse(base)
            return f"{parsed.scheme}://{parsed.netloc}{location}"
        return location

    @staticmethod
    def _extract_meta_refresh(body: str) -> Optional[str]:
        # <meta http-equiv="refresh" content="0;url=https://...">
        match = re.search(
            r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\d+\s*;\s*url=([^"\'>]+)',
            body,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return unquote(match.group(1).strip())
        match = re.search(
            r'<meta[^>]+content=["\']?\d+\s*;\s*url=([^"\'>]+)[^>]+http-equiv=["\']?refresh["\']?',
            body,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return unquote(match.group(1).strip())
        return None

    @staticmethod
    def _extract_js_redirect(body: str) -> Optional[str]:
        # window.location.href = "https://..."
        patterns = [
            r"window\.location\.href\s*=\s*[\"']([^\"']+)[\"']",
            r"window\.location\s*=\s*[\"']([^\"']+)[\"']",
            r"location\.replace\s*\(\s*[\"']([^\"']+)[\"']",
            r"location\.href\s*=\s*[\"']([^\"']+)[\"']",
        ]
        for pattern in patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                return unquote(match.group(1).strip())
        return None
