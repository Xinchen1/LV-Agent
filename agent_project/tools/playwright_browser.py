"""Playwright Browser Tool - Stable browser automation subset."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import BaseTool, ToolResult, TOOLS_REGISTRY


class PlaywrightBrowserTool(BaseTool):
    name = "browser"
    description = (
        "Browser automation with Playwright. "
        "Stable actions: navigate, scrape, screenshot, evaluate."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "screenshot", "scrape", "evaluate"],
                "description": "Browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for navigate",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for scrape",
            },
            "text": {
                "type": "string",
                "description": "Text to extract",
            },
            "javascript": {
                "type": "string",
                "description": "JavaScript code for evaluate",
            },
            "full_page": {
                "type": "boolean",
                "description": "Full page screenshot (default: false)",
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds",
                "default": 30000,
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        headless: bool = True,
        default_timeout: int = 30000,
    ):
        self.headless = headless
        self.default_timeout = default_timeout

    def execute(
        self,
        action: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        javascript: Optional[str] = None,
        full_page: bool = False,
        timeout: Optional[int] = None,
    ) -> ToolResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="Playwright not installed. Run: pip install playwright && playwright install",
            )

        timeout = timeout or self.default_timeout

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                page = browser.new_page()
                page.set_default_timeout(timeout)

                if action == "navigate":
                    if not url:
                        browser.close()
                        return ToolResult(
                            success=False,
                            output="",
                            error="url is required for navigate",
                        )
                    page.goto(url, wait_until="load")
                    title = page.title()
                    content_length = len(page.content())
                    browser.close()
                    return ToolResult(
                        success=True,
                        output=f"Navigated to: {url}\nTitle: {title}\nContent length: {content_length} bytes",
                        metadata={"url": url, "title": title},
                    )

                if action == "screenshot":
                    if url:
                        page.goto(url, wait_until="load")
                    screenshot_bytes = page.screenshot(
                        full_page=full_page,
                        type="png",
                    )
                    browser.close()
                    b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                    return ToolResult(
                        success=True,
                        output=f"Screenshot captured ({len(screenshot_bytes)} bytes, base64 length {len(b64)})",
                        metadata={"screenshot": f"data:image/png;base64,{b64}"},
                    )

                if action == "scrape":
                    if url:
                        page.goto(url, wait_until="load")
                    if not selector:
                        browser.close()
                        return ToolResult(
                            success=False,
                            output="",
                            error="selector is required for scrape",
                        )
                    page.wait_for_selector(selector, timeout=timeout)
                    elements = page.query_selector_all(selector)
                    texts = [el.text_content() or "" for el in elements]
                    browser.close()
                    return ToolResult(
                        success=True,
                        output="\n".join(texts),
                        metadata={"count": len(texts), "selector": selector},
                    )

                if action == "evaluate":
                    if not javascript:
                        browser.close()
                        return ToolResult(
                            success=False,
                            output="",
                            error="javascript is required for evaluate",
                        )
                    if url:
                        page.goto(url, wait_until="load")
                    result = page.evaluate(javascript)
                    browser.close()
                    return ToolResult(
                        success=True,
                        output=str(result),
                        metadata={"javascript": javascript[:100]},
                    )

                browser.close()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unsupported browser action: {action}",
                )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Browser error: {str(e)}")


TOOLS_REGISTRY.register(PlaywrightBrowserTool())
