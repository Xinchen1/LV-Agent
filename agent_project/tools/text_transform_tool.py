"""Text transformation tool - hashing, encoding, formatting and regex helpers (offline)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import json
import logging
import re
import unicodedata
from typing import Any, Dict, List
from urllib.parse import quote, unquote

from . import BaseTool, ToolResult, TOOLS_REGISTRY

logger = logging.getLogger(__name__)

_HASHES = {"md5", "sha1", "sha224", "sha256", "sha384", "sha512", "blake2b", "blake2s"}


def _slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^\w\s-]", "", ascii_text).strip().lower()
    return re.sub(r"[-\s]+", "-", ascii_text)


def _case(text: str, mode: str) -> str:
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    if mode == "title":
        return text.title()
    if mode == "capitalize":
        return text.capitalize()
    if mode == "snake":
        parts = re.split(r"[\s\-_.]+", text.strip())
        return "_".join(p.lower() for p in parts if p)
    if mode == "kebab":
        parts = re.split(r"[\s\-_.]+", text.strip())
        return "-".join(p.lower() for p in parts if p)
    if mode == "camel":
        parts = re.split(r"[\s\-_.]+", text.strip())
        return "".join(p[:1].upper() + p[1:] for p in parts if p)
    if mode == "pascal":
        parts = re.split(r"[\s\-_.]+", text.strip())
        return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)
    raise ValueError(f"Unknown case mode: {mode}")


class TextTransformTool(BaseTool):
    name = "text_transform"
    description = (
        "Offline text utilities: hash (md5/sha256/...), base64/url/html encode and decode, "
        "slugify, case conversion, JSON pretty/minify, counts, dedupe, sort, regex extract/replace."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["hash", "base64_encode", "base64_decode", "url_encode", "url_decode",
                         "html_escape", "html_unescape", "slug", "case", "json_format",
                         "json_minify", "count", "unique", "sort_lines", "regex_extract",
                         "regex_replace", "trim"],
                "description": "Transformation to apply.",
            },
            "text": {"type": "string", "description": "Input text."},
            "algorithm": {"type": "string", "description": "Hash algorithm (default sha256)."},
            "mode": {"type": "string",
                     "description": "Case mode: upper/lower/title/capitalize/snake/kebab/camel/pascal."},
            "indent": {"type": "integer", "description": "JSON indent (default 2)."},
            "pattern": {"type": "string", "description": "Regex pattern."},
            "replacement": {"type": "string", "description": "Regex replacement (supports \\1 groups)."},
            "group": {"type": "integer", "description": "Regex capture group (default 0)."},
            "numeric": {"type": "boolean", "description": "Numeric sort/count for sort_lines."},
            "descending": {"type": "boolean", "description": "Descending order."},
            "separator": {"type": "string", "description": "Separator for list modes."},
        },
        "required": ["action", "text"],
    }

    def execute(self, action: str, text: str = "", **kwargs) -> ToolResult:
        try:
            handler = {
                "hash": self._hash,
                "base64_encode": self._b64_encode,
                "base64_decode": self._b64_decode,
                "url_encode": self._url_encode,
                "url_decode": self._url_decode,
                "html_escape": self._html_escape,
                "html_unescape": self._html_unescape,
                "slug": self._slug_action,
                "case": self._case_action,
                "json_format": self._json_format,
                "json_minify": self._json_minify,
                "count": self._count,
                "unique": self._unique,
                "sort_lines": self._sort_lines,
                "regex_extract": self._regex_extract,
                "regex_replace": self._regex_replace,
                "trim": self._trim,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(text, **kwargs)
        except Exception as e:
            logger.debug("text_transform %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _hash(self, text: str, algorithm: str = "sha256", **kwargs) -> ToolResult:
        algo = (algorithm or "sha256").lower()
        if algo not in _HASHES or algo not in hashlib.algorithms_available:
            return ToolResult(success=False, output="", error=f"Unsupported hash: {algorithm}")
        digest = hashlib.new(algo, text.encode("utf-8")).hexdigest()
        return ToolResult(success=True, output=digest, metadata={"algorithm": algo, "digest": digest})

    def _b64_encode(self, text: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=base64.b64encode(text.encode("utf-8")).decode("ascii"))

    def _b64_decode(self, text: str, **kwargs) -> ToolResult:
        try:
            decoded = base64.b64decode(text.strip(), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as e:
            return ToolResult(success=False, output="", error=f"Invalid base64/utf-8: {e}")
        return ToolResult(success=True, output=decoded)

    def _url_encode(self, text: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=quote(text, safe=""))

    def _url_decode(self, text: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=unquote(text))

    def _html_escape(self, text: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=html.escape(text))

    def _html_unescape(self, text: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=html.unescape(text))

    def _slug_action(self, text: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=_slug(text))

    def _case_action(self, text: str, mode: str = "lower", **kwargs) -> ToolResult:
        return ToolResult(success=True, output=_case(text, mode))

    def _json_format(self, text: str, indent: int = 2, **kwargs) -> ToolResult:
        try:
            parsed = json.loads(text)
        except ValueError as e:
            return ToolResult(success=False, output="", error=f"Invalid JSON: {e}")
        return ToolResult(success=True,
                          output=json.dumps(parsed, ensure_ascii=False, indent=max(0, int(indent or 2))))

    def _json_minify(self, text: str, **kwargs) -> ToolResult:
        try:
            parsed = json.loads(text)
        except ValueError as e:
            return ToolResult(success=False, output="", error=f"Invalid JSON: {e}")
        return ToolResult(success=True, output=json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))

    def _count(self, text: str, numeric: bool = False, **kwargs) -> ToolResult:
        if numeric:
            numbers = len(re.findall(r"-?\d+(?:\.\d+)?", text))
            return ToolResult(success=True, output=str(numbers), metadata={"numbers": numbers})
        data = {
            "characters": len(text),
            "characters_no_spaces": len(re.sub(r"\s", "", text)),
            "words": len(re.findall(r"\S+", text)),
            "lines": len(text.splitlines()) if text else 0,
            "cjk_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
        }
        lines = [f"{k}: {v}" for k, v in data.items()]
        return ToolResult(success=True, output="\n".join(lines), metadata=data)

    def _unique(self, text: str, separator: str = "\n", descending: bool = False,
                **kwargs) -> ToolResult:
        sep = separator if separator else "\n"
        items = text.split(sep) if sep != "\n" else text.splitlines()
        seen = set()
        result: List[str] = []
        for item in items:
            key = item.strip()
            if key not in seen:
                seen.add(key)
                result.append(item)
        if descending:
            result = sorted(result, reverse=True)
        return ToolResult(success=True, output=sep.join(result), metadata={"count": len(result)})

    def _sort_lines(self, text: str, numeric: bool = False, descending: bool = False,
                    **kwargs) -> ToolResult:
        lines = text.splitlines()
        if numeric:
            def _key(line: str):
                match = re.search(r"-?\d+(?:\.\d+)?", line)
                return float(match.group()) if match else 0.0
            lines = sorted(lines, key=_key, reverse=descending)
        else:
            lines = sorted(lines, reverse=descending)
        return ToolResult(success=True, output="\n".join(lines), metadata={"count": len(lines)})

    def _regex_extract(self, text: str, pattern: str = "", group: int = 0, **kwargs) -> ToolResult:
        if not pattern:
            return ToolResult(success=False, output="", error="pattern is required")
        try:
            matches = re.findall(pattern, text, flags=re.MULTILINE)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")
        flat: List[Any] = []
        for m in matches:
            if isinstance(m, tuple):
                try:
                    flat.append(m[int(group)])
                except (IndexError, ValueError):
                    flat.append(m[0] if m else "")
            else:
                flat.append(m)
        output = "\n".join(str(x) for x in flat)
        return ToolResult(success=True, output=output,
                          metadata={"count": len(flat), "matches": flat})

    def _regex_replace(self, text: str, pattern: str = "", replacement: str = "",
                       **kwargs) -> ToolResult:
        if not pattern:
            return ToolResult(success=False, output="", error="pattern is required")
        try:
            result = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")
        return ToolResult(success=True, output=result)

    def _trim(self, text: str, **kwargs) -> ToolResult:
        lines = [line.rstrip() for line in text.splitlines()]
        return ToolResult(success=True, output="\n".join(lines).strip())


TOOLS_REGISTRY.register(TextTransformTool())
