"""PDF generation tool.

把 Markdown/纯文本内容渲染为 PDF 文件, 支持中文字体 (macOS Songti 提取为 TTF).
不依赖 LaTeX: 用 reportlab + Platypus 直接绘制, 支持标题/段落/列表/表格/代码块。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult

# 中文字体源 (macOS)。找不到时回退系统内置的 Courier/Helvetica。
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
_FONT_CACHE: Optional[str] = None


def _resolve_cjk_font() -> Optional[str]:
    """找到可用的中文字体 (从 TTC 提取第一个子字体到缓存目录)."""
    global _FONT_CACHE
    if _FONT_CACHE and os.path.exists(_FONT_CACHE):
        return _FONT_CACHE
    cache_dir = Path(tempfile.gettempdir()) / "lv_pdf_fonts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ttc in _FONT_CANDIDATES:
        if not os.path.exists(ttc):
            continue
        try:
            from fontTools.ttLib import TTFont
            name = Path(ttc).stem
            ttf_path = cache_dir / f"{name}_0.ttf"
            if not ttf_path.exists():
                f = TTFont(ttc, fontNumber=0)
                f.save(str(ttf_path))
            _FONT_CACHE = str(ttf_path)
            return _FONT_CACHE
        except Exception:
            continue
    return None


class PdfTool(BaseTool):
    """把文本/Markdown 内容渲染为 PDF 文件。"""

    name = "pdf_tool"
    description = (
        "Generate a PDF file from text/markdown content. "
        "Use when the user asks for a PDF report, a PDF file, or to convert content to PDF. "
        "Parameters: content (text to render), path (optional output .pdf path)."
    )
    parameters = {
        "content": {
            "type": "string",
            "description": "要渲染成 PDF 的文本/Markdown 内容",
            "required": True,
        },
        "path": {
            "type": "string",
            "description": "输出 PDF 文件路径 (默认: 当前目录/report_<timestamp>.pdf)",
            "required": False,
        },
    }

    def execute(self, content: str = "", path: str = "", **kwargs: Any) -> ToolResult:
        content = (content or "").strip()
        if not content:
            return ToolResult(success=False, output="", error="pdf_tool: content is empty")

        try:
            out_path = self._render_pdf(content, path)
            return ToolResult(
                success=True,
                output=f"PDF 已生成: {out_path}\n文件大小: {Path(out_path).stat().st_size} bytes",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"pdf_tool failed: {e}")

    # ---------- rendering ----------

    def _render_pdf(self, content: str, path: str = "") -> str:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        if not path:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"report_{ts}.pdf"
        path = os.path.abspath(os.path.expanduser(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        font_name = "Helvetica"
        cjk = _resolve_cjk_font()
        if cjk:
            try:
                pdfmetrics.registerFont(TTFont("CJKFont", cjk))
                font_name = "CJKFont"
            except Exception:
                pass

        title_style = ParagraphStyle("title", fontName=font_name, fontSize=18, leading=24, spaceAfter=10)
        h2_style = ParagraphStyle("h2", fontName=font_name, fontSize=14, leading=20, spaceBefore=8, spaceAfter=6)
        body_style = ParagraphStyle("body", fontName=font_name, fontSize=11, leading=17, spaceAfter=6)

        doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                                topMargin=18 * mm, bottomMargin=18 * mm)
        story: List[Any] = []

        def _escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                story.append(Spacer(1, 4))
                continue
            if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
                text = _escape(line.lstrip("#").strip())
                style = title_style if line.startswith("# ") else h2_style
                story.append(Paragraph(text, style))
            elif line.startswith("- ") or line.startswith("* "):
                story.append(Paragraph(f"• {_escape(line[2:])}", body_style))
            elif line.startswith("| ") and "|" in line[2:]:
                # 简单表格: 连续 | 行视为表格
                cells = [c.strip() for c in line.strip("|").split("|")]
                story.append(Paragraph(" | ".join(_escape(c) for c in cells), body_style))
            elif line.startswith("```"):
                story.append(Paragraph(_escape(line), body_style))
            else:
                story.append(Paragraph(_escape(line), body_style))

        doc.build(story)
        return path