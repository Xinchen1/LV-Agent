"""Startup banner rendering for Lv Agent.

Renders the braille portrait and brand text block. This module is self-contained
so that the legacy SuperAgent class can eventually delegate to it without
circular imports.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .renderer import Renderer


def _portrait_path() -> Path:
    """Return the path to the bundled portrait asset."""
    # banner.py lives in agent_project/ui/; project assets are at repo root.
    return Path(__file__).parent.parent.parent / "assets" / "portrait.png"


def _otsu_threshold(pixels: bytearray, total: int) -> int:
    """Pure-python Otsu threshold over a grayscale byte buffer."""
    hist = [0] * 256
    for p in pixels:
        hist[p] += 1
    sum_total = sum(i * hist[i] for i in range(256))
    wB = sumB = 0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold


def render_portrait(width_chars: Optional[int] = None) -> str:
    """Render the bundled portrait as terminal braille pixel art."""
    path = _portrait_path()
    if not path.exists():
        return ""
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except Exception:
        return ""

    try:
        if width_chars is None:
            try:
                term_cols = shutil.get_terminal_size().columns
                width_chars = max(60, min(140, term_cols - 8))
            except Exception:
                width_chars = 80

        img = Image.open(path).convert("L")
        # Higher contrast + brighter gamma: paper background becomes pure white,
        # pencil strokes stay dark and crisp -> more white area, cleaner mosaic.
        img = ImageOps.autocontrast(img, cutoff=1.0)
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = img.point(lambda p: int(255 if p > 150 else (p / 150.0) * 255))
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = ImageEnhance.Sharpness(img).enhance(2.5)

        w, h = img.size
        # Keep original height (17 rows): each cell is 2x4 px, chunky mosaic blocks.
        target_w = width_chars * 2
        target_h = max(4, int(target_w * h / w / 4) * 4)
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        pixels = bytearray(img.tobytes())

        # Threshold: near-white background stays blank (more white), darker
        # strokes become block characters.
        thresh = _otsu_threshold(pixels, len(pixels))
        sorted_px = sorted(pixels)
        pct = sorted_px[int(len(sorted_px) * 0.65)]
        thresh = int(max(thresh, pct) * 0.92)

        lines = []
        for y in range(0, target_h, 4):
            line = ""
            for x in range(0, target_w, 2):
                byte_val = 0
                block_sum = 0
                for dy in range(4):
                    for dx in range(2):
                        px = pixels[(y + dy) * target_w + (x + dx)]
                        block_sum += px
                        if px < thresh:
                            bit = dy + dx * 3 if dy < 3 else 6 + dx
                            byte_val |= 1 << bit
                if byte_val == 0:
                    # All-white 2x4 block -> blank space (inherits terminal bg).
                    line += " "
                    continue
                avg = block_sum // 8
                gray = 236 + min(19, (255 - avg) * 20 // 255)
                line += f"\033[38;5;{gray}m{chr(0x2800 + byte_val)}\033[0m"
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        return ""


def _brand_text(renderer: Renderer) -> str:
    """Return the right-hand brand text block as styled lines."""
    r = renderer
    lines = [
        "",
        r.themed("Lv agent", "brand"),
        r.themed("Lux Vita · 光与生命", "brand-dim"),
        r.themed("by cleveris research", "muted"),
        "",
        r.themed("Deep thinking, real tools.", "brand"),
        r.themed("循环深度推理 · 工具驱动 · 自我学习", "ink"),
        r.themed("Recurrent deep reasoning · tool-driven · self-learning", "muted"),
        "",
        r.themed("Capabilities", "brand"),
        r.themed("  · 多轮深度推理，拆解复杂问题", "ink"),
        r.themed("  · 调用真实工具，获取实时信息", "ink"),
        r.themed("  · 持续自我学习，越用越懂你", "ink"),
        "",
        r.themed("Mission", "brand"),
        r.themed("  · 以开源人工智能为核心", "muted"),
        r.themed("  · 让每个人都能用上最好的 AI", "muted"),
        r.themed("  · 实现智能平权", "muted"),
        "",
    ]
    return "\n".join(lines)


def render_banner(
    renderer: Renderer,
    portrait_width: int = 34,
    show_minimal: bool = False,
) -> str:
    """Render the full startup banner as a multi-line string.

    Args:
        renderer: The active Renderer instance.
        portrait_width: Width of the braille portrait in character columns.
        show_minimal: If True, render a text-only banner (for non-TTY or minimal theme).
    """
    if show_minimal or not renderer.theme.supports_color:
        rule = "-" * 40
        return (
            "Lv agent · Lux Vita (光与生命)\n"
            f"{rule}\n"
            "以开源人工智能为核心，让每一个人都能用上最好的人工智能，实现智能平权\n"
            "Open-source AI at the core — great AI for everyone, intelligence for all"
        )

    portrait = render_portrait(width_chars=portrait_width)
    text_block = _brand_text(renderer)

    if not portrait:
        return text_block

    p_lines = portrait.splitlines()
    r_lines = text_block.splitlines()
    max_lines = max(len(p_lines), len(r_lines))
    p_lines += [""] * (max_lines - len(p_lines))
    r_lines += [""] * (max_lines - len(r_lines))

    result: list[str] = []
    for p_line, r_line in zip(p_lines, r_lines):
        result.append(f" {p_line}   {r_line}")
    return "\n".join(result)


def render_system_status(
    renderer: Renderer,
    backend: str,
    model: str,
    tools: int,
    loops: int,
) -> str:
    """Render the one-line system status shown below the banner."""
    sep = renderer.muted("·")
    parts = [
        f"{renderer.muted('backend')} {renderer.ink(backend)}",
        f"{renderer.muted('model')} {renderer.ink(model)}",
        f"{renderer.muted('tools')} {renderer.ink(str(tools))}",
        f"{renderer.muted('loops')} {renderer.ink(str(loops))}",
    ]
    return f" {sep} ".join(parts)
