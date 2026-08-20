"""
Stream adapters for OpenMythos CLI output.

Provides multiple output modes:
- rich: TTY-friendly spinner, live reasoning view, syntax highlighting (default in TTY)
- plain: structured plain-text log lines without ANSI escape codes (default in non-TTY)
- json:  JSON Lines output for programmatic consumption / logging

The adapter interface mirrors the stream_callback kinds used by the agent:
  status, reasoning, tool_call, tool_result, content.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import textwrap
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\u24C2-\u24FF"
    "\U0001F200-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u20E3"
    "\u2460-\u24FF"
    "\u2600-\u26FF"
    "\U0001F780-\U0001F7FF"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "]+",
    flags=re.UNICODE,
)


def clean_runtime_text(text: str) -> str:
    """Remove emoji and collapse whitespace for compact status lines."""
    text = _EMOJI_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_content_text(text: str) -> str:
    """Remove emoji from content but keep whitespace."""
    return _EMOJI_RE.sub("", text)


def term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def fit_line(text: str, width: Optional[int] = None) -> str:
    if width is None:
        width = term_width()
    text = text.rstrip()
    if not text:
        return ""
    return text[: width - 1] + "…" if len(text) > width else text


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class StreamAdapter(ABC):
    """Abstract interface for rendering agent stream events."""

    @abstractmethod
    def emit_status(self, text: str) -> None:
        """Agent status update (e.g. 'task: ...', 'turbo · first turn')."""
        ...

    @abstractmethod
    def emit_reasoning(self, text: str) -> None:
        """Raw reasoning/thinking token."""
        ...

    @abstractmethod
    def emit_tool_call(self, text: str) -> None:
        """Tool call description."""
        ...

    @abstractmethod
    def emit_tool_result(self, text: str) -> None:
        """Tool result summary."""
        ...

    @abstractmethod
    def emit_content(self, text: str) -> None:
        """Final answer content token."""
        ...

    def finalize(self) -> None:
        """Called once after the run completes."""
        pass


# ---------------------------------------------------------------------------
# Plain adapter: structured text lines, no ANSI, pipe-friendly
# ---------------------------------------------------------------------------

class PlainStreamAdapter(StreamAdapter):
    """Structured plain-text output for non-TTY / log files / pipes."""

    def __init__(self, timestamp: bool = True, max_tool_result_len: int = 360):
        self.timestamp = timestamp
        self.max_tool_result_len = max_tool_result_len
        self._tool_header_printed = False
        self._content_started = False
        self._has_content = False
        self._last_tool_call = ""

    def _prefix(self, level: str) -> str:
        if self.timestamp:
            return f"{datetime.now().isoformat(timespec='seconds')}Z [{level.upper()}] "
        return f"[{level.upper()}] "

    def emit_status(self, text: str) -> None:
        line = clean_runtime_text(text)
        if line:
            print(f"{self._prefix('status')}{line}", flush=True)

    def emit_reasoning(self, text: str) -> None:
        # In plain mode we intentionally keep reasoning compact: only emit
        # complete sentences or significant fragments to avoid spamming logs.
        cleaned = clean_runtime_text(text)
        if cleaned and (cleaned.endswith((".", "。", "!", "?", "？", "！", ":", "：")) or len(cleaned) > 40):
            print(f"{self._prefix('reasoning')}{fit_line(cleaned)}", flush=True)

    def emit_tool_call(self, text: str) -> None:
        if not self._tool_header_printed:
            print(f"{self._prefix('tool_call')}Tools", flush=True)
            self._tool_header_printed = True
        line = clean_runtime_text(text)
        if line:
            self._last_tool_call = line
            print(f"{self._prefix('tool_call')}  -> {line}", flush=True)

    def emit_tool_result(self, text: str) -> None:
        line = clean_runtime_text(text)
        if not line:
            return
        low = line.lower()
        is_warn = (
            low.startswith(("tool error", "tool execution error", "timed out", "system stop"))
            or "blocked for safety" in low or "denied" in low or "not allowed" in low
        )
        level = "warn" if is_warn else "tool_result"
        # Summarize large tool outputs instead of flooding non-TTY logs.
        if len(line) > self.max_tool_result_len and not low.startswith(("tool error", "tool execution error")):
            summary = self._summarize_tool_result(line)
            line = f"{summary} ({len(line)} chars; full result in prompt)"
        print(f"{self._prefix(level)}  <- {fit_line(line)}", flush=True)

    def _summarize_tool_result(self, text: str) -> str:
        """Return a short summary for large tool outputs."""
        # Try to extract a web_search result count.
        if self._last_tool_call.startswith("web_search") or text.strip().startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return f"web_search returned {len(parsed)} results"
            except Exception:
                pass
        return text[: self.max_tool_result_len] + "..."

    def emit_content(self, text: str) -> None:
        if not self._content_started:
            print(f"{self._prefix('content')}", flush=True)
            self._content_started = True
        self._has_content = True
        print(clean_content_text(text), end="", flush=True)

    def finalize(self) -> None:
        if self._content_started:
            print("", flush=True)


# ---------------------------------------------------------------------------
# JSON adapter: JSON Lines for programmatic consumption
# ---------------------------------------------------------------------------

class JsonStreamAdapter(StreamAdapter):
    """JSON Lines output. Each stream event is one JSON object."""

    def __init__(self, compact: bool = True, max_tool_result_len: int = 500):
        self.compact = compact
        self.max_tool_result_len = max_tool_result_len
        self._content_buffer = []
        self._last_content_flush = time.time()
        self._buffer_lock = threading.Lock()
        self._has_content = False
        self._last_tool_call = ""

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
            "level": "INFO",
            "kind": kind,
            **payload,
        }
        if self.compact:
            print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
        else:
            print(json.dumps(event, ensure_ascii=False, indent=2), flush=True)

    def emit_status(self, text: str) -> None:
        self._flush_content()
        self._emit("status", {"text": clean_runtime_text(text)})

    def emit_reasoning(self, text: str) -> None:
        cleaned = clean_runtime_text(text)
        if cleaned:
            self._emit("reasoning", {"text": cleaned})

    def emit_tool_call(self, text: str) -> None:
        self._flush_content()
        cleaned = clean_runtime_text(text)
        self._last_tool_call = cleaned
        self._emit("tool_call", {"text": cleaned})

    def emit_tool_result(self, text: str) -> None:
        self._flush_content()
        cleaned = clean_runtime_text(text)
        success = not any(cleaned.lower().startswith(p) for p in ("tool error", "tool execution error", "timed out", "system stop"))
        # Avoid dumping huge JSON blobs into JSON Lines output.
        if len(cleaned) > self.max_tool_result_len and success:
            cleaned = self._summarize_tool_result(cleaned)
        self._emit("tool_result", {"text": fit_line(cleaned), "success": success})

    def _summarize_tool_result(self, text: str) -> str:
        if self._last_tool_call.startswith("web_search") or text.strip().startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return f"web_search returned {len(parsed)} results"
            except Exception:
                pass
        return text[: self.max_tool_result_len] + "..."

    def emit_content(self, text: str) -> None:
        self._has_content = True
        with self._buffer_lock:
            self._content_buffer.append(clean_content_text(text) if text else "")
        now = time.time()
        if now - self._last_content_flush > 0.2:
            self._flush_content()

    def _flush_content(self) -> None:
        with self._buffer_lock:
            if not self._content_buffer:
                return
            text = "".join(self._content_buffer)
            self._content_buffer.clear()
        if text:
            self._emit("content", {"text": text})
        self._last_content_flush = time.time()

    def finalize(self) -> None:
        self._flush_content()


# ---------------------------------------------------------------------------
# Rich adapter: TTY spinner, live reasoning view, syntax highlighting
# ---------------------------------------------------------------------------

class RichStreamAdapter(StreamAdapter):
    """Rich TTY output with spinner, live reasoning panel and code highlighting."""

    def __init__(self, console: Any, max_reasoning_lines: int = 3, show_thinking: bool = True):
        self.console = console
        self.max_reasoning_lines = max_reasoning_lines
        self.show_thinking = show_thinking

        from rich.live import Live
        from rich.text import Text
        self.Live = Live
        self.Text = Text

        self._live: Optional[Any] = None
        self._reasoning_buffer: list[str] = []
        self._has_reasoning = False
        self._has_content = False
        self._tool_header_printed = False
        self._last_tool_name = ""
        self._total_tokens = 0
        self._start_time = time.time()
        self._last_render = 0.0
        self._render_interval = 0.08
        self._render_lock = threading.Lock()
        self._stop_spinner_event = threading.Event()
        self._spinner_thread: Optional[threading.Thread] = None
        register_active_adapter(self)

        self._current_action = "thinking"
        self._action_since = time.time()
        self._status_text = ""

        self._highlighter = self._StreamHighlighter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_action(self, action: str) -> None:
        self._current_action = action
        self._action_since = time.time()

    def _set_status(self, text: str) -> None:
        self._status_text = text

    def _fmt_tokens(self, n: int) -> str:
        if n >= 1000:
            return f"{n // 1000}.{n % 1000 // 100}k"
        return str(n)

    def _action_label(self) -> str:
        labels = {
            "thinking": "thinking",
            "tool": "executing tool",
            "result": "processing result",
            "answering": "generating answer",
        }
        elapsed = time.time() - self._action_since
        label = labels.get(self._current_action, self._current_action)
        # status 可能已含 "thinking (step x/y)", 避免拼出重复的 "thinking · thinking"
        status_text = (self._status_text or "").strip()
        parts = []
        if status_text:
            if "thinking" in status_text.lower() and self._current_action == "thinking":
                parts.append(status_text)  # status 已含 thinking, 不再重复 label
            else:
                parts.append(status_text)
                parts.append(label)
        else:
            parts.append(label)
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = spinner_frames[int(elapsed * 10) % len(spinner_frames)]
        meta = f"{elapsed:.1f}s · {self._fmt_tokens(self._total_tokens)} tokens"
        # 工具执行/结果处理时: 用细绿色进度条替代 spinner, 直观提示"正在处理"
        # 避免长任务(如扫描大量文件)时用户误以为卡住。
        if self._current_action in ("tool", "result"):
            bar = self._thin_green_bar(elapsed)
            return f"\033[32m{bar}\033[0m \033[2m{' · '.join(parts)} · {meta}\033[0m"
        return f"\033[2m{spinner} {' · '.join(parts)} · {meta}\033[0m"

    @staticmethod
    def _thin_green_bar(elapsed: float, width: int = 24) -> str:
        """细绿色进度条: 一条细线 + 移动的绿色亮点, 表示正在处理."""
        # 亮点从左到右循环移动
        pos = int(elapsed * 4) % width
        track = ["─"] * width
        track[pos] = "●"
        # 亮点后加渐变淡绿尾迹(细)
        for i in range(1, min(4, width - pos)):
            track[pos + i] = "·"
        return "".join(track)

    def _fold_reasoning(self) -> list[str]:
        text = clean_runtime_text("".join(self._reasoning_buffer))
        width = max(term_width() - 4, 20)
        max_chars = width * self.max_reasoning_lines
        if len(text) > max_chars:
            text = text[-max_chars:]
        wrapped = [text[i : i + width] for i in range(0, len(text), width)]
        wrapped = wrapped[-self.max_reasoning_lines :]
        while len(wrapped) < self.max_reasoning_lines:
            wrapped.append("")
        return wrapped

    def _build_thinking_text(self) -> Any:
        lines = self._fold_reasoning()
        text = self.Text()
        text.append(f"{self._action_label()}\n", style="dim italic")
        if lines:
            for line in lines:
                text.append(f" {line}\n", style="dim")
        else:
            text.append(" ...\n", style="dim")
        return text

    def _update_thinking(self, force: bool = False) -> None:
        with self._render_lock:
            now = time.time()
            if not force and now - self._last_render < self._render_interval:
                return
            self._last_render = now
            if self._live is not None:
                self._live.update(self._build_thinking_text())

    def _start_thinking(self) -> None:
        if self._live is not None:
            return
        self._live = self.Live(self._build_thinking_text(), console=self.console, refresh_per_second=12, transient=True)
        self._live.start()
        self._stop_spinner_event.clear()
        self._spinner_thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._spinner_thread.start()

    def _spinner_loop(self) -> None:
        while not self._stop_spinner_event.is_set():
            self._update_thinking()
            time.sleep(self._render_interval)

    def _stop_thinking(self) -> None:
        self._stop_spinner_event.set()
        if self._spinner_thread is not None and self._spinner_thread.is_alive():
            self._spinner_thread.join(timeout=0.5)
        with self._render_lock:
            if self._live is not None:
                self._live.stop()
                self._live = None

    def pause_for_input(self) -> None:
        """Stop the spinner cleanly so a blocking input() prompt is stable on screen."""
        self._stop_thinking()

    def resume_after_input(self) -> None:
        """Restart the spinner after input() returns, if a task is still in flight."""
        if self._live is None and not self._has_content:
            self._start_thinking()
            self._update_thinking(force=True)

    def _print_tool_call(self, text: str) -> None:
        self._stop_thinking()
        text = clean_runtime_text(text)
        # 提取工具名: 兼容 "bash_exec(...)" / "bash_exec: {...}" / "bash_exec {...}" 格式
        m = re.match(r'^\s*([a-zA-Z_][\w_]*)\s*[(:{=]', text)
        self._last_tool_name = m.group(1) if m else text.split("(", 1)[0].strip()[:20]
        if not self._tool_header_printed:
            print(f"\n\033[2m──── tools ────\033[0m", flush=True)
            self._tool_header_printed = True
        # 设计方案: pending 状态前缀 ◌(灰)
        print(f"  \033[90m◌\033[0m \033[2m{text}\033[0m", flush=True)

    def _print_tool_result(self, text: str) -> None:
        self._stop_thinking()
        low = text.lower()
        # 策略提示(非错误): 安全拦截/权限拒绝等 → 用黄色而非红色(优先判断, 可能带 "Tool error:" 前缀)
        is_policy = any(k in low for k in ("blocked for safety", "denied by", "harness denied",
                                             "denied by harness", "not allowed", "not in allowed",
                                             "permission denied", "forbidden"))
        # 真错误: 执行失败/超时/系统停止
        is_error = (not is_policy) and low.startswith(("tool execution error", "timed out", "system stop", "tool error:"))
        # 多行输出(代码执行/文件读取)以代码块样式展示, 保留行结构与缩进
        if "\n" in text or is_error or is_policy:
            self._print_block_result(text, is_error, policy=is_policy)
            return
        summary = clean_runtime_text(text)
        width = max(term_width() - 6, 20)
        if len(summary) > width:
            summary = summary[: width - 1] + "…"
        # 设计方案: 状态前缀(✓成功 / ✗错误 / ◐策略)
        if is_error:
            # 浅黄错误提示(替代刺眼红), 与块状错误一致
            prefix, color = "✗", "38;5;229"
        elif is_policy:
            prefix, color = "◐", "33"
        else:
            prefix, color = "✓", "90"
        print(f" \033[{color}m{prefix}\033[0m \033[{color}m{summary}\033[0m", flush=True)

    def _print_block_result(self, text: str, is_error: bool, policy: bool = False) -> None:
        """把多行工具输出(代码执行结果/文件内容)渲染成带边框的代码块.

        设计方案: 状态前缀(成功✓/错误✗/策略◐) + 长输出自动折叠(>6行).
        """
        width = max(term_width() - 8, 40)
        max_lines = 6  # 设计文档: 超过 6 行自动折叠
        name = self._last_tool_name or "exec"
        lines = text.rstrip().split("\n")
        if policy:
            # 策略提示(安全拦截/权限拒绝): 黄色边框+内容, 不是真正的错误
            border_fg, name_fg, text_fg, mark = "33", "33", "37", "◐"
        elif is_error:
            # 浅黄错误提示(替代刺眼红): 边框/工具名/文本统一浅黄, 保留 ✗ 标记
            border_fg, name_fg, text_fg, mark = "38;5;229", "38;5;229", "38;5;229", "✗"
        else:
            border_fg, name_fg, text_fg, mark = "2", "34", "37", "✓"
        border = "─" * width
        # 状态前缀 + 工具名
        print(f" \033[{border_fg}m╭─ \033[{border_fg}m{mark}\033[0m \033[{border_fg}m{name}\033[{border_fg}m {border}\033[0m")
        shown = lines[:max_lines]
        more = len(lines) - len(shown)
        # search/replace diff 标记着色: SEARCH 红, REPLACE 绿, 分隔线暗
        search_re = re.compile(r"^\s*<{6,}\s*SEARCH\s*$", re.IGNORECASE)
        replace_re = re.compile(r"^\s*>{6,}\s*REPLACE\s*$", re.IGNORECASE)
        divider_re = re.compile(r"^\s*={6,}\s*$")
        for ln in shown:
            if len(ln) > width:
                ln = ln[: width - 1] + "…"
            if search_re.search(ln):
                print(f" \033[1;31m│ {ln}\033[0m")
            elif replace_re.search(ln):
                print(f" \033[1;32m│ {ln}\033[0m")
            elif divider_re.search(ln):
                print(f" \033[2m│ {ln}\033[0m")
            else:
                print(f" \033[{text_fg}m│ {ln}\033[0m")
        if more > 0:
            print(f" \033[{text_fg}m│ …（其余 {more} 行已折叠）\033[0m")
        if policy:
            tail = " · blocked"
        elif is_error:
            tail = " · error"
        else:
            tail = ""
        print(f" \033[{border_fg}m╰─ {len(lines)} 行{tail}\033[0m")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def emit_status(self, text: str) -> None:
        self._set_status(text)
        # 任务开始时即启动思考动画(即使无 reasoning_content, 也有 spinner 反馈)
        if not self._has_content:
            self._set_action("thinking")
            self._start_thinking()
        self._update_thinking(force=True)
        # thinking 面板已停止/已开始输出内容后, 关键状态(loop 扩展/无进展停止等)
        # 不再显示——这里将其作为独立可见行持久化输出, 避免用户看不到"增加 loop"等提示。
        # 仅当 thinking 动画已停止(_live 已清理)时才直接 print, 避免破坏 Live 渲染。
        t = (text or "").strip()
        if self._live is None and t and (
            "↑ loop" in t or "无进展" in t or "增加思考预算" in t or "still progressing" in t
        ):
            print(f" \033[90m• {t}\033[0m", flush=True)

    def emit_reasoning(self, text: str) -> None:
        if not self.show_thinking or self._has_content:
            return
        self._has_reasoning = True
        self._reasoning_buffer.append(clean_runtime_text(text))
        self._set_action("thinking")
        self._start_thinking()
        self._update_thinking(force=True)

    def emit_tool_call(self, text: str) -> None:
        self._set_action("tool")
        self._start_thinking()
        self._print_tool_call(text)

    def emit_tool_result(self, text: str) -> None:
        self._set_action("result")
        self._print_tool_result(text)

    def emit_content(self, text: str) -> None:
        if not self._has_content:
            self._has_content = True
            self._stop_thinking()
            print()  # 内容开始前留一个空行, 与提示/思考区隔开
            self._set_action("answering")
        self._highlighter.feed(clean_content_text(text) if text else text)

    def add_tokens(self, tokens: int) -> None:
        self._total_tokens += tokens

    def finalize(self) -> None:
        self._stop_thinking()
        if self._has_content:
            self._highlighter.flush()
            print("\033[0m", flush=True)

    # ------------------------------------------------------------------
    # Nested highlighter
    # ------------------------------------------------------------------

    class _StreamHighlighter:
        # 精致柔和配色: 以标准 ANSI 色为主(终端主题决定明暗), 少量 256 色暖调
        # 做点缀, 既有清晰层次又不刺眼, 适合长时间阅读。
        CODE_BG = "\033[48;5;237m"
        CODE_FG = "\033[38;5;255m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        # 层级配色: 标题 > 章节 > 列表, 由强到弱
        # 注: 不使用红/黄(用户要求去掉), 错误与重点用加粗/青色区分
        H1 = "\033[1;36m"                # 一级标题: 加粗青
        H2 = "\033[1;34m"                # 二级标题: 加粗蓝
        SECTION = "\033[1;36m"           # 章节(一、/1.): 加粗青
        LIST_MARK = "\033[2;36m"         # 列表符号: 暗青
        INLINE = "\033[36m"              # 行内代码: 青色
        URL = "\033[36;4m"               # 链接: 青下划线
        PATH = "\033[34m"                # 文件/路径: 蓝
        NUM = "\033[36m"                 # 数字/统计: 青
        ERR = "\033[1m"                  # 错误/失败: 加粗(不再用红)
        KEY = "\033[1;36m"               # 重点词: 加粗青
        QUOTE = "\033[2;90m"             # 引用/弱化: 暗灰
        DIM = "\033[2m"
        FLUSH_AT_CHARS = 24
        FORCE_FLUSH_AT_CHARS = 48
        WORD_BOUNDARY = set(" \t,.;:!?，。；：！？")

        # 行内高亮: 数字/统计、链接、路径、错误、重点词、行内代码
        # 重点词需"前后非中文字符"边界, 避免匹配到普通词中间(如"搜索结果"里的"结果")
        _HL_KEY = re.compile(
            r"(?P<code>`[^`\n]+`)"
            r"|(?P<url>https?://[^\s\"']+|www\.[^\s\"']+)"
            r"|(?P<path>(?:\.{0,2}/|~/)[\w.\-]+(?:/[\w.\-]+)*)"
            r"|(?P<num>(?<![\w])\d+(?:\.\d+)?%?(?:ms|s|GB|MB|KB|TB|倍|个|次|元|年|天|秒|%|万|亿|亿年|次/秒|QPS)?(?![\w]))"
            r"|(?P<err>Traceback|timed out|timeout|Error|错误|失败|异常)"
            r"|(?P<key>(?<![\u4e00-\u9fff])(?:最终答案|答案|结论|建议|注意|警告|最优|推荐|收益|性能|成功率)(?![\u4e00-\u9fff]))"
        )

        # 章节标题行: 一、 / 二、 / (一) 等中文序号 → 加粗金
        _SECTION_RE = re.compile(r"^\s*[一二三四五六七八九十百]+、\s*(.+)$")
        # 列表项: - / * / • / 1. / 2. 等
        _LIST_RE = re.compile(r"^\s*(?P<mark>[-*•]|\d+[.、])\s+")

        def __init__(self):
            self.in_code_block = False
            self.code_lang = ""
            self.line_buffer = ""
            # Markdown 表格缓冲: 收集 `| ... |` 行, 结束后用细线框绘制
            self._table_rows: list[list[str]] = []
            self._in_table = False

        def feed(self, token: str) -> None:
            self.line_buffer += token
            while "\n" in self.line_buffer:
                line, self.line_buffer = self.line_buffer.split("\n", 1)
                self._emit_line(line)
            # 长行中间片段提前下发, 保证逐字流畅(片段已从 line_buffer 切出,
            # 后续 _emit_line 收到的 line 即未输出部分, 不会重复也不会丢失)
            buf_len = len(self.line_buffer)
            flush_at = 0
            if buf_len >= self.FLUSH_AT_CHARS and self.line_buffer[-1] in self.WORD_BOUNDARY:
                flush_at = buf_len
            elif buf_len >= self.FORCE_FLUSH_AT_CHARS:
                flush_at = self.FORCE_FLUSH_AT_CHARS
            if flush_at:
                self._emit_fragment(self.line_buffer[:flush_at])
                self.line_buffer = self.line_buffer[flush_at:]

        def flush(self) -> None:
            if self.line_buffer:
                self._emit_fragment(self.line_buffer)
                self.line_buffer = ""

        def _emit_fragment(self, text: str) -> None:
            """流式中间片段: 原样续行输出, 不加换行."""
            if not text:
                return
            if self.in_code_block:
                print(f"{self.CODE_BG}{self.CODE_FG}{text}{self.RESET}", end="", flush=True)
                return
            if text.startswith("Final Answer:"):
                text = text[len("Final Answer:"):].lstrip()
            print(self._style_inline(text), end="", flush=True)

        def _emit_line(self, line: str) -> None:
            """完整一行: 按 Markdown 风格美化后换行."""
            stripped = line.strip()
            if self.in_code_block:
                if stripped.startswith("```"):
                    print(f"{self.DIM}└── {self.code_lang or 'code'}{self.RESET}", flush=True)
                    self.in_code_block = False
                    self.code_lang = ""
                else:
                    if line:
                        print(f"{self.CODE_BG}{self.CODE_FG}{line.rstrip()}{self.RESET}", flush=True)
                return

            if stripped.startswith("```"):
                self.in_code_block = True
                self.code_lang = stripped[3:].strip()
                print(f"{self.DIM}┌── {self.code_lang or 'code'}{self.RESET}", flush=True)
                return

            # Markdown 表格: `| a | b |` 行收集, 遇到非表格行时绘制细线框
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                # 分隔行 `| --- | --- |` 跳过, 仅用于判断列数
                # 注意: 不能 flush 已缓冲的表头——表头、分隔行、数据行应作为一个表格整体缓冲
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                    return
                self._table_rows.append(cells)
                self._in_table = True
                return
            if self._in_table:
                self._flush_table()

            if not stripped:
                print(flush=True)  # 空行 = 段落分隔
                return

            if stripped.startswith("Thought:") or stripped.startswith("Action:"):
                return
            if stripped.startswith("Final Answer:"):
                lead = len(line) - len(line.lstrip())
                colon = line.find(":", lead)
                if colon != -1:
                    line = line[colon + 1:].lstrip()
                    stripped = line.strip()

            # 引用 / 弱化行(> 开头) → 暗灰斜体感
            if stripped.startswith(">"):
                print(f"{self.QUOTE}{line.rstrip()}{self.RESET}", flush=True)
                return

            # Markdown 标题: # 一级, ##/### 二级 (隐藏 # 记号, 只显示文字)
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("#").strip()
                if level >= 2:
                    print(f"\n{self.H2}{title}{self.RESET}", flush=True)
                else:
                    print(f"\n{self.H1}{title}{self.RESET}", flush=True)
                return

            # 章节标题行(一、数据极简主义) → 加粗金
            m = self._SECTION_RE.match(stripped)
            if m and len(stripped) <= 40:
                print(f"\n{self.SECTION}{stripped}{self.RESET}", flush=True)
                return

            # 列表项: 符号用暗青, 内容走行内高亮(不再整行刷青)
            m = self._LIST_RE.match(line)
            if m:
                mark = m.group("mark")
                rest = line[m.end():]
                print(f"{self.LIST_MARK}{mark} {self.RESET}{self._style_inline(rest.rstrip())}", flush=True)
                return

            # 普通段落: 行内高亮
            print(self._style_inline(line.rstrip()), flush=True)

        def _flush_table(self) -> None:
            """用细线框绘制已缓冲的 Markdown 表格."""
            if not self._table_rows:
                return
            rows = self._table_rows
            self._table_rows = []
            self._in_table = False

            # 第一行是表头; 若第二行也是内容(无分隔行), 仍视为表头
            header = rows[0]
            body = rows[1:] if len(rows) > 1 else []

            # 列宽 = 各列内容最大宽度(宽字符按 2 计)
            def disp_w(s: str) -> int:
                w = 0
                for ch in s:
                    w += 2 if ord(ch) > 0x2E7F else 1
                return w

            def pad(s: str, width: int) -> str:
                pad_w = width - disp_w(s)
                return s + " " * max(0, pad_w)

            n_cols = max(len(header), *(len(r) for r in body)) if body else len(header)
            widths = []
            for c in range(n_cols):
                vals = []
                if c < len(header):
                    vals.append(header[c])
                for r in body:
                    if c < len(r):
                        vals.append(r[c])
                widths.append(max((disp_w(v) for v in vals), default=1) + 2)

            def border(left, mid, right) -> str:
                return left + mid.join("─" * w for w in widths) + right

            # 用细线绘制: ┌─┬─┐ / ├─┼─┤ / └─┴─┘
            print(f"{self.DIM}{border('┌', '┬', '┐')}{self.RESET}", flush=True)
            for idx, row in enumerate([header] + body):
                cells = []
                for c in range(n_cols):
                    val = row[c] if c < len(row) else ""
                    cells.append(pad(val, widths[c]))
                line = "│".join(f"{pad(c, widths[i])}" for i, c in enumerate(cells))
                if idx == 0:
                    print(f"{self.DIM}│{self.RESET}{self.BOLD}{line}{self.RESET}{self.DIM}│{self.RESET}", flush=True)
                    print(f"{self.DIM}{border('├', '┼', '┤')}{self.RESET}", flush=True)
                else:
                    styled = "│".join(self._style_inline(pad(c, widths[i])) for i, c in enumerate(cells))
                    print(f"{self.DIM}│{self.RESET}{styled}{self.DIM}│{self.RESET}", flush=True)
            print(f"{self.DIM}{border('└', '┴', '┘')}{self.RESET}", flush=True)

        def _style_inline(self, text: str) -> str:
            """行内重点着色: 数字/链接/路径/错误/重点词/行内代码/加粗."""
            def repl(m):
                if m.group("code"):
                    return f"{self.INLINE}{m.group('code')}{self.RESET}"
                if m.group("url"):
                    return f"{self.URL}{m.group('url')}{self.RESET}"
                if m.group("path"):
                    return f"{self.PATH}{m.group('path')}{self.RESET}"
                if m.group("num"):
                    return f"{self.NUM}{m.group('num')}{self.RESET}"
                if m.group("err"):
                    return f"{self.ERR}{m.group('err')}{self.RESET}"
                if m.group("key"):
                    return f"{self.KEY}{m.group('key')}{self.RESET}"
                return m.group(0)

            text = self._HL_KEY.sub(repl, text)
            # **加粗** / __加粗__ 成对处理
            text = re.sub(
                r"\*\*(?P<b>[^*\n]+)\*\*|__(?P<u>[^_\n]+)__",
                lambda m: f"{self.BOLD}{(m.group('b') or m.group('u'))}{self.RESET}",
                text,
            )
            # 清除流式分片导致的未配对 ** / __ 残片
            text = text.replace("**", "").replace("__", "")
            return text


# ---------------------------------------------------------------------------
# 非流式 markdown 高亮(供最终答案兜底打印复用, 与流式高亮一致)
# ---------------------------------------------------------------------------

def render_markdown_rich(text: str) -> str:
    """将 markdown 文本渲染为带 ANSI 颜色的字符串(非流式).

    复用 RichStreamAdapter._StreamHighlighter 的着色常量, 保证与流式输出一致:
    标题/章节/列表/代码块/行内高亮(路径、链接、数字、重点词)全部覆盖。
    返回带色字符串, 由调用方 print。
    """
    if not text:
        return text
    H = RichStreamAdapter._StreamHighlighter
    _HL_KEY = H._HL_KEY
    _SECTION_RE = H._SECTION_RE
    _LIST_RE = H._LIST_RE
    lines_out = []
    in_code = False
    code_lang = ""
    for raw in text.split("\n"):
        line = raw
        stripped = line.strip()
        if in_code:
            if stripped.startswith("```"):
                in_code = False
                lines_out.append(f"{H.DIM}└── {code_lang or 'code'}{H.RESET}")
                code_lang = ""
            else:
                lines_out.append(f"{H.CODE_BG}{H.CODE_FG}{line.rstrip()}{H.RESET}")
            continue
        if stripped.startswith("```"):
            in_code = True
            code_lang = stripped[3:].strip()
            lines_out.append(f"{H.DIM}┌── {code_lang or 'code'}{H.RESET}")
            continue
        if not stripped:
            lines_out.append("")
            continue
        if stripped.startswith(">"):
            lines_out.append(f"{H.QUOTE}{line.rstrip()}{H.RESET}")
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            style = H.H2 if level >= 2 else H.H1
            lines_out.append(f"{style}{title}{H.RESET}")
            continue
        m = _SECTION_RE.match(stripped)
        if m and len(stripped) <= 40:
            lines_out.append(f"{H.SECTION}{stripped}{H.RESET}")
            continue
        m = _LIST_RE.match(line)
        if m:
            mark = m.group("mark")
            rest = line[m.end():]
            lines_out.append(f"{H.LIST_MARK}{mark} {H.RESET}{_style_inline_hl(H, rest.rstrip())}")
            continue
        lines_out.append(_style_inline_hl(H, line.rstrip()))
    return "\n".join(lines_out)


def _style_inline_hl(H, text: str) -> str:
    """行内高亮: 对文本中的路径/链接/数字/重点词等上色."""
    if not text:
        return text
    parts = []
    pos = 0
    for m in H._HL_KEY.finditer(text):
        if m.start() > pos:
            parts.append(text[pos:m.start()])
        kind = m.lastgroup
        val = m.group(0)
        if kind == "code":
            parts.append(f"{H.INLINE}{val}{H.RESET}")
        elif kind == "url":
            parts.append(f"{H.URL}{val}{H.RESET}")
        elif kind == "path":
            parts.append(f"{H.PATH}{val}{H.RESET}")
        elif kind == "num":
            parts.append(f"{H.NUM}{val}{H.RESET}")
        elif kind == "err":
            parts.append(f"{H.ERR}{val}{H.RESET}")
        elif kind == "key":
            parts.append(f"{H.KEY}{val}{H.RESET}")
        else:
            parts.append(val)
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return "".join(parts)



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ACTIVE_RICH_ADAPTER: Optional["RichStreamAdapter"] = None


def register_active_adapter(adapter: "RichStreamAdapter") -> None:
    """Register the currently-used Rich adapter for global pause/resume."""
    global _ACTIVE_RICH_ADAPTER
    _ACTIVE_RICH_ADAPTER = adapter


def pause_active_spinner() -> None:
    """Pause the active rich spinner so raw input() won't race the Live redraw."""
    if _ACTIVE_RICH_ADAPTER is not None:
        _ACTIVE_RICH_ADAPTER.pause_for_input()


def resume_active_spinner() -> None:
    """Resume the active rich spinner after input() returns."""
    if _ACTIVE_RICH_ADAPTER is not None:
        _ACTIVE_RICH_ADAPTER.resume_after_input()


def select_stream_adapter(
    mode: Optional[str] = None,
    is_tty: Optional[bool] = None,
    console: Optional[Any] = None,
) -> StreamAdapter:
    """Select the appropriate stream adapter.

    Priority:
      1. explicit mode argument ("rich", "plain", "json")
      2. OPENMYTHOS_OUTPUT environment variable
      3. TTY detection
    """
    if mode is None:
        mode = os.environ.get("OPENMYTHOS_OUTPUT", "auto").lower()

    if is_tty is None:
        is_tty = sys.stdout.isatty()

    if mode == "json":
        return JsonStreamAdapter()
    if mode == "plain":
        return PlainStreamAdapter()
    if mode == "rich":
        return RichStreamAdapter(console=console)

    # auto: rich for TTY, plain for non-TTY
    if is_tty:
        return RichStreamAdapter(console=console)
    return PlainStreamAdapter()


