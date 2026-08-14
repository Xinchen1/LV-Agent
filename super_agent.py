#!/usr/bin/env python3
"""
Super Agent CLI - OpenMythos少年头像版启动器

启动时优先渲染少年头像像素画，并保留思考动画。
"""

import sys
import os
import re
import shutil
import subprocess
import importlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env if present.
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

# 自动补装 Pillow，保证头像渲染依赖可用。
try:
    importlib.import_module("PIL")
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "-q"])
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent / 'agent_project'))

from rich.console import Console
from rich.prompt import Prompt

from agent_project.stream_adapters import (
    PlainStreamAdapter,
    JsonStreamAdapter,
    RichStreamAdapter,
    select_stream_adapter,
)
from agent_project.terminal import style as _style
from agent_project.terminal import token as _token, set_theme as _set_theme, active_theme as _active_theme
from agent_project.ui import StatusBar

console = Console()

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


def _clean_content_text(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def _portrait_path() -> Path:
    return Path(__file__).parent / "assets" / "portrait.png"


def _otsu_threshold(pixels: "bytearray", total: int) -> int:
    """Pure-python Otsu threshold over a grayscale byte buffer.

    Finds the intensity that maximizes between-class variance, so the white
    sketch background is separated from the pencil strokes automatically.
    """
    hist = [0] * 256
    for p in pixels:
        hist[p] += 1
    sum_total = sum(i * hist[i] for i in range(256))
    wB = 0
    sumB = 0
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


def _render_portrait(width_chars: Optional[int] = None) -> str:
    """Render the bundled portrait as terminal braille pixel art.

    The source is a pencil sketch on white paper, so a fixed threshold either
    loses faint strokes or drowns the background in noise. Instead we stretch
    the histogram, gently smooth + sharpen so the hatching stays coherent, then
    pick an adaptive threshold (Otsu blended with a dark-side percentile) per
    render. The result keeps the silhouette crisp at small terminal sizes.
    """
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
        # Stretch histogram to use the full 0-255 range.
        img = ImageOps.autocontrast(img, cutoff=1.0)
        # Higher contrast + brighter gamma: paper background becomes pure white,
        # pencil strokes stay dark and crisp -> more white area, cleaner mosaic.
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = img.point(lambda p: int(255 if p > 150 else (p / 150.0) * 255))
        # Smooth noise, then re-sharpen so strokes connect into chunky blocks.
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = ImageEnhance.Sharpness(img).enhance(2.5)

        w, h = img.size
        # Keep the original height (17 rows): each character cell is 2x4 pixels,
        # but rendered with chunky mosaic blocks instead of braille dots.
        target_w = width_chars * 2
        target_h = max(4, int(target_w * h / w / 4) * 4)
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        pixels = bytearray(img.tobytes())

        # Threshold: near-white background stays blank (more white), darker
        # strokes become block characters. Pick a higher percentile so more of
        # the paper reads as white.
        thresh = _otsu_threshold(pixels, len(pixels))
        sorted_px = sorted(pixels)
        pct = sorted_px[int(len(sorted_px) * 0.65)]
        thresh = int(max(thresh, pct) * 0.92)

        # Braille dots: each dot is 1 pixel, so the mosaic blocks are ~3x smaller
        # than the chunky ▀▄█ blocks. White background stays blank (more white).
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


class SuperAgentCLI:
    def __init__(self):
        self.config_path = Path(__file__).parent / 'config.yaml'
        self.config = None
        self.agent = None
        self._tg_process = None
        self.harness_mode = False
        self._running = False
        self._last_activity = time.time()
        self._watchdog_started = False
        self._interrupt_event = threading.Event()
        self._session_tokens = 0          # 本会话累计 token(供状态栏上下文占用条)
        self._max_context_tokens = 200000  # 上下文窗口上限(设计文档默认 200K)
        self._session_start = time.time()  # 会话开始时间(供状态栏时长)
        self._input_history: list = []    # 输入历史(上下箭头翻页)
        self._history_idx: int = -1       # 当前历史索引(-1 = 新输入)
        self._drafts: list = []           # Ctrl+S 暂存的草稿栈(设计文档)
        self._setup_command_completion()

    _COMMANDS = [
        "/deep", "/research", "/model", "/config", "/theme", "/code", "/status",
        "/tools", "/sessions", "/dashboard", "/drafts", "/compress", "/harness", "/mcp", "/tg",
        "/learn", "/memskill", "/help", "/exit",
    ]

    def _setup_command_completion(self):
        """用 readline 提供命令补全: 输入 / 后按 Tab 弹出可用命令选项."""
        if not sys.stdout.isatty() or not sys.stdin.isatty():
            return
        try:
            import readline
        except Exception:
            return
        commands = self._COMMANDS

        def completer(text: str, state: int):
            if not text.startswith("/"):
                return None
            matches = [c for c in commands if c.startswith(text)]
            return matches[state] + " " if state < len(matches) else None

        try:
            readline.set_completer(completer)
            readline.parse_and_bind("tab: complete")
            readline.set_completer_delims(" \t\n")
            # 关键: 启用 bracketed paste, 让多行粘贴作为一次完整输入(而非第一个换行就截断)
            try:
                readline.parse_and_bind("set-bracketed-paste on")
            except Exception:
                pass
        except Exception:
            pass

    def _select_runner(self):
        """harness 开启时走事件溯源运行时，否则保持旧循环。"""
        if getattr(self, 'harness_mode', False) and self.agent is not None:
            from agent_project.harness.runner import make_legacy_runner
            return make_legacy_runner(self.agent, self.config)
        return self.agent.run

    def load_config(self) -> bool:
        try:
            from agent_project.config import load_config
            self.config = load_config(str(self.config_path))
            print(_style(" config loaded", "2"))
            return True
        except Exception as e:
            print(_style(f" config error: {e}", "31"))
            return False

    def show_header(self, minimal: bool = False):
        """启动头：左侧小头像，右侧品牌文字，简洁有设计感。

        Args:
            minimal: 在 plain/json 或非 TTY 模式下使用极简无 ANSI 头部。
        """
        if minimal:
            print("Lv agent · Lux Vita (光与生命)")
            print("-" * 40)
            print("Open-source AI at the core — great AI for everyone, intelligence for all")
            return

        # decorative top line
        try:
            cols = shutil.get_terminal_size().columns
        except Exception:
            cols = 80
        line = "─" * cols
        print(_style(line, "2"))

        print()
        # 自适应头像宽度: 终端越宽, 头像越大, 像素点越多(更细腻)
        _pw = 34
        if cols >= 150:
            _pw = 52
        elif cols >= 120:
            _pw = 44
        portrait = _render_portrait(width_chars=_pw)
        right_lines = [
            "",
            _style("Lv agent", "1", "38;5;188"),
            _style("Lux Vita · 光与生命", "38;5;186"),
            _style("by cleveris research", "38;5;240"),
            "",
            _style("Deep thinking, real tools.", "1", "188"),
            _style("Recurrent reasoning · tool-driven · self-learning", "38;5;245"),
            "",
            _style("Capabilities", "1", "188"),
            _style("  · Multi-turn deep reasoning for complex problems", "38;5;251"),
            _style("  · Real tools for live information", "38;5;251"),
            _style("  · Continuous self-learning, smarter over time", "38;5;251"),
            "",
            _style("Mission", "1", "188"),
            _style("  · Open-source AI at the core", "38;5;245"),
            _style("  · Great AI for everyone", "38;5;245"),
            _style("  · Intelligence for all", "38;5;245"),
            "",
        ]
        if portrait:
            p_lines = portrait.splitlines()
            max_lines = max(len(p_lines), len(right_lines))
            p_lines += [""] * (max_lines - len(p_lines))
            right_lines += [""] * (max_lines - len(right_lines))
            for p_line, r_line in zip(p_lines, right_lines):
                # one space before portrait, three spaces between
                print(f" {p_line}   {r_line}")
        else:
            for line in right_lines:
                print(f" {line}")

        print()
        # decorative bottom line
        print(_style(line, "2"))
        # 系统状态行(设计方案: model / tools / cwd)
        self._print_system_status_line()

    def _print_system_status_line(self):
        """启动横幅下的系统状态行: model / tools / cwd(参考设计方案)."""
        try:
            from agent_project.tools import TOOLS_REGISTRY
            tools_count = len(list(TOOLS_REGISTRY.list_tools()))
        except Exception:
            tools_count = 0
        model_name = "unknown"
        backend = getattr(self.config, 'backend', None) if self.config else None
        if backend == 'deepseek':
            model_name = (self.config.deepseek or {}).get('model', 'unknown')
        elif backend == 'openai':
            model_name = (self.config.openai or {}).get('model', 'unknown')
        elif backend == 'anthropic':
            model_name = (getattr(self.config, 'anthropic', {}) or {}).get('model', 'unknown')
        cwd = os.getcwd()
        sep = _style("│", "2")
        line = (
            f" {_style('model', '2')} {_style(f'{backend}/{model_name}', '188')} {sep} "
            f"{_style('tools', '2')} {_style(str(tools_count), '188')} {sep} "
            f"{_style('cwd', '2')} {_style(cwd, '245')}"
        )
        print(line)

    def initialize_agent(self) -> bool:
        from agent_project.agent import OpenMythosAgent
        try:
            self.agent = OpenMythosAgent(self.config)
            print(_style(" agents loaded", "2"))
            return True
        except Exception as e:
            print(_style(f" agent init failed: {e}", "33"))
            # Only offer interactive backend switch when stdin is a TTY.
            if not sys.stdin.isatty():
                print(_style(" non-interactive mode: cannot prompt for backend switch.", "31"))
                print(_style(" fix config.yaml or set env vars (e.g. DEEPSEEK_API_KEY) and retry.", "2"))
                return False
            print(_style(" choose a backend to continue...", "2"))
            try:
                if self.choose_and_set_model() and self.load_config():
                    self.agent = OpenMythosAgent(self.config)
                    print(_style(" agents loaded", "2"))
                    return True
            except Exception as e2:
                print(_style(f" retry failed: {e2}", "31"))
            return False

    def show_status(self):
        from agent_project.tools import TOOLS_REGISTRY
        backend = type(self.agent.backend).__name__
        model = "unknown"
        if self.config.backend == 'openai':
            model = self.config.openai.get('model', 'unknown')
        elif self.config.backend == 'anthropic':
            model = self.config.anthropic.get('model', 'unknown')
        elif self.config.backend == 'deepseek':
            model = self.config.deepseek.get('model', 'unknown')
        tools = len(list(TOOLS_REGISTRY.list_tools()))
        loops = self.config.default_thinking_loops
        sep = _style("·", "2")
        print(f" {_style('backend', '2')} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('tools', '2')} {tools} {sep} {_style('loops', '2')} {loops}")
        print()

    def format_result(self, result: Dict[str, Any]):
        metadata = result.get('metadata', {})
        duration = metadata.get('duration_ms', 0) / 1000
        loops = result.get('outer_loops', 0)
        steps = result.get('thinking_steps', 0)
        tokens = result.get('live_tokens') or result.get('session_token_usage', {}).get('last_call_tokens', 0)
        tokens_display = f"{tokens // 1000}.{tokens % 1000 // 100}k" if tokens >= 1000 else str(tokens)
        ok = bool(result.get('success'))
        status = _style("ok", "32") if ok else _style("failed", "31")
        meta = " · ".join([
            status,
            _style(f"{loops} loops", "2"),
            _style(f"{steps} steps", "2"),
            _style(f"{duration:.1f}s", "2"),
            _style(f"{tokens_display} tokens", "2"),
        ])
        # 细线分隔 + 灰色元信息, 让结果与元数据有层次
        return _style("· " + "─" * 38 + " ·", "2") + "\n " + meta
    def start_telegram(self):
        if self._tg_process is not None and self._tg_process.poll() is None:
            print(" telegram bot already running")
            return
        try:
            token = self.config.tools.telegram.get('bot_token') if self.config else None
            if not token:
                print(" error: telegram.bot_token not set in config.yaml")
                return
            script = Path(__file__).parent / 'start_telegram.py'
            python = Path(__file__).parent / '.venv' / 'bin' / 'python3'
            if not python.exists():
                python = Path(sys.executable)
            self._tg_process = subprocess.Popen(
                [str(python), str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f" telegram bot started (pid {self._tg_process.pid})")
        except Exception as e:
            print(f" failed to start telegram bot: {e}")

    def stop_telegram(self):
        if self._tg_process is None:
            print(" telegram bot not running")
            return
        try:
            self._tg_process.terminate()
            self._tg_process.wait(timeout=5)
            print(" telegram bot stopped")
        except Exception as e:
            print(f" failed to stop telegram bot: {e}")
            try:
                self._tg_process.kill()
            except Exception:
                pass
        finally:
            self._tg_process = None

    def telegram_status(self):
        if self._tg_process is None or self._tg_process.poll() is not None:
            print(" telegram bot: not running")
        else:
            print(f" telegram bot: running (pid {self._tg_process.pid})")

    def handle_learn(self, topic: str):
        """触发从最近对话学习记忆技能。"""
        if not self.agent or not getattr(self.agent, 'memskill_engine', None):
            print(" memskill engine not ready")
            return
        try:
            # 收集最近几轮作为学习素材
            recent = getattr(self, '_recent_turns', []) or []
            context = topic or "the workflow we just went through"
            prompt = (
                f"[/learn] The user wants you to learn a reusable memory skill from: {context}.\n"
                "Review the recent conversation, identify durable facts/preferences/workflows/errors, "
                "and author a memory skill following the project format.\n"
                "Save it via the memskill engine."
            )
            if recent:
                prompt += "\n\nRecent turns:\n" + "\n".join(recent[-10:])

            # 用 agent 跑一轮生成技能
            result = self.agent.run(prompt, code_mode=False)
            print(f" \033[2mlearn result: {result.get('final_answer', 'done')[:200]}\033[0m")
        except Exception as e:
            print(f" /learn failed: {e}")

    def handle_memskill(self, rest: str):
        """管理记忆技能。"""
        if not self.agent or not getattr(self.agent, 'memskill_engine', None):
            print(" memskill engine not ready")
            return
        engine = self.agent.memskill_engine
        parts = rest.split(maxsplit=1)
        cmd = parts[0].lower() if parts else 'list'
        arg = parts[1] if len(parts) > 1 else ''

        try:
            if cmd in ('', 'list', 'ls'):
                skills = engine.list_skills()
                print(f" {len(skills)} memory skills")
                for s in skills:
                    src = s.get('source', 'builtin')
                    ver = s.get('version', '')
                    print(f" \033[2m  · {s['name']} v{ver} ({src}) - {s.get('description', '')[:60]}\033[0m")
            elif cmd == 'evolve':
                evolved = engine.force_evolve()
                if evolved:
                    print(f" evolved skills: {', '.join(evolved)}")
                else:
                    print(" no evolution triggered (need more hard cases)")
            elif cmd == 'snapshot':
                tag = arg or datetime.now().strftime("%Y%m%d_%H%M%S")
                path = engine.snapshot_skills(tag=tag)
                print(f" snapshot saved: {path}")
            elif cmd == 'restore':
                if not arg:
                    print(" usage: /memskill restore <tag>")
                    return
                backup_dir = Path(engine.bank.skills_dir) / '.backups' / arg
                engine.restore_skills(backup_dir)
                print(f" restored from {backup_dir}")
            else:
                print(" usage: /memskill [list|evolve|snapshot <tag>|restore <tag>]")
        except Exception as e:
            print(f" /memskill failed: {e}")

    def set_telegram_token(self, token: str):
        if not token:
            print(" error: token cannot be empty")
            return
        try:
            text = self.config_path.read_text(encoding='utf-8')
            lines = text.splitlines(keepends=True)
            telegram_indent = None
            updated = []
            count = 0

            for line in lines:
                stripped = line.lstrip()
                indent = len(line) - len(stripped)

                if stripped.startswith('telegram:'):
                    telegram_indent = indent
                    updated.append(line)
                    continue

                if telegram_indent is not None and stripped.startswith('bot_token:'):
                    if indent > telegram_indent:
                        updated.append(' ' * telegram_indent + '  bot_token: ' + token + '\n')
                        count += 1
                        continue

                if telegram_indent is not None and stripped and indent <= telegram_indent:
                    telegram_indent = None

                updated.append(line)

            if count == 0:
                updated_text = text.rstrip() + f"\n\ntools:\n  telegram:\n    enabled: true\n    bot_token: {token}\n"
            else:
                updated_text = ''.join(updated)

            self.config_path.write_text(updated_text, encoding='utf-8')
            print(f" telegram token saved ({len(token)} chars)")
            if self.load_config() and self.config:
                print(" config reloaded")
        except Exception as e:
            print(f" failed to save telegram token: {e}")

    def _listen_for_esc(self):
        """后台线程: 任务运行时监听 ESC 键, 设置中断标志.

        用非阻塞 select 读 stdin, 检测到 \x1b(ESC)且任务在运行 → 设置 _interrupt_event。
        注意: 只读不消费(不破坏后续 input), 但 ESC 本身会被消费掉, 可接受。
        """
        try:
            import select
            fd = sys.stdin.fileno()
            while self._running and not self._interrupt_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                # 读取并检测 ESC 序列
                data = os.read(fd, 1)
                if not data:
                    continue
                if data == b"\x1b":
                    self._interrupt_event.set()
                    print(_style("  ↳ ESC 已按下, 正在中断任务...", "33"), flush=True)
                    break
        except Exception:
            pass

    def _run_with_progress(self, fn, *args, mode=None, **kwargs):
        """Run an agent function with unified stream output.

        mode:
          "rich"  -> TTY spinner + live reasoning + syntax highlighting
          "plain" -> structured plain-text lines (no ANSI)
          "json"  -> JSON Lines
          None    -> auto-detect from env/TTY

        深度研究(deep research)任务额外启用 rich Live 实时进度面板:
        顶部显示阶段/轮次/来源数/token, 替代滚动的纯文本状态行。
        """
        adapter = select_stream_adapter(mode=mode, is_tty=sys.stdout.isatty(), console=console)

        # 检测深度研究任务(设计文档: 长任务用实时面板)
        task_text = ""
        for a in args:
            if isinstance(a, str):
                task_text = a
                break
        _task_low = task_text.lower()
        is_deep_research = (
            "深度研究" in task_text or "深度调研" in task_text
            or task_text.strip().lower().startswith(("/deep", "/research"))
            or ("研究" in task_text and "报告" in task_text)
            or ("搜索" in task_text and ("报告" in task_text or "生成" in task_text))
            or ("调研" in task_text and "报告" in task_text)
        )

        # rich Live 进度面板状态
        live = None
        live_refresher = None
        live_state = {
            "stage": "初始化",
            "round": 0,
            "round_total": 0,
            "sources": 0,
            "queries": 0,
            "msg": "",
            "tokens": 0,
            "started": time.time(),
            "log": [],          # 过程日志(最近 12 条)
            "recent_queries": [],
        }

        def _log(msg: str):
            live_state["log"].append(msg)
            if len(live_state["log"]) > 12:
                live_state["log"] = live_state["log"][-12:]

        def _build_research_panel():
            from rich.panel import Panel
            from rich.console import Group
            from rich.text import Text
            # 无色简洁: 全部用默认色, 只靠结构与对齐
            elapsed = time.time() - live_state["started"]
            dots = "." * (int(elapsed * 2) % 4)

            stage_label = live_state["stage"]
            _stage_en = {
                "搜索": "Searching", "抓取正文": "Fetching", "综合报告": "Synthesizing",
                "核验": "Verifying", "完成搜索": "Done",
            }
            stage_label = _stage_en.get(stage_label, stage_label)

            # 统计: 左对齐标签 + 数值, 简洁整齐
            lines = [
                f"  Stage     {stage_label}{dots}",
                f"  Round     {live_state['round']}/{live_state['round_total']}",
                f"  Sources   {live_state['sources']}",
                f"  Queries   {live_state['queries']}",
                f"  Tokens    {live_state.get('tokens', 0):,}",
                f"  Elapsed   {int(elapsed)}s",
            ]
            # 进度条(细块, 无色)
            pct = min(100, int(live_state["sources"] / 200 * 100)) if live_state["sources"] else 0
            lines.append(f"  Progress  " + "▁" * (pct // 10) + " " * (10 - pct // 10))
            stats_text = "\n".join(lines)

            # 过程日志: 缩进对齐, 无图标
            log_lines = [f"    {ln[:64]}" for ln in live_state["log"]]
            log_panel = Panel(
                "\n".join(log_lines) if log_lines else "    (waiting...)",
                title=" Research Process ",
                padding=(0, 1),
                expand=False,
                border_style="dim",
            )

            inner = Group(
                Text(""),
                Text(stats_text),
                Text(""),
                log_panel,
            )
            return Panel(
                inner,
                title=" LV AGENT · Deep Research ",
                subtitle=" // running ",
                padding=(0, 1),
                expand=False,
                border_style="dim",
            )

        def stream_callback(kind, token):
            # ESC 中断检查: 用户在任务运行时按 ESC, 立即中断
            if self._interrupt_event.is_set():
                raise KeyboardInterrupt("user pressed ESC")
            # 深度研究: 用 Live 面板更新进度, 不打印滚动状态
            if live is not None and is_deep_research:
                tok = str(token)
                if kind == "status":
                    live_state["msg"] = tok[:100]
                    # 解析阶段/轮次/来源
                    m = re.search(r"research round (\d+)/(\d+)", tok)
                    if m:
                        live_state["round"], live_state["round_total"] = int(m.group(1)), int(m.group(2))
                        _log(f"Round {live_state['round']}/{live_state['round_total']}")
                    m = re.search(r"\((\d+) sources\)", tok)
                    if m:
                        live_state["sources"] = int(m.group(1))
                    if "enriching" in tok.lower() or "synthesiz" in tok.lower():
                        live_state["stage"] = "Synthesizing" if "synthesiz" in tok.lower() else "Fetching"
                        _log(tok[:80])
                    elif "verif" in tok.lower():
                        live_state["stage"] = "Verifying"
                        _log(tok[:80])
                    elif "research" in tok or "round" in tok:
                        live_state["stage"] = "Searching"
                    elif "no new angles" in tok.lower():
                        live_state["stage"] = "Done"
                        _log("No new angles, stopping search early")
                elif kind == "tool_call":
                    _log(tok[:90])
                    if "web_search" in tok or "search" in tok.lower():
                        live_state["queries"] += 1
                elif kind == "tool_result" and "sources" in tok:
                    m = re.search(r"total (\d+)", tok)
                    if m:
                        live_state["sources"] = int(m.group(1))
                    _log(tok[:90])
                elif kind == "tool_result":
                    _log(tok[:90])
                live.update(_build_research_panel())
                return
            if kind == "status":
                adapter.emit_status(token)
            elif kind == "reasoning":
                adapter.emit_reasoning(token)
            elif kind == "tool_call":
                adapter.emit_tool_call(token)
            elif kind == "tool_result":
                adapter.emit_tool_result(token)
            elif kind == "content":
                adapter.emit_content(token)

        def token_callback(tokens: int):
            if isinstance(adapter, RichStreamAdapter):
                adapter.add_tokens(tokens)
            if live is not None and is_deep_research:
                live_state["tokens"] = live_state.get("tokens", 0) + int(tokens)
                live.update(_build_research_panel())

        kwargs['stream_callback'] = stream_callback
        kwargs['token_callback'] = token_callback

        total_tokens = 0

        try:
            # Prime adapter so user sees immediate feedback even before first token.
            adapter.emit_status("thinking")
            self._running = True
            # 重置中断标志(上一轮可能残留)
            self._interrupt_event.clear()
            # 后台监听 ESC: 用户在任务运行时按 ESC 中断
            esc_listener = threading.Thread(target=self._listen_for_esc, daemon=True)
            esc_listener.start()
            # 深度研究: 启动 rich Live 进度面板
            if is_deep_research and sys.stdout.isatty():
                try:
                    from rich.live import Live
                    live = Live(_build_research_panel(), console=console, refresh_per_second=8, transient=True)
                    live.start()
                    # 后台刷新线程: 即使无新事件也持续刷新(计时/动画), 避免看起来"卡死"
                    def _ticker():
                        while live is not None and self._running:
                            try:
                                live.update(_build_research_panel())
                            except Exception:
                                pass
                            time.sleep(0.5)
                    live_refresher = threading.Thread(target=_ticker, daemon=True)
                    live_refresher.start()
                except Exception:
                    live = None
            result = fn(*args, **kwargs)
        except KeyboardInterrupt as e:
            # 用户 ESC 中断
            self._interrupt_event.clear()
            if live is not None:
                try:
                    live.stop()
                except Exception:
                    pass
            if isinstance(adapter, RichStreamAdapter):
                print(f"\n\033[2m任务已中断 (ESC)\033[0m", flush=True)
            else:
                print("TASK INTERRUPTED (ESC)", flush=True)
            raise
        except BaseException as e:
            if live is not None:
                try:
                    live.stop()
                except Exception:
                    pass
            if isinstance(adapter, RichStreamAdapter):
                print(f"\n\033[31mError: {type(e).__name__}: {e}\033[0m")
            else:
                print(f"ERROR {type(e).__name__}: {e}", flush=True)
            raise
        finally:
            self._running = False
            self._last_activity = time.time()
            # 深度研究面板: 正常完成时停止 Live 并打印最终统计
            if live is not None:
                try:
                    live.stop()
                except Exception:
                    pass
                print(_style(f"  ✓ Deep research done · sources {live_state.get('sources', 0)} · queries {live_state.get('queries', 0)} · tokens {live_state.get('tokens', 0)}", "2"), flush=True)
            adapter.finalize()
            # Read content flag AFTER streaming completes so all adapters can suppress
            # duplicate final-answer printing.
            has_content = getattr(adapter, '_has_content', False)
            if isinstance(adapter, RichStreamAdapter):
                total_tokens = adapter._total_tokens

        result['_content_streamed'] = has_content
        result['live_tokens'] = total_tokens
        # 累计会话 token(供状态栏上下文占用条)
        if total_tokens:
            self._session_tokens += int(total_tokens)
        return result

    def _start_idle_watchdog(self, idle_threshold: float = 90.0, cooldown_s: float = 300.0):
        """后台守护线程: 空闲超过阈值且无任务运行时, 触发静默归纳整理.

        避免干扰: 只在 idle > idle_threshold 且 _running=False 时整理一次,
        之后进入 cooldown_s 冷却, 防止高频触发。
        """

        def _watch():
            while True:
                time.sleep(15)
                try:
                    idle = time.time() - self._last_activity
                    if idle > idle_threshold and self.agent is not None and not self._running:
                        done = self.agent._idle_housekeeping()
                        if done:
                            # 安静、居中的小字脚注, 不打扰对话
                            try:
                                cols = shutil.get_terminal_size().columns
                            except Exception:
                                cols = 80
                            note = f"· idle · {done} consolidated · context compacted ·"
                            print("\033[2m" + note.center(min(cols, 88)) + "\033[0m", flush=True)
                        time.sleep(cooldown_s)
                except Exception:
                    pass

        threading.Thread(target=_watch, daemon=True, name="idle-watchdog").start()

    def _footer_line(self, width: int = 0) -> str:
        """底部状态栏: 目录 + 上下文占用条 + 命令(基于 ui.StatusBar)."""
        sb = self._status_bar_instance()
        line = sb.render(used_tokens=self._session_tokens, width=width)
        # 上下文溢出提示: >=95% 时在行尾追加红色提示(设计文档: 状态栏变红 + /compress 建议)
        if self._max_context_tokens and self._session_tokens / self._max_context_tokens >= 0.95:
            warn = _style("  ⚠️ context near limit · type /compress", "38;5;220")
            return line + warn
        return line

    def _status_bar_instance(self) -> StatusBar:
        """构建(并缓存)StatusBar 实例, 复用 renderer 保持主题一致."""
        sb = getattr(self, '_status_bar_cache', None)
        if sb is None:
            from agent_project.terminal import _renderer as _r
            sb = StatusBar(
                renderer=_r,
                max_context_tokens=self._max_context_tokens,
                session_start=getattr(self, '_session_start', time.time()),
                commands_full="/deep · /research · /model · /code · /help",
                commands_compact="/model · /code · /help",
            )
            self._status_bar_cache = sb
        return sb

    def _read_input_complete(self) -> str:
        """读取用户输入, 支持正常打字(回显/退格)与多行粘贴.

        手动读取 stdin, 保留:
        - 字符回显(用户打字能看到)
        - 退格删除
        - bracketed paste 多行完整读取
        - 单行输入回车即提交; 粘贴后回车提交(可先追加)
        """
        import termios
        import tty
        import select

        fd = sys.stdin.fileno()
        old_attr = None
        try:
            old_attr = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            try:
                return input().strip()
            except Exception:
                return ""

        buf = bytearray()
        in_paste = False
        echoed = 0  # 已回显的字节数
        try:
            while True:
                r, _, _ = select.select([fd], [], [], 0.2)
                if not r:
                    continue
                chunk = os.read(fd, 4096)
                if not chunk:
                    continue

                # ---- 逐字节状态机: 区分粘贴内容与普通输入 ----
                i = 0
                n = len(chunk)
                while i < n:
                    b = chunk[i]
                    if b == 0x1b:  # ESC
                        # 尝试匹配 ESC[200~ / ESC[201~
                        if chunk[i:i+6] == b"\x1b[200~":
                            in_paste = True
                            i += 6
                            continue
                        if chunk[i:i+6] == b"\x1b[201~":
                            in_paste = False
                            i += 6
                            continue
                        # 方向键: ESC[A 上, ESC[B 下(翻输入历史)
                        if chunk[i+1:i+2] == b"[" and chunk[i+2:i+3] in (b"A", b"B", b"C", b"D"):
                            key = chunk[i+2]
                            if key == b"A" and self._input_history:  # 上箭头 → 上一条
                                if self._history_idx < 0:
                                    self._history_idx = len(self._input_history) - 1
                                else:
                                    self._history_idx = max(0, self._history_idx - 1)
                                self._apply_history_line(buf, self._input_history[self._history_idx])
                            elif key == b"B" and self._input_history:  # 下箭头 → 下一条/新输入
                                self._history_idx += 1
                                if self._history_idx >= len(self._input_history):
                                    self._history_idx = -1
                                    self._apply_history_line(buf, "")
                                else:
                                    self._apply_history_line(buf, self._input_history[self._history_idx])
                            # C/D 左右箭头: 忽略(简单实现, 不支持光标移动)
                            i += 3
                            continue
                        # 其他 ESC 序列: 跳过 ESC 本身
                        i += 1
                        continue
                    if in_paste:
                        # 粘贴内容: 原样累积(含换行), 不提交不回显
                        buf += chunk[i:i+1]
                        i += 1
                        continue
                    # 普通输入
                    if b == 0x0a or b == 0x0d:  # 回车提交
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        return bytes(buf).decode("utf-8", errors="replace").strip()
                    elif b == 0x7f or b == 0x08:  # 退格
                        if buf:
                            while buf and (buf[-1] & 0xC0) == 0x80:
                                buf.pop()
                            if buf:
                                buf.pop()
                            echoed = len(buf)
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                    elif b == 0x03:  # Ctrl+C
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        raise KeyboardInterrupt
                    elif b == 0x04:  # Ctrl+D
                        if not buf:
                            sys.stdout.write("\r\n")
                            sys.stdout.flush()
                            raise EOFError
                        continue
                    elif b == 0x13:  # Ctrl+S: 暂存草稿(设计文档)
                        draft = bytes(buf).decode("utf-8", errors="replace").strip()
                        self._drafts.append(draft)
                        buf.clear()
                        sys.stdout.write("\r\x1b[K" + self._prompt_prefix())
                        sys.stdout.flush()
                        echoed = 0
                        print(_style(f"  ↳ 已暂存草稿 ({len(self._drafts)}), 输入 /drafts 查看", "2"), flush=True)
                        continue
                    elif b == 0x1c:  # Ctrl+\: 打开 Dashboard(设计文档)
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        return "\x1cDASHBOARD"
                    else:
                        buf += chunk[i:i+1]
                    i += 1
                # 回显新增的完整 UTF-8 字符
                if len(buf) > echoed:
                    tail = bytes(buf[echoed:])
                    try:
                        text = tail.decode("utf-8")
                        sys.stdout.write(text)
                        echoed = len(buf)
                        sys.stdout.flush()
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            try:
                if old_attr is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
            except Exception:
                pass

        return bytes(buf).decode("utf-8", errors="replace").strip()

    def _prompt_prefix(self) -> str:
        """输入提示前缀: 三段式「Lv + 路径 → 光标」(参考设计方案).

        例: Lv ~/project → _
        """
        try:
            cwd = os.getcwd()
            # 用 ~ 替代 home 前缀, 更简洁
            home = os.path.expanduser("~")
            if cwd.startswith(home):
                cwd = "~" + cwd[len(home):]
        except Exception:
            cwd = "~"
        brand = _style("Lv", "1", "188")
        path = _style(cwd, "2")
        arrow = _style("→", "2")
        return f"{brand} {path} {arrow} "

    def _apply_history_line(self, buf: bytearray, text: str) -> None:
        """替换输入 buffer 为历史行, 并刷新回显(上下箭头翻页)."""
        # 清空当前行(回退到行首 + 清到行尾)
        sys.stdout.write("\r\x1b[K")
        # 重新显示历史内容
        buf.clear()
        if text:
            buf.extend(text.encode("utf-8", errors="replace"))
        # 重画 prompt 前缀 + 当前 buffer 内容
        prefix = self._prompt_prefix()
        sys.stdout.write(prefix + text)
        sys.stdout.flush()

    def _prompt_with_footer(self) -> str:
        """显示输入框, 状态栏固定在其正下一行.

        流程(纯相对定位, 不用终端绝对行号):
          1) \r\n 确保光标到新行行首
          2) 再 \r\n 空一行(给输入框)
          3) 画状态栏一行
          4) \033[2A 上移两行回到输入框行
          5) 渲染 "You › " 并读输入
          6) 输入完成后 \r\n, 让后续输出从状态栏下方继续
        非 TTY 环境退化为普通 input。
        """
        if not sys.stdout.isatty() or not sys.stdin.isatty():
            sys.stdout.write("You › ")
            sys.stdout.flush()
            return input().strip()

        # 1) 到新行行首
        sys.stdout.write("\r\n")
        # 2) 再空一行给输入框
        sys.stdout.write("\r\n")
        # 3) 画状态栏
        sys.stdout.write(self._footer_line())
        sys.stdout.write("\r\n")
        # 4) 上移两行回到输入框行
        sys.stdout.write("\033[2A\033[G")
        sys.stdout.flush()
        # 5) 渲染输入提示: 三段式「Lv + 路径 → 光标」(参考设计方案)
        sys.stdout.write(self._prompt_prefix())
        sys.stdout.flush()
        try:
            value = self._read_input_complete()
        except (EOFError, KeyboardInterrupt):
            raise
        # 6) 输入完成, 换行让后续输出继续
        sys.stdout.write("\r\n")
        sys.stdout.flush()
        value = value.strip()
        # 多行粘贴确认: 若内容含换行, 回显总行数/总字符, 让用户确认已完整复制
        if value:
            line_count = len(value.splitlines())
            char_count = len(value)
            if line_count > 1 or char_count > 200:
                print(_style(f"  ↳ 已接收 {line_count} 行 · {char_count} 字符", "2"), flush=True)
        return value

    def _expand_file_refs(self, text: str) -> str:
        """展开 @path 文件引用为文件内容(参考设计方案).

        支持:
        - @相对路径 或 @绝对路径 → 读取文件内容拼接入提示
        - @glob 模式(如 @src/*.py) → 读取匹配的所有文件
        找不到文件时给出红色提示并保留原始文本。
        """
        import glob

        def _expand_one(ref: str) -> str:
            ref = os.path.expanduser(ref.strip())
            matches = glob.glob(ref) if any(c in ref for c in "*?[") else ([ref] if os.path.exists(ref) else [])
            if not matches:
                print(_style(f"  ⚠️ 未找到文件: {ref}", "33"), flush=True)
                return f"@{ref}"
            parts = []
            for fp in matches[:8]:
                try:
                    content = Path(fp).read_text(encoding="utf-8", errors="replace")
                    parts.append(f"--- {fp} ---\n{content}")
                except Exception as e:
                    parts.append(f"--- {fp} [读取失败: {e}] ---")
            if len(matches) > 8:
                parts.append(f"...(另有 {len(matches)-8} 个匹配未展开)")
            return "\n\n".join(parts)

        # 展开 @引用。展开结果先存占位, 全部展开完再还原, 避免内容里的 @ 被二次处理。
        expanded_cache: list = []

        def _cache_expand(ref: str) -> str:
            expanded_cache.append(_expand_one(ref))
            return f"\x00REF{len(expanded_cache)-1}\x00"

        # 优先: 含空格的文件名(以扩展名结尾, 如 "AI Agent安全使用指南.md")
        text = re.sub(r'@([^\s@，。！？!?、,;；\n]+(?:\s+[^\s@，。！？!?、,;；\n]+)*\.\w{1,10})', lambda m: _cache_expand(m.group(1)), text)
        # 其次: 普通 @引用(到空白/标点前)
        text = re.sub(r'@([^\s@，。！？!?、,;；\n]+)', lambda m: _cache_expand(m.group(1)), text)
        # 还原占位
        for i, content in enumerate(expanded_cache):
            text = text.replace(f"\x00REF{i}\x00", content)
        return text

    def _show_dashboard(self):
        """Ctrl+反斜杠 或 /dashboard: 显示 Agent Dashboard(设计文档)."""
        from agent_project.tools import TOOLS_REGISTRY
        print(_style("── Agent Dashboard ──", "1", "188"))
        if self.agent:
            try:
                self.agent.print_module_status()
            except Exception:
                pass
        print(f"  会话 token: {self._session_tokens} | 工具: {len(list(TOOLS_REGISTRY.list_tools()))}")
        if self._input_history:
            print(f"  输入历史: {len(self._input_history)} 条 | 草稿: {len(self._drafts)} 个")
        pct = int(self._session_tokens / self._max_context_tokens * 100) if self._max_context_tokens else 0
        print(f"  上下文占用: {pct}%" + (" ⚠️ 请 /compress" if pct >= 95 else ""))
        print(_style("──────────────────", "2"))

    def run_interactive(self):
        # 空闲看门狗: 用户无操作超过 idle_threshold 秒时, 静默归纳/压缩上下文
        if not self._watchdog_started:
            self._watchdog_started = True
            self._start_idle_watchdog()

        while True:
            try:
                user_input = self._prompt_with_footer()
            except Exception:
                self.stop_telegram()
                print("\n" + _style(" interrupted", "2"))
                break
            self._last_activity = time.time()

            # Ctrl+\ Dashboard 快捷键(设计文档)
            if user_input == "\x1cDASHBOARD":
                self._show_dashboard()
                continue

            # 记录输入历史(供上下箭头翻页), 排除空/重复
            if user_input and user_input != (self._input_history[-1] if self._input_history else None):
                self._input_history.append(user_input)
                if len(self._input_history) > 200:
                    self._input_history = self._input_history[-200:]
            self._history_idx = -1

            if user_input.lower() in ('exit', 'quit', 'q'):
                self.stop_telegram()
                print(_style("goodbye", "2"))
                break

            # !shell 模式: 以 ! 开头直接执行 shell, 不进入 agent turn(参考设计方案)
            elif user_input.startswith('!'):
                cmd = user_input[1:].strip()
                if not cmd:
                    print(_style(" usage: !<command>  直接执行 shell 命令", "2"))
                    continue
                print(_style(f" → {cmd}", "2", "38;5;240"))
                try:
                    import subprocess as _sp
                    _proc = _sp.run(cmd, shell=True, cwd=os.getcwd())
                except KeyboardInterrupt:
                    print(_style(" 中断", "2"))
                except Exception as e:
                    print(_style(f" shell error: {e}", "31"))
                continue

            # @文件引用: 展开 @path 为文件内容(参考设计方案), 支持相对路径与 glob
            elif '@' in user_input:
                user_input = self._expand_file_refs(user_input)

            elif user_input.lower() == '/harness':
                self.harness_mode = not self.harness_mode
                state = "on (event-sourced runtime)" if self.harness_mode else "off (legacy loop)"
                print(f" harness mode: {state}")
                continue

            elif user_input.lower() in ('/model', '/switch-model'):
                changed = self.choose_and_set_model()
                if changed:
                    print(" \033[2mreloading...\033[0m\n")
                if not self.load_config() or not self.initialize_agent():
                    print("\033[31mreload failed\033[0m\n")
                    continue
                self.show_status()
                continue

            elif user_input.lower() == '/config':
                if self.config:
                    print(f" backend: {self.config.backend}")
                    cfg_map = {
                        'openai': self.config.openai,
                        'deepseek': self.config.deepseek,
                        'anthropic': getattr(self.config, 'anthropic', {}),
                        'openmythos': getattr(self.config, 'openmythos', {}),
                    }
                    section = cfg_map.get(self.config.backend, {})
                    if section:
                        print(f" model: {section.get('model')}")
                        print(f" url: {section.get('base_url')}")
                        key = section.get('api_key')
                        print(f" auth: {'set' if key else 'not set'}")
                    print(f" loops: {self.config.default_thinking_loops}")
                    print(f" outer loops: {self.config.max_outer_loops}")
                continue

            elif user_input.lower() == '/theme':
                rest = user_input[6:].strip().lower()
                valid = ('light', 'dark', 'minimal')
                if rest not in valid:
                    from agent_project.terminal import active_theme
                    print(f" current theme: {active_theme()}")
                    print(" usage: /theme <light|dark|minimal>")
                    continue
                from agent_project.terminal import set_theme
                set_theme(rest)
                print(f" theme: {rest}")
                continue

            elif user_input.lower() == '/code':
                self.code_mode = not getattr(self, 'code_mode', False)
                status = "on" if self.code_mode else "off"
                print(f" code mode: {status}")
                continue

            elif user_input.lower() == '/status':
                if self.agent:
                    self.agent.print_module_status()
                else:
                    print(" agent not initialized")
                continue

            elif user_input.lower() == '/tools':
                from agent_project.tools import TOOLS_REGISTRY
                tools = sorted(TOOLS_REGISTRY.list_tools())
                print(f" {len(tools)} 个可用工具:")
                for name in tools:
                    tool = TOOLS_REGISTRY.get(name)
                    desc = ""
                    if tool and hasattr(tool, 'description'):
                        desc = str(tool.description).split('.')[0][:50]
                    print(f"  · {_style(name, '1', '188')} {_style(desc, '2')}")
                continue

            elif user_input.lower() == '/sessions':
                # 设计方案: 会话选择器 - 显示最近会话文件
                import glob as _g
                sess_dir = Path(__file__).parent / "data" / "harness_sessions"
                sessions = sorted(_g.glob(str(sess_dir / "*.jsonl")))[-10:] if sess_dir.exists() else []
                if not sessions:
                    print(" 暂无持久化会话")
                else:
                    print(f" {len(sessions)} 个历史会话:")
                    for s in sessions[-8:]:
                        import datetime as _dt
                        try:
                            ts = float(Path(s).stem.split('-')[0]) / 1000
                            when = _dt.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')
                        except Exception:
                            when = "?"
                        print(f"  · {_style(Path(s).name[:40], '188')} {_style(when, '2')}")
                continue

            elif user_input.lower() == '/dashboard':
                self._show_dashboard()
                continue

            elif user_input.lower() == '/drafts':
                if not self._drafts:
                    print(" 草稿栈为空 (Ctrl+S 暂存当前输入)")
                else:
                    print(f" {len(self._drafts)} 个草稿:")
                    for i, d in enumerate(reversed(self._drafts), 1):
                        print(f"  {i}. {_style(d[:80], '2')}")
                    print(" 提示: 直接粘贴草稿内容即可继续编辑")
                continue

            elif user_input.lower() == '/compress':
                if self.agent and hasattr(self.agent, 'context_engine') and self.agent.context_engine:
                    try:
                        ce = self.agent.context_engine
                        wm = ce.working_memory
                        # 计算当前 token 并压缩到 60%
                        from agent_project.context_engine import _estimate_tokens
                        current_tok = _estimate_tokens(" ".join(e.content for e in wm.events))
                        target = max(200, int(current_tok * 0.6))
                        summary = ce.compressor.compress_events(wm.events, target)
                        if summary:
                            # 用摘要替换工作记忆(保留最近的系统事件)
                            from agent_project.context_engine import WorkingMemoryEvent
                            wm.events = [WorkingMemoryEvent(
                                role="system",
                                content="[已压缩] " + summary,
                                event_type="message",
                            )]
                            # 重置会话 token 计数器到压缩后大小(压缩的是会话上下文)
                            self._session_tokens = _estimate_tokens(summary)
                            print(f" 上下文已压缩: {current_tok} token → {_estimate_tokens(summary)} token")
                        else:
                            print(" 无内容可压缩")
                    except Exception as e:
                        print(f" 压缩失败: {e}")
                else:
                    print(" 上下文引擎不可用")
                continue

            elif user_input.lower() == '/help':
                print(" /deep <topic> - run deep iterative research and generate a report")
                print(" /research <topic> - alias for /deep")
                print(" /model - switch model")
                print(" /config - show config")
                print(" /code - toggle engineering/code mode")
                print(" /status - show module health status")
                print(" /tools - list available tools")
                print(" /theme <light|dark|minimal> - switch theme")

                print(" /sessions - show recent sessions")
                print(" /dashboard - show agent dashboard")
                print(" /mcp - configure/test an MCP server")
                print(" /tg [start|stop|status|token <TOKEN>] - Telegram bot")
                print(" /learn [topic] - learn a memory skill from recent conversation")
                print(" /memskill [list|evolve|snapshot|restore <tag>] - manage memory skills")
                print(" !<command> - run shell directly (no agent turn)")
                print(" @<file> - attach file content to prompt")
                print(" /help - this help")
                print(" exit - quit")
                continue

            elif user_input.lower().startswith('/mcp'):
                rest = user_input[4:].strip()
                if not rest:
                    print(" usage: /mcp <request>")
                    print(" example: /mcp add filesystem MCP for ~/Desktop")
                    continue
                try:
                    from agent_project.tools import TOOLS_REGISTRY
                    tool = TOOLS_REGISTRY.get('mcp_setup')
                    if not tool:
                        print(" mcp_setup tool not available")
                        continue
                    result = tool.execute(request=rest)
                    print(f"\n {result.output or result.error}")
                except Exception as e:
                    print(f" mcp setup failed: {e}")
                continue

            elif user_input.lower().startswith('/tg'):
                rest = user_input[3:].strip()
                if rest.startswith('token '):
                    token = rest[6:].strip()
                    self.set_telegram_token(token)
                elif rest in ('stop', 'off'):
                    self.stop_telegram()
                elif rest in ('status',):
                    self.telegram_status()
                elif rest in ('', 'start', 'on'):
                    self.start_telegram()
                else:
                    print(" usage: /tg [start|stop|status|token <TOKEN>]")
                continue

            elif user_input.lower().startswith('/learn'):
                rest = user_input[6:].strip()
                self.handle_learn(rest)
                continue

            elif user_input.lower().startswith('/memskill'):
                rest = user_input[9:].strip()
                self.handle_memskill(rest)
                continue

            if not user_input:
                continue

            # 多段任务: 空行分隔 -> 排队; 相邻相关任务自动合并为一次执行
            segments = [s.strip() for s in re.split(r"\n\s*\n", user_input) if s.strip()]
            if len(segments) > 1:
                self._run_task_batch(segments)
                continue

            try:
                result = self._run_with_progress(
                    self._select_runner(),
                    user_input,
                    code_mode=getattr(self, 'code_mode', False),
                    mode=getattr(self, 'output_mode', None),
                )
                content_streamed = result.pop('_content_streamed', False)
                final_answer = result.get('final_answer', '')
                if not content_streamed and final_answer:
                    print(f"\n{_clean_content_text(final_answer)}")
                print(f"\n{self.format_result(result)}")
                # 深度研究报告: 完成后自动打开 HTML
                self._maybe_open_report(result)
            except KeyboardInterrupt:
                # ESC 中断任务: 回到输入循环, 不退出
                print(_style(" 任务已中断, 可以继续输入新的指令。", "2"))
            except Exception as e:
                print(f"\033[31merror: {e}\033[0m")

    def _run_task_batch(self, tasks: list):
        """多段任务排队执行: 相关任务合并成一组, 无关任务依次执行."""
        try:
            from agent_project.agent import OpenMythosAgent
            groups = OpenMythosAgent.merge_related_tasks(tasks, threshold=0.3)
        except Exception:
            groups = [[t] for t in tasks]
        runner = self._select_runner()
        total = len(groups)
        for i, group in enumerate(groups, 1):
            if total > 1:
                print(_style(f" ── 任务 {i}/{total} ──", "2"))
            combined = "\n\n".join(group)
            if len(group) > 1:
                print(_style(f" ↳ 检测到 {len(group)} 段相关任务, 已合并执行", "2"))
            try:
                result = self._run_with_progress(
                    runner,
                    combined,
                    code_mode=getattr(self, 'code_mode', False),
                    mode=getattr(self, 'output_mode', None),
                )
                content_streamed = result.pop('_content_streamed', False)
                final_answer = result.get('final_answer', '')
                if not content_streamed and final_answer:
                    print(f"\n{_clean_content_text(final_answer)}")
                print(f"\n{self.format_result(result)}")
                self._maybe_open_report(result)
            except KeyboardInterrupt:
                print(_style(" 任务批处理已中断 (ESC)", "2"))
                break
            except Exception as e:
                print(f"\033[31merror: {e}\033[0m")

    def _maybe_open_report(self, result: Dict[str, Any]):
        """深度研究报告完成后, 自动打开生成的 HTML 报告(macOS 用 open 命令)."""
        try:
            html_path = result.get("report_html_path") or (result.get("metadata") or {}).get("report_html_path")
            if not html_path:
                # 兼容: 从 report_path 推断同名 .html
                rp = result.get("report_path")
                if rp and str(rp).endswith(".md"):
                    cand = str(rp)[:-3] + ".html"
                    if os.path.exists(cand):
                        html_path = cand
            if html_path and os.path.exists(html_path):
                import subprocess as _sp
                _sp.Popen(["open", str(html_path)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                print(_style(f"  ↳ 已自动打开报告: {html_path}", "2"), flush=True)
        except Exception as e:
            print(_style(f"  ↳ 打开报告失败: {e}", "33"), flush=True)

    def run_single_task(self, task: str):
        try:
            result = self._run_with_progress(
                self._select_runner(),
                task,
                code_mode=getattr(self, 'code_mode', False),
                mode=getattr(self, 'output_mode', None),
            )
            content_streamed = result.pop('_content_streamed', False)
            final_answer = result.get('final_answer', '')
            if not content_streamed and final_answer:
                print(f"\n{_clean_content_text(final_answer)}")
            print(f"\n{self.format_result(result)}")
            self._maybe_open_report(result)
            return 0
        except Exception as e:
            import traceback
            print(f"\033[31mfailed: {e}\033[0m")
            console.print(traceback.format_exc())
            return 1

    def list_tools(self):
        try:
            from agent_project.tools import TOOLS_REGISTRY
            tools = TOOLS_REGISTRY.list_tools()
            print("\033[2mavailable tools:\033[0m")
            for name in tools:
                tool_class = TOOLS_REGISTRY.get(name)
                desc = (tool_class.__doc__ or "no description").strip().split('\n')[0][:60]
                print(f" \033[34m{name}\033[0m \033[2m{desc}\033[0m")
        except Exception as e:
            print(f"\033[31merror: {e}\033[0m")

    def run(self, argv):
        # Parse output mode early so show_header can respect --plain/--json.
        output_mode = None
        no_color = False
        for i, arg in enumerate(argv):
            if arg == '--plain':
                output_mode = 'plain'
                no_color = True
            elif arg == '--json':
                output_mode = 'json'
                no_color = True
            elif arg == '--no-color':
                console._color_system = None
                no_color = True
        self.output_mode = output_mode
        # Non-TTY consumers (pipes, files, services) should never get ANSI codes.
        if not sys.stdout.isatty():
            no_color = True
        if no_color:
            os.environ['NO_COLOR'] = '1'
        self.show_header(minimal=output_mode in ('plain', 'json') or not sys.stdout.isatty())

        fast_override = '--fast' in argv
        code_mode = '--code' in argv
        unrestricted_files = '--unrestricted-files' in argv
        restrict_files = '--no-unrestricted-files' in argv
        switch_model = '--switch-model' in argv
        loops_override = None

        i = 1
        while i < len(argv):
            if argv[i] == '--loops' and i + 1 < len(argv):
                try:
                    loops_override = int(argv[i + 1])
                except ValueError:
                    print("\033[31m--loops requires an integer\033[0m")
                    return 1
                i += 2
                continue
            if argv[i] in ('--plain', '--json', '--no-color'):
                i += 1
                continue
            i += 1

        if '--list-tools' in argv:
            if self.load_config():
                self.list_tools()
            return 0
        if '--config' in argv:
            print(" \033[2mlaunching config wizard...\033[0m\n")
            import subprocess
            wizard_path = Path(__file__).parent / "config_wizard.py"
            subprocess.run([sys.executable, str(wizard_path)])
            return 0
        if '--model' in argv:
            self.choose_and_set_model()
            return 0

        positional = []
        i = 1
        while i < len(argv):
            arg = argv[i]
            if arg.startswith('--'):
                if arg in ('--loops',):
                    i += 2
                    continue
                i += 1
                continue
            positional.append(arg)
            i += 1
        task = ' '.join(positional) if positional else None

        if not self.load_config():
            return 1

        if loops_override is not None:
            self.config.default_thinking_loops = loops_override
            self.config.max_thinking_loops = loops_override * 2

        if fast_override:
            self.config.fast_mode = True

        self.code_mode = code_mode

        if restrict_files:
            self.config.tools.file_ops['unrestricted'] = False
        elif unrestricted_files:
            self.config.tools.file_ops['unrestricted'] = True

        if self.config.tools.file_ops.get('unrestricted'):
            print(" \033[33m warning: unrestricted file access enabled\033[0m")

        if switch_model:
            self.choose_and_set_model()
            if not self.load_config():
                return 1

        if '--status' in argv:
            self.config.health.print_status_on_startup = False

        if not self.initialize_agent():
            return 1

        if '--status' in argv:
            self.agent.print_module_status(as_json='--json' in argv)
            return 0

        self.show_status()

        if task:
            return self.run_single_task(task)
        else:
            self.run_interactive()
            return 0

    def choose_and_set_model(self):
        import yaml
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f) or {}

        current_backend = cfg.get('agent', {}).get('backend', 'unknown')
        print(f"\ncurrent backend: \033[34m{current_backend}\033[0m")
        print(" [1] Anthropic")
        print(" [2] OpenAI API")
        print(" [3] OpenRouter")
        print(" [4] Local endpoint")
        print(" [5] DeepSeek")
        print(" [6] Cancel")

        choice = Prompt.ask("choice", choices=["1", "2", "3", "4", "5", "6"], default="1")
        if choice == "6":
            print("cancelled.")
            return False

        if choice == "1":
            self._configure_anthropic(cfg)
        elif choice == "2":
            self._configure_openai_direct(cfg)
        elif choice == "3":
            self._configure_openrouter(cfg)
        elif choice == "4":
            self._configure_local_endpoint(cfg)
        elif choice == "5":
            self._configure_deepseek(cfg)

        with open(self.config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print("\033[32mconfig updated\033[0m")
        return True

    def _configure_anthropic(self, cfg):
        import yaml
        print("\n\033[1mAnthropic\033[0m\n")
        api_key = Prompt.ask("API Key (sk-ant-...)").strip()
        if not api_key.startswith('sk-ant-'):
            print(" \033[33mwarning: Anthropic keys usually start with 'sk-ant-'\033[0m")

        print(" 1) claude-3-5-sonnet-20241022")
        print(" 2) claude-3-7-sonnet-20250219")
        print(" 3) claude-sonnet-4-20250514")
        print(" 4) claude-opus-4-20250514")
        print(" 5) custom...")
        model_choice = Prompt.ask("select", choices=["1", "2", "3", "4", "5"], default="3")

        models = {
            "1": "claude-3-5-sonnet-20241022",
            "2": "claude-3-7-sonnet-20250219",
            "3": "claude-sonnet-4-20250514",
            "4": "claude-opus-4-20250514",
        }

        if model_choice in models:
            model = models[model_choice]
        else:
            model = Prompt.ask("model name").strip()

        temp = Prompt.ask("temperature", default="0.7")
        try:
            temp = float(temp)
        except ValueError:
            temp = 0.7

        cfg.setdefault('agent', {})
        cfg['agent']['backend'] = 'anthropic'
        cfg['agent']['anthropic'] = {
            'api_key': api_key,
            'base_url': 'https://api.anthropic.com/v1',
            'model': model,
            'temperature': temp,
            'max_tokens': 4096,
            'timeout': 120
        }

    def _configure_openai_direct(self, cfg):
        import yaml
        print("\n\033[1mOpenAI API\033[0m\n")
        api_key = Prompt.ask("API Key (sk-...)").strip()
        model = Prompt.ask("model", default="gpt-4o-mini").strip()
        cfg.setdefault('agent', {})
        cfg['agent']['backend'] = 'openai'
        cfg['agent']['openai'] = {
            'api_key': api_key,
            'base_url': 'https://api.openai.com/v1',
            'model': model,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 4096,
            'timeout': 120
        }

    def _configure_openrouter(self, cfg):
        import yaml
        print("\n\033[1mOpenRouter\033[0m\n")
        api_key = Prompt.ask("API Key").strip()
        model = Prompt.ask("model", default="openai/gpt-4o-mini").strip()
        cfg.setdefault('agent', {})
        cfg['agent']['backend'] = 'openai'
        cfg['agent']['openai'] = {
            'api_key': api_key,
            'base_url': 'https://openrouter.ai/api/v1',
            'model': model,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 4096,
            'timeout': 120
        }

    def _configure_local_endpoint(self, cfg):
        import yaml
        print("\n\033[1mLocal endpoint\033[0m\n")
        base_url = Prompt.ask("base URL", default="http://localhost:20128").strip()
        model = Prompt.ask("model", default="oc/deepseek-v4-flash-free").strip()
        cfg.setdefault('agent', {})
        cfg['agent']['backend'] = 'openai'
        cfg['agent']['openai'] = {
            'api_key': None,
            'base_url': base_url,
            'model': model,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 4096,
            'timeout': 120
        }

    def _configure_deepseek(self, cfg):
        import yaml
        print("\n\033[1mDeepSeek\033[0m\n")
        api_key = Prompt.ask("API Key").strip()
        if not api_key:
            print(" \033[31mAPI key required\033[0m")
            return

        print(" 1) deepseek-v4-flash")
        print(" 2) deepseek-v4-pro")
        print(" 3) deepseek-chat (legacy alias)")
        print(" 4) deepseek-reasoner (legacy alias)")
        print(" 5) custom...")
        model_choice = Prompt.ask("select", choices=["1", "2", "3", "4", "5"], default="1")

        models = {
            "1": "deepseek-v4-flash",
            "2": "deepseek-v4-pro",
            "3": "deepseek-chat",
            "4": "deepseek-reasoner",
        }

        if model_choice in models:
            model = models[model_choice]
        else:
            model = Prompt.ask("model name").strip()

        temp = Prompt.ask("temperature", default="0.7")
        try:
            temp = float(temp)
        except ValueError:
            temp = 0.7

        cfg.setdefault('agent', {})
        cfg['agent']['backend'] = 'deepseek'
        cfg['agent']['deepseek'] = {
            'api_key': api_key,
            'base_url': 'https://api.deepseek.com',
            'model': model,
            'temperature': temp,
            'top_p': 0.9,
            'max_tokens': 4096,
            'timeout': 120
        }


def main():
    cli = SuperAgentCLI()
    try:
        sys.exit(cli.run(sys.argv))
    except KeyboardInterrupt:
        print("\n\033[2minterrupted\033[0m")
        sys.exit(130)


if __name__ == "__main__":
    main()
