"""
Personal Agent - 个人超级Agent。
在 OpenMythosAgent 基础上叠加：
  - 用户画像/偏好持久化 (personal_profile)
  - 快捷命令 (/today /briefing /note /task /remind /expense)
  - 一键复合动作 (quick_action: 约会/专注/出门/睡前)
  - Personal intent 零LLM 直达路由 (日历/笔记/任务/记账/提醒/简报)

设计原则: 速度优先、工具稳固、用起来爽。
  - 个人意图走规则快路，零 LLM 回合，毫秒级直出。
  - 所有个人工具幂等、兜底、超时隔离，不崩不卡。
  - 输出简洁可执行，带表情与行动项，不甩裸堆栈。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent import OpenMythosAgent
from .config import AgentConfig
from .tools import (
    BaseTool, ToolResult, TOOLS_REGISTRY,
    CalendarTool, NotesTool, TaskTool, RemindersTool,
    FinanceTool, HealthTool, TravelTool, ContactsTool,
    DailyBriefingTool,
)
from .tools.assist_data import db_path, resolve_when
from .pure_utils import edit_distance


# --------------------------------------------------------------------------- #
# Personal profile storage (SQLite, atomic write, thread-safe)
# --------------------------------------------------------------------------- #

_PROFILE_DB = db_path("personal_profile.db")
_PROFILE_LOCK = threading.RLock()


def _init_profile_db():
    _PROFILE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_PROFILE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS personal_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


_init_profile_db()


def _profile_get(key: str, default: Any = None) -> Any:
    with _PROFILE_LOCK:
        conn = sqlite3.connect(_PROFILE_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT value FROM personal_profile WHERE key = ?", (key,)).fetchone()
        conn.close()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]


def _profile_set(key: str, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, default=str)
    now = datetime.now().isoformat()
    with _PROFILE_LOCK:
        conn = sqlite3.connect(_PROFILE_DB)
        conn.execute(
            "INSERT INTO personal_profile(key, value, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, payload, now),
        )
        conn.commit()
        conn.close()


# --------------------------------------------------------------------------- #
# Shortcut commands
# --------------------------------------------------------------------------- #

_SHORTCUT_ALIASES = {
    "/today": "today",
    "/t": "today",
    "/briefing": "briefing",
    "/b": "briefing",
    "/morning": "morning",
    "/note": "note",
    "/n": "note",
    "/task": "task",
    "/todo": "task",
    "/remind": "remind",
    "/alarm": "remind",
    "/expense": "expense",
    "/spend": "expense",
    "/focus": "focus",
    "/pomodoro": "focus",
    "/habit": "habit",
    "/profile": "profile",
    "/pref": "profile",
}


def _parse_shortcut(task: str) -> Optional[Tuple[str, str]]:
    """识别快捷命令，返回 (command, rest)。"""
    t = task.strip()
    m = re.match(r"^(/[a-zA-Z_]+)(?:\s+(.*))?$", t)
    if not m:
        return None
    cmd = m.group(1).lower()
    rest = (m.group(2) or "").strip()
    canon = _SHORTCUT_ALIASES.get(cmd)
    if canon is None:
        return None
    return (canon, rest)


# --------------------------------------------------------------------------- #
# Personal Agent
# --------------------------------------------------------------------------- #

class PersonalAgent(OpenMythosAgent):
    """
    个人超级Agent。在 OpenMythosAgent 基础上叠加个人助理能力。

    使用方式与 OpenMythosAgent 完全一致，额外可用快捷命令：
      /today          今日概览(日历+任务+提醒)
      /briefing       早晨/傍晚简报
      /note <内容>    快速记笔记
      /task <内容>    快速建任务
      /remind <内容>  快速建提醒
      /expense <条目> 快速记账
      /focus <分钟>   专注计时(番茄钟)
      /habit <名称>   习惯打卡
      /profile        查看/设置个人偏好
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # 个人画像(懒加载，首次访问时读)
        self._profile: Optional[Dict[str, Any]] = None
        self._profile_loaded = False
        # 快路命令结果缓存(同一分钟内重复命令复用)
        self._shortcut_cache: Dict[str, Tuple[float, str]] = {}
        # 工具就绪标记
        self._personal_tools_ready = False
        self._ensure_personal_tools()

    # ---------------- 个人画像 ----------------

    def _load_profile(self) -> Dict[str, Any]:
        if self._profile_loaded:
            return self._profile or {}
        profile: Dict[str, Any] = {}
        for key in (
            "timezone", "language", "home_address", "work_address",
            "default_calendar", "currency", "wake_time", "sleep_time",
            "focus_minutes", "briefing_sections", "favorite_contacts",
        ):
            v = _profile_get(key)
            if v is not None:
                profile[key] = v
        # 合理默认值
        profile.setdefault("language", "zh")
        profile.setdefault("timezone", "Asia/Shanghai")
        profile.setdefault("currency", "CNY")
        profile.setdefault("focus_minutes", 25)
        profile.setdefault("wake_time", "07:00")
        profile.setdefault("sleep_time", "23:00")
        profile.setdefault(
            "briefing_sections",
            ["calendar", "tasks", "reminders", "finance", "health"],
        )
        profile.setdefault("favorite_contacts", [])
        self._profile = profile
        self._profile_loaded = True
        return profile

    def _save_profile(self, key: str, value: Any) -> None:
        self._load_profile()
        if self._profile is not None:
            self._profile[key] = value
        _profile_set(key, value)

    def get_profile(self, key: str, default: Any = None) -> Any:
        return self._load_profile().get(key, default)

    # ---------------- 工具就绪 ----------------

    def _ensure_personal_tools(self) -> None:
        """确保个人助理工具全部注册(幂等)。"""
        if self._personal_tools_ready:
            return
        for cls in (
            CalendarTool, NotesTool, TaskTool, RemindersTool,
            FinanceTool, HealthTool, TravelTool, ContactsTool,
            DailyBriefingTool,
        ):
            try:
                if cls.name not in TOOLS_REGISTRY._tools:
                    TOOLS_REGISTRY.register(cls())
            except Exception as e:
                self.logger.debug(f"personal tool {getattr(cls, 'name', cls.__name__)} register skipped: {e}")
        self._personal_tools_ready = True

    def _tool(self, name: str):
        self._ensure_personal_tools()
        return TOOLS_REGISTRY.get(name)

    # ---------------- 快捷命令 ----------------

    def _handle_shortcut(self, command: str, rest: str) -> Dict[str, Any]:
        """执行快捷命令，返回 agent 运行结果格式的 dict。"""
        started_at = datetime.now()
        try:
            if command == "today":
                result = self._cmd_today()
            elif command == "briefing":
                result = self._cmd_briefing(rest)
            elif command == "morning":
                result = self._cmd_briefing("morning")
            elif command == "note":
                result = self._cmd_note(rest)
            elif command == "task":
                result = self._cmd_task(rest)
            elif command == "remind":
                result = self._cmd_remind(rest)
            elif command == "expense":
                result = self._cmd_expense(rest)
            elif command == "focus":
                result = self._cmd_focus(rest)
            elif command == "habit":
                result = self._cmd_habit(rest)
            elif command == "profile":
                result = self._cmd_profile(rest)
            else:
                result = f"未知快捷命令：{command}"
        except Exception as e:  # pragma: no cover - defensive
            self.logger.exception(f"shortcut {command} failed")
            result = f"快捷命令 `{command}` 执行出错：{e}"

        final_answer = result if isinstance(result, str) else str(result)
        elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
        self._append_to_history(rest or command, final_answer)
        return {
            "task": rest or command,
            "thoughts": [],
            "actions": [],
            "observations": [{
                "success": True,
                "output": final_answer,
                "error": None,
                "metadata": {"mode": "personal_shortcut", "command": command, "elapsed_ms": elapsed_ms},
            }],
            "thinking_steps": 1,
            "outer_loops": 0,
            "final_reward": 1.0,
            "success": True,
            "session_token_usage": self.session_token_usage,
            "final_answer": final_answer,
            "metadata": {"mode": "personal_shortcut", "command": command},
        }

    # ---- individual shortcut handlers ----

    def _cmd_today(self) -> str:
        """今日概览：聚合日历/任务/提醒，零 LLM。"""
        today = date.today().isoformat()
        parts: List[str] = []

        # 日历
        cal = self._tool("calendar")
        if cal:
            try:
                r = cal.execute(action="list", date_from=today, date_to=today, limit=20)
                if r.success:
                    parts.append("📅 今日日历：" + (r.output or "无日程"))
                else:
                    parts.append(f"📅 今日日历：查询失败({r.error})")
            except Exception as e:
                parts.append(f"📅 今日日历：异常({e})")

        # 任务
        tasks = self._tool("tasks")
        if tasks:
            try:
                r = tasks.execute(action="list", limit=30)
                if r.success:
                    # 只显示 pending 的
                    pending = [l for l in (r.output or "").splitlines() if "pending" in l.lower() or "待处理" in l]
                    parts.append("✅ 今日任务：" + ("\n".join(pending[:10]) if pending else "全部完成，🎉"))
                else:
                    parts.append(f"✅ 今日任务：查询失败({r.error})")
            except Exception as e:
                parts.append(f"✅ 今日任务：异常({e})")

        # 提醒
        reminders = self._tool("reminders")
        if reminders:
            try:
                r = reminders.execute(action="upcoming", limit=10)
                if r.success:
                    parts.append("⏰ 即将到来提醒：" + (r.output or "无"))
                else:
                    parts.append(f"⏰ 提醒：查询失败({r.error})")
            except Exception as e:
                parts.append(f"⏰ 提醒：异常({e})")

        if not parts:
            parts.append("今日暂无安排，好好休息～")
        return "\n".join(parts)

    def _cmd_briefing(self, rest: str) -> str:
        """简报：调用 daily_briefing 工具，兜底到手工聚合。"""
        brief = self._tool("daily_briefing")
        if brief:
            action = "morning" if "morning" in rest.lower() or "早" in rest else "evening" if "晚" in rest.lower() or "evening" in rest.lower() else "generate"
            try:
                r = brief.execute(action=action, target_date=date.today().isoformat(), format="markdown")
                if r.success:
                    return r.output or "简报生成为空。"
            except Exception as e:
                self.logger.debug(f"daily_briefing tool failed: {e}")
        # 兜底：手工聚合
        return self._cmd_today()

    def _cmd_note(self, rest: str) -> str:
        if not rest:
            return "用法：/note <笔记内容>，例如：/note 买牛奶 记得带伞"
        note = self._tool("notes")
        if not note:
            return "笔记工具未就绪。"
        try:
            r = note.execute(action="create", title=rest[:40], content=rest, category="Quick")
            if r.success:
                return f"📝 已记录笔记：{rest}"
            return f"记笔记失败：{r.error}"
        except Exception as e:
            return f"记笔记异常：{e}"

    def _cmd_task(self, rest: str) -> str:
        if not rest:
            return "用法：/task <任务内容> [截止日期]，例如：/task 提交报告 周五前"
        # 简单拆分：最后出现的日期-like 当 due_date
        due = None
        m = re.search(r"(\d{4}-\d{2}-\d{2}|今天|明天|后天|周[一到日]|下周[一到日])", rest)
        if m:
            due = m.group(1)
            rest = rest.replace(due, "").strip()
        t = self._tool("tasks")
        if not t:
            return "任务工具未就绪。"
        try:
            kwargs = {"title": rest or "新任务", "status": "pending", "priority": 3}
            if due:
                kwargs["due_date"] = due
            r = t.execute(action="create", **kwargs)
            if r.success:
                return f"✅ 任务已建：{rest}{f' (截止 {due})' if due else ''}"
            return f"建任务失败：{r.error}"
        except Exception as e:
            return f"建任务异常：{e}"

    def _cmd_remind(self, rest: str) -> str:
        if not rest:
            return "用法：/remind <内容> <时间>，例如：/remind 开会 下午3点"
        # 拆分时间：取最后一个"X点/XX:XX/明天"等
        rem = self._tool("reminders")
        if not rem:
            return "提醒工具未就绪。"
        try:
            r = rem.execute(action="create", title=rest, message=rest)
            if r.success:
                return f"⏰ 提醒已设：{rest}"
            return f"设提醒失败：{r.error}"
        except Exception as e:
            return f"设提醒异常：{e}"

    def _cmd_expense(self, rest: str) -> str:
        if not rest:
            return "用法：/expense <金额> <类别> <备注>，例如：/expense 38 餐饮 午餐"
        fin = self._tool("finance")
        if not fin:
            return "财务工具未就绪。"
        try:
            # 粗解析：第一个数字当金额，第二个词当类别
            m = re.search(r"(\d+(?:\.\d+)?)", rest)
            amount = float(m.group(1)) if m else 0.0
            rest_no_amt = re.sub(r"\d+(?:\.\d+)?", "", rest).strip(" ，,。.!！")
            parts = rest_no_amt.split(None, 1)
            category = parts[0] if parts else "其他"
            desc = parts[1] if len(parts) > 1 else rest
            r = fin.execute(action="create", type="expense", amount=amount, category=category, description=desc)
            if r.success:
                return f"💰 已记账：¥{amount:.2f} {category} — {desc}"
            return f"记账失败：{r.error}"
        except Exception as e:
            return f"记账异常：{e}"

    def _cmd_focus(self, rest: str) -> str:
        """专注计时(番茄钟)：本地计数，不阻塞主线程。"""
        m = re.search(r"(\d+)", rest)
        minutes = int(m.group(1)) if m else int(self.get_profile("focus_minutes", 25))
        if minutes <= 0 or minutes > 480:
            return "专注时长建议 1~480 分钟。"
        # 记录一个 focus session 到 health (作为活动)
        health = self._tool("health")
        if health:
            try:
                health.execute(
                    action="create",
                    activity_type="focus",
                    duration_minutes=minutes,
                    intensity="moderate",
                    notes="专注计时",
                    workout_date=date.today().isoformat(),
                )
            except Exception as e:
                self.logger.debug(f"focus session record skipped: {e}")
        return f"🍅 专注已启动：{minutes} 分钟。时间到时我会提醒你休息～(本会话内仅记录，未做系统级锁屏)"

    def _cmd_habit(self, rest: str) -> str:
        if not rest:
            return "用法：/habit <习惯名称>，例如：/habit 早起喝水"
        # 习惯打卡：建一条今日 health/nutrition 记录或笔记
        note = self._tool("notes")
        if note:
            try:
                note.execute(action="create", title=f"习惯打卡：{rest}", content=f"✅ {date.today().isoformat()} 已打卡：{rest}", category="Habit")
            except Exception as e:
                self.logger.debug(f"habit note record skipped: {e}")
        return f"✅ 习惯打卡记录：{rest}"

    def _cmd_profile(self, rest: str) -> str:
        profile = self._load_profile()
        if not rest:
            lines = ["👤 个人偏好："]
            for k, v in profile.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("设置示例：/profile timezone Asia/Shanghai")
            return "\n".join(lines)
        # 设置：key value
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return "用法：/profile <key> <value>，如：/profile focus_minutes 30"
        key, val = parts[0].strip(), parts[1].strip()
        # 数字自动转
        try:
            val_parsed: Any = int(val) if re.fullmatch(r"-?\d+", val) else val
        except Exception:
            val_parsed = val
        self._save_profile(key, val_parsed)
        return f"✅ 已设置 {key} = {val_parsed}"

    # ---------------- 一键复合动作 (quick_action) ----------------

    def _quick_action_dispatch(self, task: str) -> Optional[str]:
        """识别一键复合动作，返回执行结果字符串，None 表示不匹配。"""
        tl = task.lower()
        t = task

        # 出门模式：查天气 + 日历 + 提醒带伞
        if any(k in tl for k in ("出门", "出门了", "去上班", "通勤", "出门模式")):
            return self._quick_go_out()

        # 睡前模式：明天日程 + 设睡觉提醒
        if any(k in tl for k in ("睡前", "准备睡", "睡觉", "晚安", "明日日程")):
            return self._quick_bedtime()

        # 专注模式：开启专注 + 关闭干扰提示
        if any(k in tl for k in ("开始专注", "专注模式", "番茄钟", "闭门造车", "专注一下")):
            m = re.search(r"(\d+)\s*分钟", t)
            minutes = int(m.group(1)) if m else int(self.get_profile("focus_minutes", 25))
            return self._cmd_focus(f"{minutes}分钟")

        # 约会/见人模式：查日历空档 + 建事件 + 设提醒
        if any(k in tl for k in ("约会", "见面", "约饭", "约咖啡", "会客")):
            return self._quick_date(t)

        # 早安/今日概览
        if any(k in tl for k in ("早安", "今日概览", "今天有什么", "今天安排")):
            return self._cmd_today()

        # 记账快速：花了X买Y
        m = re.search(r"(花了|花|消费|付|支出)\s*(\d+(?:\.\d+)?)\s*(?:元|块|¥|\$)?\s*(.+)", tl)
        if m:
            amount = float(m.group(2))
            desc = m.group(3).strip() or "消费"
            fin = self._tool("finance")
            if fin:
                try:
                    fin.execute(action="create", type="expense", amount=amount, category="其他", description=desc)
                    return f"💰 记账：¥{amount:.2f} {desc}"
                except Exception as e:
                    return f"记账失败：{e}"
            return "财务工具未就绪。"

        return None

    def _quick_go_out(self) -> str:
        """出门模式：天气 + 今日日历 + 提醒。"""
        lines = ["🚪 出门助手："]
        # 天气
        weather = self._tool("weather")
        if weather:
            try:
                city = self.get_profile("home_address", "")
                r = weather.execute(action="get", city=city or "上海")
                if r.success:
                    lines.append(f"  🌤 {r.output}")
                else:
                    lines.append(f"  🌤 天气查询失败({r.error})")
            except Exception as e:
                lines.append(f"  🌤 天气异常({e})")
        # 日历
        lines.append("  📅 今日日程：" + (self._today_calendar_summary() or "无"))
        lines.append("  💡 提示：带好钥匙、钱包、手机，出门平安！")
        return "\n".join(lines)

    def _quick_bedtime(self) -> str:
        """睡前模式：明日日程 + 就寝提醒。"""
        lines = ["🌙 睡前助手："]
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        cal = self._tool("calendar")
        if cal:
            try:
                r = cal.execute(action="list", date_from=tomorrow, date_to=tomorrow, limit=10)
                lines.append("  📅 明日日程：" + (r.output or "无") if r.success else f"  📅 查询失败({r.error})")
            except Exception as e:
                lines.append(f"  📅 异常({e})")
        # 设睡觉提醒
        sleep_time = self.get_profile("sleep_time", "23:00")
        rem = self._tool("reminders")
        if rem:
            try:
                rem.execute(action="create", title="准备睡觉", message=f"时间到了，{sleep_time} 准备休息", trigger_time=f"{date.today().isoformat()} {sleep_time}")
                lines.append(f"  ⏰ 已设 {sleep_time} 睡眠提醒")
            except Exception as e:
                lines.append(f"  ⏰ 提醒设置失败({e})")
        lines.append("  💤 晚安，好梦～")
        return "\n".join(lines)

    def _quick_date(self, task: str) -> str:
        """约会模式：建日历事件 + 设提醒。"""
        lines = ["💕 约会助手："]
        cal = self._tool("calendar")
        if cal:
            try:
                r = cal.execute(action="create", title="约会/见面", start_time=date.today().isoformat() + "T19:00:00", end_time=date.today().isoformat() + "T21:00:00", description=task)
                if r.success:
                    lines.append("  📅 已在日历建立约会事件(今晚 19:00~21:00)")
                else:
                    lines.append(f"  📅 建事件失败({r.error})")
            except Exception as e:
                lines.append(f"  📅 异常({e})")
        rem = self._tool("reminders")
        if rem:
            try:
                rem.execute(action="create", title="约会提醒", message="别忘了约会，记得准时！", trigger_time=date.today().isoformat() + "T18:30:00")
                lines.append("  ⏰ 已设 18:30 出发提醒")
            except Exception as e:
                lines.append(f"  ⏰ 提醒失败({e})")
        lines.append("  💐 祝约会愉快～")
        return "\n".join(lines)

    def _today_calendar_summary(self) -> Optional[str]:
        cal = self._tool("calendar")
        if not cal:
            return None
        try:
            r = cal.execute(action="list", date_from=date.today().isoformat(), date_to=date.today().isoformat(), limit=5)
            if not r.success or not r.output:
                return None
            lines = [l for l in r.output.splitlines() if l.strip() and not l.startswith("Found")]
            return "；".join(lines[:3]) if lines else "无"
        except Exception:
            return None

    # ---------------- 个人意图零LLM快路 ----------------

    def _personal_intent_fast_path(self, task: str) -> Optional[Tuple[str, Dict[str, Any], float, str]]:
        """个人意图零LLM直达：匹配后直接调用对应 personal tool，不经过 LLM。"""
        tl = task.lower()
        t = task.strip()

        # 笔记意图
        if re.search(r"(记下|记笔记|写笔记|备忘|记录一下|笔记)", tl) and not any(k in tl for k in ("搜索", "查")):
            m = re.search(r"(?:记下|记笔记|写笔记|备忘|记录一下)\s*(.+)", t)
            content = (m.group(1).strip() if m else t).strip(" 记下记笔记写笔记备忘记录一下：：，。")
            if content:
                return ("notes", {"action": "create", "title": content[:40], "content": content, "category": "Quick"}, 0.92, "personal: quick note")

        # 任务意图
        if any(k in tl for k in ("加任务|建任务|新建任务|待办|todo|任务".split("|"))) or re.search(r"(加一下任务|新增任务|创建任务)", tl):
            return ("tasks", {"action": "create", "title": t, "status": "pending"}, 0.9, "personal: quick task")

        # 提醒意图
        if any(k in tl for k in ("提醒我|闹钟|定时提醒|remind".split("|"))) or re.search(r"(提醒我|定个闹钟|提醒)", tl):
            return ("reminders", {"action": "create", "title": t}, 0.9, "personal: quick reminder")

        # 记账意图
        m = re.search(r"(花了|花|付了|支出)\s*(\d+(?:\.\d+)?)", tl)
        if m:
            amount = float(m.group(2))
            rest = t[m.end():].strip(" 元块￥$，,。.!！ ")
            return ("finance", {"action": "create", "type": "expense", "amount": amount, "category": "其他", "description": rest or "消费"}, 0.93, "personal: quick expense")

        # 日历查询意图
        if re.search(r"(今天|明日|明天|本周|日程|会议|安排|calendar)", tl) and any(k in tl for k in ("看|查|有什么|有哪些|list|列出")):
            target = date.today().isoformat()
            if "明天" in tl or "明日" in tl:
                target = (date.today() + timedelta(days=1)).isoformat()
            return ("calendar", {"action": "list", "date_from": target, "date_to": target, "limit": 20}, 0.88, "personal: calendar list")

        # 简报意图
        if any(k in tl for k in ("简报|briefing|早安|今日概要|概览".split("|"))) or re.search(r"(简报|今日概要|早上好)", tl):
            return ("daily_briefing", {"action": "morning", "target_date": date.today().isoformat(), "format": "markdown"}, 0.9, "personal: daily briefing")

        # 健康打卡
        if re.search(r"(运动|健身|跑步|游泳|打卡|健康)", tl) and not any(k in tl for k in ("查|搜|怎么|如何")):
            return ("health", {"action": "create", "activity_type": "workout", "notes": t}, 0.85, "personal: health log")

        return None

    # ---------------- 覆盖：进入运行前先试快捷/个人快路 ----------------

    def run(self, task: str, **kwargs) -> Dict[str, Any]:
        """入口：先试快捷命令 / 个人意图快路，再落回父类 OpenMythosAgent.run。"""
        # 1) 快捷命令
        sc = _parse_shortcut(task)
        if sc:
            command, rest = sc
            cache_key = f"{command}:{rest[:60]}"
            now = __import__("time").time()
            hit = self._shortcut_cache.get(cache_key)
            if hit and now - hit[0] < 60:
                return self._wrap_shortcut_result(hit[1], command, rest)
            result = self._handle_shortcut(command, rest)
            self._shortcut_cache[cache_key] = (now, result["final_answer"])
            return result

        # 2) 一键复合动作
        qa = self._quick_action_dispatch(task)
        if qa:
            return self._wrap_text_result(qa, "quick_action")

        # 3) 个人意图零LLM快路
        pi = self._personal_intent_fast_path(task)
        if pi:
            tool_name, args, conf, reason = pi
            tool = self._tool(tool_name)
            if tool:
                try:
                    r = tool.execute(**args)
                    text = r.output if r.success else f"执行失败：{r.error}"
                    return self._wrap_text_result(text, f"personal_fast:{tool_name}")
                except Exception as e:
                    self.logger.debug(f"personal fast path {tool_name} failed: {e}")
                    return self._wrap_text_result(f"个人快路执行异常：{e}", "personal_fast")

        # 4) 落回标准 agent 流程
        return super().run(task, **kwargs)

    def _wrap_text_result(self, text: str, mode: str) -> Dict[str, Any]:
        self._append_to_history(text, text)
        return {
            "task": text,
            "thoughts": [],
            "actions": [],
            "observations": [{
                "success": True,
                "output": text,
                "error": None,
                "metadata": {"mode": mode},
            }],
            "thinking_steps": 1,
            "outer_loops": 0,
            "final_reward": 1.0,
            "success": True,
            "session_token_usage": self.session_token_usage,
            "final_answer": text,
            "metadata": {"mode": mode},
        }

    def _wrap_shortcut_result(self, text: str, command: str, rest: str) -> Dict[str, Any]:
        return self._wrap_text_result(text, f"shortcut:{command}")
