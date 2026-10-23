"""
Quick Action Tool - 一键复合动作。
零 LLM：一条指令完成“出门/睡前/专注/约会/今日”多工具联动，供个人快捷入口与 MCP 复用。
内部通过 TOOLS_REGISTRY 调用 calendar/reminders/weather/finance/health/notes，失败单项兜底不影响整体。
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY


def _call(tool_name: str, **kwargs) -> ToolResult:
    tool = TOOLS_REGISTRY.get(tool_name)
    if tool is None:
        return ToolResult(success=False, output="", error=f"tool {tool_name} not registered")
    try:
        return tool.execute(**kwargs)
    except Exception as e:  # pragma: no cover - defensive per-call isolation
        return ToolResult(success=False, output="", error=f"{tool_name}: {e}")


class QuickActionTool(BaseTool):
    name = "quick_action"
    description = (
        "One-tap compound personal actions. Actions: go_out (weather+today+reminders), "
        "bedtime (tomorrow agenda+sleep reminder), focus (focus timer record), date "
        "(calendar event+reminder), today (today overview). Each sub-action degrades "
        "gracefully if a dependent tool is unavailable."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["go_out", "bedtime", "focus", "date", "today"],
                "description": "Compound action to execute.",
            },
            "minutes": {
                "type": "integer",
                "description": "Focus minutes (for 'focus', default 25).",
            },
            "detail": {
                "type": "string",
                "description": "Free-form detail/context (for 'date', 'focus').",
            },
        },
        "required": ["action"],
    }

    def execute(self, action: str, minutes: Optional[int] = None,
                detail: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            if action == "go_out":
                return self._go_out()
            if action == "bedtime":
                return self._bedtime()
            if action == "focus":
                return self._focus(minutes or 25, detail or "")
            if action == "date":
                return self._date(detail or "")
            if action == "today":
                return self._today()
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # ---- compound actions ----

    def _go_out(self) -> ToolResult:
        lines: List[str] = ["🚪 出门助手："]
        # weather
        w = _call("weather", location="上海", days=1)
        if w.success:
            lines.append(f"  🌤 {w.output.strip().splitlines()[0] if w.output else '查询中'}")
        else:
            lines.append(f"  🌤 天气查询失败({w.error})")
        # today calendar
        c = _call("calendar", action="list", date_from=date.today().isoformat(),
                  date_to=date.today().isoformat(), limit=5)
        cal_summary = self._summarize_calendar(c)
        lines.append(f"  📅 今日日程：{cal_summary or '无'}")
        lines.append("  💡 提示：带好钥匙、钱包、手机，出门平安！")
        return ToolResult(success=True, output="\n".join(lines))

    def _bedtime(self) -> ToolResult:
        lines: List[str] = ["🌙 睡前助手："]
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        c = _call("calendar", action="list", date_from=tomorrow, date_to=tomorrow, limit=10)
        if c.success:
            lines.append(f"  📅 明日日程：{c.output.strip().splitlines()[0] if c.output else '无'}")
        else:
            lines.append(f"  📅 查询失败({c.error})")
        # sleep reminder at 23:00 default
        r = _call("reminders", action="create", title="准备睡觉",
                  message="时间到了，准备休息～",
                  trigger_time=f"{date.today().isoformat()}T23:00:00")
        if r.success:
            lines.append("  ⏰ 已设 23:00 睡眠提醒")
        else:
            lines.append(f"  ⏰ 提醒设置失败({r.error})")
        lines.append("  💤 晚安，好梦～")
        return ToolResult(success=True, output="\n".join(lines))

    def _focus(self, minutes: int, detail: str) -> ToolResult:
        if minutes <= 0 or minutes > 480:
            return ToolResult(success=False, output="", error="专注时长建议 1~480 分钟")
        h = _call("health", action="log_workout", activity_type="focus",
                  duration_minutes=minutes, intensity="moderate",
                  notes=detail or "专注计时", workout_date=date.today().isoformat())
        msg = f"🍅 专注已启动：{minutes} 分钟。"
        if not h.success:
            msg += f" (健康记录跳过：{h.error})"
        msg += " 时间到时休息一下～"
        return ToolResult(success=True, output=msg)

    def _date(self, detail: str) -> ToolResult:
        lines: List[str] = ["💕 约会助手："]
        start = f"{date.today().isoformat()}T19:00:00"
        end = f"{date.today().isoformat()}T21:00:00"
        c = _call("calendar", action="create", title="约会/见面", start_time=start,
                  end_time=end, description=detail or "约会")
        if c.success:
            lines.append("  📅 已在日历建立约会事件(今晚 19:00~21:00)")
        else:
            lines.append(f"  📅 建事件失败({c.error})")
        r = _call("reminders", action="create", title="约会提醒",
                  message="别忘了约会，记得准时！",
                  trigger_time=f"{date.today().isoformat()}T18:30:00")
        if r.success:
            lines.append("  ⏰ 已设 18:30 出发提醒")
        else:
            lines.append(f"  ⏰ 提醒设置失败({r.error})")
        lines.append("  💐 祝约会愉快～")
        return ToolResult(success=True, output="\n".join(lines))

    def _today(self) -> ToolResult:
        parts: List[str] = []
        today = date.today().isoformat()
        c = _call("calendar", action="list", date_from=today, date_to=today, limit=20)
        if c.success:
            parts.append("📅 今日日历：" + (c.output or "无日程"))
        t = _call("tasks", action="list", limit=30)
        if t.success:
            pending = [l for l in (t.output or "").splitlines() if "pending" in l.lower() or "待处理" in l]
            parts.append("✅ 今日任务：" + ("\n".join(pending[:10]) if pending else "全部完成，🎉"))
        r = _call("reminders", action="upcoming", limit=10)
        if r.success:
            parts.append("⏰ 即将到来提醒：" + (r.output or "无"))
        if not parts:
            parts.append("今日暂无安排，好好休息～")
        return ToolResult(success=True, output="\n".join(parts))

    @staticmethod
    def _summarize_calendar(res: ToolResult) -> str:
        if not res.success or not res.output:
            return ""
        lines = [l for l in res.output.splitlines() if l.strip()
                 and not l.startswith("Found")]
        return "；".join(lines[:3]) if lines else "无"


TOOLS_REGISTRY.register(QuickActionTool())
