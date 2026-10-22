"""
Natural Language Parameter Extractor - Extracts tool parameters from natural language.
Converts user utterances into structured tool calls with proper parameter mapping.
"""

import re
import json
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from . import TOOLS_REGISTRY, BaseTool
from .datetime_parser import parse_datetime, extract_recurrence, ParsedDateTime, TimeGranularity


class ToolType(Enum):
    CALENDAR = "calendar"
    TASKS = "tasks"
    NOTES = "notes"
    REMINDERS = "reminders"
    CONTACTS = "contacts"
    FINANCE = "finance"
    TRAVEL = "travel"
    HEALTH = "health"
    DAILY_BRIEFING = "daily_briefing"


@dataclass
class ExtractedParameters:
    """Extracted parameters for a tool call."""
    tool_name: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    original_text: str = ""
    missing_required: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)


class ParameterExtractor:
    """Extracts tool parameters from natural language text."""
    
    # Tool-specific action keywords
    ACTION_KEYWORDS = {
        ToolType.CALENDAR: {
            "create": ["安排", "添加", "创建", "新建", "schedule", "add", "create", "new"],
            "read": ["查看", "查询", "看", "read", "view", "show"],
            "update": ["修改", "更新", "改", "update", "modify", "change"],
            "delete": ["删除", "取消", "remove", "delete", "cancel"],
            "list": ["列表", "列出", "所有", "list", "all"],
            "search": ["搜索", "找", "search", "find"],
        },
        ToolType.TASKS: {
            "create": ["添加任务", "新任务", "创建任务", "add task", "new task", "create task"],
            "read": ["查看任务", "任务详情", "read task"],
            "update": ["修改任务", "更新任务", "update task"],
            "delete": ["删除任务", "remove task", "delete task"],
            "list": ["任务列表", "所有任务", "list tasks", "my tasks"],
            "complete": ["完成", "做完", "complete", "done", "finish"],
            "reopen": ["重新打开", "reopen"],
            "search": ["搜索任务", "找任务", "search task"],
        },
        ToolType.NOTES: {
            "create": ["记笔记", "新笔记", "记录", "note", "add note", "create note"],
            "read": ["看笔记", "查看笔记", "read note"],
            "update": ["修改笔记", "update note"],
            "delete": ["删除笔记", "delete note"],
            "list": ["笔记列表", "所有笔记", "list notes"],
            "search": ["搜索笔记", "找笔记", "search note"],
            "pin": ["置顶", "pin"],
            "unpin": ["取消置顶", "unpin"],
        },
        ToolType.REMINDERS: {
            "create": ["提醒我", "设置提醒", "闹钟", "remind me", "set reminder", "alarm"],
            "read": ["查看提醒", "read reminder"],
            "update": ["修改提醒", "update reminder"],
            "delete": ["删除提醒", "取消提醒", "delete reminder"],
            "list": ["提醒列表", "所有提醒", "list reminders"],
            "complete": ["完成提醒", "complete reminder"],
            "snooze": ["延迟", "稍后", "snooze"],
        },
        ToolType.CONTACTS: {
            "create": ["添加联系人", "新联系人", "add contact", "new contact"],
            "read": ["查看联系人", "read contact"],
            "update": ["修改联系人", "update contact"],
            "delete": ["删除联系人", "delete contact"],
            "list": ["联系人列表", "所有联系人", "list contacts"],
            "search": ["搜索联系人", "找联系人", "search contact"],
            "favorite": ["收藏", "star", "favorite"],
        },
        ToolType.FINANCE: {
            "add_transaction": ["记账", "记一笔", "花了", "收入", "支出", "log expense", "log income", "spent", "earned"],
            "read_transaction": ["查看账单", "read transaction"],
            "update_transaction": ["修改账单", "update transaction"],
            "delete_transaction": ["删除账单", "delete transaction"],
            "list_transactions": ["账单列表", "流水", "list transactions"],
            "summary": ["总结", "汇总", "报表", "summary", "report"],
        },
        ToolType.TRAVEL: {
            "create_trip": ["新建旅行", "计划旅行", "plan trip", "new trip"],
            "read_trip": ["查看旅行", "read trip"],
            "add_itinerary": ["添加行程", "add itinerary"],
            "add_packing": ["添加打包", "add packing"],
            "add_booking": ["预订", "book", "add booking"],
        },
        ToolType.HEALTH: {
            "log_workout": ["运动", "锻炼", "跑步", "健身", "workout", "exercise", "run", "gym"],
            "log_nutrition": ["吃了", "早餐", "午餐", "晚餐", "零食", "喝水", "meal", "food", "ate", "water"],
            "log_sleep": ["睡觉", "睡眠", "sleep"],
            "log_vitals": ["体重", "心率", "血压", "体温", "weight", "heart rate", "blood pressure"],
            "set_goal": ["目标", "goal", "target"],
        },
        ToolType.DAILY_BRIEFING: {
            "morning": ["早报", "晨报", "早安", "morning briefing", "morning"],
            "evening": ["晚报", "晚安", "复盘", "evening briefing", "evening", "review"],
            "weekly": ["周报", "周总结", "weekly briefing", "weekly"],
        },
    }
    
    # Parameter extraction patterns per tool
    PARAM_PATTERNS = {
        ToolType.CALENDAR: {
            "title": [r'(?:安排|添加|创建|schedule|add)\s*[：:]?\s*([^，,。.]+)', r'^([^，,。.]+?)(?:\s|$)'],
            "start_time": [r'(\d{1,2})[点时:：](\d{2})?', r'at\s+(\d{1,2}):(\d{2})'],
            "location": [r'(?:在|@|location)\s*[：:]?\s*([^，,。.]{1,30})', r'@([^，,。.]{1,30})'],
            "description": [r'(?:备注|描述|note)\s*[：:]?\s*([^，,。.]+)'],
        },
        ToolType.TASKS: {
            "title": [r'(?:任务|task)\s*[：:]?\s*([^，,。.]+)', r'^(?:要|需要|必须)\s*([^，,。.]+)'],
            "project": [r'(?:项目|project)\s*[：:]?\s*([^，,。.]+)', r'\[([^\]]+)\]'],
            "priority": [r'(?:优先级|priority)\s*[：:]?\s*(\d)', r'(?:高|high|urgent|紧急)'],
            "due_date": [
                r'(?:截止|deadline|due)\s*[：:]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                r'(?:截止|deadline|due)\s*[：:]?\s*(\d{1,2}月\d{1,2}日)',
                r'(?:截止|deadline|due)\s*[：:]?\s*(\d{1,2}[-/]\d{1,2})',
                r'(?:周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六日]|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            ],
            "tags": [r'#(\w+)', r'标签\s*[：:]?\s*([^，,。.]+)'],
        },
        ToolType.NOTES: {
            "title": [r'(?:笔记|note)\s*[：:]?\s*([^，,。.]+)', r'^([^：:]+?)[：:]'],
            "content": [r'(?:内容|content)\s*[：:]?\s*([^，,。.]+)', r'[：:]\s*(.+)'],
            "category": [r'(?:分类|category)\s*[：:]?\s*([^，,。.]+)'],
            "tags": [r'#(\w+)'],
        },
        ToolType.REMINDERS: {
            "title": [r'(?:提醒我|remind me)\s*[，,]?\s*([^，,。.]+)', r'提醒\s*[：:]?\s*([^，,。.]+)'],
            "trigger_time": [r'(\d{1,2})[点时:：](\d{2})?', r'at\s+(\d{1,2}):(\d{2})'],
            "recurrence": [r'(每天|每周|每月|每年|daily|weekly|monthly|yearly)'],
        },
        ToolType.FINANCE: {
            "type": [r'(收入|income|earned)|(支出|花了|spent|expense)'],
            "amount": [r'(\d+(?:\.\d+)?)\s*(?:元|块|美元|USD|CNY|\$|¥)', r'(\d+(?:\.\d+)?)\s*(?:块|元)'],
            "category": [r'(?:分类|category)\s*[：:]?\s*([^，,。.]+)', r'(餐饮|交通|购物|工资|娱乐|医疗|food|transport|shopping|salary|entertainment|medical)'],
            "account": [r'(?:账户|account)\s*[：:]?\s*([^，,。.]+)', r'(支付宝|微信|现金|银行卡|alipay|wechat|cash|bank)'],
            "description": [r'(?:备注|note)\s*[：:]?\s*([^，,。.]+)'],
        },
        ToolType.HEALTH: {
            "activity_type": [r'(跑步|running|游泳|swimming|骑行|cycling|力量|strength|瑜伽|yoga|散步|walking|健身|gym)'],
            "duration_minutes": [r'(\d+)\s*(?:分钟|min|分)'],
            "distance_km": [r'(\d+(?:\.\d+)?)\s*(?:公里|km|千米)'],
            "calories": [r'(\d+)\s*(?:卡路里|cal|千卡|kcal)'],
            "meal_type": [r'(早餐|breakfast|午餐|lunch|晚餐|dinner|零食|snack)'],
            "food_items": [r'(?:吃了|ate|食物|food)\s*[：:]?\s*([^，,。.]+)'],
            "calories_consumed": [r'(\d+)\s*(?:卡路里|cal|千卡|kcal)'],
            "protein_g": [r'(\d+(?:\.\d+)?)\s*(?:克|g)\s*蛋白'],
            "water_ml": [r'(\d+)\s*(?:毫升|ml|毫升)\s*水'],
        },
    }
    
    def __init__(self):
        self.datetime_parser = None  # Will use module-level functions
    
    def extract(self, text: str, context: Optional[Dict] = None) -> List[ExtractedParameters]:
        """Extract all possible tool calls from text."""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        results = []
        
        # Try each tool type
        for tool_type in ToolType:
            extracted = self._extract_for_tool(tool_type, text, context)
            if extracted and extracted.confidence > 0.3:
                results.append(extracted)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def _extract_for_tool(self, tool_type: ToolType, text: str, context: Optional[Dict]) -> Optional[ExtractedParameters]:
        """Extract parameters for a specific tool type."""
        keywords = self.ACTION_KEYWORDS.get(tool_type, {})
        best_action = None
        best_confidence = 0.0
        
        # Find best matching action
        for action, kws in keywords.items():
            for kw in kws:
                if kw.lower() in text.lower():
                    confidence = len(kw) / len(text) * 0.5 + 0.3
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_action = action
        
        if not best_action:
            return None
        
        # Extract parameters
        params = self._extract_parameters(tool_type, best_action, text, context)
        
        # Add action to parameters (required by tool schemas)
        params["action"] = best_action
        
        # Check required parameters
        tool = TOOLS_REGISTRY.get(tool_type.value)
        required = []
        if tool:
            schema = tool.parameters
            required = schema.get("required", [])
        
        missing = [p for p in required if p not in params]
        
        return ExtractedParameters(
            tool_name=tool_type.value,
            action=best_action,
            parameters=params,
            confidence=best_confidence,
            original_text=text,
            missing_required=missing
        )
    
    def _extract_parameters(self, tool_type: ToolType, action: str, text: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Extract parameters for a specific tool action."""
        params = {}
        patterns = self.PARAM_PATTERNS.get(tool_type, {})
        
        # Extract common parameters using patterns
        for param_name, param_patterns in patterns.items():
            for pattern in param_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 0:
                        # No capture groups, use the whole match
                        params[param_name] = match.group(0).strip()
                    elif len(match.groups()) == 1 and match.group(1):
                        params[param_name] = match.group(1).strip()
                    elif len(match.groups()) >= 2:
                        # Handle multiple groups (e.g., hour, minute)
                        if param_name in ["start_time", "trigger_time"]:
                            hour = int(match.group(1))
                            minute = int(match.group(2)) if match.group(2) else 0
                            # Parse date context
                            dt = self._parse_datetime_context(text, hour, minute)
                            params[param_name] = dt.isoformat()
                        elif match.group(1):
                            params[param_name] = match.group(1).strip()
                    break
        
        # Tool-specific parameter extraction
        if tool_type == ToolType.CALENDAR:
            params.update(self._extract_calendar_params(text, action))
        elif tool_type == ToolType.TASKS:
            params.update(self._extract_tasks_params(text, action))
        elif tool_type == ToolType.REMINDERS:
            params.update(self._extract_reminders_params(text, action))
        elif tool_type == ToolType.FINANCE:
            params.update(self._extract_finance_params(text, action))
        elif tool_type == ToolType.HEALTH:
            params.update(self._extract_health_params(text, action))
        
        # Use context for missing parameters
        if context:
            for key, value in context.items():
                if key not in params and value is not None:
                    params[key] = value
        
        return params
    
    def _parse_datetime_context(self, text: str, hour: int, minute: int) -> datetime:
        """Parse date from text context and combine with time."""
        # Try to parse date from text
        try:
            parsed = parse_datetime(text)
            if parsed.granularity in [TimeGranularity.DAY, TimeGranularity.EXACT]:
                return parsed.start.replace(hour=hour, minute=minute)
        except (ValueError, AttributeError) as e:
            import logging
            logging.getLogger(__name__).debug("Failed to parse datetime context: %s", e)
        
        # Default to today
        return datetime.combine(date.today(), time(hour, minute))
    
    def _extract_calendar_params(self, text: str, action: str) -> Dict[str, Any]:
        params = {}
        
        # Extract end time (default 1 hour after start)
        if "start_time" in params and "end_time" not in params:
            try:
                start = datetime.fromisoformat(params["start_time"])
                params["end_time"] = (start + timedelta(hours=1)).isoformat()
            except (ValueError, TypeError) as e:
                import logging
                logging.getLogger(__name__).debug("Failed to calculate end time: %s", e)
        
        # Extract recurrence
        recurrence_match = extract_recurrence(text)
        if recurrence_match.get("is_recurring"):
            params["recurrence"] = recurrence_match["pattern"]
        
        return params
    
    def _extract_tasks_params(self, text: str, action: str) -> Dict[str, Any]:
        params = {}
        
        # Priority mapping
        if any(w in text.lower() for w in ["高", "high", "urgent", "紧急", "重要"]):
            params["priority"] = 1
        elif any(w in text.lower() for w in ["低", "low", "不急"]):
            params["priority"] = 5
        elif "priority" not in params:
            params["priority"] = 3
        
        # Tags from #hashtags
        tags = re.findall(r'#(\w+)', text)
        if tags:
            params["tags"] = tags
        
        return params
    
    def _extract_reminders_params(self, text: str, action: str) -> Dict[str, Any]:
        params = {}
        
        # Default notification methods
        params["notification_methods"] = ["popup", "sound"]
        
        # Recurrence
        recurrence_match = extract_recurrence(text)
        if recurrence_match.get("is_recurring"):
            params["recurrence"] = recurrence_match["pattern"]
        
        # Snooze minutes
        snooze_match = re.search(r'(?:延迟|snooze)\s*(\d+)\s*(?:分钟|min)', text)
        if snooze_match:
            params["snooze_minutes"] = int(snooze_match.group(1))
        
        return params
    
    def _extract_finance_params(self, text: str, action: str) -> Dict[str, Any]:
        params = {}
        
        # Determine transaction type
        if any(w in text for w in ["收入", "income", "earned", "工资", "salary"]):
            params["type"] = "income"
        elif any(w in text for w in ["支出", "花了", "spent", "expense", "买", "buy"]):
            params["type"] = "expense"
        else:
            # Default to expense for "记账"
            params["type"] = "expense"
        
        # Currency
        if any(w in text for w in ["$", "美元", "USD"]):
            params["currency"] = "USD"
        else:
            params["currency"] = "CNY"
        
        # Date (default today)
        if "transaction_date" not in params:
            params["transaction_date"] = date.today().isoformat()
        
        return params
    
    def _extract_health_params(self, text: str, action: str) -> Dict[str, Any]:
        params = {}
        
        # Date (default today)
        date_fields = ["workout_date", "nutrition_date", "sleep_date", "measured_at"]
        for field in date_fields:
            if field not in params:
                params[field] = date.today().isoformat()
        
        # Intensity for workouts
        if action == "log_workout" and "intensity" not in params:
            if any(w in text for w in ["高强度", "high intensity", "hiit", "冲刺"]):
                params["intensity"] = "high"
            elif any(w in text for w in ["低强度", "low intensity", "轻松", "easy"]):
                params["intensity"] = "low"
            else:
                params["intensity"] = "moderate"
        
        return params


# Global extractor instance
_extractor = ParameterExtractor()


def extract_tool_calls(text: str, context: Optional[Dict] = None) -> List[ExtractedParameters]:
    """Extract tool calls from natural language text."""
    return _extractor.extract(text, context)


def extract_best_tool_call(text: str, context: Optional[Dict] = None) -> Optional[ExtractedParameters]:
    """Extract the most likely tool call from text."""
    results = extract_tool_calls(text, context)
    return results[0] if results else None


# Integration with agent
def enhance_tool_call_with_nlp(tool_call: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """Enhance a tool call with NLP-extracted parameters."""
    tool_name = tool_call.get("tool_name") or tool_call.get("name")
    action = tool_call.get("action") or tool_call.get("arguments", {}).get("action")
    
    if not tool_name or not action:
        return tool_call
    
    # Extract additional parameters
    extracted = extract_best_tool_call(original_text)
    if extracted and extracted.tool_name == tool_name:
        # Merge parameters
        args = tool_call.get("arguments", {})
        for key, value in extracted.parameters.items():
            if key not in args and value is not None:
                args[key] = value
        tool_call["arguments"] = args
    
    return tool_call


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "安排明天下午3点在会议室A开会",
        "添加任务：完成报告，项目工作，优先级高，周五截止 #urgent #report",
        "记笔记：今天学到了新知识，分类工作",
        "提醒我明天上午9点喝水",
        "记一笔支出：午餐35元，支付宝，分类餐饮",
        "记录跑步：30分钟，5公里，消耗300卡路里",
        "早报",
        "新建旅行：东京5天，预算2万",
    ]
    
    for case in test_cases:
        print(f"\nInput: {case}")
        results = extract_tool_calls(case)
        for r in results[:2]:
            print(f"  Tool: {r.tool_name}, Action: {r.action}, Confidence: {r.confidence:.2f}")
            print(f"  Params: {json.dumps(r.parameters, ensure_ascii=False)}")
            if r.missing_required:
                print(f"  Missing: {r.missing_required}")