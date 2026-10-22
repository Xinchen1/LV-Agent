"""
Cross-Tool Automation Hooks - Automatic actions between personal assistant tools.
E.g., calendar event → reminder, finance budget exceeded → task, health goal missed → reminder.
"""

import json
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from . import TOOLS_REGISTRY, ToolResult


class AutomationEngine:
    """Manages cross-tool automation rules and executes them."""
    
    def __init__(self):
        self.rules: List[Dict] = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register built-in automation rules."""
        # Rule: Calendar event → Create reminder 15 min before
        self.add_rule(
            name="calendar_event_reminder",
            trigger_tool="calendar",
            trigger_action="create",
            condition=lambda result, params: True,
            action=self._create_event_reminder,
            description="Create reminder 15 minutes before calendar events"
        )
        
        # Rule: Finance expense over budget → Create task to review
        self.add_rule(
            name="budget_exceeded_task",
            trigger_tool="finance",
            trigger_action="add_transaction",
            condition=lambda result, params: params.get("type") == "expense",
            action=self._check_budget_and_create_task,
            description="Create task when category budget is exceeded"
        )
        
        # Rule: Health goal missed → Create reminder
        self.add_rule(
            name="health_goal_reminder",
            trigger_tool="health",
            trigger_action="progress",
            condition=lambda result, params: True,
            action=self._check_health_goals,
            description="Create reminders for missed health goals"
        )
        
        # Rule: Task with due date → Create reminder 1 day before
        self.add_rule(
            name="task_due_reminder",
            trigger_tool="tasks",
            trigger_action="create",
            condition=lambda result, params: params.get("due_date") is not None,
            action=self._create_task_due_reminder,
            description="Create reminder 1 day before task due date"
        )
    
    def add_rule(self, name: str, trigger_tool: str, trigger_action: str, 
                 condition: Callable, action: Callable, description: str = ""):
        """Add an automation rule."""
        self.rules.append({
            "name": name,
            "trigger_tool": trigger_tool,
            "trigger_action": trigger_action,
            "condition": condition,
            "action": action,
            "description": description,
            "enabled": True
        })
    
    def execute_rules(self, tool_name: str, action: str, result: ToolResult, params: Dict) -> List[ToolResult]:
        """Execute all matching rules for a tool action."""
        results = []
        for rule in self.rules:
            if not rule["enabled"]:
                continue
            if rule["trigger_tool"] == tool_name and rule["trigger_action"] == action:
                try:
                    if rule["condition"](result, params):
                        action_result = rule["action"](result, params)
                        if action_result:
                            results.append(action_result)
                except Exception as e:
                    # Log but don't fail the main operation
                    import logging
                    logging.getLogger(__name__).warning(f"Automation rule {rule['name']} failed: {e}")
        return results
    
    # Rule actions
    def _create_event_reminder(self, result: ToolResult, params: Dict) -> Optional[ToolResult]:
        """Create a reminder 15 minutes before calendar event."""
        if not result.success or not result.metadata.get("event_id"):
            return None
        
        event_id = result.metadata["event_id"]
        title = params.get("title", "Event")
        start_time = params.get("start_time")
        
        if not start_time:
            return None
        
        try:
            event_dt = datetime.fromisoformat(start_time)
            reminder_time = event_dt - timedelta(minutes=15)
            
            # Only create reminder if it's in the future
            if reminder_time <= datetime.now():
                return None
            
            reminders_tool = TOOLS_REGISTRY.get("reminders")
            if not reminders_tool:
                return None
            
            return reminders_tool.execute(
                "create",
                title=f"⏰ {title}",
                message=f"Starting in 15 minutes: {title}",
                trigger_time=reminder_time.isoformat(),
                notification_methods=["popup", "sound"]
            )
        except Exception:
            return None
    
    def _check_budget_and_create_task(self, result: ToolResult, params: Dict) -> Optional[ToolResult]:
        """Check if expense exceeds budget and create review task."""
        if not result.success:
            return None
        
        category = params.get("category")
        amount = params.get("amount", 0)
        
        if not category:
            return None
        
        # Check budget for this category
        from agent_project.tools.finance_tool import DB_PATH as FIN_DB
        conn = sqlite3.connect(FIN_DB)
        cursor = conn.cursor()
        
        # Get current month's budget for category
        today = date.today()
        month_start = today.replace(day=1).isoformat()
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()
        
        cursor.execute("""
            SELECT amount FROM budgets 
            WHERE category = ? AND is_active = 1 
            AND period = 'monthly' 
            AND start_date <= ? AND (end_date IS NULL OR end_date >= ?)
        """, (category, month_start, month_start))
        budget_row = cursor.fetchone()
        
        if not budget_row:
            conn.close()
            return None
        
        budget_limit = budget_row[0]
        
        # Get current month's spending in category
        cursor.execute("""
            SELECT SUM(amount) FROM transactions 
            WHERE type = 'expense' AND category = ? 
            AND transaction_date >= ? AND transaction_date < ?
        """, (category, month_start, next_month))
        spent = cursor.fetchone()[0] or 0
        
        conn.close()
        
        # If over 90% of budget, create task
        if spent >= budget_limit * 0.9:
            tasks_tool = TOOLS_REGISTRY.get("tasks")
            if not tasks_tool:
                return None
            
            return tasks_tool.execute(
                "create",
                title=f"💰 Review {category} budget",
                description=f"Spent ¥{spent:.0f} of ¥{budget_limit:.0f} budget ({spent/budget_limit*100:.0f}%). Consider adjusting spending.",
                project="Finance",
                priority=2,
                due_date=(datetime.now() + timedelta(days=1)).isoformat(),
                tags=["budget", "review", category]
            )
        
        return None
    
    def _check_health_goals(self, result: ToolResult, params: Dict) -> List[ToolResult]:
        """Check health goals and create reminders for missed ones."""
        reminders_tool = TOOLS_REGISTRY.get("reminders")
        if not reminders_tool:
            return []
        
        results = []
        metadata = result.metadata or {}
        goals = metadata.get("goals", [])
        
        for goal in goals:
            gtype = goal.get("goal_type")
            target = goal.get("target_value", 0)
            current = 0
            
            # Get current progress based on goal type
            if gtype == "daily_steps":
                current = goal.get("current", 0)  # From progress metadata
            elif gtype == "daily_water":
                current = goal.get("current", 0)
            elif gtype == "sleep_hours":
                current = goal.get("current", 0)
            
            if current > 0 and current < target * 0.8:  # Less than 80% of target
                pct = current / target * 100
                res = reminders_tool.execute(
                    "create",
                    title=f"🎯 Health Goal: {gtype}",
                    message=f"Only {pct:.0f}% of target ({current:.0f}/{target}). Keep going!",
                    trigger_time=(datetime.now() + timedelta(hours=2)).isoformat(),
                    notification_methods=["popup"]
                )
                if res:
                    results.append(res)
        
        return results
    
    def _create_task_due_reminder(self, result: ToolResult, params: Dict) -> Optional[ToolResult]:
        """Create reminder 1 day before task due date."""
        if not result.success or not result.metadata.get("task_id"):
            return None
        
        due_date_str = params.get("due_date")
        if not due_date_str:
            return None
        
        try:
            due_dt = datetime.fromisoformat(due_date_str)
            reminder_time = due_dt - timedelta(days=1)
            reminder_time = reminder_time.replace(hour=9, minute=0, second=0, microsecond=0)
            
            if reminder_time <= datetime.now():
                return None
            
            reminders_tool = TOOLS_REGISTRY.get("reminders")
            if not reminders_tool:
                return None
            
            title = params.get("title", "Task")
            return reminders_tool.execute(
                "create",
                title=f"📅 Due Tomorrow: {title}",
                message=f"Task '{title}' is due tomorrow ({due_dt.strftime('%Y-%m-%d')})",
                trigger_time=reminder_time.isoformat(),
                notification_methods=["popup", "sound"]
            )
        except Exception:
            return None


# Global automation engine instance
_AUTOMATION_ENGINE: Optional[AutomationEngine] = None


def get_automation_engine() -> AutomationEngine:
    """Get or create the global automation engine."""
    global _AUTOMATION_ENGINE
    if _AUTOMATION_ENGINE is None:
        _AUTOMATION_ENGINE = AutomationEngine()
    return _AUTOMATION_ENGINE


def trigger_automation(tool_name: str, action: str, result: ToolResult, params: Dict) -> List[ToolResult]:
    """Convenience function to trigger automation rules."""
    engine = get_automation_engine()
    return engine.execute_rules(tool_name, action, result, params)


# Monkey-patch tool execute methods to trigger automation
def _wrap_tool_execute(tool_name: str):
    """Wrap a tool's execute method to trigger automation after execution."""
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool or hasattr(tool, "_automation_wrapped"):
        return
    
    original_execute = tool.execute
    
    def wrapped_execute(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        # Extract params from args/kwargs
        params = {}
        if args and len(args) > 1:
            # First arg is usually 'action', rest are params
            pass
        params.update(kwargs)
        params["action"] = args[0] if args else kwargs.get("action", "")
        
        trigger_automation(tool_name, params.get("action", ""), result, params)
        return result
    
    tool.execute = wrapped_execute
    tool._automation_wrapped = True


def enable_automation_for_all_tools():
    """Enable automation for all personal assistant tools."""
    for tool_name in ["calendar", "tasks", "finance", "health", "reminders"]:
        _wrap_tool_execute(tool_name)


# Auto-enable when module is imported
enable_automation_for_all_tools()