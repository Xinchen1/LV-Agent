"""Tool execution, argument repair, redaction, and reward mixin for OpenMythosAgent.

Extracted from agent.py. Zero behavioural change: OpenMythosAgent keeps the same
public API by inheriting this mixin.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pure_utils import edit_distance
from .tools import TOOLS_REGISTRY, ToolCall, ToolResult


class AgentToolsMixin:
    """工具执行、参数修复、脱敏与奖励计算能力(供 OpenMythosAgent 多重继承)."""

    # ------------------------------------------------------------------ #
    # Tool argument repair
    # ------------------------------------------------------------------ #

    @staticmethod
    def _repair_tool_arguments(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """通用工具参数修复: 常见缺省/类型错误在执行前自动补齐, 减少无效调用."""
        if not isinstance(args, dict):
            return {"raw": str(args)}
        fixed = dict(args)
        if tool_name == "web_search":
            q = fixed.get("query") or fixed.get("q")
            if not isinstance(q, str) or not q.strip():
                fixed["query"] = ""
        elif tool_name == "file_ops":
            if not isinstance(fixed.get("path"), str) or not fixed.get("path", "").strip():
                fixed["path"] = "."  # 缺省列当前目录, 避免空调用报错
        elif tool_name == "python_exec":
            code = fixed.get("code") or fixed.get("program") or fixed.get("script")
            if not isinstance(code, str):
                fixed["code"] = str(code or "")
        elif tool_name == "bash_exec":
            cmd = fixed.get("command") or fixed.get("cmd")
            if not isinstance(cmd, str):
                fixed["command"] = str(cmd or "")
        elif tool_name == "glob":
            # LLM 常生成空参数 glob:{} → 缺省列当前目录所有文件
            pat = fixed.get("pattern") or fixed.get("query") or fixed.get("path")
            if not isinstance(pat, str) or not pat.strip():
                fixed["pattern"] = "**"
                fixed.setdefault("path", ".")
            elif not fixed.get("path"):
                fixed["path"] = "."
        elif tool_name == "search_files":
            pat = fixed.get("pattern") or fixed.get("query")
            if not isinstance(pat, str) or not pat.strip():
                fixed["pattern"] = ""
            fixed.setdefault("path", ".")
        elif tool_name in ("weather", "calculator", "api_call"):
            # 这些工具缺参时给空字符串兜底, 避免 TypeError
            for k in ("city", "expression", "url"):
                if k in fixed and fixed[k] is None:
                    fixed[k] = ""
        return fixed

    # ------------------------------------------------------------------ #
    # Redaction / formatting
    # ------------------------------------------------------------------ #

    _SECRET_RES = [
        re.compile(r'nvapi-[A-Za-z0-9_\-]{16,}'),
        re.compile(r'AIza[A-Za-z0-9_\-]{35}'),
        re.compile(r'sk-[A-Za-z0-9]{16,}'),
        re.compile(r'ghp_[A-Za-z0-9]{20,}'),
        re.compile(r'xox[bap]-[A-Za-z0-9\-]+'),
        re.compile(r'AKIA[0-9A-Z]{16}'),
    ]

    @classmethod
    def _redact_secrets(cls, text: str) -> str:
        """展示层脱敏: API key 明文绝不上屏(如读到的 .sh/.env 文件内容)。"""
        if not text:
            return text
        out = text
        for rx in cls._SECRET_RES:
            out = rx.sub(lambda m: m.group(0)[:7] + '••••••••', out)
        return out

    @classmethod
    def _format_tool_args(cls, arguments: Any, limit: int = 200) -> str:
        """工具调用参数的展示文本: JSON 而非 Python dict(双引号/中文可读)并截断超长值."""
        if isinstance(arguments, dict):
            try:
                text = json.dumps(arguments, ensure_ascii=False, default=str)
            except Exception:
                text = str(arguments)
        else:
            text = str(arguments)
        text = cls._redact_secrets(text)
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return text

    def _build_tool_retry_prompt(self, task: str, raw_answer: str,
                                 action: ToolCall, tool_result: ToolResult) -> str:
        """Build a corrective prompt for one-shot tool call retry."""
        return (
            "You are a helpful assistant. The previous tool call failed. "
            "Fix the arguments and call the tool again using EXACTLY JSON format.\n\n"
            f"Task: {task}\n"
            f"Your previous response: {raw_answer[:500]}\n"
            f"Failed call: {action.tool_name}({action.arguments})\n"
            f"Error: {tool_result.error or 'unknown error'}\n\n"
            "Output only the corrected tool call, e.g.:\n"
            '[TOOL:file_ops] {"action": "list", "path": "~/Desktop"} [/TOOL]'
        )

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _compute_fast_reward(self, final_answer: str, actions: List[Dict[str, Any]]) -> float:
        """快车道奖励: 基于产出质量 + 工具成败 + 置信度, 不再硬编码满分."""
        try:
            reward = 0.7
            conf = getattr(self, "_last_confidence", 0.5)
            if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                reward *= (0.6 + 0.4 * conf)
            answer = str(final_answer or "").strip()
            if not answer:
                reward *= 0.4
            elif len(answer) < 5:
                reward *= 0.7
            if actions:
                failed = sum(1 for a in actions if isinstance(a, dict) and not a.get("success", True))
                reward *= max(0.3, 1.0 - 0.5 * failed / max(1, len(actions)))
            return max(0.0, min(1.0, reward))
        except Exception:  # pragma: no cover - defensive
            return 0.7

    def _compute_reward(self, trajectory: Dict) -> float:
        """计算奖励(供经验/反思飞轮消费).

        信号分层(0~1):
        - 基础: success 布尔(0.0 / 0.5)
        - 质量: final_reward / quality_score(若已由推理引擎给出) 加权
        - 置信度: _last_confidence(内部评估) 轻度加权
        - 工具成败: 动作里失败数越多, 奖励越扣
        - 产出长度惩罚: 空答案或明显截断降权
        该分数驱动 _should_reflect/_trigger_self_improvement 的决策,
        必须反映真实执行质量而非"成功就满分"。
        """
        try:
            success = bool(trajectory.get("success", False))
            reward = 0.5 if success else 0.0

            # 质量信号: 优先用已算好的 final_reward/quality_score(0~1 归一化)
            q = trajectory.get("final_reward")
            if q is None:
                q = (trajectory.get("metadata") or {}).get("quality_score")
            if isinstance(q, (int, float)) and q > 0:
                try:
                    reward = 0.4 + 0.6 * max(0.0, min(1.0, float(q)))
                except Exception:
                    pass
                if not success:
                    reward *= 0.5

            # 置信度: 内部评估(0~1), 低置信削弱奖励
            conf = getattr(self, "_last_confidence", 0.5)
            if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                reward *= (0.6 + 0.4 * conf)

            # 工具成败: 动作列表中失败的工具调用降低奖励
            actions = trajectory.get("actions") or []
            if actions:
                failed = sum(1 for a in actions if isinstance(a, dict) and not a.get("success", True))
                reward *= max(0.2, 1.0 - 0.4 * failed / max(1, len(actions)))

            # 产出质量: 空答案/明显异常降权
            final_answer = str(trajectory.get("final_answer", "") or "").strip()
            if not final_answer:
                reward *= 0.5
            elif len(final_answer) < 5:
                reward *= 0.7

            return max(0.0, min(1.0, reward))
        except Exception:  # pragma: no cover - defensive
            return 1.0 if trajectory.get("success") else 0.0

    # ------------------------------------------------------------------ #
    # Tool call parsing
    # ------------------------------------------------------------------ #

    def _parse_output_for_action(self, text: str) -> Optional[ToolCall]:
        """Parse the first actionable tool call from model output.

        Supports: [TOOL:name] args [/TOOL], <provider>name</provider>, and
        OpenCode wrapped function call formats.
        Iterates over all parsed calls and returns the first one whose tool
        is actually registered (auto-correcting misspelled tool names),
        so an invalid/nonexistent leading call (e.g.
        <|message_model|>think...) does not block a valid later call.
        """
        for action in self._parse_all_output_actions(text):
            if TOOLS_REGISTRY.get(action.tool_name):
                return action
            # 工具名纠错: LLM 可能写错名, 尝试映射到合法工具
            corrected = self._correct_tool_name(action.tool_name)
            if corrected and TOOLS_REGISTRY.get(corrected):
                from .tools import ToolCall
                self.logger.info(f"tool name corrected: {action.tool_name!r} → {corrected}")
                return ToolCall(tool_name=corrected, arguments=action.arguments)
        return None

    def _parse_all_output_actions(self, text: str) -> List[ToolCall]:
        """从模型输出中提取所有工具调用。

        委托给 policies.ToolCallParser(单一权威实现), 避免双解析器重复维护。
        支持 [TOOL:] / XML / OpenCode / JSON action+args / tool_calls 数组 /
        function-call style 等格式。
        """
        if not text:
            return []
        from .policies import ToolCallParser
        parsed = ToolCallParser.parse_all(text)
        return [ToolCall(tool_name=name, arguments=args) for name, args in parsed]
