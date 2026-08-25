"""
StrategyDatabase - 策略提取和管理
从成功经验中提取可复用的策略规则
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

from .experience import Experience, ExperienceBuffer
from .terminal import style as _style


@dataclass
class Strategy:
    """提取的策略规则"""
    task_type: str
    pattern: str  # 任务模式描述
    conditions: Dict[str, Any]  # 触发条件
    actions: List[Dict[str, Any]]  # 推荐行动序列
    avg_loop_depth: float  # 推荐思考深度
    tool_sequence: List[str]  # 常用工具序列
    success_rate: float  # 该策略的成功率
    usage_count: int  # 被使用的次数
    example_episodes: List[str]  # 示例episode ID
    metadata: Dict[str, Any]
    id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())


class StrategyDatabase:
    """
    策略数据库
    从成功episodes中提取、存储和检索策略
    """

    def __init__(self, config):
        self.config = config.strategies
        self.storage_path = config.strategies.db_path
        self.strategies: Dict[str, Strategy] = {}

        # 确保目录存在
        from pathlib import Path
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        # 加载已有策略
        self._load()

    def update_from_experiences(self, experiences: List[Experience], experience_buffer: ExperienceBuffer):
        """从经验中提取新策略"""
        print(_style(f"  Analyzing {len(experiences)} experiences for strategy extraction...", "2"))

        # 按任务类型聚类
        clusters = self._cluster_by_task_type(experiences)

        new_strategies = []
        for task_type, eps in clusters.items():
            successful = [e for e in eps if e.trajectory.get('success', False)]
            if len(successful) < 3:
                continue  # 数据不足

            # 提取策略
            strategy = self._extract_strategy(task_type, successful, experience_buffer)
            if strategy:
                new_strategies.append(strategy)

        # 添加到数据库
        for strat in new_strategies:
            self.strategies[strat.id] = strat

        # 保存
        self._save()
        print(_style(f"  Extracted {len(new_strategies)} new strategies", "2"))

        return new_strategies

    def _cluster_by_task_type(self, experiences: List[Experience]) -> Dict[str, List[Experience]]:
        """按任务类型聚类（简化版：使用task和observed patterns）"""
        clusters = defaultdict(list)

        for exp in experiences:
            task = exp.task.lower()
            # 简单关键字分类
            if any(kw in task for kw in ['weather', 'temperature', 'rain', 'forecast']):
                cluster = 'weather_query'
            elif any(kw in task for kw in ['calculate', 'compute', 'math', 'formula']):
                cluster = 'calculation'
            elif any(kw in task for kw in ['file', 'read', 'write', 'save', 'load']):
                cluster = 'file_operation'
            elif any(kw in task for kw in ['search', 'find', 'lookup', 'about']):
                cluster = 'web_search'
            elif any(kw in task for kw in ['api', 'request', 'fetch']):
                cluster = 'api_call'
            elif any(kw in task for kw in ['python', 'code', 'exec', 'run']):
                cluster = 'code_execution'
            elif any(kw in task for kw in ['list', 'show', 'display', 'what']):
                cluster = 'list_info'
            else:
                cluster = 'general'

            clusters[cluster].append(exp)

        return clusters

    def _extract_strategy(self, task_type: str, successful: List[Experience], buffer: ExperienceBuffer) -> Optional[Strategy]:
        """从成功的episodes中提取策略"""

        # 计算平均思考深度
        avg_loops = sum(e.trajectory.get('thinking_steps', 0) for e in successful) / len(successful)

        # 提取常用工具序列
        all_tool_sequences = []
        for ep in successful:
            actions = ep.trajectory.get('actions', [])
            tool_seq = [action.get('tool_name') for action in actions if isinstance(action, dict)]
            if tool_seq:
                all_tool_sequences.append(tuple(tool_seq))

        # 最常见工具序列
        common_tool_seq = Counter(all_tool_sequences).most_common(1)
        most_common_tools = list(common_tool_seq[0][0]) if common_tool_seq else []

        # 提取条件特征
        conditions = self._extract_conditions(successful)

        # 生成模式描述
        pattern = self._generate_pattern_description(task_type, successful)

        # 行动序列模板
        actions_template = self._extract_action_template(successful)

        # 成功率
        success_rate = len(successful) / len([e for e in buffer.get_recent(1000) if e.task_type == task_type]) if buffer else 1.0

        return Strategy(
            task_type=task_type,
            pattern=pattern,
            conditions=conditions,
            actions=actions_template,
            avg_loop_depth=int(avg_loops),
            tool_sequence=most_common_tools,
            success_rate=min(success_rate, 1.0),
            usage_count=len(successful),
            example_episodes=[e.id for e in successful[:5]],
            metadata={
                "extracted_at": datetime.now().isoformat(),
                "num_examples": len(successful)
            }
        )

    def _extract_conditions(self, successful: List[Experience]) -> Dict[str, Any]:
        """提取触发条件（任务特征的统计）"""
        conditions = {
            "avg_task_length": sum(len(e.task.split()) for e in successful) / len(successful),
            "requires_tools": any(len(e.trajectory.get('actions', [])) > 0 for e in successful),
            "typical_loop_range": {
                "min": min(e.trajectory.get('thinking_steps', 0) for e in successful),
                "max": max(e.trajectory.get('thinking_steps', 0) for e in successful),
            }
        }
        return conditions

    def _generate_pattern_description(self, task_type: str, successful: List[Experience]) -> str:
        """生成模式描述"""
        examples = [e.task for e in successful[:3]]
        return f"Tasks like: {'; '.join(examples)}"

    def _extract_action_template(self, successful: List[Experience]) -> List[Dict[str, Any]]:
        """提取行动模板（最常见工具序列）"""
        # 简化：返回最常见的工具调用序列
        all_actions = []
        for ep in successful:
            for action in ep.trajectory.get('actions', []):
                if isinstance(action, dict):
                    all_actions.append(action)

        # 按工具类型分组
        by_tool = defaultdict(list)
        for action in all_actions:
            by_tool[action.get('tool_name')].append(action)

        # 取每个工具最常见的参数模式
        template = []
        for tool_name, actions in by_tool.items():
            # 简单：取第一个（实际应该统计最常见的参数组合）
            if actions:
                template.append({
                    "tool": tool_name,
                    "example_args": actions[0].get('arguments', {})
                })

        return template

    def match(self, task: str) -> Optional[Strategy]:
        """为给定任务匹配最佳策略"""
        task_lower = task.lower()

        # 简单关键字匹配（实际应该用向量相似度）
        best_match = None
        best_score = 0

        for strategy in self.strategies.values():
            # 计算匹配分数
            score = 0

            # 1. 任务类型匹配
            if strategy.task_type in task_lower:
                score += 5

            # 2. 关键字匹配
            pattern_keywords = strategy.pattern.lower().split()
            for kw in pattern_keywords:
                if kw in task_lower:
                    score += 1

            # 3. 工具需求匹配（基于任务分析）
            if score > best_score:
                best_score = score
                best_match = strategy

        # 阈值：至少匹配3分
        if best_score >= 3:
            return best_match

        return None

    def get_by_task_type(self, task_type: str) -> List[Strategy]:
        """按任务类型获取策略"""
        return [s for s in self.strategies.values() if s.task_type == task_type]

    def update_usage(self, strategy_id: str, success: bool):
        """更新策略使用统计"""
        if strategy_id in self.strategies:
            strat = self.strategies[strategy_id]
            strat.usage_count += 1
            # 移动平均更新成功率
            if strat.usage_count == 1:
                strat.success_rate = 1.0 if success else 0.0
            else:
                strat.success_rate = (strat.success_rate * (strat.usage_count - 1) + (1.0 if success else 0.0)) / strat.usage_count
            self._save()

    def get_advice_for_task(self, task: str) -> Optional[str]:
        """为任务生成策略建议文本，用于注入 prompt。"""
        strat = self.match(task)
        if not strat:
            return None
        lines = [
            f"## Recommended Strategy (from past successes, success_rate={strat.success_rate:.2f})",
            f"- Pattern: {strat.pattern}",
            f"- Suggested tool sequence: {' -> '.join(strat.tool_sequence) if strat.tool_sequence else 'none recorded'}",
            f"- Recommended thinking depth: {int(strat.avg_loop_depth)} loops",
        ]
        if strat.actions:
            lines.append("- Example actions:")
            for action in strat.actions[:3]:
                tool = action.get('tool', 'unknown')
                args = action.get('example_args', {})
                lines.append(f"  * {tool}({args})")
        return "\n".join(lines)

    def _save(self):
        """保存到磁盘 (JSON, 版本迁移友好)."""
        from pathlib import Path
        path = Path(self.storage_path) / "strategies.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: asdict(s) for sid, s in self.strategies.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self):
        """从磁盘加载 (JSON 优先, 兼容旧 pickle)."""
        from pathlib import Path
        path = Path(self.storage_path) / "strategies.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.strategies = {sid: Strategy(**s) for sid, s in data.items()}
                print(_style(f"  strategies: {len(self.strategies)} loaded", "2"))
                return
            except Exception as e:
                print(_style(f"  strategies: load failed ({e})", "2"))
                self.strategies = {}
                return
        # 旧 pickle 文件兼容
        legacy = Path(self.storage_path) / "strategies.pkl"
        if legacy.exists():
            try:
                import pickle
                with open(legacy, "rb") as f:
                    self.strategies = pickle.load(f)
                print(_style(f"  strategies: {len(self.strategies)} loaded (migrated)", "2"))
                # 迁移后立即转存 JSON
                try:
                    self._save()
                except Exception:
                    pass
                return
            except Exception as e:
                print(_style(f"  strategies: load failed ({e})", "2"))
                self.strategies = {}
