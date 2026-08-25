"""
ReflectionModule - 深度自我反思
利用OpenMythos的高n_loops进行深度自我批判和分析
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .experience import Experience

# 避免循环导入：使用字符串类型注解
# from .agent import OpenMythosAgent  # 延迟导入


@dataclass
class Reflection:
    """反思结果"""
    quality_score: int  # 1-10
    loop_depth_adequate: bool
    tool_choice_critique: str
    identified_patterns: list
    improvement_suggestions: list
    generalized_rule: str
    raw_output: str


class ReflectionModule:
    """
    让Agent深度反思自己的行为
    利用OpenMythos的循环架构：高n_loops进行深入分析
    """

    REFLECTION_PROMPT_TEMPLATE = """You are a meta-cognitive analyzer. Review this agent execution critically.

TASK: {task}

EXECUTION TRACE:
{trace}

RESULT: {result}
SUCCESS: {success}

THINKING DEPTH USED: {n_loops} loops

Analyze deeply (use your full reasoning capacity):

1. THINKING QUALITY: Was {n_loops} loops adequate? Too shallow? Overthinking?
2. TOOL SELECTION: Were the right tools used? Any wrong, missing, or redundant tool calls?
3. ERROR PATTERNS: If failed, what went wrong? Premature action? Incorrect parsing? Tool misuse?
4. EFFICIENCY: Could fewer loops achieve same result? Any wasted iterations?
5. STRATEGY: What general principle or pattern can be extracted for similar future tasks?

Consider multiple angles. Be thorough. Use your recurrent thinking to examine the trace step by step.

Return JSON in this EXACT format:
{{
    "quality_score": <integer 1-10>,
    "loop_depth_adequate": <boolean>,
    "tool_choice_critique": "<specific critique of tool usage>",
    "identified_patterns": ["pattern1", "pattern2", ...],
    "improvement_suggestions": [
        {{"suggestion": "...", "reasoning": "...", "priority": "high/medium/low"}},
        ...
    ],
    "generalized_rule": "<concise rule for future similar tasks>",
    "failure_mode": "<if failed, categorize: reasoning_error|tool_error|parsing_error|insufficient_loops|other>",
    "confidence_in_reflection": <float 0-1>
}}

Focus on EXTRACTABLE KNOWLEDGE that can improve future performance.
Do NOT just summarize. Identify CAUSES and SOLUTIONS."""

    def __init__(self, agent: 'OpenMythosAgent', config):
        self.agent = agent
        self.config = config
        self.reflections_path = config.reflection.reflections_path

        # 确保目录存在
        from pathlib import Path
        Path(self.reflections_path).mkdir(parents=True, exist_ok=True)

    BATCH_REFLECTION_TEMPLATE = """You are a meta-cognitive analyzer. Review {n} agent executions critically.

{episodes}

For EACH episode above, analyze (in the same order):
1. THINKING QUALITY: Was the loop depth adequate? Too shallow? Overthinking?
2. TOOL SELECTION: Right tools used? Any wrong, missing, or redundant calls?
3. ERROR PATTERNS: If failed, what went wrong? Premature action? Parsing? Tool misuse?
4. EFFICIENCY: Could fewer loops achieve the same? Wasted iterations?
5. STRATEGY: What general principle can be extracted for similar future tasks?

{history_note}Return a JSON ARRAY with exactly {n} objects, one per episode, each in this EXACT format:
[
  {{
    "quality_score": <integer 1-10>,
    "loop_depth_adequate": <boolean>,
    "tool_choice_critique": "<specific critique>",
    "identified_patterns": ["p1", "p2"],
    "improvement_suggestions": [{{"suggestion": "...", "reasoning": "...", "priority": "high"}}],
    "generalized_rule": "<concise rule for future similar tasks>",
    "failure_mode": "<reasoning_error|tool_error|parsing_error|insufficient_loops|other>",
    "confidence_in_reflection": <float 0-1>
  }},
  ...
]
Focus on EXTRACTABLE KNOWLEDGE that improves future performance. Identify CAUSES and SOLUTIONS, not summaries."""

    def reflect(self, episode: Experience) -> Reflection:
        """对单个episode进行深度反思"""

        # 格式化trace
        trace = self._format_trace(episode)

        prompt = self.REFLECTION_PROMPT_TEMPLATE.format(
            task=episode.task,
            trace=trace,
            result=episode.trajectory.get('final_reward', 'N/A'),
            success=episode.trajectory.get('success', False),
            n_loops=episode.trajectory.get('thinking_steps', 0)
        )

        # 关键：使用更高的n_loops进行反思
        reflection_loops = self.config.reflection.thinking_loops_for_reflection

        # 生成反思（不做工具调用，纯思考）
        reflection_tokens = self.agent.backend.generate(
            prompt,
            n_loops=reflection_loops,
            temperature=0.3,  # 降低随机性，提高分析质量
            max_tokens=1024
        )

        # Handle both token IDs (OpenMythosBackend) and text (OpenAI/DeepSeek backends)
        if isinstance(reflection_tokens, str):
            reflection_text = reflection_tokens
        else:
            reflection_text = self.agent.tokenizer.decode(reflection_tokens)

        # 提取JSON
        try:
            json_start = reflection_text.find('{')
            json_end = reflection_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = reflection_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                # 如果找不到JSON，尝试解析整个文本（回退）
                data = self._parse_reflection_text(reflection_text)
        except json.JSONDecodeError as e:
            # 解析失败，记录原始输出
            data = {
                "quality_score": 5,
                "loop_depth_adequate": None,
                "tool_choice_critique": "Failed to parse reflection JSON",
                "identified_patterns": [],
                "improvement_suggestions": [],
                "generalized_rule": "",
                "failure_mode": "parsing_error",
                "confidence_in_reflection": 0.0
            }

        reflection = Reflection(
            quality_score=data.get('quality_score', 5),
            loop_depth_adequate=data.get('loop_depth_adequate', False),
            tool_choice_critique=data.get('tool_choice_critique', ''),
            identified_patterns=data.get('identified_patterns', []),
            improvement_suggestions=data.get('improvement_suggestions', []),
            generalized_rule=data.get('generalized_rule', ''),
            raw_output=reflection_text
        )

        # 保存反思到文件（可选）
        if self.config.reflection.save_reflections:
            self._save_reflection(episode.id, reflection)

        return reflection

    def _format_trace(self, episode: Experience) -> str:
        """格式化执行轨迹用于反思prompt"""
        lines = []
        trajectory = episode.trajectory

        if 'thoughts' in trajectory and trajectory['thoughts']:
            lines.append("THOUGHTS:")
            for i, thought in enumerate(trajectory['thoughts'][:5]):  # 最多5条
                lines.append(f"  Step {i+1}: {thought[:200]}...")  # 截断

        if 'actions' in trajectory:
            lines.append("\nACTIONS:")
            for i, action in enumerate(trajectory['actions']):
                lines.append(f"  Step {i+1}: {action}")

        if 'observations' in trajectory:
            lines.append("\nOBSERVATIONS:")
            for i, obs in enumerate(trajectory['observations']):
                obs_preview = str(obs)[:200] + ("..." if len(str(obs)) > 200 else "")
                lines.append(f"  Step {i+1}: {obs_preview}")

        lines.append(f"\nTotal thinking steps: {trajectory.get('thinking_steps', 'N/A')}")
        lines.append(f"Final success: {trajectory.get('success', False)}")

        return "\n".join(lines)

    def _parse_reflection_text(self, text: str) -> Dict[str, Any]:
        """解析非JSON格式的反思文本（降级方案）"""
        # 简单的关键字解析
        result = {
            "quality_score": 5,
            "loop_depth_adequate": False,
            "tool_choice_critique": "",
            "identified_patterns": [],
            "improvement_suggestions": [],
            "generalized_rule": "",
            "failure_mode": "unknown",
            "confidence_in_reflection": 0.5
        }

        # 尝试提取分数
        if "quality_score" in text or "score" in text:
            import re
            match = re.search(r'score["\s:]+(\d+)', text.lower())
            if match:
                result['quality_score'] = int(match.group(1))

        # 提取建议
        if "suggestion" in text or "improve" in text:
            lines = text.split('\n')
            for line in lines:
                if any(kw in line.lower() for kw in ['suggest', 'improve', 'should', 'could']):
                    result['improvement_suggestions'].append({
                        "suggestion": line.strip(),
                        "reasoning": "",
                        "priority": "medium"
                    })

        return result

    def _save_reflection(self, episode_id: str, reflection: Reflection):
        """保存反思到文件"""
        from pathlib import Path
        filename = Path(self.reflections_path) / f"{episode_id}.json"

        data = {
            "episode_id": episode_id,
            "reflection": reflection.__dict__,
            "timestamp": datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def batch_reflect(self, episodes: List[Experience], group_size: int = 3) -> List[Reflection]:
        """批量反思: 每组合并为一次 LLM 调用 (省 token/时间), 并注入历史 lessons 提供参照."""
        reflections = []
        # 每组最多 group_size 个 episode, 保持单案例分析质量同时减少 LLM 调用次数。
        for i in range(0, len(episodes), group_size):
            group = episodes[i:i + group_size]
            try:
                refs = self._reflect_group(group)
                reflections.extend(refs)
            except Exception as e:
                print(f"Reflection group failed: {e}")
                for ep in group:
                    reflections.append(self._fallback_reflection(ep, f"Reflection error: {e}"))
        return reflections

    def _reflect_group(self, episodes: List[Experience]) -> List[Reflection]:
        """对一组 episode 执行一次批量反思 LLM 调用."""
        # 组装每组文本
        parts = []
        for idx, ep in enumerate(episodes, start=1):
            trace = self._format_trace(ep)
            parts.append(
                f"EPISODE {idx}\n"
                f"TASK: {ep.task}\n"
                f"EXECUTION TRACE:\n{trace}\n"
                f"RESULT: {ep.trajectory.get('final_reward', 'N/A')}\n"
                f"SUCCESS: {ep.trajectory.get('success', False)}\n"
            )
        episodes_text = "\n\n".join(parts)

        # 注入历史 lessons: 让反思基于已有经验, 而不是从零开始。
        history_note = ""
        try:
            lessons = self.agent.experience_buffer.get_lessons(episodes[0].task, k=3) if self.agent.experience_buffer else []
            if lessons:
                lesson_lines = [f"- {l.condition}: {l.action}" for l in lessons]
                history_note = "PREVIOUS LESSONS (from past reflections):\n" + "\n".join(lesson_lines) + "\n\n"
        except Exception:
            history_note = ""

        prompt = self.BATCH_REFLECTION_TEMPLATE.format(
            n=len(episodes),
            episodes=episodes_text,
            history_note=history_note,
        )
        try:
            raw = self.agent.backend.generate(
                prompt, n_loops=3, temperature=0.3, max_tokens=2048
            )
        except Exception as e:
            raise RuntimeError(f"batch reflection LLM failed: {e}")

        text = raw if isinstance(raw, str) else self.agent.tokenizer.decode(raw)
        return self._parse_batch_reflections(text, episodes)

    def _parse_batch_reflections(self, text: str, episodes: List[Experience]) -> List[Reflection]:
        """解析批量反思输出: 期望 JSON 数组; 失败时回退逐个解析或默认."""
        import json
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
                if isinstance(data, list) and data:
                    refs = []
                    for item, ep in zip(data, episodes):
                        if not isinstance(item, dict):
                            refs.append(self._fallback_reflection(ep, "non-dict batch item"))
                            continue
                        refs.append(self._build_reflection(item, ep, text))
                    # 若数组比 episode 少, 补齐
                    while len(refs) < len(episodes):
                        refs.append(self._fallback_reflection(episodes[len(refs)], "missing in batch"))
                    return refs
        except Exception:
            pass
        # 回退: 逐个用单案例 reflect (代价高但保底)
        return [self.reflect(ep) for ep in episodes]

    def _build_reflection(self, data: Dict[str, Any], ep: Experience, raw_text: str) -> Reflection:
        """从解析后的 dict 构造 Reflection."""
        reflection = Reflection(
            quality_score=data.get('quality_score', 5),
            loop_depth_adequate=data.get('loop_depth_adequate', False),
            tool_choice_critique=data.get('tool_choice_critique', ''),
            identified_patterns=data.get('identified_patterns', []),
            improvement_suggestions=data.get('improvement_suggestions', []),
            generalized_rule=data.get('generalized_rule', ''),
            raw_output=raw_text
        )
        if self.config.reflection.save_reflections:
            self._save_reflection(ep.id, reflection)
        return reflection

    def _fallback_reflection(self, ep: Experience, reason: str) -> Reflection:
        """解析/调用失败时的默认反思."""
        return Reflection(
            quality_score=0,
            loop_depth_adequate=False,
            tool_choice_critique=reason,
            identified_patterns=[],
            improvement_suggestions=[],
            generalized_rule="",
            raw_output=""
        )
