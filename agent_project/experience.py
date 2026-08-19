"""
ExperienceBuffer - 存储和检索Agent交互经验
向量化存储，用于相似任务检索和经验反思
"""

import json
import uuid
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import AgentConfig
from .terminal import style as _style


@dataclass
class Experience:
    """单个经验片段"""
    task: str
    task_type: str
    trajectory: Dict[str, Any]
    id: str = ""
    # trajectory fields:
    # - thoughts: list of thinking outputs (optional)
    # - actions: list of tool calls
    # - observations: list of tool results
    # - thinking_steps: int (n_loops used)
    # - final_reward: float
    # - success: bool
    # - metadata: dict
    embedding: Optional[List[float]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Lesson:
    """从经验中提取的可操作教训"""
    task_pattern: str
    condition: str
    action: str
    outcome: str
    success: bool
    id: str = ""
    source_episode_id: Optional[str] = None
    timestamp: str = ""
    usage_count: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())


class ExperienceBuffer:
    """
    经验缓冲区，支持向量检索
    存储格式：ChromaDB + JSON备份
    """

    def __init__(self, config: AgentConfig):
        self.config = config.experience
        self.storage_path = Path(config.experience.vector_db_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._mode = "memory"
        self._memory_store = []

        self._lessons: List[Lesson] = []
        self._lessons_path = self.storage_path / "lessons.json"
        self._load_lessons()

        self._cache: Dict[str, Experience] = {}

        self._episodes_path = self.storage_path / "episodes.json"
        self._load_episodes()

    def add_episode(self, task: str, trajectory: Dict[str, Any], task_type: str = "unknown"):
        """添加一个episode到经验库"""
        exp = Experience(
            id=str(uuid.uuid4()),
            task=task,
            task_type=task_type,
            trajectory=trajectory
        )

        self._memory_store.append(exp)

        # 更新缓存
        self._cache[exp.id] = exp

        # 自动清理
        self._maybe_prune()

        # Persist to JSON
        self._save_episodes()

        return exp.id

    def get_similar(
        self,
        task: str,
        k: int = 5,
        success_only: bool = True,
        task_type: Optional[str] = None
    ) -> List[Experience]:
        """检索相似的成功案例（keyword fallback）"""
        task_lower = task.lower()
        scored = []
        for exp in self._memory_store:
            score = 0.0
            task_words = task_lower.split()
            exp_words = exp.task.lower().split()
            for w in task_words:
                if len(w) > 1 and w in exp_words:
                    score += 1.0
            if success_only and not exp.trajectory.get('success', False):
                continue
            if task_type and exp.task_type != task_type:
                continue
            if score > 0:
                scored.append((score, exp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:k]]

    def get_by_task_type(self, task_type: str, success_only: bool = True, limit: int = 100) -> List[Experience]:
        """按任务类型检索"""
        all_exps = list(self._cache.values()) if self._cache else self._memory_store
        filtered = [e for e in all_exps if e.task_type == task_type]
        if success_only:
            filtered = [e for e in filtered if e.trajectory.get('success', False)]
        filtered.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered[:limit]

    def get_recent(self, n: int = 100, success_only: bool = False) -> List[Experience]:
        """获取最近的N个episodes"""
        all_exps = list(self._cache.values()) if self._cache else self._memory_store
        filtered = [e for e in all_exps if not success_only or e.trajectory.get('success', False)]
        filtered.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered[:n]

    def get_failures(self, n: int = 50) -> List[Experience]:
        """获取失败案例（用于反思）"""
        recent = self.get_recent(n, success_only=False)
        return [e for e in recent if not e.trajectory.get('success', False)]

    def add_lesson(self, lesson: Lesson):
        """添加一条教训并持久化"""
        # 去重：相同 condition + action 组合只保留最新一条
        self._lessons = [
            l for l in self._lessons
            if not (l.condition == lesson.condition and l.action == lesson.action)
        ]
        self._lessons.append(lesson)
        self._save_lessons()

    def get_lessons(self, task: str, success_only: Optional[bool] = None, k: int = 5) -> List[Lesson]:
        """检索与当前任务相关的教训"""
        task_lower = task.lower()
        scored = []
        for lesson in self._lessons:
            score = 0
            pattern_words = lesson.task_pattern.lower().split()
            for word in pattern_words:
                if len(word) > 2 and word in task_lower:
                    score += 1
            if success_only is not None and lesson.success != success_only:
                continue
            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in scored[:k]]

    def get_all_lessons(self) -> List[Lesson]:
        """返回所有教训（用于调试/导出）"""
        return list(self._lessons)

    def _load_lessons(self):
        """从磁盘加载教训"""
        try:
            if self._lessons_path.exists():
                data = json.loads(self._lessons_path.read_text(encoding="utf-8"))
                self._lessons = [Lesson(**item) for item in data]
        except Exception:
            self._lessons = []

    def _save_lessons(self):
        """保存教训到磁盘"""
        try:
            data = [asdict(l) for l in self._lessons]
            self._lessons_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def count(self) -> int:
        """总episode数量"""
        return len(self._cache) if self._cache else len(self._memory_store)
    
    def _maybe_prune(self):
        """如果超过限制，删除旧的记录"""
        if not self.config.max_episodes:
            return
        all_exps = list(self._cache.values()) if self._cache else self._memory_store
        if len(all_exps) > self.config.max_episodes:
            all_exps.sort(key=lambda e: e.timestamp)
            to_delete = int(self.config.max_episodes * 0.1)
            for exp in all_exps[:to_delete]:
                self._cache.pop(exp.id, None)
                if exp in self._memory_store:
                    self._memory_store.remove(exp)

    def _load_episodes(self):
        """Load episodes from JSON backup in memory mode."""
        if not self._episodes_path.exists():
            return
        try:
            data = json.loads(self._episodes_path.read_text(encoding="utf-8"))
            for item in data:
                exp = Experience(**item)
                self._cache[exp.id] = exp
                if exp not in self._memory_store:
                    self._memory_store.append(exp)
        except Exception:
            pass

    def _save_episodes(self):
        """Persist episodes to JSON in memory mode."""
        try:
            experiences = [asdict(exp) for exp in (self._cache.values() if self._cache else self._memory_store)]
            self._episodes_path.write_text(json.dumps(experiences, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def save_json_backup(self, path: str):
        """导出为JSON（备份或分析用）"""
        experiences = [asdict(exp) for exp in (self._cache.values() if self._cache else self._memory_store)]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(experiences, f, indent=2, ensure_ascii=False)

    def load_from_json(self, path: str):
        """从JSON导入"""
        with open(path, 'r') as f:
            experiences = json.load(f)

        for exp_data in experiences:
            exp = Experience(**exp_data)
            if not self._cache:
                self._cache = {}
            self._cache[exp.id] = exp
            if exp not in self._memory_store:
                self._memory_store.append(exp)
