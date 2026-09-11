"""
Fast-Path Dispatcher and Routing Logic for LV Agent.

Extracts fast paths out of OpenMythosAgent:
1. LocationFastPath: Directly locate projects/folders/files by name to avoid blind directory listing.
2. SimpleQueryClassifier: Fast-path determination for greetings, concise Q&A, and simple conversational turns.
3. FastPathRouter: Centralized coordinator for dispatching requests to appropriate fast paths.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# =====================================================================
# Precompiled Regular Expressions for Performance
# =====================================================================

_RE_COMPOUND_ACTION = re.compile(
    r'(修改|改为|改成|改动|更新|优化|完善|改进|增强|重构|修复|调整|删除|去除|运行|执行|启动|安装|部署|重写|改写|免登|不用登录|去登录|跳过登录)'
)

_RE_PUNCTUATION_SPLIT = re.compile(r'[，。！？、；;,.!?]')
_RE_SUFFIX_STRIP = re.compile(r'(项目|文件夹|目录|文件)\s*$')
_RE_ALPHANUMERIC = re.compile(r'[a-zA-Z_\-0-9]+')
_RE_VERB_EXTRACT = re.compile(
    r'(?:查找|找一下|搜索|搜一下|查一下|看看有没有|看看|看下|找|查|搜)\s*([\w\s\-_.]+?)(?:项目|文件夹|目录|文件)?\s*$'
)
_RE_FOUND_COUNT = re.compile(r'Found\s+(\d+)\s+file')

_DOC_MARKERS = [
    "文章", "笔记", "报告", "这篇", "那篇", "文档", "资料", "研究", "总结",
    "article", "note", "report", "doc", "paper"
]

_INFO_SEARCH_MARKERS = [
    "新闻", "资讯", "消息", "动态", "最新", "信息", "资料", "教程",
    "怎么", "如何", "教程", "介绍", "是什么", "怎么做", "天气",
    "news", "update", "info", "how to", "what is", "weather",
    "股票", "行情", "价格", "比分", "比赛",
    "发布时间", "发布日期", "发布", "时间", "什么时候", "release", "when",
]

_NEWS_MARKERS = ["新闻", "资讯", "动态", "最新", "消息", "news", "update"]

_LOCATION_VERBS = [
    "查找", "找一下", "搜索", "搜一下", "看看有没有", "看看", "看下", "在哪里", "在哪", "位于", "找", "查", "搜"
]

_LOCATION_SUFFIXES = ["项目", "文件夹", "目录", "文件"]
_LOCATION_MARKERS = ["下的", "里面", "中的", "上的", "里的"]
_LOCATION_ALIASES_ORDERED = ["下载文件夹", "桌面文件夹", "文档文件夹", "下载", "桌面", "文档"]

_PATH_MAPPINGS = [
    ("文档文件夹", "Documents"), ("文档目录", "Documents"), ("文档", "Documents"),
    ("桌面文件夹", "Desktop"), ("桌面", "Desktop"),
    ("下载文件夹", "Downloads"), ("下载", "Downloads"),
]

# Simple query classifier regexes
_MOD_VERBS = (
    "加", "加上", "加入", "添加", "增加", "修改", "改", "更新", "优化",
    "完善", "改进", "增强", "重构", "修复", "调整", "删除", "去除",
    "移动", "移", "移到", "移动到", "挪", "剪切", "搬到",
    "add", "modify", "update", "improve", "optimize", "refactor", "fix",
    "move", "mv", "relocate", "rename", "重命名"
)

_MOVE_VERBS = ("移动", "移到", "移动到", "挪", "剪切", "搬到", "move", "mv", "relocate")
_MOVE_TARGETS = ("到", "文件夹", "目录", "文件", "进去", "放到", "放进", "移入", "移至", "目录", "folder", "dir", "path")

_MOD_TARGETS = (
    "功能", "特性", "文件", "代码", "程序", "脚本", "游戏", "项目", "音效",
    ".py", ".js", ".ts", ".md", ".txt", "snake", "贪吃", "贪食"
)

_SIMPLE_GREETING_PATTERNS = [
    re.compile(r'^(你好|嗨|哈喽|hello|hi|hey|在吗|在嘛|您好|早上好|晚上好|下午好)[!!??\.,\s]*$', re.IGNORECASE),
    re.compile(r'^(谢谢|感谢|不客气|再见|拜拜|goodbye|bye)[!!??\.,\s]*$', re.IGNORECASE),
    re.compile(r'^(好的|行|可以|ok|okay|yes|no|嗯|哦|啊|对|不对|没错)[!!??\.,\s]*$', re.IGNORECASE),
    re.compile(r'^(谢|对)[!!??,,.!\s]*$', re.IGNORECASE),
]

# 自我能力查询：无需工具，直接回答
_SELF_CAPABILITY_PATTERN = re.compile(r'(你都有?什么功能|你能(做|干)什么|你会(什么|哪些)|介绍.*功能|有哪些功能)', re.IGNORECASE)

_READ_FOLDER_PATTERN = re.compile(r"(看下|看一下|看看|查看|浏览|打开|读一下|读取)\s*[^，。！？!?]{1,40}?(文件夹|目录|folder|dir)")
_ANALYZE_FOLDER_PATTERN = re.compile(r"(分析|剖析|解析)\s*[^，。！？!?]{1,30}")
_ANALYZE_NAME_FIRST_PATTERN = re.compile(r"[^，。！？!?]{1,30}?(分析下|分析一下|分析|剖析|解析)$")

_DEEP_KEYWORDS = [
    # 中文核心动作词
    '搜索', '查找', '调研', '研究', '考察', '融资', '股价', '行情', '资料',
    '文件', '代码', '程序', '分析', '比较', '设计', '实现',
    '部署', '调试', '测试', '优化', '解释', '总结', '翻译', '天气', '新闻',
    '写', '读', '创建', '构建', '获取', '调用', '运行', '执行', '列', '览', '示',
    '查', '改', '编', '发', '搜', '算', '整理', '归档', '分类',
    '规划', '对比', '评估', '推荐', '方案', '步骤', '流程', '架构', '原理', '机制',
    # 中文语义触发
    '联网', '实时', '最新', '现在', '今天', '目前',
    '几个', '多少', '列表', '清单', '报告',
    '项目', '文件夹', '目录', '桌面', '下载',
    # 英文动作词
    'search', 'find', 'lookup', 'research', 'investigate', 'file', 'code', 'program',
    'analyze', 'compare', 'describe', 'design', 'implement', 'deploy', 'debug', 'test', 'optimize',
    'explain', 'summarize', 'translate', 'weather', 'news', 'write', 'read', 'create',
    'build', 'fetch', 'call', 'run', 'execute', 'list', 'show', 'display',
    'plan', 'evaluate', 'recommend', 'architecture', 'workflow', 'compare',
]

_QA_EXCLUDE = (
    '文件', '代码', '程序', '项目', '目录', '文件夹', '脚本',
    '.py', '.js', '.ts', '.md', 'bug', '报错', '错误',
    '库', '包', '函数', '接口', '报修'
)

_QA_PATTERNS = [
    re.compile(r"^什么(是|叫|叫做)\s*\S+", re.IGNORECASE),
    re.compile(r"^\S{1,20}?(是|叫|叫做)\s*什么\??$", re.IGNORECASE),
    re.compile(r"^(解释|解释一下|简单解释|用.*话解释|介绍一下|说说|聊聊|讲讲|给我解释)\s*[^，。！？]{1,25}?", re.IGNORECASE),
]

_HISTORY_INTRO_KEYWORDS = [
    "历史", "发展", "演进", "起源", "诞生", "成名",
    "人工智能", "AI 历史", "简介", "概述", "回顾",
    "介绍", "背景", "来历"
]


# =====================================================================
# 1. Location Fast Path
# =====================================================================

class LocationFastPath:
    """定位快路：直接根据名称定位本地项目/文件夹/文件，避免盲目全盘扫描或多轮循环。"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_target_and_root(self, task: str) -> Optional[Tuple[str, str]]:
        """从任务文本中提取目标文件名/目录名以及搜索根路径。若不匹配定位特征则返回 None。"""
        task_lower = task.lower()

        # 1. 复合任务守卫：定位只是第一步，后面伴随修改/运行等动作时不走 fast path
        if _RE_COMPOUND_ACTION.search(task):
            return None

        has_verb = any(v in task for v in _LOCATION_VERBS)
        has_suffix = any(s in task for s in _LOCATION_SUFFIXES)
        has_location_marker = any(m in task for m in _LOCATION_MARKERS)

        # 检查是否包含显式的 "<location>的<target>" 模式
        has_explicit_location_target = False
        for alias in _LOCATION_ALIASES_ORDERED:
            pattern = re.compile(re.escape(alias) + r"(?:项目|文件夹|目录|文件)?\s*的\s*(.+)", re.IGNORECASE)
            if pattern.search(task):
                has_explicit_location_target = True
                break

        if not (has_verb or has_location_marker or has_explicit_location_target):
            return None
        if not has_suffix and not _RE_ALPHANUMERIC.search(task):
            return None

        # 排除文档阅读意图: "看下 X 文章/笔记/报告" 是读文档而非定位
        if any(m in task for m in _DOC_MARKERS) and not has_location_marker and not has_explicit_location_target:
            return None

        # 排除搜索信息/新闻意图
        if not has_suffix and any(m in task_lower for m in _INFO_SEARCH_MARKERS):
            return None
        if any(m in task_lower for m in _NEWS_MARKERS) and not has_location_marker and not has_explicit_location_target:
            return None

        # 解析搜索根路径
        root = str(Path.home())
        for alias, target in _PATH_MAPPINGS:
            if alias in task:
                root = str(Path.home() / target)
                break

        # 显式绝对路径优先: 任务中若出现真实存在的绝对路径, 以它为根.
        # 否则默认 home 递归 glob 会扫全盘导致卡死(真实教训: home 全盘 crawl hang).
        for _cand in re.findall(r"(/[^\s'\"“”`]+)", task):
            _p = Path(_cand.rstrip("/"))
            try:
                if _p.exists():
                    root = str(_p if _p.is_dir() else _p.parent)
                    break
            except Exception:
                continue

        # 提取目标名称
        target_name = None

        # 优先显式结构: "<location>的<target>"
        for alias in _LOCATION_ALIASES_ORDERED:
            pattern = re.compile(re.escape(alias) + r"(?:项目|文件夹|目录|文件)?\s*的\s*(.+)", re.IGNORECASE)
            m = pattern.search(task)
            if m:
                target_name = _RE_PUNCTUATION_SPLIT.split(m.group(1).strip(), maxsplit=1)[0].strip()
                target_name = _RE_SUFFIX_STRIP.sub('', target_name).strip()
                break

        # Marker 回退: 下的/里面/中的/上的/里的
        if not target_name:
            for marker in _LOCATION_MARKERS:
                if marker in task:
                    idx = task.rfind(marker)
                    target_name = _RE_PUNCTUATION_SPLIT.split(task[idx + len(marker):].strip(), maxsplit=1)[0].strip()
                    target_name = _RE_SUFFIX_STRIP.sub('', target_name).strip()
                    break

        # 动词回退: 动词与后缀之间
        if not target_name:
            m = _RE_VERB_EXTRACT.search(task)
            if m:
                target_name = m.group(1).strip()

        if not target_name:
            return None

        # 清洗引号和别名前缀
        target_name = target_name.strip('"\'`“”').lstrip("的").strip()
        for alias in _LOCATION_ALIASES_ORDERED:
            if target_name.lower().startswith(alias.lower()):
                target_name = target_name[len(alias):].lstrip("/\\的").strip()
                break
        target_name = target_name.lstrip("/\\的").strip()

        if not target_name or len(target_name) < 2 or len(target_name) > 40:
            return None

        return target_name, root

    def execute(
        self,
        task: str,
        status_callback: Optional[Callable[[Optional[Any], str], None]] = None,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        glob_tool: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """执行定位快路检索。有真实命中返回结果字典，否则返回 None。"""
        parsed = self.extract_target_and_root(task)
        if not parsed:
            return None

        target_name, root = parsed

        if status_callback:
            status_callback(stream_callback, f"locating '{target_name}' under {root}")

        search_pattern = f"**/*{target_name}*"
        try:
            if glob_tool is None:
                from .tools import GlobTool
                glob_tool = GlobTool()

            roots = []
            if root != str(Path.home()):
                roots.append(root)
            if str(Path.cwd()) not in roots:
                roots.append(str(Path.cwd()))
            # home 全盘递归 crawl 已删: 云盘/mount/socket 下 scandir 会卡死,
            # Python 层超时对此无效(真实教训: smoke 全盘 hang).
            # 无明确范围的模糊查询直接回落主循环, 由工具链(带超时)慢慢找.

            output_parts = []
            for r in roots:
                try:
                    # 遍历自带超时(防 home 级大目录卡死); 首个有命中的根即停,
                    # 不再扫剩余根(此前全扫, 慢且易 hang).
                    res = glob_tool.execute(pattern=search_pattern, path=r, max_results=40)
                    count = 0
                    try:
                        count = int((res.metadata or {}).get("count", 0))
                    except Exception:
                        m0 = _RE_FOUND_COUNT.search(res.output or "")
                        count = int(m0.group(1)) if m0 else 0

                    if res.success and count > 0 and res.output:
                        output_parts.append(f"[{r}]\n{res.output}")
                        break
                except Exception:
                    continue

            if not output_parts:
                return None

            output = "\n\n".join(output_parts)
            final = f"Found matches for '{target_name}':\n{output}"
            return {
                "final_answer": final,
                "success": True,
                "outer_loops": 1,
                "thinking_steps": 1,
                "metadata": {"duration_ms": 0, "fast_path": True, "location_fast_path": True},
            }
        except Exception as e:
            self.logger.debug(f"location fast path failed: {e}")
            return None


# =====================================================================
# 2. Simple Query Classifier
# =====================================================================

class SimpleQueryClassifier:
    """简单查询分类器：智能判定任务是否属于无需复杂工具编排与多轮深度推理的轻量查询。"""

    def is_simple_query(
        self,
        task: str,
        is_continuation_fn: Optional[Callable[[str], bool]] = None,
        is_pure_nudge_fn: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """判定 task 是否为简单对话/问答查询。"""
        task = task.strip()
        task_lower = task.lower()

        # 0) 空输入直接 fast
        if not task:
            return True

        # 0.5) 移动/重命名类或文件修改意图 → deep
        if any(v in task for v in _MOVE_VERBS) and any(k in task for k in _MOVE_TARGETS):
            return False
        if any(v in task for v in _MOD_VERBS) and any(k in task for k in _MOD_TARGETS):
            return False

        # 1.5) 有实质内容的延续追问 → deep (纯催促除外)
        if is_continuation_fn and is_pure_nudge_fn:
            try:
                if is_continuation_fn(task) and not is_pure_nudge_fn(task) and len(task.strip()) > 8:
                    return False
            except Exception:
                pass

        # 2) 纯问候/礼貌/确认 → 先于关键词匹配
        for pattern in _SIMPLE_GREETING_PATTERNS:
            if pattern.match(task_lower):
                return True

        # 2.1) 自我能力查询 → fast（直接回答，无需工具）
        if _SELF_CAPABILITY_PATTERN.search(task):
            return True

        # 2.5) 明确"看/读 X 文件夹/目录" → fast
        if _READ_FOLDER_PATTERN.search(task):
            return True

        # 2.6) "分析 X 文件夹/项目" → fast
        if _ANALYZE_FOLDER_PATTERN.search(task) and (
            "文件夹" in task or "目录" in task or "项目" in task or "代码库" in task or "仓库" in task
            or "folder" in task_lower or "dir" in task_lower or "project" in task_lower
        ):
            return True

        # 2.7) "X 分析下/分析一下" (名字在前) → fast
        if _ANALYZE_NAME_FIRST_PATTERN.search(task):
            return True

        # 2.8) 纯知识问答豁免: "什么是X / X是什么 / 解释一下X概念"
        if not any(e in task for e in _QA_EXCLUDE):
            if any(p.match(task) for p in _QA_PATTERNS):
                return True

        # 3) 动作/工具/深度推理关键词 → deep
        if any(kw in task_lower for kw in _DEEP_KEYWORDS):
            return False

        # 4) 超短任务(<=10字符)且无深度关键词 → fast
        if len(task) <= 10:
            return True

        # 5) 11-15字符的查询 - 查看数量类短查询通常需要工具
        if len(task) <= 15:
            if re.search(r'几个\s*[文件文件夹目录个]', task_lower):
                return False
            return True

        # 6) 较长任务(>40字符) → deep
        if len(task) > 40:
            return False

        # 中等长度(16-40字符)且无明确动作关键词
        if any(k in task_lower for k in _HISTORY_INTRO_KEYWORDS):
            tool_keywords = ["文件", "代码", "搜索", "创建", "项目", "目录"]
            if not any(k in task_lower for k in tool_keywords):
                return True

        return True


# =====================================================================
# 3. Unified Fast Path Router
# =====================================================================

class FastPathRouter:
    """快路总调度器：统一管理定位快路、简单查询识别等调度决策。"""

    def __init__(
        self,
        status_emitter: Optional[Callable[[Optional[Any], str], None]] = None,
        logger: Optional[logging.Logger] = None,
        cache_get: Optional[Callable[[str, Tuple[Any, ...], Callable[[], Any]], Any]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.status_emitter = status_emitter
        self.cache_get = cache_get
        self.location_fast_path = LocationFastPath(logger=self.logger)
        self.simple_classifier = SimpleQueryClassifier()

    def is_simple_query(
        self,
        task: str,
        is_continuation_fn: Optional[Callable[[str], bool]] = None,
        is_pure_nudge_fn: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """带缓存的简单查询判断。"""
        if self.cache_get:
            return self.cache_get(
                "is_simple",
                (task,),
                lambda: self.simple_classifier.is_simple_query(
                    task,
                    is_continuation_fn=is_continuation_fn,
                    is_pure_nudge_fn=is_pure_nudge_fn,
                )
            )
        return self.simple_classifier.is_simple_query(
            task,
            is_continuation_fn=is_continuation_fn,
            is_pure_nudge_fn=is_pure_nudge_fn,
        )

    def try_location_fast_path(
        self,
        task: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        glob_tool: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """尝试走定位快路。"""
        return self.location_fast_path.execute(
            task=task,
            status_callback=self.status_emitter,
            stream_callback=stream_callback,
            token_callback=token_callback,
            glob_tool=glob_tool,
        )


# Global helper instance
_default_simple_classifier = SimpleQueryClassifier()


def is_simple_query(
    task: str,
    is_continuation_fn: Optional[Callable[[str], bool]] = None,
    is_pure_nudge_fn: Optional[Callable[[str], bool]] = None,
) -> bool:
    """便利函数：判断是否为简单查询。"""
    return _default_simple_classifier.is_simple_query(
        task,
        is_continuation_fn=is_continuation_fn,
        is_pure_nudge_fn=is_pure_nudge_fn,
    )
