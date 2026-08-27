"""Pure static utilities extracted from agent.py — zero dependency on Agent instance.

All functions here are self-contained and importable without constructing OpenMythosAgent.
Agent 类中的同名方法已改为代理到本模块(向后兼容)。
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Task tokenizer (原 OpenMythosAgent._TASK_STOP / _TASK_STOP_CHARS)
# ---------------------------------------------------------------------------

_TASK_STOP: Set[str] = set(
    "调研 分析 比较 设计 实现 部署 调试 测试 优化 搜索 查找 计算 解释 总结 翻译 推荐 评估 研究 "
    "查看 告诉 帮我 请问 一下 一个 这个 那个 这些 那些 然后 以及 还有 现在 今天 目前 最新 情况 "
    "内容 资料 信息 报告 输出 生成 列出 介绍 说明 关于 对于 进行 需要 是否 什么 怎么 如何 为什么 "
    "给我 继续 再 写 写一 写个".split()
)

_TASK_STOP_CHARS: Set[str] = set(
    "的了么呢吗吧啊呀哦嗯好对是而在与及为之于以从到把被让给就都也还再这那请问帮你它"
    "分分析析一一上下看看查算写做找说说讲讲谈问要能会想希望需要请叫个位种些样次回遍点"
)


# ---------------------------------------------------------------------------
# Task similarity helpers
# ---------------------------------------------------------------------------

def task_tokens(task: str) -> Set[str]:
    """把任务拆成"主题特征": 英文词 + 由内容字符组成的中文双字(剔除动作/虚词)."""
    task = (task or "").lower()
    toks: Set[str] = set(re.findall(r"[a-z0-9]+", task))
    cjk = [c for c in re.findall(r"[一-鿿]", task) if c not in _TASK_STOP_CHARS]
    for i in range(len(cjk) - 1):
        bg = cjk[i] + cjk[i + 1]
        if bg not in _TASK_STOP:
            toks.add(bg)
    return toks


def task_similarity(a: str, b: str) -> float:
    """两个任务的相关度(0-1). 共享英文主题词(AI/Python等)视为强关联."""
    ta, tb = task_tokens(a), task_tokens(b)
    if not ta or not tb:
        return 0.0
    shared = ta & tb
    if not shared:
        return 0.0
    sim = len(shared) / min(len(ta), len(tb))
    if any(t.isascii() and len(t) >= 2 for t in shared):
        sim = max(sim, 0.3)
    return sim


def merge_related_tasks(
    tasks: List[str], threshold: float = 0.2
) -> List[List[str]]:
    """多段任务分组: 相邻且相关度 >= threshold 的任务合并成一组.

    - 相关(同一主题/目标) → 合并为一次执行, 避免重复搜索/推理
    - 无关 → 各自成组, 排队依次执行
    """
    groups: List[List[str]] = []
    for t in tasks:
        t = (t or "").strip()
        if not t:
            continue
        if groups and task_similarity(groups[-1][-1], t) >= threshold:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


# ---------------------------------------------------------------------------
# Intent & query classification
# ---------------------------------------------------------------------------

# 纯催促词集合(不含有意义动作)
_PURE_NUDGE_PATTERNS: List[str] = [
    "继续", "接著", "接着", "继续啊", "继续呀", "接着啊",
    "继续嘛", "接着嘛", "继续吧", "接着吧",
    "继续做", "接着做", "继续弄", "接着弄",
    "继续写", "接着写", "继续看", "接着看",
    "go on", "continue", "keep going", "keep it up",
    "往下", "下一步", "下一步呢", "接下来呢", "然后呢",
    "然后啊", "然后呀", "请继续", "请接着",
]


def is_pure_nudge(task: str) -> bool:
    """判断是否为纯催促/续接词(不含明确指令或新信息)."""
    t = (task or "").strip().lower()
    if not t:
        return False
    # 含具体动作(动词+名词)的不是纯催促
    if re.search(r'[一-鿿]{1,4}\s*[一-鿿]{0,4}[\s]*(的|了|吗|吧|呢|啊|呀)', t):
        pass
    for pattern in _PURE_NUDGE_PATTERNS:
        if pattern in t:
            return True
    return False


def is_ultra_short_ambiguous(task: str) -> bool:
    """判断是否为超短歧义输入(≤4字符且含代词或无意义)."""
    t = (task or "").strip()
    if not t or len(t) <= 2:
        return True
    if len(t) <= 4 and re.search(r'^[这那她它他好嗯]$', t):
        return True
    return False


def is_continuation_query(task: str, is_pure_nudge_fn: Callable[[str], bool] = is_pure_nudge) -> bool:
    """判断是否为"继续上次讨论/接着刚才话题"这类延续性对话请求.

    Parameters
    ----------
    task : 用户输入文本
    is_pure_nudge_fn : 纯催促判断函数 (默认用本模块的 is_pure_nudge)
    """
    task_lower = (task or "").strip().lower()
    if not task_lower:
        return False
    continuation_markers = [
        '继续', '接着', '再聊', '再继续', '继续聊', '继续讲', '继续讨论',
        '继续刚才', '继续我们', '继续上次', '接着刚才', '接着上次', '接着我们',
        '接下来呢', '然后呢', '继续之前',
        'continue', 'go on', 'as we were', 'continue the discussion',
    ]
    if any(m in task_lower for m in continuation_markers):
        action_verbs = [
            '下载', '克隆', 'clone', '安装', 'install', '运行', '跑', 'run',
            '构建', 'build', '执行', '写', '写入', '保存', 'save', '导出',
            '搜索', 'search', '查', '生成', '生成报告', 'report', '部署',
            'deploy', '编译', 'compile', '拉', 'pull', 'git', '打开', '读取',
            'read', '分析', 'analyze', '继续完成', '继续执行', '接着做', '继续做',
            'continue the task', 'continue working', '继续下载', '继续克隆',
        ]
        if any(v in task_lower for v in action_verbs):
            return False
        return True
    if re.search(r'(昨天|上次|之前|刚才).{0,8}(讨论|话题|对话|聊|health|健康)', task_lower):
        return True
    if len(task_lower) <= 18 and re.match(
        r'^(这个|那个|该|此|这类|这种|这样|上面|刚才|之前).{2,}', task_lower
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Tool result analysis
# ---------------------------------------------------------------------------

# 目录/文件列表风格的正则
_LISTING_PATTERN = re.compile(r'^(d\s|[-rwxdr]\S*\s+|\S+\s+\d+\s*$)')


def needs_tool_summary(raw_output: str, is_continuation: bool) -> bool:
    """判断工具原始输出是否应转成自然语言总结.

    目录/文件列表、JSON 等结构化转储直接当答案会显得"只读没回复",
    此时应基于观察生成一句自然语言回答; 延续性对话请求也一律总结。
    """
    if not raw_output:
        return False
    if is_continuation:
        return True
    lines = [ln for ln in raw_output.splitlines() if ln.strip()]
    if len(lines) >= 2:
        listing_like = sum(1 for ln in lines if _LISTING_PATTERN.match(ln))
        if listing_like >= max(1, len(lines) // 2):
            return True
    if len(raw_output) > 600:
        return True
    return False


def tool_returns_listing(action: Any) -> bool:
    """工具调用是否返回目录/文件列表这类结构化数据."""
    from .tools import ToolResult  # 懒导入, 避免循环
    if not isinstance(action, ToolResult):
        return False
    text = getattr(action, "output", "") or getattr(action, "result", "") or ""
    if not text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4:
        listing_like = sum(1 for ln in lines if _LISTING_PATTERN.match(ln))
        return listing_like >= max(2, len(lines) // 2)
    return False


def skip_tool_retry(action: Any, tool_result: Any) -> bool:
    """判断是否应跳过工具重试(永久性错误)."""
    err = str(getattr(tool_result, "error", "") or "").lower()
    perm_errors = [
        "permission denied", "access denied", "not permitted",
        "operation not permitted",
    ]
    return any(p in err for p in perm_errors)


# ---------------------------------------------------------------------------
# Tool name fuzzy correction
# ---------------------------------------------------------------------------

_TOOL_ALIASES: Dict[str, str] = {
    "calc": "calculator", "math": "calculator", "calculate": "calculator",
    "calulator": "calculator", "calclator": "calculator",
    "write_file": "file_ops", "read_file": "file_ops", "file": "file_ops",
    "files": "file_ops", "open_file": "file_ops", "read": "file_ops", "write": "file_ops",
    "code": "python_exec", "python": "python_exec", "execute_python": "python_exec",
    "pthon": "python_exec", "python_exe": "python_exec", "run_code": "python_exec",
    "run": "bash_exec", "shell": "bash_exec", "terminal": "bash_exec", "command": "bash_exec",
    "bash": "bash_exec", "sh": "bash_exec", "exec": "bash_exec", "bash_exe": "bash_exec",
    "find": "glob", "glob_search": "glob", "list_files": "glob",
    "grep": "search_files", "search_files": "search_files",
    "api": "api_call", "http": "api_call", "request": "api_call", "rest": "api_call",
    "git_ops": "git", "gitops": "git",
    "weather_tool": "weather", "forecast": "weather",
    "file_search": "search_files",
}


def correct_tool_name(alias_or_name: str, valid_tools: Set[str]) -> Optional[str]:
    """工具名纠错: 尝试把 LLM 写错/臆造的工具名映射到最接近的合法工具."""
    from .pure_utils import edit_distance  # 懒导入
    if not alias_or_name:
        return None
    stripped = alias_or_name.strip().lower()
    if stripped in _TOOL_ALIASES:
        return _TOOL_ALIASES[stripped]
    for v in valid_tools:
        v_l = v.lower()
        if stripped in v_l or v_l in stripped:
            return v
    best, best_dist = None, 3
    for v in valid_tools:
        d = edit_distance(stripped, v.lower())
        len_ok = d <= 2 or (len(v) and abs(len(stripped) - len(v)) / max(len(v), 1) <= 0.4)
        if d < best_dist and len_ok:
            best, best_dist = v, d
    if best:
        return best
    norm = re.sub(r'[_\-\s]', '', stripped)
    for v in valid_tools:
        if re.sub(r'[_\-\s]', '', v.lower()) == norm:
            return v
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_output_for_action(
    text: str,
    parse_all_fn: Callable[[str], List[Tuple[str, Dict]]],
    valid_tools: Set[str],
    correct_fn: Callable[[str, Set[str]], Optional[str]] = correct_tool_name,
) -> Optional[Tuple[str, Dict]]:
    """Parse the first valid tool call from model output.

    Parameters
    ----------
    text : raw model output
    parse_all_fn : function that extracts (name, args) pairs from text
    valid_tools : set of registered tool names
    correct_fn : fuzzy corrector function
    """
    from .tools import ToolCall
    for name, args in parse_all_fn(text):
        if name in valid_tools:
            return (name, args)
        corrected = correct_fn(name, valid_tools)
        if corrected and corrected in valid_tools:
            return (corrected, args)
    return None


# ---------------------------------------------------------------------------
# Search helpers (thin delegates)
# ---------------------------------------------------------------------------

def extract_search_keywords(task: str, use_llm: bool = False) -> str:
    """Delegate to research_report.extract_search_keywords_hybrid."""
    from .research_report import extract_search_keywords_hybrid
    return extract_search_keywords_hybrid(task, use_llm=use_llm)


def ground_search_query(orig_keywords: str, generated_query: str) -> str:
    """Delegate to research_report.ground_search_query."""
    from .research_report import ground_search_query
    return ground_search_query(orig_keywords, generated_query)