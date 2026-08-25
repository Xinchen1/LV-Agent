"""意图识别的纯规则函数(无 self / 无 LLM 依赖).

从 agent.py 的 OpenMythosAgent 中抽取出来, 便于独立单元测试,
并保持原有实例方法作为薄委托包装(见 agent.py 中的 _is_* 静态方法),
因此对外部调用方与既有测试完全透明。
"""

import re


def is_pure_nudge(task: str) -> bool:
    """纯催促/喂词(如"继续""go on")判定, 不含真实任务信息."""
    if not task:
        return False
    t = task.strip().lower()
    # 长度很短(<=8) 且只含催促词 → 纯催促
    nudge_only = re.fullmatch(
        r"[继续接着再来啦哈吧嗯好.。!！?？\s]*?(继续|接着|再|go on|continue|继续做|继续啊|继续吧|继续嘛|接着做)[继续接着再来啦哈吧嗯好.。!！?？\s]*",
        t,
    )
    if len(t) <= 8 and nudge_only:
        return True
    # 常见纯催促短句
    if t in {"继续", "继续啊", "继续吧", "接着", "接着呢", "继续做", "继续啊！", "go on", "continue", "continue please"}:
        return True
    return False


def is_ultra_short_ambiguous(task: str) -> bool:
    """超短输入(≤4字符)且不含明确动作关键词: 大概率是问候/打字误触.

    这类输入(如 "nih")不应触发工具调用——模型会把 "nih" 幻觉解释成
    "NIH 医学研究院" 之类然后去搜索, 一旦搜不到就整屏报错。
    应直接友好澄清或问候。

    注意: "是的/是/对/好/嗯" 等确认/回应词**不**算歧义——用户常用来
    回应上一轮的问题(如"需要我打开吗?" → "是的"), 必须放行让模型
    基于历史继续执行, 否则会打断对话连续性。
    """
    t = (task or "").strip()
    if not t or len(t) > 4:
        return False
    tl = t.lower()
    # 明确短命令/问候 → 正常走对话流程, 不拦截
    if tl.startswith("/") or tl in {"hi", "hey", "yo", "hello", "你好", "嗨", "哈喽", "在吗", "在", "help"}:
        return False
    # 确认/回应/否定词 → 放行(常是回复上一轮问题, 需结合历史继续)
    if tl in {
        "是", "是的", "对", "对的", "对呀", "好", "好的", "好呀", "嗯", "嗯嗯", "嗯呢",
        "行", "可以", "要", "要的", "需要", "当然", "没问题", "有", "有的",
        "yes", "yeah", "yep", "y", "sure", "ok", "okay", "fine",
        "不是", "不对", "不要", "不用", "不需要", "没", "没有", "no", "n",
    }:
        return False
    # 含动作词 → 放行(可能真是搜索/打开等意图)
    action_words = ["搜", "查", "天气", "新闻", "股票", "股价", "时间", "日期",
                    "找", "找找", "看", "看看", "找一下", "看一下",
                    "ls", "dir", "list", "search", "find", "read", "open", "show", "cat"]
    if any(w in tl for w in action_words):
        return False
    return True


def is_folder_read_intent(task: str) -> bool:
    """判断是否为"阅读/读/查看文件夹"意图(不包含具体文件名)."""
    t = (task or "").strip()
    if not t:
        return False
    # 纯指代: "阅读" / "读一下" / "查看" 等, 无文件名/无搜索词
    if re.fullmatch(r"(阅读|读|读一下|查看|看一下|看看|浏览|打开|看看内容)", t):
        return True
    # 含文件夹/目录词且无具体文件
    if re.search(r"(阅读|读|查看|浏览).{0,6}(文件夹|目录|folder|directory)", t):
        return True
    # 看下 X 文件夹/目录(带文件夹名, 如"看下 health os 这个文件夹")
    if re.search(r"(看下|看一下|看看|查看|浏览|打开|读一下|读取|里面)\s*[^，。！？!?]{1,40}?(文件夹|目录|folder|dir)", t):
        return True
    # 分析 X 文件夹/项目: "分析下 grok-build"、"grok-build 分析下" 等
    if re.search(r"(分析|剖析|解析)\s*[^，。！？!?]{1,40}?(文件夹|目录|项目|代码库|仓库|folder|dir|project)", t) or \
       re.search(r"[^，。！？!?]{1,40}?(分析下|分析一下|分析|剖析|解析)", t):
        return True
    # "分析下 grok-build"(动词在前, 无"文件夹"词): 分析 + 后跟具体名
    if re.search(r"^(分析下|分析一下|分析|剖析|解析)\s*[^，。！？!?]{1,40}", t):
        return True
    return False
