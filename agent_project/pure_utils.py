"""无状态纯工具函数(无 self / cls / 配置依赖).

从 agent.py 的 OpenMythosAgent 抽取的纯逻辑: 工具输出预判、失败重试跳过
判定、以及编辑距离拼写纠错。便于独立测试, 并在 agent.py 中保留薄委托包装,
对调用方与既有测试透明。
"""


def tool_returns_listing(action) -> bool:
    """执行前预判工具是否会产生"目录/文件列表"类原始输出.

    若会在展示时被总结, 则先不直接播放原始内容, 避免先播列表又播总结。
    """
    if action is None:
        return False
    name = getattr(action, "tool_name", "")
    args = getattr(action, "arguments", {}) or {}
    if name == "file_ops":
        return str(args.get("action", "")).lower() in ("list", "find", "grep")
    if name in ("bash_exec", "run_code", "python_exec"):
        return True
    return False


def skip_tool_retry(action, tool_result) -> bool:
    """判断工具失败后是否不值得重试(重试会复现相同失败并重复刷屏).

    联网搜索无结果/超时类错误重试同一查询毫无意义, 直接走自然语言兜底。
    """
    name = getattr(action, "tool_name", "")
    err = str(getattr(tool_result, "error", "") or "").lower()
    if name == "web_search" and any(
        k in err for k in ("no search results", "no results", "无结果", "没有找到", "nothing found", "timed out", "timeout")
    ):
        return True
    return False


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离(用于工具名拼写纠错)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
