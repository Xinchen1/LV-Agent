"""意图分类器: 规则启发式 + LLM 软匹配的独立类.

从 OpenMythosAgent._classify_intent / _llm_intent_classify 抽取而来,
便于独立测试(用 mock backend)。backend 在每次 classify 调用时传入,
避免测试中途替换 a.backend 后分类器仍持有旧 backend 的陈旧引用。
"""

import re
from typing import Any, Callable, Dict, Optional, Tuple

from .tools import TOOLS_REGISTRY


class IntentClassifier:
    def __init__(self, correct_tool_name: Callable[[str], Optional[str]], logger):
        self._correct_tool_name = correct_tool_name
        self._logger = logger

    def classify(
        self, task: str, backend
    ) -> Optional[Tuple[str, Dict[str, Any], float, str]]:
        """启发式意图分类器.

        基于任务文本关键词 + 模式识别, 在 LLM 生成之上做一层确定性兜底:
        当置信度 >= 阈值时, 直接注入对应的工具调用, 避免 LLM 幻觉/犹豫导致漏检。

        Returns:
            (tool_name, arguments, confidence, reason) 或 None(未达阈值)。
        """
        if not task:
            return None
        t = task.strip()
        tl = t.lower()

        # -2) 显式"查询我的记忆/关于我"意图 → 特殊标记, 禁止转去 web_search
        #     这类问题不依赖外部网络, 答案来自注入的记忆上下文。
        #     必须排在 web_search 意图(含"是什么"匹配)之前, 否则会被误判成搜索。
        #     排除"X是什么意思"(问词义, 应走 QA 定义类回答)与"搜索/查一下"(明确要联网)。
        if re.search(
            r'(我的记忆|记得我|记住我|关于我|了解我|我的情况|我的背景|我们聊过|你记得|你的记忆里|在你记忆|核心记忆|你了解我|还记得我|记不记得我|你认识我吗|你知道我)',
            t, re.IGNORECASE,
        ) and not any(k in tl for k in ("搜索", "查一下", "查下", "查询", "最新", "新闻", "search")) and not re.search(
            r'(是什么意思|什么含义|是什么概念|什么意思)', t, re.IGNORECASE,
        ) and not re.match(r'^(什么(是|叫|叫做)|啥(是|叫)|什么是|啥是)\s*\S+', t, re.IGNORECASE):
            return ("__memory_query__", {}, 0.95, "detected memory-query intent (answer from injected memory, no web search)")

        # -1) 看/读文件夹意图: "看下 X 文件夹/目录" → 定位并在当前目录下 list
        # 支持 "桌面上的 X" 等跨目录文件夹访问
        if any(k in tl for k in ("看下", "看一下", "看看", "查看", "浏览", "读一下", "读取", "打开") ) and any(
            k in tl for k in ("文件夹", "目录", "folder", "dir", "目录结构", "里面", "内容")
        ):
            m = re.search(r'(?:看下|看一下|看看|查看|浏览|读一下|读取|打开|里面)\s*[^，。！？!?]{1,40}?(?:文件夹|目录|folder|dir)', t)
            fname = None
            if m:
                raw = m.group(0)
                # 提取文件夹名(去动词/类型词/指代词)
                for v in ("看下", "看一下", "看看", "查看", "浏览", "读一下", "读取", "打开", "里面", "这个", "那个", "的"):
                    raw = raw.replace(v, "")
                raw = raw.replace("文件夹", "").replace("目录", "").replace("folder", "").replace("dir", "").strip()
                if raw:
                    fname = raw
            
            # 额外支持 "桌面上的 X" 这种路径前缀模式
            m2 = re.search(r'(?:桌面上?|桌面的?|桌面实验)\s*(?:的)?\s*([^，。！？!\s]+)', t, re.IGNORECASE)
            if m2 and not fname:
                name_part = m2.group(1).strip()
                if name_part:
                    fname = f"Desktop/{name_part}"
                    return ("file_ops", {"action": "list", "path": fname}, 0.9, "detected folder-read intent with desktop path")
            
            if fname:
                return ("file_ops", {"action": "list", "path": fname}, 0.9, "detected folder-read intent")

        # -0.34) 移动/剪切文件意图 → bash_exec mv (需要真正执行文件系统操作)
        _move_verbs = ("移动到", "移到", "移动", "挪到", "挪", "剪切", "搬到", "移入", "移至", "move", "mv", "relocate")
        if any(v in tl for v in _move_verbs):
            # 提取源(用户提到的具体文件名/目录名)与目标
            target = None
            src_name = None
            m = re.search(r"(?:到|移至|移入|放到|放进|移到)\s*([^，。！？\s]+)", t)
            if m:
                target = m.group(1).strip().strip(chr(34) + chr(39))
            m2 = re.search(r"([\w\u4e00-\u9fff\-.]+\.(?:md|txt|py|js|ts|json|yaml|yml|csv|html|pdf))", t)
            if m2:
                src_name = m2.group(1)
            if target:
                cmd = f'mv "{src_name}" "{target}/" && echo moved' if src_name else f'mkdir -p "{target}" && echo ready'
                return ("bash_exec", {"command": cmd}, 0.85, "detected move-file intent")

        # -0.35) PDF 生成意图 → pdf_tool (优先于 file_ops, 因为用户明确要 PDF 文件)
        if any(k in tl for k in ("pdf", "pdf文件", "转成pdf", "转pdf", "做成pdf", "输出pdf",
                                 "pdf报告", "pdf版", "生成pdf", "导出pdf")):
            pdf_content = t.strip()
            # 提取输出路径(可选)
            pdf_path = None
            m = re.search(r"(?:到|存到|保存到|输出到)\s*([^，。！？\s]+?\.pdf)", tl)
            if m:
                pdf_path = m.group(1)
            args = {"content": pdf_content}
            if pdf_path:
                args["path"] = pdf_path
            return ("pdf_tool", args, 0.92, "detected pdf-generation intent")

        # -0.4) 保存/写入意图优先 → file_ops write(必须在"分析X文件夹"之前,
        #       否则"保存到 lv 文件夹"会被"分析"正则误判为 list)
        if any(k in tl for k in ("保存到", "存到", "保存成", "写进", "写入", "保存", "写成", "整理成",
                                  "新建", "创建", "写一篇", "保存为", "输出到文件")):
            fname = None
            m = re.search(r"(?:保存到|存到|写进|写入|保存成|输出到|整理成|保存为)\s*[“\"']?([^，。！？\s]+)", t)
            if m:
                cand = m.group(1)
                if re.search(r"\.(?:md|txt|py|json|yaml|yml|csv|html)$", cand):
                    fname = cand
                else:
                    fname = cand.rstrip("文件夹目录") + "/"
            if fname is None:
                m2 = re.search(r"([\w\u4e00-\u9fff\-\.]{1,40}\.(?:md|txt|py|json|yaml|yml|csv|html))", t)
                if m2:
                    fname = m2.group(1)
            args = {}
            if fname:
                fname = re.sub(r"^(帮|请|麻烦|把|将|存|保存)", "", fname)
                args["path"] = fname.strip()
            return ("file_ops", args, 0.9, "detected file-creation intent")

        # -0.5) 分析 X 文件夹/项目/代码库: 提取名称, 转 file_ops list(然后自动读内容)
        if any(k in tl for k in ("分析", "剖析", "解析", "analyze")) and any(
            k in tl for k in ("文件夹", "目录", "项目", "代码库", "仓库", "repo", "folder", "dir", "project")
        ) or re.search(r'(分析|剖析|解析)\s*[^，。！？!?]{1,30}', t) or re.search(
            r'[^，。！？!?]{1,30}?(分析|剖析|解析)\s*$', t
        ):
            name = re.sub(r'^(帮我|请|麻烦)?\s*(分析一下|分析下|分析|剖析|解析)\s*', '', t)
            # 去掉尾部动作词与类型词 (支持 "grok-build 分析下" 这类名字在前)
            name = re.sub(r'(分析一下|分析下|分析|剖析|解析)[，。！？!?]?$', '', name)
            name = re.sub(r'(文件夹|目录|项目|代码库|仓库|的内容|里面|repo|folder|project|dir).*$', '', name, flags=re.IGNORECASE)
            name = name.strip(' 的，。！？!?、')
            if name and len(name) <= 40:
                return ("file_ops", {"action": "list", "path": name}, 0.85, "detected folder-analysis intent")

        # 0) 先判断代码执行/命令意图, 避免 "print(1+1)" 这类被算术规则误判为 calculator
        if any(k in tl for k in ("python", "写代码", "运行代码", "执行代码", "写个程序", "写个脚本", "跑一下", "run code", "execute python", "写段代码")):
            return ("python_exec", {"code": ""}, 0.85, "detected code-execution intent")
        if any(k in tl for k in ("npm install", "pip install", "git clone", "shell", "终端", "命令行", "bash ", "apt install", "brew install", "运行命令", "执行命令", "cargo build")):
            return ("bash_exec", {}, 0.85, "detected shell-command intent")
        if any(k in tl for k in ("找到", "查找文件", "找一下", "文件在哪里", "定位", "where is", "which file", "find the", "找找")):
            return ("glob", {}, 0.8, "detected file-find intent")

        # -0.3) 修改/优化现有文件意图 → 先 read 目标文件(让模型看到内容后真正修改)
        # 如"给贪吃蛇加变速""修改 snake_game.py""在游戏里加音效"
        mod_verbs = ("加", "加上", "加入", "添加", "增加", "修改", "改", "更新", "优化", "完善", "改进", "增强",
                     "重构", "修复", "调整", "删除", "去除", "add", "modify", "update", "improve", "optimize",
                     "enhance", "refactor", "fix", "change", "edit")
        has_mod_verb = any(v in task for v in mod_verbs)
        # 目标文件: 提取 .py/.js/.md 等文件名, 或"给XX"/"在XX里" 的 XX
        target = None
        m = re.search(r'([\w\u4e00-\u9fff\-\.\/]{1,80}\.(?:py|js|ts|md|txt|json|yaml|yml|sh|html|css|c|cpp|rs|go|java))', task)
        if m:
            target = m.group(1)
        if target is None:
            m = re.search(r'(给|对|在|把)\s*([\w\u4e00-\u9fff\-\.]{1,40}?)(?:这个|那个)?(?:文件|代码|程序|项目|脚本|游戏|软件)', task)
            if m:
                target = m.group(2)
        if target is None and has_mod_verb:
            # "加功能/加音效/优化" 等无明确目标的修改意图 → 注入 file_ops read 当前目录(让模型找目标文件)
            return ("file_ops", {"action": "list", "path": "."}, 0.75, "detected file-modify intent (locate target first)")
        if has_mod_verb and target and len(target) <= 60:
            return ("file_ops", {"action": "read", "path": target}, 0.8, "detected file-modify intent (read target first)")

        # 1) 计算意图: 含算式模式(数字+运算符) → calculator
        if re.search(r'\d[\d\s]*[+\-*/^%]\s*\d', t):
            expr = re.sub(r'[^\d+\-*/().%^ \s]', '', t)
            # 提取最长的数学表达式片段
            m = re.search(r'[-+*/()\d.^%\s]{3,}', expr)
            if m and any(ch.isdigit() for ch in m.group(0)):
                return ("calculator", {"expression": m.group(0).strip()}, 0.95, "detected arithmetic expression")

        # 2) 天气意图 → weather
        if any(k in tl for k in ("天气", "气温", "温度", "weather", "temperature")):
            # 提取城市名(如果有)
            city = None
            m = re.search(r'([\u4e00-\u9fff]{2,6}(?:市|城市))', t)
            if m:
                city = m.group(1)
            args = {"city": city} if city else {}
            return ("weather", args, 0.9, "detected weather intent")

        # 3) 创建文件/文章意图 → file_ops write
        if any(k in tl for k in ("新建", "创建", "写一篇", "写一篇文章", "创建文章", "新建文章",
                                  "保存为", "输出到文件", "写个文件", "新建文件", "创建文件",
                                  "保存到", "存到", "保存成", "写进", "写入", "保存", "写成", "整理成")):
            # 尝试从任务中提取文件名(带扩展名)
            # 优先: "叫/为/名字是/标题是/叫 名字.md" 形式
            fname = None
            m = re.search(r'(?:叫|为|名字|名称|标题是|标题为)\s*[“\"\']?([\w\u4e00-\u9fff\-\.]{1,40}\.(?:md|txt|py|json|yaml|yml|csv|html))', t)
            if m:
                fname = m.group(1)
            else:
                # 其次: 独立出现的 .md 文件名(空格/句末分隔, 排除前面整段中文)
                m2 = re.search(r'(?<=[\s，。！？,，])?([\w\u4e00-\u9fff\-\.]{1,40}\.(?:md|txt|py|json|yaml|yml|csv|html))(?![\w\u4e00-\u9fff])', t)
                if m2:
                    # 若匹配到的是中英文混合(含2个以上汉字且前面紧贴汉字), 截取扩展名前合理长度
                    cand = m2.group(1)
                    # 文件名不应包含"新建/创建/写一篇"等动词
                    for v in ("新建", "创建", "写一篇", "写一篇文章", "创建文章", "新建文章"):
                        if v in cand:
                            cand = cand.split(v)[-1]
                            break
                    fname = cand
            args = {}
            if fname:
                # 剥离可能的引导语
                fname = re.sub(r'^(帮|请|麻烦|帮我|请帮我|在|到|目录|当前目录|放|存|保存)?(新建|创建|写一篇|写一篇文章|一个|一篇文章|把|将)?', '', fname)
                args["path"] = fname
            return ("file_ops", args, 0.9, "detected file-creation intent")

        # 4) 明确文件读取(带扩展名) → file_ops read
        if re.search(r'[\w\u4e00-\u9fff\-\.]+\.[a-zA-Z0-9]{1,10}', t) and any(
            k in tl for k in ("读取", "读一下", "打开", "看下", "看看", "read", "open", "查看")
        ):
            m = re.search(r'([\w\u4e00-\u9fff\-\.]+\.[a-zA-Z0-9]{1,10})', t)
            if m:
                return ("file_ops", {"action": "read", "path": m.group(1)}, 0.85, "detected file-read intent")

        # 4b) "看/读 X 文章/文件"(无扩展名, 如"看下 research 这篇文章"): 提取名称, 用 glob 定位
        if any(k in tl for k in ("看下", "看一下", "看看", "读取", "读一下", "打开", "查看", "读", "看")) and any(
            k in tl for k in ("文章", "这篇", "那篇", "文件", "文档", "报告", "笔记", "article", "doc", "file", "report", "note")
        ):
            # 提取名称: 去动词/类型词后剩余的词组
            name = re.sub(r'^(帮我|请|麻烦)?\s*(看下|看一下|看看|读取|读一下|打开|查看|读|看)\s*', '', t)
            name = re.sub(r'(这篇|那篇|文章|文件|文档|报告|笔记|的?内容|the|article|doc|file|report|note).*$', '', name, flags=re.IGNORECASE)
            name = name.strip(' 的，。！？!?、')
            if name and len(name) <= 40:
                return ("glob", {"pattern": f"*{name}*"}, 0.8, "detected article/file-read intent (no extension)")

        # 5) 明确搜索意图 → web_search
        if any(k in tl for k in ("搜索", "查一下", "查下", "查询", "最新", "新闻", "资讯", "search", "look up", "find out", "多少钱", "是什么")):
            # 提取搜索词: 去掉动作前缀
            q = re.sub(r'^(帮我|请|麻烦)?\s*(搜索一下|搜索|查找一下|查找|查一下|查下|查询|搜一下|帮我搜一下|search for|look up|find out)\s*[:：]?\s*', '', t)
            # 清理残留的"一下/一遍/帮我"等引导虚词
            q = re.sub(r'^(一下|一遍|帮我|请|麻烦)\s*', '', q)
            q = q.strip(' 。！!？?')
            if q and len(q) >= 2:
                return ("web_search", {"query": q}, 0.8, "detected web-search intent")

        # LLM 前置意图分析兜底: 规则未命中且有明确动作/需求时,
        # 用一次简短 LLM 调用判断真实意图(应对规则盲区, 如"帮我整理vault")。
        # 长度≥5 是为了避免问候/闲聊等极短输入也触发 LLM 调用(省 token)。
        if len(t) >= 5:
            llm_intent = self._llm_classify(t, backend)
            if llm_intent:
                return llm_intent

        return None

    def _llm_classify(
        self, task: str, backend
    ) -> Optional[Tuple[str, Dict[str, Any], float, str]]:
        """LLM 软匹配意图分类(规则未命中/置信度不足时的升级层).

        特点:
        - 精准理解用户意图: 从候选工具选最匹配的并给出参数
        - 支持拆分复合意图: 如"查资料并保存到文件" → 主意图 web_search,
          附带 write 子意图(存到 returned secondary, 供调用方二次执行)
        - 结构化 JSON 校验 + 工具名归一化, 减少幻觉
        返回 (tool_name, arguments, confidence, reason) 或 None。
        """
        tools_desc = ", ".join(TOOLS_REGISTRY.list_tools())
        prompt = (
            "你是意图分类器。精准判断用户请求最应该用哪个工具完成。\n"
            "可用工具: " + tools_desc + "\n"
            "工具说明: web_search 查网络/新闻/资料; read_web 打开网页读正文; file_ops 本地文件读写; "
            "bash_exec 执行命令; glob 按文件名找文件; search_files 搜文件内容; calculator 计算; "
            "weather 天气; git 仓库; python_exec 运行代码; github_search 搜GitHub。\n"
            "规则: 一个请求可能包含多个动作(如'搜索A并保存到B')。输出\n"
            "{\"tool\": \"主工具\", \"args\": {主工具参数}, \"reason\": \"一句话原因\", "
            "\"secondary\": {\"tool\": \"次工具\", \"args\": {...}} 或 null}\n"
            "主工具 = 用户最核心的动作; secondary = 紧随其后的附加动作(如保存/写入)。\n"
            "若无法确定(闲聊/无明确动作), 输出 {\"tool\": \"none\"}\n\n"
            f"用户请求: {task}\n"
            "JSON:"
        )
        try:
            raw = backend.generate(prompt, n_loops=1, temperature=0.0, max_tokens=250)
            import json as _json
            import re as _re
            m = _re.search(r'\{.*\}', str(raw or ""), _re.DOTALL)
            if not m:
                return None
            payload = _json.loads(m.group(0))
            tool_name = payload.get("tool", "")
            if not tool_name or tool_name == "none":
                return None
            # 工具名归一化(大小写/别名)
            canon = None
            if TOOLS_REGISTRY.get(tool_name):
                canon = tool_name
            elif TOOLS_REGISTRY.get(tool_name.lower()):
                canon = tool_name.lower()
            else:
                canon = self._correct_tool_name(tool_name)
                if canon and not TOOLS_REGISTRY.get(canon):
                    canon = None
            if not canon:
                return None
            args = payload.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            # 复合意图: 记录 secondary 供调用方后续执行
            sec = payload.get("secondary") or {}
            if isinstance(sec, dict) and sec.get("tool"):
                sec_name = sec.get("tool")
                if TOOLS_REGISTRY.get(sec_name) or TOOLS_REGISTRY.get(sec_name.lower()):
                    sec_args = sec.get("args", {}) or {}
                    if isinstance(sec_args, dict):
                        args["_secondary"] = (sec_name, sec_args)
            conf = 0.75
            reason = payload.get("reason", "llm intent classify")
            return (canon, args, conf, reason)
        except Exception as e:
            self._logger.debug(f"llm intent classify failed: {e}")
            return None
