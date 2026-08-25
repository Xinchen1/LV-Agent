"""
response_filter - 回复清洗与流式路由 (从 agent.py 抽出的独立单元)

去 God Object 第一步: 把"fast path 回复清洗"与"流式回调路由"这两个自洽、与
OpenMythosAgent 生命周期无关的单元移到这里, 让 agent.py 只负责编排。

本模块:
  - clean_fast_answer: 剥离 think 标签残留 / 尾回声 / 泄露的英文自言自语。
  - strip_leaked_reasoning + is_mostly_english + STRIP_SELFTALK_RE: 泄露清洗子逻辑。
  - StreamRouter: 流式 token 路由, 区分 reasoning / content / tool 标记, 抑制 <think>
    与 [TOOL:...] 直接透出给用户。
"""

import re


# 英文"自言自语"特征词: 模型未用 <think> 标签时泄露的元推理
STRIP_SELFTALK_RE = re.compile(
    r'\b(we need to|we should|the user asks?|the user (wants|said|is|means)|according to|'
    r'i should|let me|the assistant|the response|this is (a|the)|based on|to answer (the|this)|'
    r'(in|respond in) (chinese|english)|we can (explain|answer)|the (current|final) answer|'
    r'continue (from|exactly)|do not repeat|the instructions|as per (user|the))\b',
    re.IGNORECASE,
)


def is_mostly_english(s: str, threshold: float = 0.7) -> bool:
    """判断一段文字是否以英文(ASCII 字母)为主. 用于区分'泄露的英文推理'与'正常中文回答中的英文词'."""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if ord(c) < 128]
    return len(ascii_letters) / len(letters) >= threshold


# 行首"自言自语"特征词: 用于流式逐行过滤模型泄露到正文前的英文元推理行。
# 只匹配强元推理标记(we need to / the user / let me / to answer / according to ...),
# 不放宽到裸 "I/We/So" 等, 避免误删正常的英文回答句。
_LEAK_LINE_RE = re.compile(
    r'(?i)^\s*(?:we need to|we should|the user|the model|the assistant|the response|'
    r'let me|let us|i should|i need to|to answer|according to|based on|'
    r'in order to|as per|ok[,.]?|okay[,.]?|sure[,.]?|alright[,.]?|hmm|well|'
    r'now,|so,|first,|next,|here is|here are|this is)\b'
)


def is_leaked_self_talk(line: str) -> bool:
    """判断一行是否为模型未用 <think> 标签时泄露到正文的英文"自言自语"行。

    用于流式渲染时逐行抑制(而非整段清洗): 行以内英文为主, 且以 we/i/the user/...
    等元推理特征词开头, 视为泄露, 不展示给用户。中文回答中的英文词不会命中
    (不以这些词开头), 代码块行也不会命中(高亮器对代码块提前返回)。
    """
    s = line.strip()
    if not s:
        return False
    if not is_mostly_english(s):
        return False
    return bool(_LEAK_LINE_RE.match(s))


def strip_leaked_reasoning(text: str) -> str:
    """剥离模型未用 <think> 标签时泄露到正文开头的英文元推理(自我对话).

    典型症状: 模型先写一段英文 "We need to answer the user's question... / The user asks...
    然后才切到用户语言(如中文)给出真正答案, 两者往往以换行分隔。

    处理逻辑: 从开头扫描, 找到第一个"后面紧接非 ASCII 答案"的换行; 若换行前的前缀
    以英文为主且含有自言自语特征词(we need to / the user asks / according to ...),
    则把整段前缀(英文推理, 含其中可能嵌入的用户原话引用)删掉, 只保留非英文答案。

    若整段几乎都是英文(用户用英文提问、正常英文回答)则不动, 避免误删。
    """
    if not text:
        return text
    # 注: 不在此处对"整段是否英文"做早退。纯英文回答不含非 ASCII 文字块,
    # 下面的循环自然找不到可剥离的目标, 不会误删。早退反而会让"英文推理 + 简短中文答案"
    # 这类混合回复(英文占比高)漏网。
    for m in re.finditer(r'\n', text):
        after = text[m.end():]
        # 换行后的内容确实以非 ASCII 文字(中文等)开头 => 视为答案起点
        if re.search(r'^[^\x00-\x7f]{4,}', after):
            prefix = text[:m.start()]
            if is_mostly_english(prefix) and STRIP_SELFTALK_RE.search(prefix):
                answer = after.lstrip()
                return answer if answer else text
            # 前缀不像自言自语 => 不再往前找, 保守保留原文
            break
    return text


def clean_fast_answer(text: str) -> str:
    """清理 fast path 回复中的多余内容: think 标签残留与尾部回声.

    模型(如 deepseek-reasoner)有时输出 <thinking>...</thinking> 或回答后再开一个
    think 块(回声), 被截断后留下 <th 残片或 "可以/我/好的" 等短回声。
    """
    if not text:
        return text
    # 0. 剥离可能残留在正文的置信度行(置信度只在思考内部, 不给用户看)
    text = re.sub(r'\s*(?:置信度|confidence)\s*[:：]\s*0?\.\d{1,2}\s*[。]?\s*$', '', text).strip()
    text = re.sub(r'\s*(?:置信度|confidence)\s*[:：]\s*0?\.\d{1,2}\s*[。]?', '', text).strip()
    # 1. 去掉完整 think 块(支持 <think>/<thinking>)
    text = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    # 1b. 防御:模型未用 <think> 标签、把英文自我推理混进正文时, 剥离开头的英文"自言自语"块
    text = strip_leaked_reasoning(text)
    # 2. 去掉尾部未闭合的标签残片(如 <th / <thin / <thinki)
    text = re.sub(r'\s*<t[a-z]*\s*$', '', text).strip()
    # 3. 去掉尾部回声: 结尾若重复了紧邻其前的短词/短句, 只保留一段
    text = re.sub(r'(\S.{0,14}?)\s*\n?\s*\1\s*$', r'\1', text, flags=re.DOTALL)
    return text.strip()


class StreamRouter:
    """流式回调路由器,支持 native reasoning 和 <think> 标签.

    额外抑制原始 [TOOL:...] 标签直接泄漏到 content 流;这些标签会保留在
    content_parts 中供后续工具解析,但不会被用户看到.
    """

    def __init__(self, cb, reasoning_parts, content_parts):
        self.cb = cb
        self.reasoning_parts = reasoning_parts
        self.content_parts = content_parts
        self.state = 'start'  # start, think, content, tool_call_raw
        self.buffer = ''
        self.MAX_START_BUFFER = 30
        # Marker detection: keep buffer small so we don't delay real content too long
        self._tool_marker = '[TOOL:'
        self._tool_end_marker = '[/TOOL]'

    def _append(self, kind, text):
        """Record text in the corresponding part list without emitting to callback."""
        if not text:
            return
        if kind == 'reasoning':
            self.reasoning_parts.append(text)
        else:
            self.content_parts.append(text)

    def _emit(self, kind, text):
        """Emit text to callback and record it in parts."""
        if text:
            self.cb(kind, text)
            self._append(kind, text)

    def _can_still_be_think_tag(self):
        """判断当前缓冲区是否仍有可能匹配 <think>/<thinking>(空缓冲不算)"""
        prefix = self.buffer.lstrip()[:9]
        if not prefix:
            return False
        return ('<think>'.startswith(prefix) or '<thinking>'.startswith(prefix)) and len(prefix) < 9

    def _can_still_be_tool_marker(self):
        """判断当前缓冲区是否仍有可能匹配 [TOOL:(纯空白不算, 避免吞掉正文开头)"""
        stripped = self.buffer.lstrip()
        if not stripped:
            return False
        return self._tool_marker.startswith(stripped[:len(self._tool_marker)])

    def on_token(self, kind, token):
        if kind == 'reasoning':
            # native reasoning:直接透传
            if self.state == 'start':
                # 清空可能存在的 start 缓冲
                if self.buffer.strip():
                    self._emit('content', self.buffer)
                self.buffer = ''
            self.state = 'content'
            self._emit('reasoning', token)
            return

        # kind == 'content'
        if self.state == 'start':
            self.buffer += token
            stripped = self.buffer.lstrip()
            if stripped.startswith(('<think>', '<thinking>')):
                self.state = 'think'
                self.buffer = stripped[len('<think>'):]
            elif stripped.startswith(self._tool_marker):
                self.state = 'tool_call_raw'
                self.buffer = stripped
            elif len(self.buffer) >= self.MAX_START_BUFFER or not self._can_still_be_think_tag():
                self.state = 'content'
                self._emit('content', self.buffer)
                self.buffer = ''

        elif self.state == 'think':
            self.buffer += token
            close = self.buffer.find('</think')  # 同时匹配 </think> 与 </thinking>
            if close != -1:
                end = self.buffer.find('>', close)
                if end != -1:
                    self._emit('reasoning', self.buffer[:close])
                    self.state = 'content'
                    self.buffer = self.buffer[end + 1:]
                    if self.buffer:
                        self._emit('content', self.buffer)
                        self.buffer = ''

        elif self.state == 'content':
            # While emitting content, detect an inline [TOOL: marker and suppress it.
            if self._tool_marker in token or self._can_still_be_tool_marker():
                self.buffer += token
                if self._tool_marker in self.buffer:
                    # Split at the marker: emit any leading content, then suppress the rest
                    idx = self.buffer.find(self._tool_marker)
                    if idx > 0:
                        self._emit('content', self.buffer[:idx])
                    self.state = 'tool_call_raw'
                    self.buffer = self.buffer[idx:]
                return
            if '<think' in token or self._can_still_be_think_tag() or (token.startswith('<') and not self.buffer):
                # 内容后模型又开新的 think 块(常见回声), 转 think 状态抑制其泄漏为正文
                self.buffer += token
                if '<think' in self.buffer:
                    idx = self.buffer.find('<think')
                    if idx > 0:
                        self._emit('content', self.buffer[:idx])
                    self.state = 'think'
                    self.buffer = self.buffer[idx + len('<think'):]
                elif not self._can_still_be_think_tag():
                    # 攒够了但不是 think 标签(如数学符号 "< 5"), 作为普通内容吐出
                    self._emit('content', self.buffer)
                    self.buffer = ''
                return
            self._emit('content', token)

        elif self.state == 'tool_call_raw':
            self.buffer += token
            if self._tool_end_marker in self.buffer:
                end_idx = self.buffer.find(self._tool_end_marker) + len(self._tool_end_marker)
                raw_call = self.buffer[:end_idx]
                # Keep the raw call in content_parts for downstream parsing but don't emit it.
                self._append('content', raw_call)
                self.buffer = self.buffer[end_idx:]
                self.state = 'content'
                if self.buffer:
                    self._emit('content', self.buffer)
                    self.buffer = ''
            elif '\n\n' in self.buffer:
                # Heuristic: tool call without [/TOOL] ends at a blank line
                parts = self.buffer.split('\n\n', 1)
                self._append('content', parts[0])
                self.buffer = parts[1]
                self.state = 'content'
                if self.buffer:
                    self._emit('content', self.buffer)
                    self.buffer = ''

    def finalize(self):
        if self.buffer:
            if self.state == 'think':
                self._emit('reasoning', self.buffer)
            elif self.state == 'tool_call_raw':
                # Preserve raw tool call for parser but don't leak it as content
                self._append('content', self.buffer)
            else:
                # content 态残留 buffer 尚未逐字透出过, emit 补上(避免丢尾)
                self._emit('content', self.buffer)
            self.buffer = ''
