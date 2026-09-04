"""
Model Backend Support:
- OpenAICompatBackend: OpenAI-compatible cloud API (OpenAI format)
- OpenAIBackend: OpenAI-compatible endpoints (local or cloud)
- DeepSeekBackend: Official DeepSeek API (OpenAI-compatible)
- AnthropicBackend: Native Anthropic Messages API
- OpenMythosBackend: Local OpenMythos model (offline)
"""

import json
import os
import random
import time
from typing import Dict, Any, Optional, List, Callable

import requests

from .terminal import style as _style


_TIKTOKEN_ENC = None


def _get_tiktoken():
    """Cached tiktoken encoder — avoids repeated get_encoding() calls."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = False
    return _TIKTOKEN_ENC


def _estimate_tokens(text: str) -> int:
    """Estimate token count. Prefer tiktoken; fall back to character heuristic."""
    if not text:
        return 0
    enc = _get_tiktoken()
    if enc:
        return len(enc.encode(text))
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return max(1, cjk // 2 + (len(text) - cjk) // 4)


def _extract_http_status(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from common API exception shapes."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            return status
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        status = body.get("status") or body.get("status_code")
        if isinstance(status, int):
            return status
    # Try string parsing as fallback.
    msg = str(exc)
    if "429" in msg:
        return 429
    return None


class AnthropicBackend:
    """Native Anthropic Messages API backend."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = None
        self.last_total_tokens = 0

        print(_style("  Anthropic backend", "2"))
        print(f"    {_style('model', '2')} {model}")
        print(f"    {_style('url', '2')}   {base_url}")

    @property
    def client(self):
        """Lazy load anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic backend. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        n_loops: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        **kwargs
    ) -> str:
        """Generate text via Anthropic Messages API."""
        # Anthropic uses a top-level system parameter; we split it from the user prompt.
        system_prompt = self._build_system_prompt(n_loops)
        messages = [{"role": "user", "content": prompt}]

        # Drop OpenAI-specific args that Anthropic does not accept.
        kwargs.pop("stream_callback", None)
        kwargs.pop("top_p", None)

        use_stream = stream or bool(stream_callback) or bool(token_callback)

        full_prompt_text = system_prompt + "\n" + prompt
        if token_callback:
            try:
                token_callback(_estimate_tokens(full_prompt_text))
            except Exception:
                pass

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if use_stream:
                    with self.client.messages.stream(
                        model=self.model,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature or self.temperature,
                        system=system_prompt,
                        messages=messages,
                        **kwargs
                    ) as stream_resp:
                        content_parts = []
                        for text in stream_resp.text_stream:
                            content_parts.append(text)
                            if stream_callback:
                                stream_callback("content", text)
                            if token_callback:
                                try:
                                    token_callback(_estimate_tokens(text))
                                except Exception:
                                    pass
                        content = "".join(content_parts)
                        if not content:
                            raise ValueError("Empty streaming response from Anthropic")
                        return content

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    system=system_prompt,
                    messages=messages,
                    **kwargs
                )

                usage = getattr(response, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    self.last_total_tokens = input_tokens + output_tokens
                    if token_callback:
                        try:
                            token_callback(int(self.last_total_tokens))
                        except Exception:
                            pass

                content_parts = []
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        content_parts.append(getattr(block, "text", ""))
                content = "".join(content_parts)
                if not content:
                    raise ValueError("Empty response from Anthropic")
                return content

            except Exception as e:
                status = _extract_http_status(e)
                is_rate_limit = status == 429
                if attempt < max_attempts - 1:
                    if is_rate_limit:
                        wait = 5 * (2 ** attempt)
                        print(_style(f"  Anthropic rate limit (429), retrying in {wait}s", "2"))
                    else:
                        wait = 2 ** attempt
                        print(_style(f"  Anthropic API error, retrying in {wait}s: {e}", "2"))
                    time.sleep(wait)
                else:
                    if is_rate_limit:
                        print(_style(f"  Anthropic rate limit (429) after {max_attempts} attempts. Please wait and retry.", "31"))
                    else:
                        print(_style(f"  Anthropic API error after {max_attempts} attempts: {e}", "31"))
                    raise

    def _build_system_prompt(self, n_loops: int) -> str:
        """Build system prompt similar to other backends."""
        base = """You are a deep-thinking AI agent with access to tools.

Your core capability: **Deep Reasoning**
- Before taking any action, think step by step.
- Analyze the problem thoroughly.
- Plan before executing.

Available tools (use the exact name):
- web_search: Search the web for the latest/real-time information. (query: string)
  e.g. web_search(query="今天的AAPL收盘价")
- calculator: Perform pure arithmetic. (expression: string)
  e.g. calculator(expression="(12+7)*3")
- python_exec: Execute Python code for computation/processing. (code: string)
  e.g. python_exec(code="print(2**10)")
- file_ops: Read/write/list/search files. (action: "read"|"write"|"list"|"grep", path: string, content?: string)
  e.g. file_ops(action="read", path="config.yaml") ; 新建文章: file_ops(action="write", path="分析.md", content="...")
- bash_exec: Run shell commands (install, git clone, build, etc.). (command: string)
  e.g. bash_exec(command="git clone https://github.com/x/y.git 目标目录")
- glob: Find files by name pattern. (pattern: string, path?: string)
  e.g. glob(pattern="**/*.md", path="~/Desktop")
- api_call: Make HTTP requests to allowed hosts. (url: string, method: string, headers?: dict, data?: dict)

Guidelines:
1. THINK deeply before acting.
2. When uncertain, use tools to gather information.
3. After receiving observation, analyze it and decide next step.
4. Continue until task is complete.
5. When the user asks to create/write/save a new article or file, call file_ops with action="write", then verify by reading it back.

CONTEXT COMPREHENSION:
- When the user provides a long pasted text, large codebase, or multi-turn conversation, first identify and summarize the key context: the user's goal, constraints, prior decisions, and any unresolved questions.
- Before answering, briefly restate your understanding of the context to ensure alignment.
- If the context exceeds your effective window, focus on the most recent and most relevant parts, and ask the user for clarification rather than hallucinating.
"""
        if n_loops >= 16:
            depth = "VERY HIGH: extensive reasoning, consider 3-5 approaches, self-critique."
        elif n_loops >= 8:
            depth = "HIGH: thorough but concise, evaluate 2-3 approaches."
        else:
            depth = "MODERATE: quick but careful reasoning, 1-2 approaches."
        return base + f"\nTHINKING DEPTH: {depth}\n"


class OpenMythosBackend:
    """
    OpenMythos本地模型后端（保留用于对比或离线使用）
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", timeout: int = 120, **kwargs):
        print("  loading OpenMythos model...")
        try:
            from open_mythos.main import OpenMythos, MythosConfig
            self.OpenMythos = OpenMythos
            self.MythosConfig = MythosConfig
        except ImportError:
            raise ImportError("OpenMythos not installed. Run: pip install -e ../OpenMythos-main")

        cfg = MythosConfig(
            vocab_size=32000,
            dim=kwargs.get('dim', 256),
            n_heads=kwargs.get('n_heads', 8),
            max_seq_len=4096,
            max_loop_iters=kwargs.get('max_loops', 8),
            prelude_layers=2,
            coda_layers=2,
            attn_type=kwargs.get('attention_type', 'mla'),
            n_experts=kwargs.get('n_experts', 8),
            n_shared_experts=kwargs.get('n_shared_experts', 1),
            n_experts_per_tok=kwargs.get('n_experts_per_tok', 2),
            expert_dim=kwargs.get('expert_dim', 64),
        )

        self.model = OpenMythos(cfg)
        self.model.eval()
        self.model.to(device)

        # 加载权重（如果有）
        if model_path:
            import torch
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)

        # Tokenizer
        try:
            from open_mythos.tokenizer import MythosTokenizer
            self.tokenizer = MythosTokenizer()
            print("  Using MythosTokenizer")
        except ImportError:
            print(_style("  MythosTokenizer not available, using fallback", "2"))
            self.tokenizer = self._create_simple_tokenizer()


    def _create_simple_tokenizer(self):
        """简单tokenizer回退"""
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 32000
            def encode(self, text):
                return [ord(c) % 32000 for c in text]
            def decode(self, tokens):
                return ''.join(chr(int(t) % 256) for t in tokens)
        return SimpleTokenizer()

    def generate(
        self,
        prompt: str,
        n_loops: int = 8,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        使用OpenMythos生成
        注意：OpenMythos原生支持n_loops（内部循环）
        """
        import torch
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        with torch.no_grad():
            # 关键：使用OpenMythos的n_loops参数进行深度推理
            logits = self.model.forward(
                input_ids,
                n_loops=n_loops,
                return_hidden=False
            )
            # 解码
            tokens = logits[0].argmax(dim=-1).cpu().tolist()
            # 生成完整序列（简化：这里只返回预填充的结果）
            # 实际应该用generate方法
            output = self.tokenizer.decode(tokens)

        return output


class OpenAIBackend:
    """
    OpenAI-compatible backend.
    Supports any OpenAI API-compatible endpoint:
    - Local: http://localhost:20128 (your example)
    - OpenRouter: https://openrouter.ai/api/v1
    - LiteLLM: http://localhost:4000
    - Ollama: http://localhost:11434/v1
    - Any server implementing /v1/chat/completions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        timeout: int = 120,
        bypass_proxy: bool = False,
    ):
        """
        Initialize OpenAI-compatible backend.

        Args:
            api_key: API key (None or "skip" for endpoints without auth)
            base_url: API base URL (e.g., "http://localhost:20128" or "https://openrouter.ai/api/v1")
            model: Model name your endpoint expects (e.g., "oc/deepseek-v4-flash-free")
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            bypass_proxy: When True, ignore system/env proxy settings and connect
                directly. Useful for domestic APIs (e.g. DeepSeek) that are
                reachable without a VPN/proxy, avoiding intermittent "connection
                refused" errors when the local proxy (Clash/V2Ray) is down.
        """
        self.api_key = api_key if api_key and api_key != "skip" else None
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.bypass_proxy = bypass_proxy

        # Connection pool and health check
        self._client = None
        self._consecutive_errors = 0
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes

        print(_style("  OpenAI-compatible backend", "2"))
        print(f"    {_style('model', '2')} {model}")
        if base_url:
            print(f"    {_style('url', '2')}   {base_url}")
        if self.api_key:
            print(f"    {_style('auth', '2')}  {'*' * 12}...")
        else:
            print(f"    {_style('auth', '2')}  none")

    @property
    def client(self):
        """Lazy load openai client"""
        if self._client is None:
            try:
                import openai
                kwargs = dict(
                    base_url=self.base_url,
                    api_key=self.api_key or "skip",  # openai lib requires non-empty string
                )
                if self.bypass_proxy:
                    # Bypass system/env proxy (Clash/V2Ray at 127.0.0.1:7891 etc.)
                    # so domestic APIs like DeepSeek connect directly and don't
                    # fail when the local proxy is down.
                    try:
                        import httpx
                        kwargs["http_client"] = httpx.Client(
                            trust_env=False,
                            timeout=httpx.Timeout(self.timeout, connect=10.0),
                            # Connection pool keep-alive: prevent stale connections after idle
                            limits=httpx.Limits(
                                max_keepalive_connections=5,
                                keepalive_expiry=30.0,
                            ),
                        )
                    except ImportError:
                        pass
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI-compatible backend. "
                    "Install with: pip install openai"
                )
        return self._client

    def _build_system_prompt(self, n_loops: int) -> str:
        """
        构建system prompt，引导模型输出OpenMythos期望的[TOOL:]格式
        
        Args:
            n_loops: 思考深度参数
            
        Returns:
            System prompt字符串
        """
        # cancri fast (DeepSeek-V4-Flash @ AMD): 超级智能编程专家约束
        expert_prefix = ""
        if self.model == "DeepSeek-V4-Flash":
            expert_prefix = """You are "cancri fast" — a super-intelligent programming expert agent.
Hard constraints:
- You are first and foremost a world-class software engineer: precise, efficient, production-oriented.
- Always prefer correct, runnable code over prose. When asked to code, output the full implementation, then verify it (run/build/test via tools when available).
- Follow repo conventions; make minimal, reviewable diffs; never break existing behavior without explicit reason.
- Explain briefly (what/why), then act. No filler, no hallucinating APIs — check signatures via tools/search before use.
- If requirements are ambiguous, state assumptions explicitly and proceed with the most reasonable choice.

"""
        base = expert_prefix + """You are a deep-thinking AI agent with access to tools.

Your core capability: **Deep Reasoning**
- Before taking any action, think step by step in latent space
- Analyze the problem thoroughly
- Consider multiple approaches
- Plan before executing

TOOL CALLING FORMAT (MANDATORY):
When you need to use a tool, output it in this EXACT format:

[TOOL:tool_name]
{"argument": "value", ...}
[/TOOL]

Examples:
[TOOL:web_search]
{"query": "今天的AAPL收盘价"}
[/TOOL]

[TOOL:calculator]
{"expression": "(12+7)*3"}
[/TOOL]

[TOOL:file_ops]
{"action": "read", "path": "config.yaml"}
[/TOOL]

[TOOL:bash_exec]
{"command": "git clone https://github.com/x/y.git 目标目录"}
[/TOOL]

[TOOL:python_exec]
{"code": "print(2**10)"}
[/TOOL]

[TOOL:glob]
{"pattern": "**/*.md", "path": "~/Desktop"}
[/TOOL]

Available tools (use the exact name):
- web_search: Search the web for the latest/real-time information. (query: string)
- calculator: Perform pure arithmetic. (expression: string)
- python_exec: Execute Python code for computation/processing. (code: string)
- file_ops: Read/write/list/search files. (action: "read"|"write"|"list"|"grep", path: string, content?: string)
- bash_exec: Run shell commands (install, git clone, build, etc.). (command: string)
- glob: Find files by name pattern. (pattern: string, path?: string)
- api_call: Make HTTP requests to allowed hosts. (url: string, method: string, headers?: dict, data?: dict)

Guidelines:
1. THINK deeply before acting. Use your full reasoning capacity.
2. When uncertain, use tools to gather information.
3. After receiving observation, analyze it and decide next step.
4. Continue until task is complete.
5. When the user asks to create/write/save a new article or file, call file_ops with action="write", then verify by reading it back.
6. FILE READ BEFORE SUMMARIZE: If user asks to summarize/extract/提炼要点 from a file, you MUST first call file_ops action='read' with the exact path, get the file content, then summarize. NEVER summarize from memory or from a directory listing alone. Directory listing is not content.
7. MEMORY RECALL DISCIPLINE: When a local file exists in the working directory, prioritize fresh file_ops read over experience memory. Do not hallucinate content from past episodes.
8. SUMMARY FORMAT AND VALIDATION: For 提炼要点 tasks, output must contain at least sections: 公司概况 / 核心产品 / 技术里程碑. After generating, self-check: if any section missing or content is generic, you MUST re-read the file with file_ops action='read' and regenerate. No rollback is allowed without fresh read.

CONTEXT COMPREHENSION:
- When the user provides a long pasted text, large codebase, or multi-turn conversation, first identify and summarize the key context: the user's goal, constraints, prior decisions, and any unresolved questions.
- Before answering, briefly restate your understanding of the context to ensure alignment.
- If the context exceeds your effective window, focus on the most recent and most relevant parts, and ask the user for clarification rather than hallucinating.
"""

        # 根据n_loops调整思考深度指导
        # OpenMythos 思路: 循环深度推理(Recurrent-Depth Reasoning)
        # 把"循环迭代+ACT自适应停止"翻译给文本模型, 让它模拟潜空间循环思考:
        # 每轮 = 压缩当前理解 -> 反思 -> 更新"思维状态", 自主决定是否继续深挖。
        if n_loops >= 16:
            depth_guide = """
THINKING DEPTH: RECURRENT-DEPTH REASONING (n_loops=16+, 深层循环)
执行 OpenMythos 式循环深度推理(在"思维状态空间"迭代, 而非单次线性回答):
1. ROUND 1(首次前向): 快速通读问题, 形成初步理解。压缩为一行"当前状态"。
2. ROUND 2..N(循环迭代): 每一轮都基于"上一轮状态"继续——
   · 修正: 指出上一轮结论的漏洞/未考虑因素
   · 深化: 从不同视角(因果/反事实/类比)重新审视
   · 扩张: 尝试 3-5 种不同思路, 含最不可能的那种
   · 验证: 假设是否成立?证据链是否完整?
   每轮结束时, 把最新理解压缩成新的"当前状态", 供下一轮使用。
3. ACT 自适应停止: 每轮自问"状态是否收敛?"——若连续两轮状态无实质变化
   (无新发现/无新视角/无矛盾), 立即停止循环, 否则继续, 最多到 n_loops。
4. 收敛后: 基于最终"思维状态"输出答案。
关键: 这是循环递归, 不是罗列要点。后一轮必须建立在前一轮之上。
"""
        elif n_loops >= 8:
            depth_guide = """
THINKING DEPTH: RECURRENT REASONING (n_loops=8-15, 中等循环深度)
1. ROUND 1: 初步理解, 记录"当前状态"。
2. ROUND 2: 从反方向审视——我哪里可能错了?遗漏了什么?
3. ROUND 3: 若状态有变化则继续深化(因果/反事实), 否则收敛。
每轮更新"当前状态", 以最近一轮为基础继续, 不重复已确认结论。
"""
        else:
            depth_guide = """
THINKING DEPTH: LIGHT (n_loops<8, 快速收敛)
- 一轮通读 -> 一轮快速核查(有无明显漏洞/缺失) -> 若无大问题即输出。
- 不展开多轮循环, 保持效率。
"""

        return base + depth_guide

    def _ensure_connection(self) -> None:
        """Lightweight health check: recreate client if connection is stale."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return
        self._last_health_check = time.time()
        try:
            # Use a minimal request to verify connection with short timeout
            import httpx
            resp = httpx.get(f"{self.base_url.rstrip('/')}/models", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            # Connection likely stale (idle timeout, proxy down, Ollama unloaded model)
            # Force recreate on next property access
            self._client = None

    def _handle_error(self, e: Exception) -> bool:
        """Handle errors with retry logic. Returns True if should retry."""
        self._consecutive_errors += 1
        status = _extract_http_status(e)
        is_rate_limit = status == 429
        is_transient = status in (408, 429, 500, 502, 503, 504) or status is None
        if is_transient:
            wait = min(2 ** self._consecutive_errors, 30)
            print(_style(f"  Connection issue, retrying in {wait}s...", "2"))
            time.sleep(wait)
            self._client = None  # Force reconnect
            return True
        return False

    def _should_retry(self, e: Exception) -> bool:
        """Determine if an error is retryable.

        Retry only on transient failures:
          - network-level errors (ConnectionError / TimeoutError)
          - HTTP 408 (timeout), 429 (rate limit), 5xx (server errors)
          - status is None (unclassified connection-level failure)
        Permanent 4xx client errors — including 401/403 auth failures — are NOT
        retried: they will never succeed, so retrying just wastes time/budget.
        """
        if isinstance(e, (ConnectionError, TimeoutError)):
            return True
        status = _extract_http_status(e)
        if status is None:
            return True
        if status in (408, 429, 500, 502, 503, 504):
            return True
        # Permanent client errors: do not retry.
        return False

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff with jitter: base * 2^attempt + jitter"""
        base = min(2 ** self._consecutive_errors, 30)
        jitter = random.uniform(0, 0.5)
        return min(base + jitter, 30)

    def _reset_errors(self):
        """Reset consecutive error counter on success."""
        self._consecutive_errors = 0

    def _is_ollama(self) -> bool:
        """Heuristic: is this backend pointing at a local Ollama server?"""
        bu = (self.base_url or "").lower()
        return "11434" in bu or "ollama" in bu

    def generate(
        self,
        prompt: str,
        n_loops: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,  # ignored for text mode
        stream_callback: Optional[Callable] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        **kwargs
    ) -> str:
        """
        Generate text in OpenMythos-compatible format using [TOOL:] syntax.

        This backend always uses text mode with custom tool format to match
        the OpenMythos agent architecture.

        Args:
            prompt: User prompt
            n_loops: Reasoning depth (affects system prompt)
            temperature: Override temperature
            max_tokens: Override max_tokens
            tools: Optional tool definitions for native function calling
            **kwargs: Additional API arguments

        Returns:
            Generated text string (may include [TOOL:] calls)
        """
        # stream_callback/token_callback are handled internally; do not forward them to the API
        kwargs.pop("stream_callback", None)

        def _report_usage(resp):
            if token_callback is None:
                return
            total = None
            usage = getattr(resp, "usage", None)
            if usage is not None:
                total = getattr(usage, "total_tokens", None)
                if total is None:
                    total = getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
            if total:
                try:
                    token_callback(int(total))
                except Exception:
                    pass

        def _single_call(msgs, stream: bool):
            create_args = dict(
                model=self.model,
                messages=msgs,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens,
                timeout=self.timeout,
                stream=stream,
                **kwargs,
            )
            # 本地 Ollama: 钉住模型常驻显存, 消除冷启动(~5.5s)的等待
            if self._is_ollama():
                create_args["keep_alive"] = -1
            completion = self.client.chat.completions.create(**create_args)
            if stream:
                content_parts = []
                finish_reason = None
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta is None:
                        continue
                    text = getattr(delta, "content", None) or ""
                    reasoning_text = getattr(delta, "reasoning_content", None) or ""
                    if text:
                        content_parts.append(text)
                        if stream_callback:
                            try:
                                stream_callback("content", text)
                            except Exception:
                                pass
                        if token_callback:
                            try:
                                token_callback(_estimate_tokens(text))
                            except Exception:
                                pass
                    elif reasoning_text:
                        # 与非流式路径 content or reasoning 兜底对齐: reasoning 仅在无 content 时并入
                        content_parts.append(reasoning_text)
                        if stream_callback:
                            try:
                                stream_callback("reasoning", reasoning_text)
                            except Exception:
                                pass
                        if token_callback:
                            try:
                                token_callback(_estimate_tokens(reasoning_text))
                            except Exception:
                                pass
                result = "".join(content_parts)
                if not result:
                    raise ValueError("Empty streaming response from model")
                return result.strip(), finish_reason

            # Non-streaming
            _report_usage(completion)
            message = completion.choices[0].message
            content = getattr(message, 'content', None) or ""
            reasoning = getattr(message, 'reasoning_content', None) or ""
            result = content or reasoning or ""
            finish_reason = getattr(completion.choices[0], 'finish_reason', None)
            return result.strip(), finish_reason

        system_prompt = self._build_system_prompt(n_loops)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        use_stream = bool(stream_callback) or bool(token_callback)

        # Report estimated prompt tokens immediately so the counter starts moving.
        full_prompt_text = system_prompt + "\n" + prompt
        if token_callback:
            try:
                token_callback(_estimate_tokens(full_prompt_text))
            except Exception:
                pass

        # Retry logic with smart status-code handling.
        # Ensure connection is fresh (handles idle timeout, proxy down, Ollama model unload)
        self._ensure_connection()
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Auto-continue if the model stopped due to length (max_tokens hit)
                max_continuations = 2
                collected_parts = []
                for continuation in range(max_continuations + 1):
                    result, finish_reason = _single_call(messages, use_stream)
                    collected_parts.append(result)
                    if finish_reason != "length":
                        break
                    # Build continuation prompt: keep system + original user + truncated assistant output
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": result},
                        {"role": "user", "content": "Continue exactly from where you left off. Do not repeat what has already been said."}
                    ]
                    if token_callback:
                        try:
                            token_callback(_estimate_tokens("[continuation triggered]"))
                        except Exception:
                            pass

                self._consecutive_errors = 0
                return "\n".join(collected_parts).strip()

            except Exception as e:
                status = _extract_http_status(e)
                # Authentication / authorization errors should fail fast.
                if status in (401, 403):
                    raise RuntimeError(
                        f"OpenAI API authentication failed ({status}). "
                        "Please check your API key and base_url."
                    ) from e
                # Rate limit: exponential backoff.
                if status == 429:
                    wait = 5 * (2 ** attempt)
                    print(_style(f"  rate limit (429), retrying in {wait}s", "2"))
                    time.sleep(wait)
                    continue
                # Other retryable errors.
                if self._should_retry(e):
                    wait = self._calculate_backoff(attempt)
                    print(_style(f"  API error, retrying in {wait:.1f}s: {e}", "2"))
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"OpenAI API error: {e}") from e


    def generate_native(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        n_loops: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream_callback: Optional[Callable] = None,
        token_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """原生 Function Calling: 返回结构化 {content, tool_calls}.

        相比文本协议(generate + 正则解析), 这是确定性路径:
        - 传 tools 给 API, 模型返回结构化的 tool_calls(name + arguments JSON)
        - 无需猜测格式, 无需 1300 行解析器
        - 无工具调用时返回纯 content

        stream_callback 非空时开启流式: 边生成边回调 content/reasoning,
        避免慢速远程模型下"长时间空白 -> 一次性蹦出全文"的体感卡顿。
        若端点不支持 tools+stream, 自动回退到非流式。

        Returns:
            {"content": str, "tool_calls": [{"name": str, "arguments": dict}],
             "_streamed_content": bool}
            _streamed_content 供调用方抑制重复透出。
        """
        stream = bool(stream_callback)
        try:
            # Ensure connection is fresh (handles idle timeout, proxy down, Ollama model unload)
            self._ensure_connection()
            messages = [
                {"role": "system", "content": self._build_system_prompt(n_loops)},
                {"role": "user", "content": prompt},
            ]
            payload: Dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens,
                timeout=self.timeout,
            )
            if tools:
                payload["tools"] = tools
            if stream:
                payload["stream"] = True
            payload.update(kwargs)
            # 本地 Ollama: 钉住模型常驻显存, 消除冷启动(~5.5s)的等待
            if self._is_ollama():
                payload["keep_alive"] = -1

            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            tool_slots: Dict[int, Dict[str, str]] = {}
            streamed_content = False

            if stream:
                try:
                    for chunk in self.client.chat.completions.create(**payload):
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        text = getattr(delta, "content", None)
                        if text:
                            content_parts.append(text)
                            streamed_content = True
                            stream_callback("content", text)  # type: ignore[union-attr]
                            if token_callback:
                                token_callback(_estimate_tokens(text))
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            stream_callback("reasoning", reasoning)  # type: ignore[union-attr]
                            if token_callback:
                                token_callback(_estimate_tokens(reasoning))
                        for tc_delta in (getattr(delta, "tool_calls", None) or []):
                            idx = getattr(tc_delta, "index", 0) or 0
                            slot = tool_slots.setdefault(idx, {"name": "", "arguments": ""})
                            fn = getattr(tc_delta, "function", None)
                            if fn is not None:
                                nm = getattr(fn, "name", None)
                                if nm:
                                    slot["name"] += nm
                                ar = getattr(fn, "arguments", None)
                                if ar:
                                    slot["arguments"] += ar
                    self._consecutive_errors = 0
                    return self._finalize_native(
                        "".join(content_parts) or "".join(reasoning_parts),
                        list(tool_slots.values()),
                        streamed_content,
                    )
                except Exception:
                    # 端点不支持 tools+stream, 回退到非流式
                    pass

            # 非流式(回退或 stream 关闭)
            non_stream = {k: v for k, v in payload.items() if k != "stream"}
            completion = self.client.chat.completions.create(**non_stream)
            message = completion.choices[0].message
            content = getattr(message, "content", None) or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
            if not content and reasoning:
                content = reasoning

            tool_calls = []
            raw_tcs = getattr(message, "tool_calls", None) or []
            for tc in raw_tcs:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                name = getattr(fn, "name", "") or ""
                args_raw = getattr(fn, "arguments", "{}") or "{}"
                arguments = {}
                try:
                    arguments = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw or {})
                except Exception:
                    arguments = {"_raw": args_raw}
                tool_calls.append({"name": name, "arguments": arguments})

            self._consecutive_errors = 0
            return self._finalize_native(content, tool_calls, False)
        except Exception as e:
            # 失败时降级: 返回空 tool_calls, 由调用方回退文本协议
            raise RuntimeError(f"generate_native failed: {e}") from e

    @staticmethod
    def _finalize_native(content: str, raw_tcs: List[Dict[str, str]], streamed_content: bool) -> Dict[str, Any]:
        tool_calls = []
        for tc in raw_tcs:
            name = tc.get("name", "") or ""
            args_raw = tc.get("arguments", "{}") or "{}"
            arguments = {}
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw or {})
            except Exception:
                arguments = {"_raw": args_raw}
            tool_calls.append({"name": name, "arguments": arguments})
        return {
            "content": content,
            "tool_calls": tool_calls,
            "_streamed_content": streamed_content,
        }


class DeepSeekBackend(OpenAIBackend):
    """
    Official DeepSeek API backend.
    DeepSeek uses an OpenAI-compatible /v1/chat/completions endpoint.
    Docs: https://platform.deepseek.com/api_docs/
    """

    KNOWN_MODELS = {
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        timeout: int = 120,
        bypass_proxy: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            bypass_proxy=bypass_proxy,
        )
        if model not in self.KNOWN_MODELS:
            print(_style(f"  warning: model '{model}' is not in known DeepSeek models: {sorted(self.KNOWN_MODELS)}", "33"))
        print(_style("  DeepSeek backend", "2"))
        print(f"    {_style('model', '2')} {model}")
        print(f"    {_style('url', '2')}   {base_url}")
        if self.bypass_proxy:
            print(f"    {_style('proxy', '2')} bypassed (direct connection)")

def get_backend():
    """Factory to get current backend instance based on prefs > config > env."""
    import json, os
    from pathlib import Path
    # 1) 偏好文件
    pref_path = Path(__file__).parent.parent / 'data' / 'config' / 'model_prefs.json'
    pref_data = {}
    try:
        if pref_path.exists():
            pref_data = json.loads(pref_path.read_text(encoding='utf-8'))
    except Exception:
        pref_data = {}
    # 2) config
    try:
        from .config import load_config
        cfg = load_config()
        agent_cfg = getattr(cfg, "agent", None)
        cfg_base_url = getattr(agent_cfg.openai, "base_url", None) if agent_cfg else None
        cfg_model = getattr(agent_cfg.openai, "model", None) if agent_cfg else None
        cfg_api_key = getattr(agent_cfg.openai, "api_key", None) if agent_cfg else None
    except Exception:
        cfg_base_url = cfg_model = cfg_api_key = None
    # 3) 优先级合并
    base_url = pref_data.get('base_url') or cfg_base_url or os.getenv('OPENAI_BASE_URL', '')
    model = pref_data.get('model') or cfg_model or os.getenv('OPENAI_MODEL', '')
    api_key = pref_data.get('api_key') or cfg_api_key or os.getenv('OPENAI_API_KEY', '')
    if not base_url or not model:
        raise RuntimeError('模型后端未配置: 请先执行 /model 选择模型，或设置 OPENAI_BASE_URL / OPENAI_MODEL 环境变量。当前不会回退到本地 Ollama。')
    from .model_backends import OpenAIBackend
    try:
        backend = OpenAIBackend(api_key=api_key or '', base_url=base_url, model=model)
        # 打印当前使用模型
        try:
            print(f" backend OpenAIBackend · model {model} · base_url {base_url}")
        except Exception:
            pass
        return backend
    except Exception:
        class DummyBackend:
            def generate(self, prompt, max_tokens=256, temperature=0.3):
                return ""
        return DummyBackend()