"""
OpenMythosAgent - 深度思考智能体
利用OpenMythos循环架构 + 工具调用 + 自我改进
**UPGRADED: World-class agent with planning, reasoning engine, memory, and self-correction**
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError

_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent_bg_")

# Base imports only: heavy modules are loaded lazily inside _init_advanced_modules()
# to keep agent startup fast, especially when memory/planning/reflection are disabled.
from .config import AgentConfig
from .tools import TOOLS_REGISTRY, BaseTool, ToolResult, ToolCall
from .research_report import is_research_report_task, generate_research_report
from .health import ModuleHealthChecker, ModuleStatus
from .execution_engine import ExecutionContext, ExecutionEngine
from .policies import DirectPolicy
from .terminal import style as _style

try:
    from rich.console import Console
    from rich.prompt import Prompt
    _HAS_RICH_PROMPT = True
except Exception:  # pragma: no cover - rich is in requirements-core but keep fallback
    Console = None  # type: ignore
    Prompt = None  # type: ignore
    _HAS_RICH_PROMPT = False

# Project root (parent of agent_project package) for safe default file outputs.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OpenMythosAgent:
    """
    深度思考Agent
    - 利用OpenMythos循环架构进行深度推理
    - 支持工具调用
    - 经验存储和检索
    - 自我反思和改进
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        # 日志优先初始化，确保后续模块加载错误能写入文件
        self._setup_logging()

        # 初始化模型后端(支持DeepSeek或OpenMythos本地)
        self.backend = self._load_model_backend()

        # 核心高级模块占位符（延迟加载）
        self.experience_buffer = None
        self.strategy_db = None
        self.reflection_module = None
        self.planner = None
        self.reasoning_engine = None
        self.memory_manager = None
        self.self_correction = None
        self.context_engine = None
        self.memskill_engine = None

        # 跨轮次短期对话历史(用于上下文记忆)
        # 持久化上限调大: sqlite 全量存档, 这里保留更大窗口防止旧对话被过早丢弃
        self.max_history_turns = 200
        self.history_path = Path(__file__).parent.parent / "data" / "conversation_history.json"
        self.conversation_history: List[Dict[str, str]] = self._load_history()

        # Stable session identifier for long-term memory.
        import uuid
        self.session_id = str(uuid.uuid4())

        # 统计
        self.episodes_completed = 0
        self._last_reflection_episode = 0  # 距上次反思已完成的 episode 数(冷却) 
        self._last_confidence = 0.5         # 最近一轮的内部置信度评估(不展示给用户)
        self.outer_loop_counter = 0
        self.total_corrections = 0

        # 自我修正状态:上一次的纠正动作会影响下轮参数
        self._pending_correction: Optional[Dict[str, Any]] = None
        self._strategy_override: Optional[str] = None
        self._last_matched_strategy_id: Optional[str] = None
        self._code_mode_override: bool = False

        # 动态 token 统计（连接真实 LLM 调用）
        self.session_token_usage = {"total": 0, "stream": [], "last_call_tokens": 0}

        # 轻量级每轮缓存，避免对同一任务重复进行正则/文件扫描/策略匹配
        self._method_cache: Dict[Tuple[str, ...], Any] = {}

        # 工具结果缓存，防止同一轮中重复执行相同工具调用（如 super loop 反复读取同一文件）
        self._tool_result_cache: Dict[str, ToolResult] = {}

        # Harness 能力内核（可选）：启用后所有工具调用先过策略门。
        # 必须在 _init_advanced_modules 之前构建，因为 ReasoningEngine 会引用它。
        self._harness_kernel = self._build_harness_kernel()

        # 初始化高级模块（按需导入重依赖）
        self._init_advanced_modules()

        # 工具注册
        self.tools = TOOLS_REGISTRY
        self._setup_tools()

        # Health check: gives users visibility into which modules are ready/degraded.
        self.health_checker = ModuleHealthChecker(config)
        self.module_status = self.health_checker.check_all()

        print(_style("  agent ready", "2"))
        print(f"    {_style('backend', '2')}  {type(self.backend).__name__}")
        print(f"    {_style('tools', '2')}    {len(list(TOOLS_REGISTRY.list_tools()))}")
        print(f"    {_style('loops', '2')}    {config.max_outer_loops}")
        if config.health.print_status_on_startup:
            self.print_module_status()

    # ============ 初始化辅助方法 ============

    def _build_harness_kernel(self):
        """Build the capability kernel when config.harness.enabled; else None."""
        cfg = getattr(self.config, "harness", None)
        if not cfg or not cfg.enabled:
            from .tools import set_harness_kernel
            set_harness_kernel(None)
            return None
        from .harness.kernel import (
            permissive_policy,
            registry_executor,
            safe_default_policy,
            Kernel,
        )
        from .harness.approval import console_approval

        if cfg.policy == "permissive":
            policy = permissive_policy()
        else:
            root = cfg.workspace_root or str(Path(__file__).resolve().parent.parent)
            policy = safe_default_policy(root)
        kernel = Kernel(
            policy=policy,
            executor=registry_executor(TOOLS_REGISTRY),
            ask=console_approval,
            allowlist_path=cfg.allowlist_path,
        )
        from .tools import set_harness_kernel
        set_harness_kernel(kernel)
        self.logger.info(f"harness kernel: enabled (policy={cfg.policy})")
        return kernel

    def print_module_status(self, as_json: bool = False) -> None:
        """Print a human-readable or JSON health table for all advanced modules."""
        if as_json:
            print(self.module_status.to_dict())
            return

        modules = self.module_status.modules
        rows = [["module", "status", "dependency", "fallback"]]
        for m in modules:
            rows.append([
                m.name,
                m.status.value,
                m.dependency or "-",
                m.fallback or "-",
            ])

        # Simple fixed-width rendering without requiring tabulate.
        widths = [max(len(str(r[i])) for r in rows) for i in range(4)]
        for i, row in enumerate(rows):
            line = "  ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row))
            if i == 0:
                print(f"  {_style(line, '2')}")
                print(f"  {_style('-' * (sum(widths) + 6), '2')}")
            else:
                m = modules[i - 1]
                color_code = {
                    ModuleStatus.READY: "32",
                    ModuleStatus.DEGRADED: "33",
                    ModuleStatus.DISABLED: "2",
                    ModuleStatus.FAILED: "31",
                }.get(m.status, "")
                print(f"  {_style(line, color_code)}")
                if self.config.health.show_install_hints and m.install_hint:
                    print(f"    {_style('hint: ' + m.install_hint, '2')}")

    def _init_advanced_modules(self):
        """Initialize advanced modules lazily to keep startup fast."""

        def _ready(name: str, detail: str = ""):
            msg = f"{name}: ready"
            if detail:
                msg += f" ({detail})"
            self.logger.info(msg)

        def _failed(name: str, exc: Exception):
            self.logger.warning(f"{name}: failed ({exc})")

        # Episodic memory buffer: only load when memory/reflection features are active.
        if self.config.memory.enabled or self.config.reflection.enabled:
            try:
                from .experience import ExperienceBuffer
                self.experience_buffer = ExperienceBuffer(self.config)
                _ready("experience")
            except Exception as e:
                _failed("experience", e)

        # Strategy database: lightweight, but still loaded only when reasoning/planning/correction need it.
        try:
            from .strategies import StrategyDatabase
            self.strategy_db = StrategyDatabase(self.config)
        except Exception as e:
            _failed("strategy", e)

        # Reflection module
        if self.config.reflection.enabled:
            try:
                from .reflection import ReflectionModule
                self.reflection_module = ReflectionModule(self, self.config)
                _ready("reflection")
            except Exception as e:
                _failed("reflection", e)

        # Planning
        if self.config.planning.enabled:
            try:
                from .planning import Planner
                self.planner = Planner(
                    model_backend=self.backend,
                    tokenizer=getattr(self.backend, 'tokenizer', None),
                    config=self.config
                )
                _ready("planning")
            except Exception as e:
                _failed("planning", e)

        # Reasoning Engine
        if self.config.reasoning.enabled:
            try:
                from .reasoning import ReasoningEngine
                self.reasoning_engine = ReasoningEngine(
                    model_backend=self.backend,
                    tokenizer=getattr(self.backend, 'tokenizer', None),
                    config=self.config,
                    loop_controller=None,
                    harness_kernel=self._harness_kernel,
                    per_turn_cache=self._tool_result_cache,
                )
                _ready("reasoning")
            except Exception as e:
                _failed("reasoning", e)

        # Memory Manager: keep a raw MemoryManager for file/session memory,
        # and optionally use LLM Wiki as the semantic memory for ContextEngine.
        self.memory_manager: Optional[Any] = None
        self._raw_memory_manager: Optional[Any] = None
        self._wiki_manager: Optional[Any] = None

        if self.config.memory.enabled:
            wiki_available = False
            wiki_manager = None
            try:
                from .wiki_memory import create_memory_manager as create_wiki_manager, LLMWikiManager, PassthroughBackendClient
                wiki_available = True
            except Exception:
                create_wiki_manager = None  # type: ignore

            # Raw memory manager always has file + session memory layers.
            try:
                from .memory import MemoryManager
                raw_mm = MemoryManager(
                    kg_storage=self.config.memory.kg_storage_path,
                    episodic_storage=self.config.memory.episodic_storage_path,
                    embedding_model=self.config.memory.embedding_model,
                    file_memory_path=self.config.memory.file_memory_path,
                    user_memory_path=self.config.memory.user_memory_path,
                    sqlite_session_path=self.config.memory.sqlite_session_path,
                    project_root=str(Path(__file__).resolve().parent.parent),
                )
                self._raw_memory_manager = raw_mm
                self.memory_manager = raw_mm
                _ready("memory")
            except Exception as e:
                _failed("memory", e)
                raw_mm = None

            # Optional LLM Wiki semantic layer.
            if wiki_available and create_wiki_manager is not None and raw_mm is not None:
                try:
                    llm_client = None
                    if self.backend is not None:
                        try:
                            llm_client = PassthroughBackendClient(self.backend)
                        except Exception:
                            llm_client = None
                    wiki_manager = create_wiki_manager(
                        kg_storage=self.config.memory.kg_storage_path,
                        episodic_storage=self.config.memory.episodic_storage_path,
                        embedding_model=self.config.memory.embedding_model,
                        llm_client=llm_client,
                    )
                    self._wiki_manager = wiki_manager
                    mode = "llm" if llm_client else "keyword"
                    _ready("wiki_memory", mode)
                except Exception as e:
                    _failed("wiki_memory", e)
                    wiki_manager = None

        # Context Engine (unified memory & context facade)
        if self.config.memory.enabled:
            try:
                from .context_engine import ContextEngine
                self.context_engine = ContextEngine(
                    self.config,
                    backend=self.backend,
                    episodic_memory=self.experience_buffer,
                    semantic_memory=self._wiki_manager or self.memory_manager,
                )
                # Keep the raw memory manager for file/session memory calls.
                self._raw_memory_manager = self._raw_memory_manager or self.memory_manager
                # Redirect legacy memory_manager calls to the unified engine
                self.memory_manager = self.context_engine
                # 重启后回填持久化对话历史到工作记忆, 恢复跨会话记忆
                try:
                    self.context_engine.seed_history(getattr(self, 'conversation_history', None) or [])
                except Exception as e:
                    self.logger.warning(f"seed history failed {e}")
                _ready("context")
            except Exception as e:
                _failed("context", e)

        # Self-Correction
        if self.config.self_correction.enabled:
            try:
                from .self_correction import SelfCorrectionModule
                self.self_correction = SelfCorrectionModule(self.config.self_correction)
                _ready("correction")
            except Exception as e:
                _failed("correction", e)

        # MemSkill: self-evolving memory skills
        if self.config.memory.enabled:
            try:
                from .memskill import MemSkillEngine

                def _memskill_llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
                    return self.backend.generate(
                        prompt=prompt,
                        n_loops=1,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ).strip()

                self.memskill_engine = MemSkillEngine(
                    llm_call=_memskill_llm_call,
                    skills_dir=self.config.memory.kg_storage_path.replace("kg_store", "memory_skills"),
                    embedding_model=self.config.memory.embedding_model,
                    top_k=getattr(self.config.memory, "memskill_top_k", 3),
                    hard_case_buffer_size=getattr(self.config.memory, "memskill_hard_case_buffer", 50),
                    evolution_interval=getattr(self.config.memory, "memskill_evolution_interval", 5),
                )
                _ready("memskill")
            except Exception as e:
                _failed("memskill", e)

        # Skill engine: reusable mini-agent prompt templates with /skill command.
        try:
            from .skills import SkillEngine

            def _skill_llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
                return self.backend.generate(
                    prompt=prompt,
                    n_loops=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ).strip()

            self.skill_engine = SkillEngine(
                skills_dir="./data/skills",
                embedding_model=self.config.memory.embedding_model,
                llm_call=_skill_llm_call,
            )
            _ready("skills", f"{len(self.skill_engine.list())} loaded")
        except Exception as e:
            _failed("skills", e)
            self.skill_engine = None

    def _load_model_backend(self):
        """
        加载模型后端
        支持:
        - DeepSeek - 推荐,使用deepseek-chat等
        - OpenAI-compatible (本地端点或OpenRouter等)
        - OpenMythos本地模型 (离线)
        """
        # 延迟导入,避免不必要的依赖
        from .model_backends import OpenAIBackend, AnthropicBackend, OpenMythosBackend, DeepSeekBackend

        if self.config.backend == "deepseek":
            # DeepSeek official API (OpenAI-compatible)
            ds_cfg = self.config.deepseek
            api_key = ds_cfg.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError(
                    "DeepSeek backend selected but no API key provided. "
                    "Set agent.deepseek.api_key in config.yaml or DEEPSEEK_API_KEY env var."
                )
            backend = DeepSeekBackend(
                api_key=api_key,
                base_url=ds_cfg.get('base_url', 'https://api.deepseek.com'),
                model=ds_cfg.get('model', 'deepseek-chat'),
                temperature=ds_cfg.get('temperature', self.config.temperature),
                top_p=ds_cfg.get('top_p', 0.9),
                max_tokens=ds_cfg.get('max_tokens', 4096),
                timeout=ds_cfg.get('timeout', 120),
                bypass_proxy=ds_cfg.get('bypass_proxy', True),
            )
            backend.tokenizer = self._create_simple_tokenizer()
            return backend

        elif self.config.backend == "openai":
            """OpenAI-compatible endpoint (local or cloud)"""
            openai_cfg = self.config.openai
            api_key = openai_cfg.get('api_key') or os.getenv('OPENAI_API_KEY')
            
            # api_key can be None for local endpoints without auth
            if api_key == "skip" or api_key is None:
                api_key = None

            base_url = openai_cfg.get('base_url')
            if not base_url:
                raise ValueError(
                    "OpenAI backend selected but no base_url provided. "
                    "Set agent.openai.base_url in config.yaml (e.g., http://localhost:20128)"
                )

            model = openai_cfg.get('model', 'gpt-4o-mini')
            if not model:
                raise ValueError(
                    "OpenAI backend selected but no model specified. "
                    "Set agent.openai.model in config.yaml"
                )

            backend = OpenAIBackend(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=openai_cfg.get('temperature', self.config.temperature),
                top_p=openai_cfg.get('top_p', 0.9),
                max_tokens=openai_cfg.get('max_tokens', 4096),
                timeout=openai_cfg.get('timeout', 120),
            )
            backend.tokenizer = self._create_simple_tokenizer()
            return backend

        elif self.config.backend == "openmythos":
            # OpenMythos local backend
            openmythos_cfg = self.config.openmythos
            backend = OpenMythosBackend(
                model_path=openmythos_cfg.get('model_path'),
                device=openmythos_cfg.get('device', 'cpu'),
                dim=openmythos_cfg.get('dim', 256),
                n_heads=openmythos_cfg.get('n_heads', 8),
                max_loops=openmythos_cfg.get('max_loops', 8),
                attention_type=openmythos_cfg.get('attention_type', 'gqa'),
                n_experts=openmythos_cfg.get('n_experts', 8),
                n_shared_experts=openmythos_cfg.get('n_shared_experts', 1),
                n_experts_per_tok=openmythos_cfg.get('n_experts_per_tok', 2),
                expert_dim=openmythos_cfg.get('expert_dim', 64),
            )
            return backend

        elif self.config.backend == "openrouter":
            # OpenRouter (OpenAI-compatible gateway to many models)
            or_cfg = self.config.openrouter
            api_key = or_cfg.get('api_key') or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenRouter backend selected but no API key provided. "
                    "Set agent.openrouter.api_key in config.yaml or OPENROUTER_API_KEY env var."
                )
            backend = OpenAIBackend(
                api_key=api_key,
                base_url=or_cfg.get('base_url', 'https://openrouter.ai/api/v1'),
                model=or_cfg.get('model', 'anthropic/claude-sonnet-4'),
                temperature=or_cfg.get('temperature', self.config.temperature),
                top_p=or_cfg.get('top_p', 0.9),
                max_tokens=or_cfg.get('max_tokens', 4096),
                timeout=or_cfg.get('timeout', 120),
            )
            backend.tokenizer = self._create_simple_tokenizer()
            return backend

        elif self.config.backend == "anthropic":
            # Anthropic Messages API (direct)
            ant_cfg = self.config.anthropic
            api_key = ant_cfg.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "Anthropic backend selected but no API key provided. "
                    "Set agent.anthropic.api_key in config.yaml or ANTHROPIC_API_KEY env var."
                )
            try:
                from .model_backends import AnthropicBackend
            except ImportError:
                raise ImportError(
                    "anthropic backend requires 'anthropic' package. "
                    "Install with: pip install anthropic"
                )
            backend = AnthropicBackend(
                api_key=api_key,
                base_url=ant_cfg.get('base_url', 'https://api.anthropic.com/v1'),
                model=ant_cfg.get('model', 'claude-sonnet-4-20250514'),
                temperature=ant_cfg.get('temperature', self.config.temperature),
                max_tokens=ant_cfg.get('max_tokens', 4096),
                timeout=ant_cfg.get('timeout', 120),
            )
            return backend

    def _create_simple_tokenizer(self):
        """Create a fallback tokenizer for testing only."""
        print(_style("  fallback tokenizer (not for production OpenMythos models)", "2"))

        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 32000

            def encode(self, text):
                return [ord(c) % 32000 for c in text]

            def decode(self, tokens):
                result = []
                for t in tokens:
                    t = int(t) % 256
                    if 32 <= t < 127:
                        result.append(chr(t))
                    else:
                        result.append('?')
                return ''.join(result)

            def __call__(self, text, return_tensors=None):
                import torch
                ids = torch.tensor([self.encode(text)])
                if return_tensors == "pt":
                    return {"input_ids": ids}
                return ids

        return SimpleTokenizer()

    def _setup_tools(self):
        """配置工具"""
        enabled_tools = self.config.tools.enabled_tools

        # 根据配置调整工具参数
        if 'web_search' in enabled_tools:
            provider = self.config.tools.web_search.get('provider', 'duckduckgo')
            from .tools import WebSearchTool
            TOOLS_REGISTRY._tools['web_search'] = WebSearchTool(
                provider=provider,
                api_key=os.getenv('SERPAPI_KEY'),
                config=dict(self.config.tools.web_search),
            )

        if 'python_exec' in enabled_tools:
            timeout = self.config.tools.code_exec.get('timeout', 10)
            from .tools import PythonExecTool
            TOOLS_REGISTRY._tools['python_exec'] = PythonExecTool(timeout=timeout)

        if 'file_ops' in enabled_tools:
            allowed_dirs = self.config.tools.file_ops.get('allowed_dirs', ['./data', './workspace'])
            max_file_size = self.config.tools.file_ops.get('max_file_size', 1048576)
            unrestricted = self.config.tools.file_ops.get('unrestricted', False)
            from .tools import FileOpsTool
            TOOLS_REGISTRY._tools['file_ops'] = FileOpsTool(
                allowed_dirs=allowed_dirs,
                max_file_size=max_file_size,
                unrestricted=unrestricted
            )

        if 'api_call' in enabled_tools:
            allowed_hosts = self.config.tools.api_call.get('allowed_hosts', [])
            timeout = self.config.tools.api_call.get('timeout', 30)
            from .tools import ApiCallTool
            TOOLS_REGISTRY._tools['api_call'] = ApiCallTool(allowed_hosts=allowed_hosts, timeout=timeout)

        if 'bash_exec' in enabled_tools:
            default_timeout = self.config.tools.bash_exec.get('timeout', 120)
            max_timeout = self.config.tools.bash_exec.get('max_timeout', 600)
            default_cwd = self.config.tools.bash_exec.get('default_cwd', str(Path(__file__).parent.parent))
            from .tools import BashExecTool
            TOOLS_REGISTRY._tools['bash_exec'] = BashExecTool(
                default_timeout=min(default_timeout, max_timeout),
                max_timeout=max_timeout,
                default_cwd=default_cwd,
            )

        if 'search_files' in enabled_tools:
            from .tools import GrepTool
            TOOLS_REGISTRY._tools['search_files'] = GrepTool()

        if 'glob' in enabled_tools:
            from .tools import GlobTool
            TOOLS_REGISTRY._tools['glob'] = GlobTool()

        # MCP servers: discover and register external tools
        self.mcp_manager = None
        self.mcp_orchestrator = None
        mcp_config = self.config.mcp
        if mcp_config.get('enabled', False):
            try:
                from .tools.mcp_client import register_mcp_tools, McpOrchestrator
                from .tools.mcp_setup import get_global_mcp_manager
                manager = get_global_mcp_manager()
                servers = dict(mcp_config.get('servers', {}))
                # When native file_ops is unrestricted, the MCP filesystem server
                # is restricted to its configured root and will conflict with
                # arbitrary-path access. Skip it in that case.
                file_ops_unrestricted = self.config.tools.file_ops.get('unrestricted', False)
                if file_ops_unrestricted and 'filesystem' in servers:
                    print(_style("  mcp filesystem: disabled (native file_ops unrestricted)", "2"))
                    del servers['filesystem']
                # 懒加载: 只注册服务器配置(不启动进程), 工具异步后台预加载, 不阻塞启动
                for name, cfg in servers.items():
                    if not manager.servers.get(name):
                        manager.add_server(name, cfg)
                self.mcp_manager = manager
                self.mcp_orchestrator = McpOrchestrator(manager)
                # 后台异步发现并注册 MCP 工具(启动不等待, 完成后 tools 可用)
                def _lazy_mcp_tools():
                    try:
                        # 后台启动每个服务器(懒加载的核心: 不阻塞主线程), 初始化完成后注册工具
                        for name, conn in list(manager.servers.items()):
                            try:
                                conn._ensure_process()
                            except Exception as e:
                                self.logger.debug(f"lazy mcp {name} start failed: {e}")
                        register_mcp_tools(manager)
                    except Exception as e:
                        self.logger.debug(f"lazy mcp tools failed: {e}")
                import threading
                threading.Thread(target=_lazy_mcp_tools, daemon=True).start()
            except Exception as e:
                print(_style(f"  mcp: failed ({e})", "2"))

        print(_style(f"  tools: {', '.join(enabled_tools)}", "2"))
        # 确保所有内置工具已注册（idempotent）
        try:
            from .tools import register_builtin_tools
            register_builtin_tools()
        except Exception as e:
            print(f"register_builtin_tools: {e}")


    def _setup_logging(self):
        """设置日志：文件完整记录，控制台极简淡化（无颜色、无 ANSI）."""
        from pathlib import Path
        import logging

        log_path = Path(self.config.logging.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("OpenMythosAgent")
        logger.setLevel(self.config.logging.level)
        logger.propagate = False
        # 避免重复添加 handler（例如测试或重初始化时）
        if logger.handlers:
            logger.handlers.clear()

        # 文件日志：保留完整时间、级别、logger 名
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(self.config.logging.level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if self.config.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.logging.level)

            class _PlainFormatter(logging.Formatter):
                _tags = {
                    logging.DEBUG: "debug  ",
                    logging.INFO: "       ",
                    logging.WARNING: "warn   ",
                    logging.ERROR: "error  ",
                }

                def format(self, record: logging.LogRecord) -> str:
                    tag = self._tags.get(record.levelno, record.levelname.lower()[:6].ljust(6))
                    msg = record.getMessage().replace('\n', ' ').replace('\r', '')
                    return f"{tag}{msg}"

            console_handler.setFormatter(_PlainFormatter())
            logger.addHandler(console_handler)

        # 压低第三方库的嘈杂日志
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        self.logger = logger

    def _emit_status(self, stream_callback, message: str):
        """Send a dynamic status update to the UI when a stream callback is available."""
        if stream_callback:
            try:
                stream_callback("status", message)
            except Exception:
                pass

    def _cache_get(self, namespace: str, key: Tuple[Any, ...], factory: Callable[[], Any]) -> Any:
        """Lightweight per-turn cache for deterministic, repeated computations."""
        def _make_hashable(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            if isinstance(value, (list, tuple)):
                return tuple(_make_hashable(v) for v in value)
            if isinstance(value, set):
                return frozenset(_make_hashable(v) for v in value)
            if isinstance(value, dict):
                return tuple(sorted((str(k), _make_hashable(v)) for k, v in value.items()))
            return str(value)

        full_key = (namespace,) + tuple(_make_hashable(k) for k in key)
        if full_key not in self._method_cache:
            self._method_cache[full_key] = factory()
        return self._method_cache[full_key]

    def _log_to_file(self, message: str):
        """Write a log record only to file handlers, bypassing console output."""
        import logging
        record = logging.LogRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            message,
            (),
            None,
        )
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.level <= logging.INFO:
                handler.emit(record)

    def _status(self, stream_callback, message: str):
        """File-only log + dynamic UI status (no console clutter)."""
        self._log_to_file(message)
        self._emit_status(stream_callback, message)

    def _run_deep_research(
        self,
        task: str,
        stream_callback: Optional[callable] = None,
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Run the deep-research report workflow with config.research settings."""
        self._status(stream_callback, "deep research mode")
        # Strip the "/deep" or "/research" prefix for a cleaner topic if present.
        clean_task = task
        lower = task.lstrip().lower()
        for prefix in ("/deep", "/research"):
            if lower.startswith(prefix):
                rest = task.lstrip()[len(prefix):].strip()
                if rest:
                    clean_task = f"深度研究 {rest}"
                break

        # 延续性指代解析: 输入含代词(它/这个/那个)或延续话术时,
        # 必须先从对话历史解析真实主题, 否则会拿"它/报告"这种无意义词去搜索。
        from .research_report import extract_research_topic
        _raw_topic = extract_research_topic(clean_task)
        _continuation_markers = ("我的意思", "的意思", "你也", "你也去", "然后整合", "整合起来",
                                 "接着", "继续刚才", "继续上", "同上面", "和刚才那个",
                                 "同刚才", "和上次", "像刚才", "接着刚才")
        # 代词/指代: "深度分析它" "分析这个" "研究那个" 等
        _pronoun_present = bool(re.search(r"(深度分析|分析|研究|调研)\s*(它|这个|那个|这些|那些|该系统|这个系统|那个系统)", clean_task))
        _needs_history_topic = (
            _pronoun_present
            or _raw_topic in ("它", "报告", "这个", "那个", "系统", "该", "此")
            or len(_raw_topic) <= 1
            or (any(m in clean_task for m in _continuation_markers) and any(m in clean_task for m in _continuation_markers))
        )
        if _needs_history_topic:
            # 从历史提取上一轮真实主题(结合上下文理解指代)
            prev_topic = self._infer_continuation_topic(clean_task)
            if prev_topic:
                self.logger.info(f"deep research referential: '{clean_task[:30]}' → topic '{prev_topic}'")
                clean_task = f"深度研究 {prev_topic}"

        report_result = generate_research_report(
            clean_task,
            backend=self.backend,
            config=self.config,
            stream_callback=stream_callback,
            token_callback=token_callback,
            output_dir=_PROJECT_ROOT / "reports",
        )
        final_answer = report_result.get("final_answer", "")
        self.last_report_path = report_result.get("report_path")

        # 后置核验: 深度研究必须真实产出报告文件(非空)且有来源, 否则视为失败并补救。
        # 防止搜索失败(sources=0)时 agent 谎报"研究完成"。
        _report_path = report_result.get("report_path") or report_result.get("metadata", {}).get("report_path")
        _sources_count = (report_result.get("metadata", {}) or {}).get("sources_count", 0)
        _report_valid = False
        if _report_path:
            try:
                from pathlib import Path as _P
                rp = _P(_report_path)
                _report_valid = rp.exists() and rp.stat().st_size > 200 and _sources_count > 0
            except Exception:
                _report_valid = False
        if not _report_valid:
            # 深度研究未真正完成(无报告/空报告/无来源): 降级为普通深度回答 + 提示重试
            self.logger.warning(f"deep research verification FAILED: path={_report_path} sources={_sources_count}")
            if _report_path:
                try:
                    from pathlib import Path as _P
                    rp = _P(_report_path)
                    if rp.exists() and rp.stat().st_size > 200:
                        # 文件在但无来源: 至少告知用户文件位置
                        final_answer = (final_answer or "") + f"\n\n(注: 报告已保存, 但本次搜索来源较少, 内容可能不够全面。)"
                    else:
                        final_answer = (final_answer or "") + "\n\n⚠ 深度研究的报告未能完整生成(搜索未返回结果)。请稍后重试, 或换个表述再试。"
                except Exception:
                    pass
            else:
                final_answer = (final_answer or "") + "\n\n⚠ 深度研究的报告未能生成(搜索未返回结果)。请稍后重试。"

        # Persist report summary to file memory.
        raw_mm = getattr(self, "_raw_memory_manager", None)
        if raw_mm is not None and hasattr(raw_mm, "remember_file"):
            try:
                topic = clean_task[:80]
                content = f"Report saved to {self.last_report_path}\n\n{final_answer[:1000]}"
                raw_mm.remember_file(topic, content, source="memory", append=True)
            except Exception as e:
                self.logger.debug(f"file memory persist failed: {e}")

        self._append_to_history(task, final_answer)
        return {
            "task": task,
            "thoughts": [],
            "actions": [],
            "observations": [{
                "success": report_result.get("success", False),
                "output": final_answer,
                "error": None,
                "metadata": report_result.get("metadata", {}),
            }],
            "thinking_steps": report_result.get("metadata", {}).get("verification_rounds", self.config.research.verification_rounds) + 1,
            "outer_loops": self.config.research.iterative_rounds,
            "final_reward": 1.0 if report_result.get("success") else 0.0,
            "success": report_result.get("success", False),
            "session_token_usage": self.session_token_usage,
            "final_answer": final_answer,
            "report_path": report_result.get("report_path"),
            "report_html_path": report_result.get("report_html_path"),
            "report_markdown": report_result.get("report_markdown"),
            "metadata": {
                "mode": "deep_research",
                **report_result.get("metadata", {}),
            },
        }

    def _infer_continuation_topic(self, task: str) -> Optional[str]:
        """从最近对话历史推断延续性请求的真实研究主题.

        当用户说"我的意思是你也去搜索整合""继续刚才那个"等延续话术时,
        主题应从上一轮对话提取, 而不是把整句当搜索词。
        """
        try:
            from .research_report import extract_research_topic as _ext
            # 1. 最近一轮用户真实话题(排除纯延续话术 + 含代词/指代的话术)
            #    含代词的消息(它/这个/那个/这些)本身依赖上文, 不能作为主题源。
            _pronoun_re = re.compile(r"(它|这个|那个|这些|那些|该系统|这个系统|那个系统|上面|刚才)")
            for entry in reversed(self.conversation_history):
                ut = str(entry.get("user", "") or "").strip()
                if not ut:
                    continue
                if any(m in ut for m in ("我的意思", "的意思", "你也", "然后整合", "整合起来",
                                         "继续刚才", "接着刚才", "和刚才那个", "像刚才")):
                    continue
                # 跳过含代词的消息(它/这个/那个 指代上文, 不能当主题)
                if _pronoun_re.search(ut) and len(ut) <= 20:
                    continue
                topic = _ext(ut)
                if topic and len(topic) <= 12 and not _pronoun_re.search(topic):
                    return topic
                import re as _re
                cands = [c for c in re.findall(r"[\u4e00-\u9fff]{2,8}(?:AI|ai|Agent|agent)?", topic) if len(c) >= 2]
                if cands:
                    return cands[0]
            # 2. 回退: 从最近 assistant 消息提取真实实体(含具体系统/项目名)
            for entry in reversed(self.conversation_history):
                at = str(entry.get("assistant", "") or "").strip()
                if not at:
                    continue
                # 提取看起来像专有名词的词(大写/带连接符/数字, 或中文长词)
                import re as _re2
                for cand in _re2.findall(r"[A-Z][A-Za-z0-9_-]{2,30}|[\u4e00-\u9fff]{3,12}(?:系统|架构|项目|平台|框架)", at):
                    if len(cand) >= 3 and not _pronoun_re.search(cand):
                        return cand
            # 3. 兜底: 用户当前话术里"我的意思是X"中的 X
            m = re.search(r"(?:我的意思是|的意思就是|就是说)\s*(.+)", task)
            if m:
                return m.group(1).strip()[:40]
            return None
        except Exception as e:
            self.logger.debug(f"infer continuation topic failed: {e}")
            return None

    # ============ 快速路径 ============

    def _is_simple_query(self, task: str) -> bool:
        """智能判断是否为简单对话型查询,无需深度推理/工具编排."""
        return self._cache_get("is_simple", (task,), lambda: self._compute_is_simple_query(task))

    def _compute_is_simple_query(self, task: str) -> bool:
        """简单查询判断的实际计算逻辑.

        优先级(高→低):
        1) 空输入 → fast
        2) 明确的纯问候/礼貌/确认短句 → fast (不包含任何动作关键词)
        3) 包含动作/工具/深度推理关键词 → deep (无论多短)
        4) 超短(<10字)且无动作关键词 → fast
        5) 超长(>40字) → deep
        6) 中等长度且无明确工具意图 → fast

        关键: 步骤3)必须在长度判断之前,否则短查询如"搜索AI"(4字)会被误走 fast 路径,
            导致模型在 _run_simple 中不能正确使用工具.
        """
        task = task.strip()
        task_lower = task.lower()

        # 0) 空输入直接 fast
        if not task:
            return True

        # 0.5) 修改/优化文件意图 → deep (需走主循环执行 read→apply_diff→verify)
        # "给XX加功能""修改XX文件""在XX里加YY" 等必须真正改动文件, 不能 fast 单次回答
        _mod_verbs = ("加", "加上", "加入", "添加", "增加", "修改", "改", "更新", "优化",
                      "完善", "改进", "增强", "重构", "修复", "调整", "删除", "去除",
                      "add", "modify", "update", "improve", "optimize", "refactor", "fix")
        if any(v in task for v in _mod_verbs) and any(
            k in task for k in ("功能", "特性", "文件", "代码", "程序", "脚本", "游戏", "项目", "音效",
                                ".py", ".js", ".ts", ".md", ".txt", "snake", "贪吃", "贪食")
        ):
            return False

        # 2) 纯问候/礼貌/确认 → 先于关键词检查匹配,避免被误判为需要工具
        simple_patterns = [
            r'^(你好|嗨|哈喽|hello|hi|hey|在吗|在嘛|您好|早上好|晚上好|下午好)[!!??\.,\s]*$',
            r'^(谢谢|感谢|不客气|再见|拜拜|goodbye|bye)[!!??\.,\s]*$',
            r'^(好的|行|可以|ok|okay|yes|no|嗯|哦|啊|对|不对|没错)[!!??\.,\s]*$',
            r'^(谢|对)[!!??,,.!\s]*$',
        ]
        for pattern in simple_patterns:
            if re.match(pattern, task_lower, re.IGNORECASE):
                return True

        # 2.5) 明确"看/读 X 文件夹/目录" → fast
        #     (_run_simple 有自动 list+read 文件夹逻辑; 走 fast 可避免主循环全盘 find/只列不读)
        if re.search(r"(看下|看一下|看看|查看|浏览|打开|读一下|读取)\s*[^，。！？!?]{1,40}?(文件夹|目录|folder|dir)", task):
            return True
        # 2.6) "分析 X 文件夹/项目" → fast (同上, 走自动读文件夹逻辑)
        if re.search(r"(分析|剖析|解析)\s*[^，。！？!?]{1,30}", task) and (
            "文件夹" in task or "目录" in task or "项目" in task or "代码库" in task or "仓库" in task
            or "folder" in task.lower() or "dir" in task.lower() or "project" in task.lower()
        ):
            return True
        # 2.7) "X 分析下/分析一下" (名字在前) → fast
        if re.search(r"[^，。！？!?]{1,30}?(分析下|分析一下|分析|剖析|解析)$", task):
            return True

        # 3) 动作/工具/深度推理关键词 → deep (无论长度多少!)
        #    这是最高优先级的"必须使用工具"判断,必须排在长度判断之前.
        #    否则短查询如"搜索AI"(4字)/"天气"(2字)/"写代码"(3字) 会被误走 fast 路径.
        deep_keywords = [
            # 中文核心动作词
            '搜索', '查找', '调研', '研究', '考察', '融资', '股价', '行情', '资料',
            '文件', '代码', '程序', '分析', '比较', '设计', '实现',
            '部署', '调试', '测试', '优化', '解释', '总结', '翻译', '天气', '新闻',
            '写', '读', '创建', '构建', '获取', '调用', '运行', '执行', '列', '览', '示',
            '查', '改', '编', '发', '搜', '算',
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
        if any(kw in task_lower for kw in deep_keywords):
            return False

        # 4) 超短任务(<=10字符)且无深度关键词 → fast
        if len(task) <= 10:
            return True

        # 5) 11-15字符的查询 - 查看数量类短查询通常需要工具
        if len(task) <= 15:
            if re.search(r'几个\s*[文件文件夹目录个]', task_lower):
                return False
            return True

        # 6) 较长任务(>40字符) → deep (保守起见避免漏走工具)
        if len(task) > 40:
            return False

        # 中等长度(16-40字符)且无明确动作关键词,视为简单问题 → fast
        return True

    def _classify_intent(self, task: str) -> Optional[Tuple[str, Dict[str, Any], float, str]]:
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

        # -1) 看/读文件夹意图: "看下 X 文件夹/目录" → 定位并在当前目录下 list
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
            if fname:
                return ("file_ops", {"action": "list", "path": fname}, 0.9, "detected folder-read intent")

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

        # 5.5) LLM 前置意图分析兜底: 规则未命中且有明确动作/需求时,
        # 用一次简短 LLM 调用判断真实意图(应对规则盲区, 如"给我找点AI资料").
        # 仅当任务长度适中且有内容(避免问候/闲聊也触发, 慢且费 token)。
        if len(t) >= 6 and not self._is_simple_query(t):
            llm_intent = self._llm_intent_classify(t)
            if llm_intent:
                return llm_intent

        return None

    def _llm_intent_classify(self, task: str) -> Optional[Tuple[str, Dict[str, Any], float, str]]:
        """LLM 软匹配意图分类(规则未命中/置信度不足时的升级层).

        特点:
        - 精准理解用户意图: 从候选工具选最匹配的并给出参数
        - 支持拆分复合意图: 如"查资料并保存到文件" → 主意图 web_search,
          附带 write 子意图(存到 returned secondary, 供调用方二次执行)
        - 结构化 JSON 校验 + 工具名归一化, 减少幻觉
        返回 (tool_name, arguments, confidence, reason) 或 None。
        """
        from .tools import TOOLS_REGISTRY
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
            raw = self.backend.generate(prompt, n_loops=1, temperature=0.0, max_tokens=250)
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
            self.logger.debug(f"llm intent classify failed: {e}")
            return None

    @staticmethod
    def _is_pure_nudge(task: str) -> bool:
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

    def _last_user_task(self) -> Optional[str]:
        """返回历史中最近一条用户消息(排除纯催促/命令词)."""
        for entry in reversed(self.conversation_history):
            msg = (entry.get("user") or "").strip()
            if not msg:
                continue
            if self._is_pure_nudge(msg):
                continue
            low = msg.lower()
            if low in {"quit", "exit", "q"} or low.startswith(("/", "!")):
                continue
            return msg
        return None

    def _is_continuation_query(self, task: str) -> bool:
        """判断是否为"继续上次讨论/接着刚才话题"这类延续性对话请求.

        延续性请求应优先基于历史对话/记忆直接回答, 而不是重新列出目录或
        读文件(除非用户明确点名某个具体文件)。
        """
        task_lower = (task or "").strip().lower()
        if not task_lower:
            return False
        # 明确的延续动作词
        continuation_markers = [
            '继续', '接着', '再聊', '再继续', '继续聊', '继续讲', '继续讨论',
            '继续刚才', '继续我们', '继续上次', '接着刚才', '接着上次', '接着我们',
            '接下来呢', '然后呢', '继续之前',
            'continue', 'go on', 'as we were', 'continue the discussion',
        ]
        if any(m in task_lower for m in continuation_markers):
            # 关键: 若"继续"之后紧跟具体动作(下载/克隆/安装/运行/构建/写/保存/执行/搜索),
            # 这是"继续执行某个任务", 不是"延续聊天" → 必须走工具执行
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
        # 指代历史讨论: 昨天/上次/之前/刚才 + 讨论/话题/对话/健康
        if re.search(r'(昨天|上次|之前|刚才).{0,8}(讨论|话题|对话|聊|health|健康)', task_lower):
            return True
        # 短句中以中文指代词开头（这个/那个/该/此/这类/这种/这样）且没有明确引入新实体，
        # 大概率在追问前文话题，应优先基于历史回答而非重新搜索/列目录。
        if len(task_lower) <= 18 and re.match(
            r'^(这个|那个|该|此|这类|这种|这样|上面|刚才|之前).{2,}', task_lower
        ):
            return True
        return False

    def _needs_tool_summary(self, raw_output: str, is_continuation: bool) -> bool:
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
            listing_like = sum(
                1 for ln in lines
                if re.match(r'^(d\s|[-rwxdr]\S*\s+|\S+\s+\d+\s*$)', ln)
            )
            if listing_like >= max(1, len(lines) // 2):
                return True
        if len(raw_output) > 600:
            return True
        return False

    @staticmethod
    def _tool_returns_listing(action: Any) -> bool:
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

    @staticmethod
    def _is_ultra_short_ambiguous(task: str) -> bool:
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

    @staticmethod
    def _skip_tool_retry(action: Any, tool_result: Any) -> bool:
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

    def _tool_failure_fallback(self, task: str, tool_result: Any) -> str:
        """工具调用失败时的自然语言兜底回复, 避免把原始错误直接甩给用户.

        对 File not found: 自动列出父目录, 提供真实可用的文件名,
        而不是直接放弃(模型常把目录+模糊词当文件名, 如 'lv/全部')。
        """
        err = str(getattr(tool_result, "error", "") or "未知错误").strip()[:120]
        err_lower = err.lower()
        if any(k in err_lower for k in ("no search results", "no results", "无结果", "没有找到", "nothing found")):
            return "抱歉，这次没有搜到相关结果。你可以换个关键词，或告诉我具体想了解什么，我再帮你查。"
        if any(k in err_lower for k in ("timed out", "timeout")):
            return "抱歉，这次查询超时了。你可以稍后再试，或换一个更具体的说法。"
        # File not found: 尝试列出父目录, 让用户/模型看到真实文件名
        if "not found" in err_lower or "file does not exist" in err_lower or "不存在" in err:
            try:
                m = re.search(r"not found:\s*(.+)", err) or re.search(r"不存在:\s*(.+)", err)
                if m:
                    path = m.group(1).strip().strip("'\"")
                    from pathlib import Path as _P
                    p = _P(path)
                    cand = p.parent if p.parent.exists() and p.parent.is_dir() else (p if p.is_dir() else None)
                    if cand is not None:
                        try:
                            entries = sorted(
                                [e.name for e in cand.iterdir()],
                                key=lambda x: (not _P(cand / x).is_dir(), x.lower()),
                            )[:15]
                            if entries:
                                listing = "，".join(entries)
                                return (
                                    f"抱歉，没找到文件：{path}。"
                                    f"该目录下现有的内容是：{listing}。"
                                    f"你可以告诉我具体要操作哪个文件，或直接说'保存全部到 lv'让我新建。"
                                )
                        except Exception:
                            pass
            except Exception:
                pass
        return f"抱歉，刚才的操作没能完成（{err}）。你可以换个说法再试一次，或告诉我具体想做什么。"


    def _summarize_tool_answer(self, task: str, action: Any, raw_output: str,
                               token_callback: Optional[Callable[[int], None]] = None) -> str:
        """基于工具观察生成面向用户问题的自然语言回答, 避免只甩出原始列表."""
        prompt = (
            "You are a helpful assistant. A tool was called to answer the user's question.\n\n"
            f"User question: {task}\n\n"
            f"Tool: {action.tool_name}({action.arguments})\n\n"
            f"Tool result:\n{raw_output[:3000]}\n\n"
            "Give a concise, natural-language answer to the user's question based ONLY on the "
            "tool result. Do not dump the raw result verbatim; summarize the key points the user "
            "cares about. If the result is a directory/file listing, tell the user what's relevant "
            "in it and answer the question directly. Final Answer:"
        )
        try:
            answer = self.backend.generate(
                prompt, n_loops=1, temperature=0.4, max_tokens=1024,
                token_callback=token_callback,
            ).strip()
        except Exception as e:
            self.logger.warning(f"tool summary failed {e}")
            return raw_output
        cleaned = self._clean_fast_answer(answer)
        return cleaned or raw_output

    def _resolve_filename_in_task(self, task: str) -> Optional[str]:
        """从简单文件查询中提取文件名并解析为存在的绝对路径."""
        return self._cache_get("resolve_file", (task,), lambda: self._compute_resolve_filename_in_task(task))

    def _compute_resolve_filename_in_task(self, task: str) -> Optional[str]:
        """文件名解析的实际计算逻辑."""
        # Match common filename patterns: name.ext, name(1).ext, name_v2.ext, etc.
        candidates = re.findall(r"[\w\-\(\)\[\]\.]+\.[a-zA-Z0-9]{1,10}", task)
        candidates = [c for c in candidates if "." in c and not c.endswith((".",))]
        # Strip common read/view verb prefixes that \w captured (e.g. '看下xxx.md').
        read_prefixes = ['读取', '读', '查看', '看', '打开', '显示', '列出',
                         'read', 'show', 'view', 'open', 'list']
        cleaned = []
        for c in candidates:
            stripped = c
            for prefix in read_prefixes:
                if stripped.lower().startswith(prefix.lower()):
                    stripped = stripped[len(prefix):]
                    break
            stripped = stripped.strip()
            if stripped and "." in stripped:
                cleaned.append(stripped)
        candidates = cleaned
        if not candidates:
            return None

        search_roots = []
        # 1. Current working directory
        search_roots.append(Path.cwd())
        # 2. Desktop
        search_roots.append(Path.home() / "Desktop")
        # 3. Home
        search_roots.append(Path.home())
        # 4. Recent conversation paths (from working memory / history)
        for hist in getattr(self, "conversation_history", [])[-5:]:
            for text in (hist.get("user", ""), hist.get("assistant", "")):
                for cand in re.findall(r"[\w\-\(\)\[\]\.]+\.[a-zA-Z0-9]{1,10}", text):
                    if cand not in candidates:
                        candidates.append(cand)

        for cand in candidates:
            # Direct path
            if cand.startswith("/") or cand.startswith("~"):
                p = Path(cand).expanduser().resolve()
                if p.is_file():
                    return str(p)
            # Search in common roots
            for root in search_roots:
                p = (root / cand).resolve()
                if p.is_file():
                    return str(p)
                # Also try case-insensitive match
                try:
                    for fp in root.iterdir():
                        if fp.is_file() and fp.name.lower() == cand.lower():
                            return str(fp.resolve())
                except (PermissionError, OSError):
                    continue
        return None

    def _resolve_open_path(self, task: str) -> Optional[str]:
        """Resolve a file/folder path when the user asks to open it."""
        task_lower = task.lower().strip()
        open_intent = any(kw in task_lower for kw in ['打开', 'open', '开启', 'launch'])
        if not open_intent:
            return None

        # 1) Try to extract an explicit filename/path from the task.
        explicit_candidates = self._extract_filename_candidates(task)
        if explicit_candidates:
            # If the user named a specific file, only open it if it exists.
            for cand in explicit_candidates:
                p = Path(cand).expanduser().resolve()
                if p.exists():
                    return str(p)
            return None

        # 2) If the task mentions "report" / "文件" / "报告" without a path,
        #    or is a bare open command, fall back to the most recently generated report path.
        bare_open = task_lower in {'打开', 'open', '你打开', '请打开', '帮我打开', '给我打开'}
        if getattr(self, 'last_report_path', None) and (
            bare_open
            or any(kw in task_lower for kw in ['报告', 'report', '文件', 'file', '它', '这个'])
        ):
            p = Path(self.last_report_path).expanduser().resolve()
            if p.exists():
                return str(p)

        # 3) Search history for the most recent absolute path that still exists.
        for hist in reversed(getattr(self, 'conversation_history', [])):
            for text in (hist.get('user', ''), hist.get('assistant', '')):
                for cand in re.findall(r'[\w\-\(\)\[\]\./~]+\.[a-zA-Z0-9]{1,10}', text):
                    p = Path(cand).expanduser().resolve()
                    if p.exists():
                        return str(p)
        return None

    def _extract_filename_candidates(self, task: str) -> List[str]:
        """Return filename/path candidates mentioned in the task, without requiring existence."""
        candidates = re.findall(r"[\w\-\(\)\[\]\./~]+\.[a-zA-Z0-9]{1,10}", task)
        read_prefixes = ['读取', '读', '查看', '看', '打开', '显示', '列出',
                         'read', 'show', 'view', 'open', 'list']
        cleaned = []
        for c in candidates:
            stripped = c
            for prefix in read_prefixes:
                if stripped.lower().startswith(prefix.lower()):
                    stripped = stripped[len(prefix):]
                    break
            stripped = stripped.strip()
            if stripped and "." in stripped:
                cleaned.append(stripped)
        return cleaned

    def _read_folder_markdown(self, folder: str) -> str:
        """读取文件夹下的 markdown 文件内容, 供总结时参考.

        设计文档: 看文件夹必须 list 后 read 内容, 基于内容回答而非只列目录.
        读取前 3 个 .md 文件的内容(截断到合理长度)。
        """
        if not folder:
            return ""
        try:
            from pathlib import Path
            p = Path(folder).expanduser()
            if not p.is_dir():
                return ""
            md_files = sorted(p.glob("*.md"))[:3]
            if not md_files:
                return ""
            parts = []
            for f in md_files:
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")[:1200]
                    parts.append(f"--- {f.name} ---\n{content}")
                except Exception:
                    parts.append(f"--- {f.name} [读取失败] ---")
            return "\n\n".join(parts)
        except Exception:
            return ""

    @staticmethod
    def _is_folder_read_intent(task: str) -> bool:
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

    def _recent_folder_from_history(self, task: str) -> Optional[str]:
        """从最近对话历史中找出最近提到的、仍存在的目录路径.

        用于"阅读"这类指代性请求: 用户可能刚说"看下桌面的xx文件夹",
        然后只回"阅读", 需要自动补全到那个目录。
        """
        # 优先: 历史中最新的绝对路径(目录)
        for hist in reversed(getattr(self, 'conversation_history', [])):
            for text in (hist.get('user', ''), hist.get('assistant', '')):
                # 匹配绝对路径或含 / 的相对路径片段
                for m in re.finditer(r'(/(?:Users|home|Volumes|Applications|Desktop|Downloads|Documents)/[^\s，。,!?）)]+)', text or ""):
                    cand = m.group(1).strip()
                    if len(cand) < 2:
                        continue
                    p = Path(cand).expanduser()
                    if p.is_dir():
                        return str(p.resolve())
        # 次优: 历史中提到的"桌面/下载/文档"等别名目录名
        for hist in reversed(getattr(self, 'conversation_history', [])):
            for text in (hist.get('user', ''), hist.get('assistant', '')):
                m = re.search(r'(桌面|下载|文档|Downloads|Desktop|Documents)\S{0,40}', text or "")
                if m:
                    name = m.group(0).strip(" ，,。.!?）)]}")
                    for root in [Path.home() / "Desktop", Path.home() / "Downloads",
                                 Path.home() / "Documents"]:
                        p = root / name.replace("桌面", "").replace("下载", "").replace("文档", "").strip("/")
                        if p.exists():
                            return str(p.resolve())
        # 第三: 从当前任务本身提取"看下 X 文件夹"中的文件夹名, 在当前工作目录下定位
        m = re.search(r'(?:看下|看一下|看看|查看|浏览|打开|读一下|读取|里面)\s*[^，。！？!?]{1,40}?(?:文件夹|目录|folder|dir)', task or "")
        if m:
            raw = m.group(0)
            for v in ("看下", "看一下", "看看", "查看", "浏览", "打开", "读一下", "读取", "里面", "这个", "那个", "的"):
                raw = raw.replace(v, "")
            raw = raw.replace("文件夹", "").replace("目录", "").replace("folder", "").replace("dir", "").strip()
            if raw:
                for root in [Path.cwd(), Path.home(), Path.home() / "Desktop"]:
                    if not root.is_dir():
                        continue
                    p = root / raw
                    if p.is_dir():
                        return str(p.resolve())
                    # 大小写不敏感匹配(用户说"health os", 目录叫 HealthOS)
                    try:
                        raw_norm = re.sub(r'[\s\-_]+', '', raw.lower())
                        for child in root.iterdir():
                            if child.is_dir():
                                child_norm = re.sub(r'[\s\-_]+', '', child.name.lower())
                                if child_norm == raw_norm or (raw_norm and raw_norm in child_norm):
                                    return str(child.resolve())
                    except Exception:
                        pass
        return None

    def _get_skill_context(self, task: str) -> str:
        """Return the rendered prompt of the active skill, if any."""
        engine = getattr(self, "skill_engine", None)
        if engine is None:
            return ""
        return engine.render_active(task)

    def _get_skill_tool_hint(self) -> List[str]:
        """Return preferred tools declared by the active skill."""
        engine = getattr(self, "skill_engine", None)
        if engine is None:
            return []
        return engine.get_active_tools_hint()

    def _should_inject_memory(self, task: str) -> bool:
        """判断是否需要注入长时记忆/语义记忆上下文.

        Turbo 模式: 首次对话或简单问答跳过记忆注入,减少首次响应延迟.
        """
        # 首次对话: 不注入历史记忆
        if not getattr(self, 'conversation_history', None):
            return False
        # 简单问答: 不需要外部记忆
        if self._is_simple_query(task):
            return False
        return True

    def _should_plan(self, task: str) -> bool:
        """判断任务是否值得做分解规划.简单/单步任务跳过规划,避免 JSON 解析开销."""
        return self._cache_get("should_plan", (task,), lambda: self._compute_should_plan(task))

    def _compute_should_plan(self, task: str) -> bool:
        """规划判断的实际计算逻辑."""
        task = task.strip()
        task_lower = task.lower()

        # 单步文件/搜索/计算类任务不需要规划
        single_step_patterns = [
            r'^(读取?|列出?|查看|显示|打开|读|list|read|show|display)\s+',
            r'(当前文件夹|当前目录|这个文件夹|这个目录|当前文件)',
            r'^\s*(计算|算一下|求|calculate|compute)\s+',
            r'^\s*(搜索|查找|search|find)\s+',
        ]
        for pattern in single_step_patterns:
            if re.search(pattern, task_lower, re.IGNORECASE):
                return False

        # 包含多步骤/复杂关键词 → 需要规划
        planning_keywords = [
            '计划', '方案', '步骤', '流程', '策略', '分解', '实现', '开发', '项目',
            'plan', 'strategy', 'steps', 'workflow', 'implement', 'develop', 'project',
            '设计', '架构', '搭建', '部署', '优化', '重构',
            'design', 'architecture', 'build', 'deploy', 'optimize', 'refactor',
        ]
        if any(kw in task_lower for kw in planning_keywords):
            return True

        # 超短任务(<=12 字符)不需要规划
        if len(task) <= 12:
            return False

        # 中等长度任务默认不规划,交给 ReAct/CoT 处理
        return False

    def _is_coding_task(self, task: str) -> bool:
        """判断是否为软件工程/编码任务,适合启用 code_mode 进行高速精确编辑."""
        return self._cache_get("is_coding", (task, self._code_mode_override), lambda: self._compute_is_coding_task(task))

    def _compute_is_coding_task(self, task: str) -> bool:
        """编码任务判断的实际计算逻辑."""
        if getattr(self, '_code_mode_override', False):
            return True
        task_lower = task.lower()
        coding_keywords = [
            '代码', 'coding', 'program', 'programming', 'write code', 'implement',
            '开发', '函数', 'function', 'class', 'module', 'refactor', '重构',
            '修复', 'fix', 'bug', 'debug', '调试', '添加', 'add', '修改', 'change',
            '优化', 'optimize', '测试', 'test', '单元测试', 'unittest', 'feature',
            'file_ops', 'apply_diff', 'project', '项目', '仓库', 'repo', 'git',
        ]
        return any(kw in task_lower for kw in coding_keywords)

    def _is_file_analysis_task(self, task: str) -> bool:
        """判断是否为需要结构化步骤的文件/项目分析任务(但不需要 LLM 分解规划).

        优化后:更积极地识别文件/目录查询,支持中文和英文双语.
        """
        return self._cache_get("is_file_analysis", (task,), lambda: self._compute_is_file_analysis_task(task))

    def _compute_is_file_analysis_task(self, task: str) -> bool:
        """文件分析任务判断的实际计算逻辑."""
        task_lower = task.lower()

        # 0. 排除关于文件读取/分析机制的元问题(不是真的要让 Agent 读文件)
        meta_patterns = [
            r"你.*文件.*机制",
            r"文件.*机制",
            r"文件.*机制.*厉害",
            r"文件.*机制.*怎么",
            r"文件.*机制.*如何",
            r"怎么.*读取文件",
            r"如何.*读取文件",
            r"读取文件.*厉害",
            r"读取文件.*怎么",
            r"读取文件.*如何",
            r"机制.*怎么样",
            r"机制.*好不好",
            r"机制.*厉害",
            r"原理.*怎么",
            r"原理.*如何",
            r"文件.*原理",
            r"文件.*怎么.*工作",
            r"文件.*如何.*工作",
        ]
        if any(re.search(p, task_lower) for p in meta_patterns):
            return False

        # 核心关键词(中文+英文)
        analysis_keywords = [
            '分析', '总结', 'summarize', 'analyze', '阅读', 'read', 'check', '检查',
            '项目', 'project', '文件夹', 'folder', '目录', 'directory',
            '文件', 'file', '代码', 'code', '源码', 'source', '構造', '结构', 'overview',
            '查看', 'show', 'list', '列', '览', '介绍', 'intro', 'explore',
        ]

        # 深度操作关键词
        deep_keywords = [
            '分析', '分析文件', '分析项目', '总结', '概览', 'overview', '结构',
            '检查', '检查文件', '核对', 'verify', 'compare', '对比', '比较',
            '读取', '读取文件', '查看文件', '浏览', 'browse', 'inspect',
            'list', '列', '览', '显示', 'show', '列出',
        ]

        # 路径/文件相关关键词
        path_keywords = [
            '路径', 'path', '目录', 'directory', 'folder', '文件夹', '项目',
            '文件', 'code', 'source', 'src', 'lib', 'package',
            './', '../', '/', '.py', '.js', '.ts', '.json', '.md', '.txt',
        ]

        # 检查是否包含文件/目录/项目关键词
        has_file_keywords = any(kw in task_lower for kw in ['文件', 'folder', 'directory', 'project', '代码', 'code'])

        # 检查路径模式(更宽松)
        path_patterns = [
            r'[/\\](?:[\w\s/.\-_~]+)',  # Unix/Windows 路径
            r'~\/',  # 用户主目录
            r'[\w\-]+\.(py|js|ts|json|md|txt|css|html|go|rs|java|c|cpp|h)',  # 文件扩展名
            r'(?:[a-zA-Z0-9_\-.]+/)+[a-zA-Z0-9_\-.]+',  # 相对路径风格
        ]
        has_path_pattern = any(re.search(p, task) for p in path_patterns)

        # 检查是否包含深度关键词
        has_deep_kw = any(kw in task_lower for kw in deep_keywords)

        # 检查是否包含分析关键词
        has_analysis_kw = any(kw in task_lower for kw in analysis_keywords)

        # 1. 路径 + 深度关键词 → 文件分析
        if has_path_pattern and has_deep_kw:
            return True

        # 2. 文件/目录关键词 + 分析关键词 → 文件分析
        if has_file_keywords and has_analysis_kw:
            return True

        # 3. 项目/文件夹 + 查看/分析 关键词 → 文件分析
        if any(kw in task_lower for kw in ['项目', 'folder', 'directory']):
            if any(kw in task_lower for kw in ['分析', '查看', '浏览', 'list', '列', '览']):
                return True

        # 4. 简单文件/目录查询(小于 30 字符)→ 文件分析
        if len(task) <= 30 and has_file_keywords:
            return True

        # 5. 检查是否包含路径 + 项目/文件关键词
        if has_path_pattern and ('project' in task_lower or 'file' in task_lower or 'folder' in task_lower):
            return True

        return False

    def _run_simple(self, task: str, memory_context: str = "",
                    history_context: str = "",
                    stream_callback: Optional[callable] = None,
                    token_callback: Optional[Callable[[int], None]] = None,
                    is_continuation: Optional[bool] = None) -> Dict[str, Any]:
        """对简单查询执行单次 LLM 调用快速返回,仍保留简短思考过程"""
        started_at = datetime.now()

        if is_continuation is None:
            is_continuation = self._is_continuation_query(task)

        # 纯催促型延续(如"继续啊/接着做/go on"): 用户是要"继续执行上一个任务",
        # 不是"延续聊天"。此时把最近一次用户任务重新注入, 让 LLM 真正行动而非空回复。
        if is_continuation and self._is_pure_nudge(task):
            last_task = self._last_user_task()
            if last_task:
                is_continuation = False  # 走工具执行路径
                task = f"继续执行上一条任务: {last_task}\n(若已完成则直接确认完成结果; 若未完成则现在真正执行并给出结果)"
                self.logger.info(f"continuation nudge -> re-execute last task: {last_task[:60]}")

        system_parts = [
            "You are Lv Super Agent, a helpful assistant with memory of past conversations.",
            "Use the Recent Conversation and Relevant Memory below to answer.",
            "If the user refers to '他们/她们/它们/他/她/它/刚才/之前/这个/那个/该/此/这类/这种', resolve the reference from the conversation history. "
            "For example, if the user asks '这个榜单权威吗' after discussing OSWorld, '这个榜单' means OSWorld.",
            "First think briefly inside <think>...</think> tags, then respond naturally and concisely.",
            "",
            f"Today is {datetime.now().strftime('%Y-%m-%d')} ({datetime.now().strftime('%A')}). "
            f"Current year: {datetime.now().year}. When searching or answering time-sensitive questions, "
            f"use the current date/year rather than outdated knowledge. Do NOT write older years like 2025 or 2024 "
            f"into search queries unless the user explicitly asks for that year.",
            "",
            "When a request requires external knowledge (search/weather/news/stock), web pages, files, folders, code, or program execution — use a tool call.",
            "Tool calls use the exact format (each on its own line, no surrounding fences):",
            "",
            "  [TOOL:tool_name]",
            "  {JSON arguments}",
            "",
            "Available tools:",
            "- web_search(query, max_results=5) → 联网搜索最新信息(天气/新闻/股价/资料/产品/公司)",
            "- read_web(url) → 打开网页提取正文",
            "- file_ops(action, path, ...) → action ∈ [read, write, list, exists, grep, analyze]",
            "- glob(pattern, path='.', max_results=100) → 按文件名模式定向查找文件(如 **/*.py)",
            "- search_files(pattern, path, ...) → 在文件中搜索文本内容(grep)",
            "- python_exec(code, lang, timeout=30) → 执行 python/bash/python_file 代码",
            "- bash_exec(command, timeout=120) → 执行任意 shell 命令(含 git clone/install/运行)",
            "- git(command, repository) → git 操作: clone/init/status/add/commit/push/pull 等",
            "",
            f"当前工作目录: {os.getcwd()} (文件操作用相对该目录的路径, 不要凭空编造绝对路径)",
            "",
            "Task routing (根据用户问题自行判断优先工具组合):",
            "- 写代码/调试/重构 → python_exec, bash_exec, file_ops; 思考中列算法步骤、边界情况。",
            "- 查资料/新闻/股价 → web_search; 优先官方来源(官网 > 新闻 > 社交媒体)。",
            "- 总结文档/写报告 → file_ops/read_web 拉取原文, 按 大纲→关键点→润色 三步组织。",
            "- 比较/评估/方案 → 收集多方信息后列出对比维度再给结论, 不要只给单一来源。",
            "",
            "Tool-use loop:",
            "- 工具调用后在本轮思考中核对: 预期结果 vs 实际结果; 若不符则换关键词或换工具, 不要重复相同调用。",
            "- 工具返回错误/无结果时, 用自然语言说明, 并基于已有信息给出合理回答; 不要把原始错误甩给用户。",
            "",
            "Self-assessment:",
            "- 在 <think> 思考的**末尾**写一句置信度自我评估, 格式: 置信度: 0.XX (0~1 两位小数)。",
            "- 若置信度 < 0.6, 在思考中说明主要原因(信息不足/工具失败/冲突证据)。",
            "- 置信度只写在 <think> 内部, **绝对不要**出现在给用户的最终正文里。",
            "",
            "Rules:",
            "- 1 tool call per turn; after tool result, answer based on it in the SAME turn (do not ask the user to re-run).",
            "- Prefer your own knowledge for obvious facts. For anything time-sensitive or uncertain, use web_search.",
            "- For file/request tasks, output the tool call FIRST before any explanation.",
            "- DOWNLOADING CODE / CLONING REPOS: 用户说'下载源码/克隆仓库/git下载'时, 直接用 git(command='clone', repository='https://github.com/OWNER/REPO.git') 或 bash_exec('git clone ... 目标目录'); 不要用 api_call 只查信息。先确认目标目录存在, 再真正执行 clone, 完成后汇报实际结果。",
            "- READING FOLDERS: 当用户说'阅读/读/查看 某个文件夹'时, 必须 list 后再 read 其中的文件内容, 然后基于内容回答。绝不能只 list 目录就回复。",
            "- READING A FILE: 用户说'看/读/打开/查看 X文件/我的XX'时, 先确定文件路径(用 glob 或 file_ops 定向查找, 不要全盘 find), 然后 read 文件内容并基于内容回答。绝不能只列出路径不读内容。",
            "- FINDING FILES: 用 glob(pattern, path) 定向查找, 不要用 bash find 全盘扫描(会刷屏权限错误)。只在用户明确要求全盘搜索时才用 bash find, 并加 2>/dev/null 屏蔽权限报错。",
            "- CREATING FILES/ARTICLES: 用户说'新建/创建/写一篇/保存一篇文章/把XX放进去'时, 必须直接调用 file_ops(action='write', path='<当前目录/文件名.md>', content='<完整内容>') 真正创建文件。写完用 file_ops(action='read', path=...) 验证内容已写入。绝不能只 list 目录或只说'好的'而不实际 write。",
            "- MODIFYING FILES: 用户说'给XX加功能/改一下/修改/更新/优化XX'时, 必须先 read 目标文件, 然后直接调用 file_ops(action='apply_diff', path=..., diff='<<<<<<< SEARCH\n原文\n=======\n新文\n>>>>>>> REPLACE') 或 file_ops(action='write', path=..., content='完整新内容') 真正修改文件。修改后 read 验证。绝不能只描述'应该怎么改'而不实际执行修改。",
            "- Final answer must be in concise, natural Chinese (do not repeat tool result verbatim unless asked).",
            "- 最终完成时, 用大白话/通俗易懂的话总结结论或结果(像跟朋友解释一样, 不用术语堆砌, 让非技术的人也能听懂你做了什么、得到什么)。",
        ]
        if is_continuation:
            system_parts.append(
                "The user is CONTINUING a previous discussion. Answer directly from the "
                "Recent Conversation / Relevant Memory above. Do NOT start a new web_search, "
                "do NOT list the current directory, and do NOT read files unless the user "
                "explicitly asks for fresh information or names a specific file to open."
            )
        skill_tools = self._get_skill_tool_hint()
        if skill_tools:
            system_parts.append("")
            system_parts.append(f"Preferred tools for this task: {', '.join(skill_tools)}")
        context_parts = []
        if history_context:
            context_parts.append(history_context)
        if memory_context:
            context_parts.append("## Relevant Memory:\n" + memory_context)
        skill_context = self._get_skill_context(task)
        if skill_context:
            context_parts.append("## Active Skill Instructions:\n" + skill_context)
        # 快车道也注入用户画像, 让简单问答同样按用户偏好行动
        if self.context_engine:
            try:
                pf = self.context_engine.user_profile.format(max_tokens=300)
                if pf and pf.strip():
                    context_parts.append(pf)
            except Exception:
                pass
        # 快车道也做跨会话历史召回, 简单问"我之前说的XX"也能想起
        # (用更高相关性门槛+最多1条, 避免无关旧对话干扰简单问答)
        try:
            past = self._recall_past_conversations(task, k=1, min_relevance=0.25)
            if past:
                context_parts.append(past)
        except Exception:
            pass
        if context_parts:
            system_parts.append("\n" + "\n\n".join(context_parts))
        prompt = "\n".join(system_parts) + f"\n\nUser: {task}\nAssistant:"

        # 流式回调包装:支持 native reasoning_content 与 <think> 标签两种思考来源
        reasoning_parts = []
        content_parts = []

        if stream_callback:
            # fast path 的 content 先缓冲、清洗后再平滑重放:
            # 模型常把 <thinking> 或答案片段混进 content/reasoning, 直接实时透出会产生
            # "的，你好"/尾部回声等多余内容。
            # 这里只透出工具/状态事件; reasoning(思考过程)默认不实时显示,
            # 仅在交互式深度思考(非简单问答)时透出, 保证简洁输出。
            # reasoning 到达时给一个轻量"思考中"状态, 避免用户看到空白以为卡死。
            _thinking_shown = {"v": False}
            def _buffer_user_cb(kind, text):
                if kind in ('tool_call', 'tool_result', 'error'):
                    stream_callback(kind, text)
                elif kind == 'status':
                    stream_callback(kind, text)
                elif kind == 'reasoning' and not _thinking_shown["v"]:
                    _thinking_shown["v"] = True
                    stream_callback('status', 'thinking')
                # 'reasoning' 与 'content' 的正文都不实时透出, 由最终清洗后的答案统一重放

            router = self._create_stream_router(_buffer_user_cb, reasoning_parts, content_parts)
            internal_callback = router.on_token
        else:
            internal_callback = None

        # Dynamic max_tokens: simple greetings/questions fit in 512, but anything
        # involving search results, code, or multi-sentence answers needs more room.
        fast_max_tokens = 512
        task_lower = task.lower()
        # 记忆召回类问题(昨天/上次/之前聊了什么)需要列出多个话题, 需要更多空间
        _memory_recall = bool(re.search(r'(昨天|上次|之前|刚才|还记得|我们聊|话题|对话历史)', task))
        if any(k in task_lower for k in ['搜索', '搜', '查找', '查', 'report', '报告', '总结', '分析', '写', '代码', 'code']):
            fast_max_tokens = 4096
        elif _memory_recall:
            fast_max_tokens = 2048
        elif len(task) > 40 or '?' in task or '？' in task:
            fast_max_tokens = 2048

        # 原生 Function Calling 优先: 后端支持时直接拿结构化 tool_calls,
        # 转成文本协议格式供后续 _parse_output_for_action 复用, 避免模型猜格式。
        raw_answer = None
        try:
            if hasattr(self.backend, "generate_native"):
                native = self.backend.generate_native(
                    prompt,
                    tools=TOOLS_REGISTRY.get_openai_tools(),
                    n_loops=1,
                    temperature=self.config.temperature,
                    max_tokens=fast_max_tokens,
                )
                tcs = native.get("tool_calls") or []
                if tcs:
                    parts = []
                    for tc in tcs:
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})
                        parts.append(f"[TOOL:{name}] {json.dumps(args, ensure_ascii=False)} [/TOOL]")
                    reasoning = (native.get("content") or "").strip()
                    raw_answer = (reasoning + "\n" if reasoning else "") + " ".join(parts)
                elif (native.get("content") or "").strip():
                    raw_answer = native["content"].strip()
        except Exception as e:
            self.logger.debug(f"fast native FC unavailable, falling back: {e}")
            raw_answer = None
        if raw_answer is None:
            try:
                raw_answer = self.backend.generate(
                    prompt,
                    n_loops=1,
                    temperature=self.config.temperature,
                    max_tokens=fast_max_tokens,
                    stream_callback=internal_callback,
                    token_callback=token_callback
                )
            except Exception as e:
                # 稳定性: 后端调用失败(重试耗尽/非连接错误)不让整轮崩溃, 降级为友好提示
                self.logger.error(f"fast generate failed: {type(e).__name__}: {e}")
                raw_answer = f"抱歉, 生成暂时失败({type(e).__name__}), 请稍后重试。"

        # 动态 token 统计：从后端读取真实使用量
        tokens_used = getattr(self.backend, "last_total_tokens", None)
        if tokens_used is not None:
            self.session_token_usage["last_call_tokens"] = int(tokens_used)
            self.session_token_usage["total"] += int(tokens_used)
            self.session_token_usage["stream"].append({"call": "fast_path", "tokens": int(tokens_used)})

        if stream_callback:
            router.finalize()

        raw_answer = raw_answer.strip() or "你好!有什么可以帮你的吗?"

        # 兜底:如果流式过程中没有拿到任何 reasoning,尝试从完整回复中解析 think 标签(<think>/<thinking>)
        if not reasoning_parts:
            m = re.search(r'<think(?:ing)?>(.*?)</think(?:ing)?>', raw_answer, flags=re.DOTALL | re.IGNORECASE)
            if m:
                reasoning_text = m.group(1).strip()
                answer_text = (raw_answer[:m.start()] + raw_answer[m.end():]).strip()
                if stream_callback:
                    stream_callback('status', 'thinking')
                reasoning_parts.append(reasoning_text)
                content_parts = [answer_text]
            else:
                content_parts.append(raw_answer)

        answer_text = "".join(content_parts).strip()
        # 当模型只返回 reasoning 或内容未被路由时,回退到原始输出(去掉 think 标签)
        if not answer_text:
            answer_text = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', raw_answer, flags=re.DOTALL | re.IGNORECASE).strip()
        # 清理模型输出的多余内容(think 标签残留 / 尾部回声)
        answer_text = self._clean_fast_answer(answer_text)

        # 抽取置信度: 从 <think> 思考中提取(内部评估, 不展示给用户), 供 self_correction 使用
        self._last_confidence = self._extract_confidence(raw_answer)
        self.logger.info(f"confidence: {self._last_confidence}")

        # 如果模型输出了工具调用,尝试直接执行并返回结果(失败时自动纠正一次)
        action = self._parse_output_for_action(answer_text)
        actions = []
        final_answer = ""
        ambiguous_reply = ""

        # 意图分类器兜底: LLM 没生成有效工具调用、生成错误工具, 或生成的工具与
        # 用户意图明显冲突(如"查新闻"却去 find 全盘扫描)时,
        # 用确定性规则注入正确的工具调用(置信度高的场景)。
        _classifier_override = False
        classified = self._classify_intent(task)
        # 规则层高置信(>=0.8)或 LLM 升级层(conf 0.75)都视为可信意图, 可 override
        if classified and classified[2] >= 0.7:
            c_tool, c_args, c_conf, c_reason = classified
            if action is None or not TOOLS_REGISTRY.get(action.tool_name):
                _classifier_override = True
            else:
                # 模型生成的工具与意图严重冲突 → 强制纠正
                a_name = action.tool_name
                _mismatch = False
                if c_tool == "web_search" and a_name in ("bash_exec", "find", "glob", "search_files"):
                    _mismatch = True
                if c_tool == "glob" and a_name == "web_search":
                    _mismatch = True
                if _mismatch:
                    self.logger.info(f"intent mismatch: model chose {a_name}, classifier wants {c_tool}; overriding")
                    _classifier_override = True
            if _classifier_override:
                tool_name, intent_args, conf, reason = classified
                self.logger.info(f"intent classifier: {tool_name} (conf={conf}, {reason})")
                # 创建文件意图: 需要确认文件名, 若分类器提取不到就提示而非乱建
                if tool_name == "file_ops" and intent_args.get("action") == "list" and intent_args.get("path"):
                    # 文件夹读取意图: 用真实目录解析(大小写不敏感), 解析不到才用原始名
                    resolved_folder = self._recent_folder_from_history(task)
                    if resolved_folder:
                        intent_args["path"] = resolved_folder
                    from .tools import ToolCall
                    action = ToolCall(tool_name="file_ops", arguments=intent_args)
                elif tool_name == "file_ops" and not intent_args.get("path") and any(
                    k in task for k in ("新建", "创建", "写一篇", "保存为", "输出到文件", "写个文件", "新建文件", "创建文件")
                ):
                    # 没提取到文件名 → 询问用户或使用默认名
                    fname_match = re.search(r'([\w\u4e00-\u9fff\-\.]{1,60}\.(?:md|txt|py|json|yaml|yml|csv|html))', task)
                    path = fname_match.group(1) if fname_match else None
                    if path:
                        intent_args["path"] = path
                        from .tools import ToolCall
                        action = ToolCall(tool_name="file_ops", arguments={"action": "write", "path": path})
                elif tool_name == "glob" and intent_args.get("pattern"):
                    # "看 X 文章"(无扩展名)意图: 用 glob 定位实际文件, 然后直接 read
                    try:
                        from .tools import GlobTool
                        g = GlobTool()
                        res = g.execute(pattern=intent_args.get("pattern", "**"), path=".", max_results=5)
                        if res.success and res.output:
                            # 从 glob 结果提取第一个 .md/.txt 文件路径
                            first_file = None
                            for line in res.output.splitlines():
                                line = line.strip()
                                if line and not line.startswith("Found"):
                                    cand = line.split(" (")[0].strip()
                                    p = Path(cand)
                                    if p.is_file() and p.suffix.lower() in (".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".html", ".py"):
                                        first_file = str(p.resolve())
                                        break
                            if first_file:
                                from .tools import ToolCall
                                action = ToolCall(tool_name="file_ops", arguments={"action": "read", "path": first_file})
                                self.logger.info(f"article glob resolved: {first_file}")
                            else:
                                from .tools import ToolCall
                                action = ToolCall(tool_name="file_ops", arguments={"action": "read", "path": "."})
                        else:
                            from .tools import ToolCall
                            action = ToolCall(tool_name="file_ops", arguments={"action": "read", "path": "."})
                    except Exception:
                        pass
                else:
                    from .tools import ToolCall
                    action = ToolCall(tool_name=tool_name, arguments=intent_args)

        # 超短歧义输入(如 "nih"): 模型容易幻觉成某个机构/缩写去搜索,
        # 一旦搜不到就整屏报错。这里直接不走工具, 用自然语言澄清/问候。
        if action and self._is_ultra_short_ambiguous(task):
            action = None
            ambiguous_reply = "你好！刚才的输入好像没输完整（可能是打字误触）。可以再说一遍吗，或直接告诉我你想做什么？"

        # For simple file-read tasks, bypass model uncertainty and read directly.
        simple_file_path = None
        if self._is_simple_query(task) and not is_continuation:
            simple_file_path = self._resolve_filename_in_task(task)
            if simple_file_path and (action is None or action.tool_name != 'file_ops' or action.arguments.get('action') != 'read'):
                from .tools import ToolCall
                action = ToolCall(tool_name='file_ops', arguments={'action': 'read', 'path': simple_file_path})

        # For "open" / "打开" requests, bypass the model and open the file directly.
        open_path = self._resolve_open_path(task)
        if open_path and (action is None or action.tool_name != 'file_ops' or action.arguments.get('action') != 'open'):
            from .tools import ToolCall
            action = ToolCall(tool_name='file_ops', arguments={'action': 'open', 'path': open_path})

        # 阅读文件夹意图: "阅读/读/查看" + 上文提到过具体目录(如"看下桌面的xx文件夹")
        # -> 自动补全为读取该目录, 避免模型只 list 不 read。
        if action is None and self._is_folder_read_intent(task) and not is_continuation:
            folder = self._recent_folder_from_history(task)
            if folder:
                from .tools import ToolCall
                action = ToolCall(tool_name='file_ops', arguments={'action': 'list', 'path': folder})

        if action:
            # 目录/文件列表等原始输出不适合直接播放, 需要总结后再给用户。
            # 执行前根据工具与参数预判, 避免先播完原始列表又播一遍总结。
            needs_summary = is_continuation or self._tool_returns_listing(action)
            tool_result, final_answer = self._execute_tool_and_observe(action, stream_callback, suppress_content=needs_summary)
            actions.append({
                'tool_name': action.tool_name,
                'arguments': action.arguments,
                'success': tool_result.success,
                'output': tool_result.output[:500],
                'error': tool_result.error,
            })
            # 原始工具输出(目录/文件列表等)不适合直接当答案:
            # 基于观察生成一句自然语言回复, 避免"只读文件没回复"。
            # (最终答案会在下方统一平滑重放, 这里只需替换 final_answer)
            if tool_result.success and self._needs_tool_summary(tool_result.output, is_continuation):
                # 看文件夹场景: list 成功后自动读取目录下的 markdown 文件内容
                # 设计文档要求"list 后再 read 文件内容, 基于内容回答", 避免只列不读
                list_path = None
                if action.tool_name == 'file_ops' and action.arguments.get('action') == 'list':
                    list_path = action.arguments.get('path', '')
                elif action.tool_name == 'bash_exec' and self._is_folder_read_intent(task):
                    # 模型用 bash ls 代替 file_ops list 时同样自动读内容(分析文件夹场景)
                    cmd = str(action.arguments.get('command', ''))
                    # 只在 "ls" 之后的 token 里找第一个非选项/非重定向的目标路径
                    seg = re.split(r'[;&|]', cmd)[0]
                    after_ls = seg.split('ls', 1)[1] if 'ls' in seg else ''
                    path = None
                    for tok in after_ls.split():
                        if tok.startswith('-') or tok == '2>' or tok.endswith('/dev/null'):
                            continue
                        path = tok.strip().strip('"\'')
                        break
                    if path and not path.startswith('2>'):
                        list_path = path
                if list_path:
                    folder_contents = self._read_folder_markdown(list_path)
                    if folder_contents:
                        tool_result.output = tool_result.output + "\n\n" + folder_contents
                summary = self._summarize_tool_answer(task, action, tool_result.output, token_callback)
                if summary and not self._is_truncated_answer(summary):
                    final_answer = summary
            # One-shot retry on tool failure to recover from bad args/format
            if not tool_result.success:
                if self._skip_tool_retry(action, tool_result):
                    # 联网无结果/超时: 重试只会复现相同失败并重复刷屏, 直接自然兜底
                    final_answer = self._tool_failure_fallback(task, tool_result)
                else:
                    retry_prompt = self._build_tool_retry_prompt(task, answer_text, action, tool_result)
                    retry_answer = self.backend.generate(
                        retry_prompt,
                        n_loops=1,
                        temperature=self.config.temperature,
                        max_tokens=512,
                        stream_callback=internal_callback,
                        token_callback=token_callback
                    ).strip()
                    retry_action = self._parse_output_for_action(retry_answer)
                    if retry_action:
                        retry_needs_summary = is_continuation or self._tool_returns_listing(retry_action)
                        tool_result2, final_answer = self._execute_tool_and_observe(
                            retry_action, stream_callback, suppress_content=retry_needs_summary
                        )
                        actions.append({
                            'tool_name': retry_action.tool_name,
                            'arguments': retry_action.arguments,
                            'success': tool_result2.success,
                            'output': tool_result2.output[:500],
                            'error': tool_result2.error,
                        })
                        if tool_result2.success and self._needs_tool_summary(tool_result2.output, is_continuation):
                            summary = self._summarize_tool_answer(task, retry_action, tool_result2.output, token_callback)
                            if summary and not self._is_truncated_answer(summary):
                                final_answer = summary
                        elif not tool_result2.success:
                            final_answer = self._tool_failure_fallback(task, tool_result2)
                    else:
                        final_answer = self._tool_failure_fallback(task, tool_result)
        else:
            final_answer = answer_text

        # 超短歧义输入: 用澄清/问候回复, 不被模型的幻觉工具文本覆盖
        if ambiguous_reply:
            final_answer = ambiguous_reply

        # 如果模型完全没有返回可展示内容,使用友好兜底
        if not final_answer:
            final_answer = "你好!有什么可以帮你的吗?"

        # 安全兜底:清理泄漏到最终回复中的 [TOOL:...] 标签(快速路径不展示工具调用原文)
        # 1) 闭合的 [TOOL:...]...[/TOOL] 块
        final_answer = re.sub(r'\[TOOL:\w+\].*?\[/TOOL\]', '', final_answer, flags=re.DOTALL)
        # 2) 未闭合的 [TOOL:...] + JSON 参数 {…}
        final_answer = re.sub(r'\[TOOL:\w+\]\s*\{[^}]*\}', '', final_answer)
        # 3) 残留的 [TOOL:...] 或 [/TOOL] 碎片
        final_answer = re.sub(r'\[/?TOOL:?\w*\]', '', final_answer)
        # 保留段落换行, 只压缩行内多余空白
        final_answer = re.sub(r'[ \t]+', ' ', final_answer)
        final_answer = re.sub(r'\n{3,}', '\n\n', final_answer).strip()
        if not final_answer:
            final_answer = "你好!有什么可以帮你的吗?"

        # 截断检测: 流被掐断时后端可能返回残缺内容(如 "The"), 补一次完整重试,
        # 避免残缺回答被写进历史、污染后续所有追问。
        if self._is_truncated_answer(final_answer):
            self.logger.warning(f"fast answer looks truncated: {final_answer!r}; retrying once")
            try:
                retry_raw = self.backend.generate(
                    f"{prompt}\n\n注意: 上一条回答异常(截断/过短), 请重新给出完整回答。\nUser: {task}\nAssistant:",
                    n_loops=1,
                    temperature=self.config.temperature,
                    max_tokens=fast_max_tokens,
                    token_callback=token_callback,
                )
                retry_answer = self._clean_fast_answer(str(retry_raw or "")).strip()
                if retry_answer and not self._is_truncated_answer(retry_answer):
                    final_answer = retry_answer
            except Exception as e:
                self.logger.debug(f"fast retry failed {e}")

        # "光说不做"检测: 模型只承诺要做(如"我先看一下HealthOS文件")却未真正执行工具
        # -> 重试一次, 强制它立即调用工具并给出结果
        # 增强: 也检测"声称要搜索/查找/查看/分析但实际无工具调用"(知行合一)
        _promised_but_no_action = self._is_promise_response(final_answer) and not actions
        if not _promised_but_no_action and not actions:
            # 说了"搜索/查找/查看/分析/打开/读取/下载/克隆"等动作词 + 未来时承诺, 却没有任何工具调用
            _claim_words = ["搜索", "查找", "查看", "分析", "打开", "读取", "下载", "克隆",
                            "search", "look up", "fetch", "read", "open", "analyze", "clone",
                            "查询", "调查", "查一下", "找找", "调研", "研究一下",
                            "整理", "保存", "存到", "写入", "写进", "存为", "创建文件", "写成",
                            "写成文章", "生成文件", "写一篇", "保存到", "输出到",
                            "整理成", "整理好", "汇总", "记录", "存档",
                            "write", "save", "create file", "write file", "output to"]
            _promise_words = ["我会", "我将", "我先", "稍后", "准备", "接下来", "下一步", "再去", "随后",
                              "马上", "立刻", "这就", "等一等", "别急", "我现在", "我这就",
                              "我来", "现在就来", "好的马上", "那我", "那我继续", "继续", "接着",
                              "先换", "换个", "换更", "再试", "再搜", "再查", "等我", "待会"]
            _promised_but_no_action = any(w in final_answer for w in _claim_words) and any(
                w in final_answer for w in _promise_words
            )
        if _promised_but_no_action:
            self.logger.warning(f"promise/claim without tool use: {final_answer!r}; forcing execution")
            try:
                retry_raw = self.backend.generate(
                    f"{prompt}\n\n注意: 你刚才只是说要做什么, 但没有真正执行。"
                    f"请立即调用对应工具完成该操作, 并基于工具结果直接回答, 不要只承诺。\nUser: {task}\nAssistant:",
                    n_loops=1,
                    temperature=self.config.temperature,
                    max_tokens=fast_max_tokens,
                    token_callback=token_callback,
                )
                retry_answer = self._clean_fast_answer(str(retry_raw or "")).strip()
                if retry_answer:
                    # 重试后若模型真正输出了工具调用, 重新解析并直接执行
                    retry_action = self._parse_output_for_action(retry_answer)
                    if retry_action and TOOLS_REGISTRY.get(retry_action.tool_name):
                        self.logger.info(f"promise retry produced tool call: {retry_action.tool_name}")
                        try:
                            r_needs_summary = is_continuation or self._tool_returns_listing(retry_action)
                            r_result, r_answer = self._execute_tool_and_observe(
                                retry_action, stream_callback, suppress_content=r_needs_summary
                            )
                            if r_result.success:
                                # 后置核验: 写入类动作, 验证文件确实存在且非空
                                verified = True
                                verify_note = ""
                                if retry_action.tool_name == "file_ops":
                                    _fa = (retry_action.arguments or {}).get("action")
                                    _fp = (retry_action.arguments or {}).get("path")
                                    if _fa in ("write", "apply_diff") and _fp:
                                        try:
                                            exists = TOOLS_REGISTRY.get("file_ops").execute(
                                                action="exists", path=_fp
                                            )
                                            if exists.success and exists.output.strip().lower() == "true":
                                                verify_note = " (已核验文件存在)"
                                            else:
                                                verified = False
                                        except Exception:
                                            pass
                                final_answer = r_answer or f"已执行: {r_result.output[:300]}"
                                if verified:
                                    final_answer += verify_note
                                actions.append({
                                    'tool_name': retry_action.tool_name,
                                    'arguments': retry_action.arguments,
                                    'success': True,
                                    'output': r_result.output[:500],
                                    'error': r_result.error,
                                    'verified': verified,
                                })
                            else:
                                # 后置核验: 工具执行失败, 不能直接返回承诺文字
                                self.logger.warning(f"promise retry tool FAILED: {r_result.error}")
                                final_answer = self._tool_failure_fallback(task, r_result)
                        except Exception as e:
                            self.logger.debug(f"promise retry tool exec failed {e}")
                    elif not self._is_promise_response(retry_answer):
                        final_answer = retry_answer
                    else:
                        # 后置感知: 重试后仍是空承诺(模型坚持不执行工具)
                        # 直接按意图注入工具调用, 不再依赖模型自觉。
                        self.logger.warning(f"promise retry STILL promise-only: {retry_answer!r}; forcing via intent classifier")
                        injected = self._classify_intent(task)
                        if injected and injected[2] >= 0.7:
                            injected_tool, injected_args, _, _ = injected
                            from .tools import ToolCall as _TC
                            forced = _TC(tool_name=injected_tool, arguments=injected_args)
                            try:
                                f_needs_summary = is_continuation or self._tool_returns_listing(forced)
                                f_result, f_answer = self._execute_tool_and_observe(
                                    forced, stream_callback, suppress_content=f_needs_summary
                                )
                                if f_result.success:
                                    final_answer = f_answer or f"已执行: {f_result.output[:300]}"
                                    actions.append({
                                        'tool_name': forced.tool_name,
                                        'arguments': forced.arguments,
                                        'success': True,
                                        'output': f_result.output[:500],
                                        'error': f_result.error,
                                    })
                                else:
                                    final_answer = f"抱歉，执行时遇到问题：{f_result.error or '未知错误'}"
                            except Exception as e:
                                self.logger.debug(f"forced intent exec failed {e}")
            except Exception as e:
                self.logger.debug(f"promise retry failed {e}")

        # 清洗后平滑重放干净答案(实时流已缓冲, 此处统一输出, 避免 think 残留/回声)
        if stream_callback and final_answer:
            for i in range(0, len(final_answer), 8):
                stream_callback("content", final_answer[i:i + 8])
                time.sleep(0.003)

        if self.context_engine and not self._is_truncated_answer(final_answer):
            self.context_engine.observe_assistant(final_answer)

        if getattr(self, "skill_engine", None):
            self.skill_engine.report_outcome(True)

        completed_at = datetime.now()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        # 快车道也学习用户偏好(之前 fast path 不更新画像, 随口说的偏好全丢了)。
        # 异步后台执行: 内部可能做一次 LLM 抽取, 若同步执行会阻塞在返回提示符
        # 之前, 造成"输出结束后卡顿"。
        try:
            self._update_user_profile_async(task, final_answer)
        except Exception:
            pass

        return {
            'task': task,
            'thoughts': [],
            'actions': actions,
            'observations': [{
                'success': True,
                'output': final_answer,
                'error': None,
                'metadata': {'fast_path': True}
            }],
            'thinking_steps': 1,
            'outer_loops': 1,
            'final_reward': 1.0,
            'success': True,
            'session_token_usage': self.session_token_usage,
            'final_answer': final_answer,
            'metadata': {
                'mode': 'fast',
                'strategy': 'direct',
                'started_at': started_at.isoformat(),
                'completed_at': completed_at.isoformat(),
                'duration_ms': duration_ms,
                'fast_path': True,
                'confidence': getattr(self, '_last_confidence', 0.5),
        'tokens': self.session_token_usage.get('last_call_tokens', 0)
        }
        }

    def _extract_confidence(self, raw_output: str) -> float:
        """从模型原始输出(含 <think>)中提取置信度, 供内部自校正使用.

        默认 0.5; 仅在 think 中发现"置信度: X.XX"格式时返回实际值。
        置信度仅供内部评估, 不会展示给用户。
        """
        if not raw_output:
            return 0.5
        # 置信度通常写在 think 内; 直接全局搜索一次即可(think 内也属于 raw_output)。
        cm = re.search(r'(?:置信度|confidence)\s*[:：]\s*(0?\.\d{1,2})', raw_output, re.IGNORECASE)
        if cm:
            try:
                val = float(cm.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5

    def _clean_fast_answer(self, text: str) -> str:
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
        # 2. 去掉尾部未闭合的标签残片(如 <th / <thin / <thinki)
        text = re.sub(r'\s*<t[a-z]*\s*$', '', text).strip()
        # 3. 去掉尾部回声: 结尾若重复了紧邻其前的短词/短句, 只保留一段
        text = re.sub(r'(\S.{0,14}?)\s*\n?\s*\1\s*$', r'\1', text, flags=re.DOTALL)
        return text.strip()

    # 常见合法短答(不计入"疑似截断")
    _COMMON_SHORT_ANSWERS = {"ok", "okay", "yes", "no", "hi", "hello", "hey", "thanks",
                             "good", "fine", "sure", "done", "bye", "n", "y", "好的", "好", "嗯", "行", "可以", "是"}

    # "光说不做"模式: 只承诺要做某事, 却没真正执行(如"我先看一下文件"/"让我查一下")
    _PROMISE_RE = re.compile(
        r"(让我|我来|我先|我准备|我来看|我去|先看|先查|先确认|先检查|回头|接下来|稍后|然后我|我再|我会|我将|准备去|随后|下一步|马上|立刻|这就|等一等|别急|好的马上|这就帮你|我现在|我这就|现在来|正在整理|整理成|保存到|存到|那我|那我继续|继续|接着|先换|换个|换更|再试|再搜|再查|等我|待会)"
    )
    # "只思考不行动"模式: 输出全是 <think>/思考性文字, 没有任何工具调用
    _THINK_ONLY_RE = re.compile(
        r"(?:让我|我需要|应该|先|接下来|首先|最后|计划|步骤|搜索|查找|研究).{0,80}(?:来做|完成|进行|执行|实现|搜索|查找|研究|获取)",
        re.IGNORECASE,
    )

    @classmethod
    def _is_promise_response(cls, text: str) -> bool:
        """检测"只说要做什么、没真正做"的空承诺回复.

        例: "我先看一下 HealthOS 里生成了哪些文件" —— 说了要做, 却没调用工具。
        若回答带出了实质结论(较长且有结论性标点/关键词), 不算空承诺。
        """
        t = (text or "").strip()
        if not t:
            return False
        # 纯 think 输出(无结论/无行动): 视为"空思考"
        if t.startswith(("<think", "<thinking")) or cls._THINK_ONLY_RE.search(t):
            if not re.search(r"[。！？]|结论|结果|如下|完成|已经|好了|目录|文件", t):
                return True
        if not cls._PROMISE_RE.search(t):
            return False
        # 有实际内容/结论 => 不是空承诺
        # 收紧: 仅当长度较长(>100)且含明确结论性词才放行; 单纯句号/短承诺仍视为空承诺
        if len(t) > 100 and re.search(r"结论|结果|如下|完成|已经|好了|目录|文件|发现|建议", t):
            return False
        return True

    @classmethod
    def _is_truncated_answer(cls, text: str) -> bool:
        """判断回答是否疑似流截断(过短/残缺). 正常短答("好"/"ok")不算.

        流被掐断时后端可能把残缺内容(如 "The")当完整返回;
        这类碎片若写进历史会污染后续所有追问。
        """
        t = (text or "").strip()
        if not t:
            return True
        if t in cls._COMMON_SHORT_ANSWERS:
            return False
        # 纯 ASCII、无标点、很短 -> 大概率是截断碎片("The"/"And"/"So")
        if len(t) <= 6 and all(ord(c) < 128 for c in t) and not any(c in ".,;:!?()\"'" for c in t):
            return True
        return False

    def _update_user_profile(self, task: str, final_answer: str):
        """从一次交互中学习用户偏好/事实(供快车道与全路径共用)."""
        ce = getattr(self, "context_engine", None)
        if ce is None:
            return
        try:
            # 简短/无实质内容的轮次(问候等)只跑启发式, 跳过 LLM 抽取, 提速
            text = f"{task}{final_answer}"
            llm_client = None
            if len(text) >= 40:
                client = getattr(self, "_profile_llm_client", None)
                if client is None:
                    from .wiki_memory import PassthroughBackendClient
                    client = PassthroughBackendClient(self.backend)
                    self._profile_llm_client = client
                llm_client = client
            ce.user_profile.update_from_interaction(task, final_answer, llm_client=llm_client)
        except Exception as e:
            self.logger.debug(f"profile update failed {e}")

    def _update_user_profile_async(self, task: str, final_answer: str):
        """后台异步学习用户偏好, 避免同步 LLM 调用阻塞回复返回."""
        try:
            _THREAD_POOL.submit(self._update_user_profile, task, final_answer)
        except Exception as e:
            self.logger.debug(f"async profile update failed {e}")

    def _idle_housekeeping(self) -> int:
        """静默整理: 在用户无操作的空隙调用, 归纳旧对话并压缩活动窗口.

        - 把最旧的若干轮归纳为记忆摘要(LLM), 摘要存入长期记忆(不丢)
        - 从活动对话窗口裁剪这些旧轮次, 让后续 prompt 更短更快
        - 全量历史仍在 sqlite(turns 表), 随时可跨会话召回
        返回本次整理的动作数; 0 表示无需整理。
        """
        lock = getattr(self, "_housekeeping_lock", None)
        if lock is None:
            import threading
            self._housekeeping_lock = threading.Lock()
            lock = self._housekeeping_lock
        if not lock.acquire(blocking=False):
            return 0  # 已有整理在进行中
        try:
            actions = 0
            history = getattr(self, "conversation_history", None) or []
            # 触发阈值: 活动窗口超过 keep_max 轮才整理
            keep_max = 40
            compact_batch = 20
            if len(history) > keep_max:
                old = history[:compact_batch]
                summary = self._summarize_history(old)
                if summary:
                    self._store_idle_memory(summary)
                    actions += 1
                self.conversation_history = history[compact_batch:]
                self._save_history()
                actions += 1
                self.logger.info(f"idle housekeeping: trimmed {compact_batch} old turns, summarized {len(summary)} chars")
            # 工作记忆压缩(由 ContextEngine 决定是否折叠旧事件)
            if self.context_engine:
                try:
                    ce = self.context_engine
                    budget = getattr(ce, "max_total_context", 8000)
                    if getattr(ce, "working_memory", None) is not None and \
                       len(ce.working_memory.events) > 120:
                        ce.working_memory.events = ce.working_memory.events[-120:]
                        actions += 1
                except Exception:
                    pass
            return actions
        finally:
            lock.release()

    def _summarize_history(self, turns: List[Dict[str, Any]]) -> str:
        """把一段对话归纳成精简中文摘要(供长期记忆)."""
        if not turns:
            return ""
        try:
            text = "\n".join(
                f"{t.get('user', '')[:150]} → {t.get('assistant', '')[:100]}"
                for t in turns[-15:]
            )
            prompt = (
                "把下面这段对话归纳成 100 字以内的中文要点(话题、结论、用户偏好、待办事项)。"
                "只输出摘要, 不要解释。\n\n对话:\n" + text
            )
            raw = self.backend.generate(prompt, n_loops=1, temperature=0.3, max_tokens=300)
            return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL | re.IGNORECASE).strip()[:500]
        except Exception as e:
            self.logger.debug(f"history summarize failed: {e}")
            return ""

    def _store_idle_memory(self, summary: str) -> bool:
        """把归纳摘要存入长期记忆(优先 wiki/记忆管理器, 兜底追加到 memory.md)."""
        try:
            mm = getattr(self, "_wiki_manager", None) or getattr(self, "_raw_memory_manager", None)
            if mm is not None and hasattr(mm, "remember"):
                try:
                    mm.remember(text=summary, context="历史对话归纳")
                    return True
                except Exception:
                    pass
        except Exception:
            pass
        try:
            p = Path(__file__).resolve().parent.parent / "data" / "memory.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(f"\n\n## 对话归纳 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{summary}\n")
            return True
        except Exception:
            return False

    def _recall_past_conversations(self, task: str, k: int = 3, min_relevance: float = 0.12) -> str:
        """按当前任务搜索全部历史对话(跨 session), 让 agent"随时想得起来".

        对话已全量持久化在 data/sessions.db(turns 表)。这里对检索结果做
        **相关性门槛**: 只用与当前任务特征重叠足够高的旧轮次, 避免把
        只含一个关键词的无关旧对话塞进 prompt 造成意图混乱。
        """
        raw_mm = getattr(self, "_raw_memory_manager", None)
        store = getattr(raw_mm, "session_store", None)
        if store is None or not task:
            return ""
        try:
            turns = store.search(task, session_id="", k=k * 6)
        except Exception:
            return ""
        scored = []
        for t in turns:
            if getattr(t, "session_id", None) == self.session_id:
                continue
            content = str(t.content or "")
            if not content:
                continue
            sim = self.task_similarity(task, content)
            if sim >= min_relevance:
                scored.append((sim, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        # 去重: 内容过于相近的旧轮次只保留一条
        kept = []
        for sim, t in scored:
            if any(self.task_similarity(t.content, kt.content) > 0.8 for _, kt in kept):
                continue
            kept.append((sim, t))
            if len(kept) >= k:
                break
        if not kept:
            return ""
        lines = ["## 历史相关对话(与当前任务相关, 仅作参考):"]
        for sim, t in kept[:k]:
            role = "用户" if t.role == "user" else "助手"
            content = str(t.content or "")[:160]
            if content:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _repair_tool_arguments(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """通用工具参数修复: 常见缺省/类型错误在执行前自动补齐, 减少无效调用."""
        if not isinstance(args, dict):
            return {"raw": str(args)}
        fixed = dict(args)
        if tool_name == "web_search":
            q = fixed.get("query") or fixed.get("q")
            if not isinstance(q, str) or not q.strip():
                fixed["query"] = ""
        elif tool_name == "file_ops":
            if not isinstance(fixed.get("path"), str) or not fixed.get("path", "").strip():
                fixed["path"] = "."  # 缺省列当前目录, 避免空调用报错
        elif tool_name == "python_exec":
            code = fixed.get("code") or fixed.get("program") or fixed.get("script")
            if not isinstance(code, str):
                fixed["code"] = str(code or "")
        elif tool_name == "bash_exec":
            cmd = fixed.get("command") or fixed.get("cmd")
            if not isinstance(cmd, str):
                fixed["command"] = str(cmd or "")
        elif tool_name == "glob":
            # LLM 常生成空参数 glob:{} → 缺省列当前目录所有文件
            pat = fixed.get("pattern") or fixed.get("query") or fixed.get("path")
            if not isinstance(pat, str) or not pat.strip():
                fixed["pattern"] = "**"
                fixed.setdefault("path", ".")
            elif not fixed.get("path"):
                fixed["path"] = "."
        elif tool_name == "search_files":
            pat = fixed.get("pattern") or fixed.get("query")
            if not isinstance(pat, str) or not pat.strip():
                fixed["pattern"] = ""
            fixed.setdefault("path", ".")
        elif tool_name in ("weather", "calculator", "api_call"):
            # 这些工具缺参时给空字符串兜底, 避免 TypeError
            for k in ("city", "expression", "url"):
                if k in fixed and fixed[k] is None:
                    fixed[k] = ""
        return fixed

    # 任务相似度: 过滤常见动作/虚词, 避免"分析一下X"与"分析一下Y"被误判相关
    _TASK_STOP = set(
        "调研 分析 比较 设计 实现 部署 调试 测试 优化 搜索 查找 计算 解释 总结 翻译 推荐 评估 研究 "
        "查看 告诉 帮我 请问 一下 一个 这个 那个 这些 那些 然后 以及 还有 现在 今天 目前 最新 情况 "
        "内容 资料 信息 报告 输出 生成 列出 介绍 说明 关于 对于 进行 需要 是否 什么 怎么 如何 为什么 "
        "给我 继续 再 写 写一 写个".split()
    )
    # 动作/虚词字符: 在组双字前先剔除, 聚焦"主题内容"而非"动作"
    _TASK_STOP_CHARS = set(
        "的了么呢吗吧啊呀哦嗯好对是而在与及为之于以从到把被让给就都也还再这那请问帮我你它"
        "分分析析一一上下看看查算写做找说说讲讲谈问要能会想希望需要请叫个位种些样次回遍点"
    )

    @staticmethod
    def task_tokens(task: str) -> set:
        """把任务拆成"主题特征": 英文词 + 由内容字符组成的中文双字(剔除动作/虚词)."""
        task = (task or "").lower()
        toks = set(re.findall(r"[a-z0-9]+", task))
        cjk = [c for c in re.findall(r"[\u4e00-\u9fff]", task) if c not in OpenMythosAgent._TASK_STOP_CHARS]
        for i in range(len(cjk) - 1):
            bg = cjk[i] + cjk[i + 1]
            if bg not in OpenMythosAgent._TASK_STOP:
                toks.add(bg)
        return toks

    @classmethod
    def task_similarity(cls, a: str, b: str) -> float:
        """两个任务的相关度(0-1). 共享英文主题词(AI/Python等)视为强关联."""
        ta, tb = cls.task_tokens(a), cls.task_tokens(b)
        if not ta or not tb:
            return 0.0
        shared = ta & tb
        if not shared:
            return 0.0
        sim = len(shared) / min(len(ta), len(tb))
        # 共享一个明确的英文主题词(≥2字母) = 强主题关联
        if any(t.isascii() and len(t) >= 2 for t in shared):
            sim = max(sim, 0.3)
        return sim

    @classmethod
    def merge_related_tasks(cls, tasks: List[str], threshold: float = 0.2) -> List[List[str]]:
        """多段任务分组: 相邻且相关度 >= threshold 的任务合并成一组.

        - 相关(同一主题/目标) → 合并为一次执行, 避免重复搜索/推理
        - 无关 → 各自成组, 排队依次执行
        """
        groups: List[List[str]] = []
        for t in tasks:
            t = (t or "").strip()
            if not t:
                continue
            if groups and cls.task_similarity(groups[-1][-1], t) >= threshold:
                groups[-1].append(t)
            else:
                groups.append([t])
        return groups

    def _create_stream_router(self, user_callback, reasoning_parts, content_parts):
        """创建流式回调路由器,支持 native reasoning 和 <think> 标签.

        额外抑制原始 [TOOL:...] 标签直接泄漏到 content 流;这些标签会保留在
        content_parts 中供后续工具解析,但不会被用户看到.
        """

        class StreamRouter:
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
                if not self.buffer:
                    return
                if self.state == 'think':
                    self._emit('reasoning', self.buffer)
                elif self.state == 'tool_call_raw':
                    # Preserve raw tool call for parser but don't leak it as content
                    self._append('content', self.buffer)
                else:
                    self._emit('content', self.buffer)
                self.buffer = ''

        return StreamRouter(user_callback, reasoning_parts, content_parts)

    # ============ 核心Agent循环 ============

    def run(
        self,
        task: str,
        mode: str = 'production',
        stream_callback: Optional[callable] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        code_mode: bool = False
    ) -> Dict[str, Any]:
        """
        运行Agent(主入口)

        Args:
            task: 用户任务
            mode: 'production' | 'reflection' | 'exploration'
            stream_callback: 可选的流式回调函数,接收 (kind, token) 参数
            token_callback: 可选的 token 用量回调函数,接收 (tokens) 参数
            code_mode: 强制启用编码模式(更激进的工程化工作流)

        Returns:
            包含完整轨迹和结果的字典
        """
        # Heavy enum/helpers are imported lazily inside the main entry point.
        try:
            from .planning import PlanningStrategy, create_simple_plan
        except Exception:
            PlanningStrategy = None  # type: ignore
            create_simple_plan = None  # type: ignore
        from .reasoning import ReasoningStrategy

        # 新一轮开始时清空每轮缓存
        self._method_cache.clear()
        self._tool_result_cache.clear()

        # 本会话轮次计数(进程级):启动后第一个问题走极速 Turbo,后续恢复正常深度
        is_session_first_turn = (getattr(self, '_session_turn_count', 0) == 0)
        self._session_turn_count = getattr(self, '_session_turn_count', 0) + 1
        # 关键: 仅当没有持久化历史时才视为"全新首轮"。若磁盘已有对话历史
        # (如进程重启), 首问应恢复正常记忆注入, 而非硬编码跳过 memory。
        has_persisted_history = bool(getattr(self, 'conversation_history', None))

        # 在指代消解之前先判断是否为延续性追问,避免 task 被改写后丢失
        # "这个/那个/该"等指代特征,导致后续错误地重新搜索。
        is_continuation_query = self._is_continuation_query(task)

        # 利用上下文补全不完整/歧义输入;如果无法补全则直接澄清
        resolved_task, clarification = self._resolve_ambiguous_task(task)
        # 首次对话跳过指代澄清,直接进入极速 Turbo 路径,保证首响最快
        if clarification is not None and not is_session_first_turn:
            # 优先尝试使用已解析的上下文继续;如果确实无上下文才澄清
            if resolved_task == task:
                return {
                    "final_answer": clarification,
                    "success": True,
                    "outer_loops": 0,
                    "thinking_steps": 0,
                    "metadata": {"clarification": True, "original_task": task},
                }
            task = resolved_task
        self._status(stream_callback, f"task: {task[:80]}{'...' if len(task) > 80 else ''}")
        self._code_mode_override = code_mode
        self._current_task = task  # 用于后续 web_search query 锚定校正:避免模型把用户关键词"跑偏"改写

        # Prompt-injection guard: log and short-circuit obvious jailbreak attempts.
        if getattr(self.config, "harness", None) and self.config.harness.prompt_injection_scan:
            try:
                from .harness.prompt_guard import guard_input
                guard_result = guard_input(task, source="user")
                if guard_result.triggered:
                    self.logger.warning(
                        "Prompt guard triggered: category=%s pattern=%r",
                        guard_result.category,
                        guard_result.matched_pattern,
                    )
                    return {
                        "final_answer": (
                            "I can't process this request: it triggered the prompt-injection guard "
                            f"(category: {guard_result.category}). Please rephrase your request."
                        ),
                        "success": False,
                        "outer_loops": 0,
                        "thinking_steps": 0,
                        "metadata": {"prompt_guard": guard_result.to_dict()},
                    }
            except Exception as e:
                self.logger.debug(f"prompt guard error: {e}")

        # Skill slash commands: /skill list, /skill <name>, /skill create, etc.
        if task.lstrip().lower().startswith("/skill") and getattr(self, "skill_engine", None):
            remaining, result = self.skill_engine.handle_slash(task)
            if result.get("type") in ("info", "error", "skill_loaded"):
                output = result.get("output", "")
                if remaining:
                    # If a skill was activated with a trailing task, run the task with the skill loaded.
                    task = remaining
                else:
                    return {
                        "final_answer": output,
                        "success": result.get("type") != "error",
                        "outer_loops": 0,
                        "thinking_steps": 0,
                        "metadata": {"skill_command": result},
                    }

        # Deep research + Markdown report workflow: handles requests like
        # "搜索 XXX 并生成深度调研报告" or "输出 markdown 文件".
        # Also handles explicit "/deep <topic>" and "/research <topic>" commands.
        if is_research_report_task(task) or task.lstrip().lower().startswith(("/deep", "/research")):
            return self._run_deep_research(
                task,
                stream_callback=stream_callback,
                token_callback=token_callback,
            )

        if self.context_engine:
            self.context_engine.observe_user(task)

        # Auto-detect coding tasks early so planning/loop decisions can skip overhead
        code_mode = self._is_coding_task(task)
        if code_mode:
            self._status(stream_callback, "code mode")

        # Holds the ReasoningTrace when the reasoning engine is used; otherwise None.
        trace = None

        # ===== 首次对话:强制极速 Turbo 模式 =====
        # 启动后首问必须最快响应 —— 无论问题复杂度,跳过记忆注入/规划/多轮推理,
        # 走单次 LLM 调用快速路径(n_loops=1, 流式),最低首响延迟。
        # 后续对话恢复正常深度推理(含编码任务的完整 ReAct 循环)。
        # 例外: 若磁盘已持久化历史(进程重启后的首问),不跳过记忆注入。
        # fast_mode: true 才走"首轮极速 turbo"; false 时首轮也走完整深度推理。
        if mode == 'production' and is_session_first_turn and not has_persisted_history and self.config.fast_mode:
            self._status(stream_callback, "turbo · first turn")
            history_context = self._format_history_context()  # 首次为空
            result = self._run_simple(
                task,
                memory_context="",  # 首次无历史记忆可注入
                history_context=history_context,
                stream_callback=stream_callback,
                token_callback=token_callback,
                is_continuation=is_continuation_query,
            )
            self._append_to_history(
                task,
                result.get('final_answer')
                or (result.get('observations', [{}])[-1].get('output', '') if result.get('observations') else ''),
            )
            return result

        # Retrieve wiki memory context for this task (turbo mode skips on first turn / simple queries)
        memory_mode = "turbo" if self._is_simple_query(task) else ("deep" if self._is_complex_task(task) else "normal")
        memory_context = self._get_memory_context(task, mode=memory_mode) if self._should_inject_memory(task) else ""
        if memory_context:
            self._status(stream_callback, f"memory {len(memory_context)} chars")

        # Build short-term conversation history context
        history_context = self._format_history_context()

        # Fast path for simple conversational queries
        if mode == 'production' and self._is_simple_query(task):
            self._status(stream_callback, "fast path")
            result = self._run_simple(
                task,
                memory_context=memory_context,
                history_context=history_context,
                stream_callback=stream_callback,
                token_callback=token_callback,
                is_continuation=is_continuation_query,
            )
            self._append_to_history(task, result.get('final_answer') or result.get('observations', [{}])[-1].get('output', ''))
            return result

        # Fast path for location/locate queries: avoid blind directory listing.
        location_result = self._try_location_fast_path(
            task, stream_callback=stream_callback, token_callback=token_callback
        )
        if location_result:
            self._append_to_history(task, location_result.get('final_answer', ''))
            return location_result

        # 1. Use advanced planning if available (skip for simple tasks and code mode)
        plan = None
        needs_plan = (
            self.planner and self.config.planning.enabled and mode != 'reflection'
            and not code_mode and self._should_plan(task)
        )
        if needs_plan and PlanningStrategy is not None:
            try:
                plan = self.planner.create_plan(
                    task,
                    strategy=PlanningStrategy(self.config.planning.default_strategy),
                    optimize=self.config.planning.optimize_plans,
                    max_subtasks=self.config.planning.max_subtasks
                )
                self._status(stream_callback, f"plan {len(plan.nodes)} tasks")
            except Exception as e:
                self.logger.warning(f"plan failed {e}")

        # Fallback: structured file/folder analysis tasks benefit from a lightweight
        # plan even when full LLM decomposition is skipped.
        if plan is None and self.planner and self.config.planning.enabled and create_simple_plan is not None and self._is_file_analysis_task(task):
            plan = create_simple_plan(task)
            self._status(stream_callback, "plan lightweight")

        # 2. Determine thinking loops (adaptive or from plan)
        plan_context = ""
        if plan:
            root_loops = plan.nodes['task_0'].assigned_loops if 'task_0' in plan.nodes else self.config.default_thinking_loops
            max_node_loops = max(n.assigned_loops for n in plan.nodes.values())
            n_loops = min(max(root_loops, max_node_loops, len(plan.nodes) * 3), self.config.max_thinking_loops)
            # Embed the plan so the reasoning engine can follow it
            plan_lines = ["## Plan (follow this sequence):"]
            for idx, node_id in enumerate(plan.topological_sort(), 1):
                node = plan.nodes[node_id]
                plan_lines.append(f"{idx}. [{node_id}] {node.description}")
            plan_context = "\n".join(plan_lines)
        else:
            # Determine thinking loops (simple fallback)
            if mode == 'reflection':
                n_loops = self.config.reflection.thinking_loops_for_reflection
            else:
                n_loops = self._decide_thinking_loops(task, mode)

        # Apply pending correction from previous run (self-correction feedback loop)
        if self._pending_correction:
            rec_loops = self._pending_correction.get('recommended_loops')
            if rec_loops:
                n_loops = min(max(n_loops, rec_loops), self.config.max_thinking_loops)
            strat_override = self._pending_correction.get('strategy_override')
            if strat_override:
                self._strategy_override = strat_override
            self._status(stream_callback, f"correction loops {n_loops}")

        self._status(stream_callback, f"loops {n_loops}")

        # 3. Retrieve similar cases for few-shot (if available)
        similar_cases = self._retrieve_similar_cases(task)
        if similar_cases:
            self._status(stream_callback, f"similar {len(similar_cases)}")

        # 4. Build initial prompt
        if self.reasoning_engine and self.config.reasoning.enabled:
            # Use advanced reasoning engine
            # Default to SUPER_AGENT for the adaptive meta-loop (reflection + replanning).
            # Reflection mode keeps the configured strategy so it can be explicitly studied.
            reasoning_strategy = ReasoningStrategy.SUPER_AGENT
            if mode == 'reflection':
                reasoning_strategy = ReasoningStrategy(self.config.reasoning.default_strategy)
            if self._strategy_override:
                try:
                    reasoning_strategy = ReasoningStrategy(self._strategy_override)
                except ValueError:
                    pass
            # Retrieve strategy advice from past successes
            strategy_advice = self.strategy_db.get_advice_for_task(task) if self.strategy_db else ""
            context_parts = []
            if history_context:
                context_parts.append(history_context)
            if memory_context:
                # memory_context 已含 profile/semantic/episodic/lessons
                # (build_system_context 内置), 无需再单独塞 lessons_context 造成重复
                context_parts.append(memory_context)
            skill_context = self._get_skill_context(task)
            if skill_context:
                context_parts.append("## Active Skill Instructions:\n" + skill_context)
            if strategy_advice:
                context_parts.append(strategy_advice)
            if plan_context:
                context_parts.append(plan_context)
            if context_parts:
                task_with_context = "\n\n".join(context_parts) + f"\n\n## Current Task:\n{task}"
            # MCP orchestrator: dynamically select relevant tools and suggest combinations
            all_tools = TOOLS_REGISTRY.get_tools_dict()
            available_tools = all_tools
            skill_tools = self._get_skill_tool_hint()
            if skill_tools:
                # Boost skill-preferred tools to the front of the available map.
                ordered: Dict[str, str] = {}
                for name in skill_tools:
                    if name in all_tools and name not in ordered:
                        ordered[name] = all_tools[name]
                ordered.update(all_tools)
                available_tools = ordered
            tool_suggestion_context = ""
            if self.mcp_orchestrator:
                try:
                    available_tools = self.mcp_orchestrator.recommend_tools(task, all_tools)
                    combo = self.mcp_orchestrator.suggest_combination(task)
                    if combo:
                        tool_suggestion_context = f"## Suggested Tool Combination: {' -> '.join(combo)}"
                    self.logger.info(f"mcp {len(available_tools)} tools")
                except Exception as e:
                    self.logger.warning(f"mcp failed {e}")

            # Re-apply skill-preferred tool ordering after MCP orchestrator.
            if skill_tools:
                ordered_after_mcp: Dict[str, str] = {}
                for name in skill_tools:
                    if name in available_tools and name not in ordered_after_mcp:
                        ordered_after_mcp[name] = available_tools[name]
                ordered_after_mcp.update(available_tools)
                available_tools = ordered_after_mcp

            # Inject tool combination hint into context if not already present
            if tool_suggestion_context and tool_suggestion_context not in task_with_context:
                task_with_context = tool_suggestion_context + "\n\n" + task_with_context

            # Engineering tasks need more ReAct steps for read -> edit -> verify -> fix
            if code_mode:
                n_loops = max(n_loops, 6)
                self._status(stream_callback, f"loops {n_loops} code")

            trace = self.reasoning_engine.reason(
                task=task_with_context,
                available_tools=available_tools,
                strategy=reasoning_strategy,
                custom_loops=n_loops,
                stream_callback=stream_callback,  # 传递流式回调
                token_callback=token_callback,    # 传递 token 用量回调
                code_mode=code_mode
            )

            # 外层闭环: 质量不达标时在 max_outer_loops 预算内重试(尝试不同策略), 实现"反馈→重规划→重试"
            # 不再只重试一次, 而是用完外层循环预算或质量达标为止。
            outer_budget = max(1, self.config.max_outer_loops)
            outer_attempt = 1
            self.outer_loop_counter = 1
            quality = getattr(trace, 'quality_score', 1.0)
            retry_strategies = [ReasoningStrategy.VERIFICATION, ReasoningStrategy.SELF_CONSISTENCY]
            while quality < 0.45 and outer_attempt < outer_budget and not code_mode and mode != 'reflection':
                strat = retry_strategies[(outer_attempt - 1) % len(retry_strategies)]
                self._status(stream_callback, f"retry {outer_attempt}/{outer_budget} with {strat.value}")
                retry_trace = self.reasoning_engine.reason(
                    task=task_with_context,
                    available_tools=available_tools,
                    strategy=strat,
                    custom_loops=max(n_loops, 3) + outer_attempt,
                    stream_callback=stream_callback,
                    token_callback=token_callback,
                    code_mode=code_mode
                )
                outer_attempt += 1
                self.outer_loop_counter = outer_attempt
                if getattr(retry_trace, 'quality_score', 0.0) > quality:
                    trace = retry_trace
                    quality = getattr(retry_trace, 'quality_score', 0.0)
                else:
                    # 质量没提升, 换下一个策略继续; 若两个策略都试过仍不行则停
                    if outer_attempt > len(retry_strategies):
                        break

            trajectory = {
                'task': task,
                'thoughts': [s.content for s in trace.steps],
                'actions': [{'tool_name': t} for t in trace.tools_used],
                'observations': trace.observations if hasattr(trace, 'observations') else [],
                'thinking_steps': trace.total_loops,
                'outer_loops': trace.outer_loops,
                'final_reward': trace.quality_score,
                'success': trace.success,
                'final_answer': trace.final_answer,
                'metadata': {
                    'mode': mode,
                    'started_at': datetime.now().isoformat(),
                    'strategy': trace.strategy.value,
                    'duration_ms': trace.duration_ms
                }
            }
        else:
            # Unified fallback: single-shot DirectPolicy loop via ExecutionEngine
            direct_ctx = ExecutionContext(
                task=task,
                available_tools=TOOLS_REGISTRY.get_tools_dict(),
                config=self.config,
                max_steps=n_loops,
                stream_callback=stream_callback,
                token_callback=token_callback,
                code_mode=code_mode,
                extra_context=plan_context or "",
                history_context=history_context,
            )
            engine = ExecutionEngine(
                model_backend=self.backend,
                config=self.config,
                harness_kernel=self._harness_kernel,
                per_turn_cache=self._tool_result_cache,
            )
            exec_trace = engine.run(DirectPolicy(), direct_ctx)
            trajectory = {
                'task': task,
                'thoughts': [s.reasoning for s in exec_trace.steps],
                'actions': [
                    {'tool_name': c.tool_name, 'arguments': c.arguments}
                    for s in exec_trace.steps for c in s.tool_calls
                ],
                'observations': [{'success': True, 'output': o, 'error': None} for o in exec_trace.observations],
                'thinking_steps': len(exec_trace.steps),
                'outer_loops': len(exec_trace.steps),
                'final_reward': exec_trace.quality_score,
                'success': exec_trace.success,
                'final_answer': exec_trace.final_answer or "",
                'metadata': {
                    'mode': mode,
                    'started_at': datetime.now().isoformat(),
                    'strategy': 'direct',
                    'duration_ms': exec_trace.duration_ms,
                    **exec_trace.metadata,
                },
            }

        # Append current turn to short-term history
        final_answer = trajectory.get('final_answer') or ''
        if not final_answer and trajectory.get('observations'):
            final_answer = str(trajectory['observations'][-1])[:500]
        self._append_to_history(task, final_answer)

        if self.context_engine and not self._is_truncated_answer(final_answer):
            self.context_engine.observe_assistant(final_answer)

        # 5. Self-correction evaluation
        if self.self_correction and self.config.self_correction.enabled and trajectory:
            try:
                metrics, correction = self.self_correction.process_execution(
                    trace=trace,
                    task=task,
                    strategy=trajectory.get('metadata', {}).get('strategy', 'unknown'),
                    tools_used=[a.get('tool_name', 'unknown') for a in trajectory.get('actions', [])]
                )
                trajectory['metadata']['quality_metrics'] = metrics.__dict__
                if correction:
                    trajectory['metadata']['correction_applied'] = correction.__dict__
                    self.total_corrections += 1
                    # Persist correction so the NEXT task can adapt parameters
                    self._pending_correction = {
                        'recommended_loops': correction.recommended_loops,
                        'strategy_override': correction.strategy_override,
                        'action_type': correction.action_type,
                        'description': correction.description,
                    }
                else:
                    self._pending_correction = None
            except Exception as e:
                self.logger.warning(f" Self-correction failed: {e}")
                self._pending_correction = None

        # 6. Consolidate into long-term memory (semantic + profile via ContextEngine, async)
        if self.context_engine and self.config.memory.enabled and trajectory:
            try:
                self.context_engine.consolidate(task, trajectory)
            except Exception as e:
                self.logger.warning(f" ContextEngine consolidation failed: {e}")

        # 7. Wrap up
        trajectory['thinking_steps'] = trajectory.get('thinking_steps', 0) or trajectory.get('outer_loops', 0) * n_loops
        trajectory['final_reward'] = trajectory.get('final_reward', self._compute_reward(trajectory))
        trajectory['metadata']['completed_at'] = datetime.now().isoformat()

        # 8. Store experience and trigger self-improvement
        self._store_experience(trajectory)
        self.episodes_completed += 1

        # Update strategy usage statistics if a strategy was matched and applied
        if self._last_matched_strategy_id and self.strategy_db:
            try:
                self.strategy_db.update_usage(
                    self._last_matched_strategy_id,
                    success=trajectory.get('success', False)
                )
            except Exception as e:
                self.logger.warning(f"strategy usage update failed {e}")

        if self._should_reflect():
            self._trigger_self_improvement()

        # 9. Auto-extract entities & build knowledge graph (wiki pages)
        # Run asynchronously so that LLM-based memory extraction does not block the response.
        if self.memory_manager and self.config.memory.enabled:
            try:
                content = json.dumps(trajectory, indent=2, ensure_ascii=False)

                def _remember_work():
                    page_ids = self.memory_manager.remember(
                        text=content,
                        context=task,
                        auto_link=True
                    )
                    self._log_to_file(f"memory pages: {len(page_ids)}")
                    trajectory['metadata']['memory_pages'] = page_ids

                    # MemSkill: skill-conditioned memory extraction + evolution
                    if self.memskill_engine:
                        try:
                            outcome = trajectory.get('final_answer', '') or str(trajectory.get('observations', [])[-1]) if trajectory.get('observations') else ''
                            result = self.memskill_engine.learn_from_interaction(
                                task=task,
                                trajectory=trajectory,
                                outcome=outcome,
                                success=bool(trajectory.get('success')),
                                memory_backend=self.memory_manager,
                            )
                            self._log_to_file(
                                f"memskill ops={len(result.get('operations', []))} "
                                f"skills={','.join(result.get('skills_used', []))} "
                                f"evolved={','.join(result.get('evolved_skills', []))}"
                            )
                            trajectory['metadata']['memskill'] = result
                        except Exception as e:
                            self._log_to_file(f"MemSkill learning failed: {e}")

                _THREAD_POOL.submit(_remember_work)
            except Exception as e:
                self.logger.warning(f"   Memory remember scheduling failed: {e}")

        if getattr(self, "skill_engine", None):
            self.skill_engine.report_outcome(bool(trajectory.get('success')))

        return trajectory

    # ============ 辅助方法 ============

    def _decide_thinking_loops(self, task: str, mode: str) -> int:
        """自适应决定思考深度:根据任务长度、关键词和工具需求调整"""
        return self._cache_get("thinking_loops", (task, mode), lambda: self._compute_decide_thinking_loops(task, mode))

    def _compute_decide_thinking_loops(self, task: str, mode: str) -> int:
        """思考深度决策的实际计算逻辑."""
        if mode == 'reflection':
            return self.config.reflection.thinking_loops_for_reflection

        task_lower = task.lower()
        base = self.config.default_thinking_loops

        # 查询策略库
        strategy = self.strategy_db.match(task) if self.strategy_db else None
        self._last_matched_strategy_id = None
        if strategy:
            base = int(strategy.avg_loop_depth)
            self._last_matched_strategy_id = strategy.id
            self._log_to_file(f"strategy loops {base}")

        complexity_score = 0

        # 任务长度
        if len(task) > 200:
            complexity_score += 3
        elif len(task) > 80:
            complexity_score += 2
        elif len(task) > 40:
            complexity_score += 1

        # 需要多步推理/工具的关键词
        multi_step_keywords = [
            '比较', '对比', '分析', '设计', '实现', '部署', '调试', '优化',
            'compare', 'analyze', 'design', 'implement', 'deploy', 'debug', 'optimize',
            '总结', '解释', '评估', '推荐', '方案',
            'summarize', 'explain', 'evaluate', 'recommend', 'plan',
        ]
        if any(kw in task_lower for kw in multi_step_keywords):
            complexity_score += 1

        # 明确需要搜索/外部数据
        tool_keywords = ['搜索', '查找', 'search', 'find', 'lookup', '天气', '新闻', 'weather', 'news']
        if any(kw in task_lower for kw in tool_keywords):
            complexity_score += 1

        loops = base + complexity_score

        # 随机探索
        if mode == 'exploration':
            import random
            loops = random.randint(4, self.config.max_thinking_loops)

        return min(int(loops), self.config.max_thinking_loops)

    def _retrieve_similar_cases(self, task: str, k: int = 3) -> List[Experience]:
        """检索相似成功案例"""
        if not self.experience_buffer:
            return []  # 经验库未启用/加载失败时静默跳过, 不刷警告
        try:
            return self.experience_buffer.get_similar(task, k=k, success_only=True)
        except Exception as e:
            self.logger.debug(f"retrieve similar cases skipped: {e}")
            return []

    def _is_complex_task(self, task: str) -> bool:
        """判断是否需要 deep 模式（最大上下文、完整记忆）."""
        task_lower = task.lower()
        complex_signals = [
            "架构", "重构", "设计", "实现", "优化", "调试", "分析", "研究",
            "architecture", "refactor", "design", "implement", "optimize", "debug", "analyze", "research",
        ]
        return any(sig in task_lower for sig in complex_signals) and len(task) > 30

    def _try_location_fast_path(self, task: str, stream_callback=None, token_callback=None) -> Optional[Dict[str, Any]]:
        """Directly locate projects/folders/files by name to avoid blind directory listing."""
        task_lower = task.lower()
        # Only trigger for locate/find intents
        locate_verbs = ["查找", "找一下", "搜索", "搜一下", "看看有没有", "看看", "看下", "在哪里", "在哪", "位于", "找", "查", "搜"]
        locate_suffixes = ["项目", "文件夹", "目录", "文件"]
        has_verb = any(v in task for v in locate_verbs)
        has_suffix = any(s in task for s in locate_suffixes)
        has_location_marker = any(m in task for m in ["下的", "里面", "中的", "上的", "里的"])

        # Detect explicit "<location>的<target>" pattern (e.g. "桌面文件夹的report")
        location_aliases_ordered = ["下载文件夹", "桌面文件夹", "文档文件夹", "下载", "桌面", "文档"]
        has_explicit_location_target = False
        for alias in location_aliases_ordered:
            pattern = re.compile(re.escape(alias) + r"(?:项目|文件夹|目录|文件)?\s*的\s*(.+)", re.IGNORECASE)
            if pattern.search(task):
                has_explicit_location_target = True
                break

        if not (has_verb or has_location_marker or has_explicit_location_target):
            return None
        if not has_suffix and not re.search(r'[a-zA-Z_\-0-9]+', task):
            return None

        # 排除明显的"搜索信息/新闻/资料"意图: 这些应走 web_search, 而非 find 文件
        # (如"查ai新闻""搜一下最新的资讯""查查天气"等)
        info_search_markers = ["新闻", "资讯", "消息", "动态", "最新", "信息", "资料", "教程",
                               "怎么", "如何", "教程", "介绍", "是什么", "怎么做", "天气",
                               "news", "update", "info", "how to", "what is", "weather",
                               "股票", "行情", "价格", "比分", "比赛"]
        if has_suffix is False and any(m in task_lower for m in info_search_markers):
            return None
        # 即便含"文件夹/文件"后缀, 若是搜索网络信息意图也不触发 find
        if any(m in task_lower for m in ["新闻", "资讯", "动态", "最新", "消息", "news", "update"]) and \
           not has_location_marker and not has_explicit_location_target:
            return None

        # Resolve search root
        root = str(Path.home())
        for alias, target in [
            ("文档文件夹", "Documents"), ("文档目录", "Documents"), ("文档", "Documents"),
            ("桌面文件夹", "Desktop"), ("桌面", "Desktop"),
            ("下载文件夹", "Downloads"), ("下载", "Downloads"),
        ]:
            if alias in task:
                root = str(Path.home() / target)
                break

        # Extract target name:
        # 1) Prefer explicit pattern: "<location>的<target>" or "<location><suffix>的<target>"
        # 2) Then markers like "下的/里面/中的/上的/里的"
        # 3) Finally fallback to verb-based extraction.
        target_name = None

        # Pattern: "下载文件夹的claude code" -> target = "claude code"
        # Handles: 下载/桌面/文档/下载文件夹/桌面文件夹/文档文件夹 + optional 项目/文件夹/目录/文件 + 的
        for alias in location_aliases_ordered:
            pattern = re.compile(re.escape(alias) + r"(?:项目|文件夹|目录|文件)?\s*的\s*(.+)", re.IGNORECASE)
            m = pattern.search(task)
            if m:
                target_name = m.group(1).strip()
                target_name = re.sub(r'(项目|文件夹|目录|文件)\s*$', '', target_name).strip()
                break

        # Marker-based fallback: part after 下的/里面/中的/上的/里的
        if not target_name:
            for marker in ["下的", "里面", "中的", "上的", "里的"]:
                if marker in task:
                    idx = task.rfind(marker)
                    target_name = task[idx + len(marker):].strip()
                    target_name = re.sub(r'(项目|文件夹|目录|文件)\s*$', '', target_name).strip()
                    break

        # Verb-based fallback: between verb and optional suffix
        # Order verbs longest-first to avoid "搜索" being split as "搜" + "索...".
        if not target_name:
            m = re.search(r'(?:查找|找一下|搜索|搜一下|查一下|看看有没有|看看|看下|找|查|搜)\s*([\w\s\-_.]+?)(?:项目|文件夹|目录|文件)?\s*$', task)
            if m:
                target_name = m.group(1).strip()

        if not target_name:
            return None

        # Clean common prefixes/suffixes and Chinese alias roots
        target_name = target_name.strip('"\'`“”')
        for alias in location_aliases_ordered:
            if target_name.lower().startswith(alias.lower()):
                target_name = target_name[len(alias):].lstrip("/\\")
                break
        if not target_name or len(target_name) < 2:
            return None

        import shlex
        safe_root = shlex.quote(root)
        # Treat spaces/hyphens/underscores as wildcards so "claude code" matches "claude-code" and "claude_code".
        search_name = target_name.replace(" ", "*").replace("-", "*").replace("_", "*")
        # Escape single quotes for shell single-quoted string.
        escaped_name = search_name.replace("'", "'\"'\"'")
        cmd = f"find {safe_root} -maxdepth 4 -iname '*{escaped_name}*' -print | head -40"

        self._status(stream_callback, f"locating '{target_name}' in {root}")
        bash_tool = TOOLS_REGISTRY.get("bash_exec")
        if bash_tool is None:
            return None
        try:
            tool_result = bash_tool.execute(command=cmd)
            output = tool_result.output or tool_result.error or ""
            # bash_exec returns '(no output)' when stdout is empty; treat it as empty.
            if output.strip().lower() == "(no output)":
                output = ""
            # If nothing found in the specific location, quickly widen to home directory.
            if not output.strip() and root != str(Path.home()):
                home_cmd = f"find {shlex.quote(str(Path.home()))} -maxdepth 4 -iname '*{escaped_name}*' -print | head -40"
                tool_result = bash_tool.execute(command=home_cmd)
                output = tool_result.output or tool_result.error or ""
                if output.strip().lower() == "(no output)":
                    output = ""
                if output.strip():
                    output = f"(expanded search from {root} to home)\n{output}"
            if not output.strip():
                output = f"No results found for '{target_name}' under {root} or home directory."
            final = f"Found matches for '{target_name}':\n{output}"
            return {
                "final_answer": final,
                "success": True,
                "outer_loops": 1,
                "thinking_steps": 1,
                "metadata": {"duration_ms": 0},
            }
        except Exception as e:
            return None

    def _get_memory_context(self, task: str, max_pages: int = 3, mode: str = "normal") -> str:
        """Unified memory recall via ContextEngine, with raw MemoryManager fallback,
        plus cross-session past-conversation recall so the agent 'remembers' old chats."""
        if not self.config.memory.enabled:
            return ""
        ctx = ""
        if self.context_engine:
            try:
                # include_history=False: 最近对话由 history_context 单独注入,
                # 避免 working 历史在 memory_context 里重复出现(重复即噪音)。
                ctx = self.context_engine.build_system_context(
                    task, include_history=False, k=max_pages, mode=mode
                )
            except Exception as e:
                self.logger.warning(f"context engine memory context failed {e}")
        else:
            raw_mm = getattr(self, "_raw_memory_manager", None)
            if raw_mm is not None and hasattr(raw_mm, "augment_prompt"):
                try:
                    base = "You are Lv Super Agent. Use the following memory context to inform your response."
                    ctx = raw_mm.augment_prompt(base, task, session_id=self.session_id, max_tokens=1024)
                except Exception as e:
                    self.logger.warning(f"raw memory augment failed {e}")
        # 跨会话历史对话召回: 让"随时想得起来"成为现实
        past = self._recall_past_conversations(task)
        if past:
            ctx = (ctx + "\n\n" + past).strip() if ctx else past
        return ctx

    def _format_history_context(self, max_turns: int = 4) -> str:
        """Return working-memory conversation context via ContextEngine."""
        if self.context_engine:
            try:
                ctx = self.context_engine.working_memory.format_for_prompt(max_tokens=1200)
                if ctx and ctx.strip():
                    return ctx
            except Exception:
                pass
        # 工作记忆为空(如回填失败)时, 回退到持久化的 conversation_history
        return self._legacy_format_history_context(max_turns)

    def _legacy_format_history_context(self, max_turns: int = 4) -> str:
        """Legacy fallback using conversation_history list."""
        if not self.conversation_history:
            return ""
        recent = self.conversation_history[-max_turns:]
        lines = ["## Recent Conversation (summarized — verify facts with current tools):"]
        for turn in recent:
            user = turn.get("user", "")[:200]
            assistant = turn.get("assistant", "")[:200]
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {assistant}{' ...' if len(turn.get('assistant', '')) > 200 else ''}")
        return "\n".join(lines)

    def _extract_lessons(self, trajectory: Dict[str, Any]) -> List[Lesson]:
        """从一次执行轨迹中提取可复用的教训."""
        from .experience import Lesson
        lessons = []
        task = trajectory.get('task', '')
        task_lower = task.lower()
        success = trajectory.get('success', False)
        actions = trajectory.get('actions', [])
        observations = trajectory.get('observations', [])
        final_answer = str(trajectory.get('final_answer', '') or '')

        tool_names = []
        for action in actions:
            if isinstance(action, dict):
                tool_names.append(action.get('tool_name', ''))
            elif isinstance(action, ToolCall):
                tool_names.append(action.tool_name)

        # 失败模式:重复调用同一工具
        if not success and len(tool_names) >= 3:
            from collections import Counter
            most_common = Counter(tool_names).most_common(1)
            if most_common and most_common[0][1] >= 3:
                lessons.append(Lesson(
                    task_pattern=task,
                    condition="repeated same tool call 3+ times",
                    action="stop repeating; choose a different tool or file; verify path before calling again",
                    outcome="task failed due to infinite loop",
                    success=False,
                    source_episode_id=None
                ))

        # 失败模式:没有读取文件就给出最终答案(项目/文件分析任务)
        has_file_ops = any(name == 'file_ops' for name in tool_names)
        if not success and any(kw in task_lower for kw in ['分析', '项目', '文件', '总结', 'summarize', 'analyze', 'project', 'file']):
            if not has_file_ops and len(final_answer) > 50:
                lessons.append(Lesson(
                    task_pattern=task,
                    condition="file/project analysis without reading files",
                    action="ALWAYS use file_ops to list and read relevant files before giving final answer",
                    outcome="premature final answer with no evidence",
                    success=False,
                    source_episode_id=None
                ))

        # 成功模式:使用了 file_ops 序列完成任务
        if success and has_file_ops and len(tool_names) >= 2:
            lessons.append(Lesson(
                task_pattern=task,
                condition="task requires reading files",
                action="first list directory with file_ops, then read key files with offset/limit",
                outcome="successfully gathered evidence before answering",
                success=True,
                source_episode_id=None
            ))

        # 失败模式:路径错误
        obs_text = " ".join(str(o) for o in observations)
        if not success and ('does not exist' in obs_text or 'not found' in obs_text.lower()):
            lessons.append(Lesson(
                task_pattern=task,
                condition="file or tool path error",
                action="use absolute paths; verify with list/exists before read; do not repeat the wrong path",
                outcome="tool call returned path error",
                success=False,
                source_episode_id=None
            ))

        return lessons

    def _load_history(self) -> List[Dict[str, str]]:
        """从磁盘加载跨会话短期对话历史."""
        try:
            if self.history_path.exists():
                data = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data[-self.max_history_turns:]
        except Exception as e:
            self.logger.warning(f"history load failed {e}")
        return []

    def _save_history(self):
        """保存短期对话历史到磁盘."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(
                json.dumps(self.conversation_history, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            self.logger.warning(f"history save failed {e}")

    def _resolve_ambiguous_task(self, task: str) -> Tuple[str, Optional[str]]:
        """
        利用对话上下文补全不完整的用户输入和代词指代。

        当用户输入像"下载文件夹有"这种缺少宾语的句子,或"分析这个系统"这种
        使用代词的句子时,尝试从最近对话中推断隐含对象。如果无法推断,返回澄清问题。

        Returns:
            (resolved_task, clarification_question)
            clarification_question 为 None 表示已成功补全,可直接使用 resolved_task。
        """
        task = task.strip()
        if not task:
            return task, "你想让我做什么?"

        # 1. 代词/指代消解
        # 检测"这个/那个/它/该/此/这些/那些/前者/后者/上述(+系统/项目/文件夹/文件/代码)?"
        pronoun_pattern = re.compile(
            r"(?:这个|那个|这些|那些|前者|后者|上述|它|该|此)(?:系统|项目|文件夹|文件|代码|东西)?",
            re.IGNORECASE,
        )
        has_pronoun = bool(pronoun_pattern.search(task))

        # 从历史中提取最近的主题/实体
        recent_turns = getattr(self, "conversation_history", [])[-6:]
        recent_user_msgs = [t.get("user", "").strip() for t in recent_turns]
        recent_assistant_msgs = [t.get("assistant", "").strip() for t in recent_turns]
        recent_text = "\n".join(recent_user_msgs + recent_assistant_msgs)

        if has_pronoun:
            resolved = self._resolve_pronoun_to_entity(task, recent_text)
            if resolved:
                return resolved, None
            return task, "你提到的'这个/那个'具体指什么项目或文件?"

        # 2. 跳过已知的完整问句(尤其是关于 Agent 自身能力、问候、疑问句等)
        complete_question_patterns = [
            r"^你[能可]做[什么到].*",
            r"^你[有都].*[什么么能].*",
            r"^你[是谁怎样].*",
            r"^你叫[什么].*",
            r"^你好.*",
            r"^帮助$",
            r"^help$",
            r"^\?.*",
            # 疑问助词结尾的完整问句("你知道我吗"/"然后呢"等)不得判为不完整
            r"^.*[吗么呢]\s*$",
            # 疑问词开头的完整问句
            r"^(怎么|如何|为什么|哪些|什么|多少|哪个|谁|何时|哪里|几|是不是|有没有|能否|能不能|可不可以|为啥|为何|什么叫).*",
        ]
        if any(re.match(p, task, re.IGNORECASE) for p in complete_question_patterns):
            return task, None

        # 3. 检测是否是不完整句: 仅当明显"悬空"(列举类动词结尾、缺少对象)时判定,
        #    不再用笼统的"以吗/么/呢/是/在结尾"作判据(那会误伤完整问句)。
        incomplete_markers = [
            r"^.+?文件夹(?:里|里面)?\s*有\s*$",
            r"^.+?目录(?:里|里面)?\s*有\s*$",
            r"^.+?文件(?:里|里面)?\s*有\s*$",
            r"^.+?里面\s*有\s*$",
            r"^.+?下有\s*$",
        ]
        is_incomplete = any(re.match(p, task, re.IGNORECASE) for p in incomplete_markers)
        if not is_incomplete:
            return task, None

        # 3. 尝试推断隐含对象
        # 如果当前提到"下载/桌面/文档"且历史中有"claude code",补全为查找/分析问题
        location_aliases = ["下载文件夹", "下载", "桌面", "桌面文件夹", "文档", "文档文件夹"]
        has_location = any(alias in task for alias in location_aliases)

        if has_location:
            target = self._extract_most_recent_entity(recent_text)
            if target:
                # 构造更完整的任务
                if "有" in task or "有没有" in task:
                    resolved = f"{task.rstrip('有').rstrip()}有没有'{target}'项目/文件?如果有,请分析它。"
                else:
                    resolved = f"{task} '{target}'"
                return resolved, None

        # 4. 无法推断,返回澄清问题
        if has_location:
            return task, f"你是指'{task}'里面有什么,还是想让我查找/分析某个具体项目?"
        return task, f"'{task}'似乎没有说完,你想让我做什么?"

    _PRONOUN_SUB_RE = re.compile(
        r"(?:这个|那个|这些|那些|前者|后者|上述|它|该|此)(?:[\u4e00-\u9fa5a-zA-Z0-9\-_]{1,8})?",
        re.IGNORECASE,
    )

    def _resolve_pronoun_to_entity(self, task: str, recent_text: str) -> Optional[str]:
        """把代词(这个/那个/它等)解析为最近提到的具体路径或项目名。"""
        # 1. 优先找完整文件/文件夹路径(最近提到优先)
        # 支持绝对路径 + 相对路径(../xxx、./xxx、IDE/super-ide), 相对路径结合 cwd 解析成绝对路径
        # (?<!\.) 防止 ./ 从 ../ 的第二个点处开始匹配(正则回溯误匹配出 ./IDE/...)
        path_patterns = [
            r"(/Users/[^\s\n\"'<>|]+)",
            r"(~/[^\s\n\"'<>|]+)",
            r"(/home/[^\s\n\"'<>|]+)",
            r"(/Volumes/[^\s\n\"'<>|]+)",
            r"(\.\./[^\s\n\"'<>|]+)",
            r"(?<!\.)(\./[^\s\n\"'<>|]+)",
        ]
        paths = []
        for pat in path_patterns:
            for m in re.finditer(pat, recent_text):
                p = m.group(1).strip()
                # 只去掉尾部标点(用 rstrip), 避免把相对路径前导的 .. 剥掉
                p = p.rstrip(".,;:!?，。、")
                if len(p) > 2:
                    paths.append(p)
        # 取最近一个(文本中越靠后越近)
        if paths:
            target_path = paths[-1]
            # 相对路径(../、./)结合当前工作目录解析成绝对路径, 供后续工具直接使用
            if target_path.startswith(("../", "./")):
                try:
                    target_path = str((Path.cwd() / target_path).resolve())
                except Exception:
                    pass
            # 若相对 cwd 解析出的路径不存在(如 ../IDE/super-ide 相对 agent 目录),
            # 尝试常见根目录(Downloads/Desktop/Documents)组合, 找到真实存在的位置。
            if target_path.startswith(("../", "./")) and not Path(target_path).exists():
                rel = Path(target_path)
                for root in [Path.home() / "Downloads", Path.home() / "Desktop",
                             Path.home() / "Documents", Path.cwd().parent]:
                    cand = root / rel
                    if cand.exists():
                        target_path = str(cand.resolve())
                        break
            # 去掉尾部的压缩包或具体文件,保留项目目录(如果目标是目录)
            path_obj = Path(target_path)
            if path_obj.is_file():
                target_path = str(path_obj.parent)
            return self._PRONOUN_SUB_RE.sub(
                f"'{target_path}'",
                task,
                count=1,
            )

        # 1.5 实体名是目录: 从历史中定位该目录的实际路径(大小写不敏感 + 名字包含匹配),
        #    避免"分析它"把 super-ide 解析成裸名字后 agent 去错误目录搜索。
        target = self._extract_most_recent_entity(recent_text)
        if target:
            located = self._locate_project_dir(target)
            if located:
                return self._PRONOUN_SUB_RE.sub(f"'{located}'", task, count=1)
            return self._PRONOUN_SUB_RE.sub(
                f"'{target}'",
                task,
                count=1,
            )
        return None

    def _locate_project_dir(self, name: str) -> Optional[str]:
        """在常见根目录下定位名字匹配的项目/文件夹(大小写/连字符归一).

        返回首个存在的目录绝对路径; 找不到返回 None。
        搜索范围: 各根目录的直接子目录 + 直接子目录的一级子目录(两级),
        足够覆盖 Downloads/IDE/super-ide 这类嵌套, 又不至于全盘扫描。
        """
        if not name:
            return None
        name_norm = re.sub(r"[\s\-_]+", "", name.lower())
        if not name_norm:
            return None
        roots = [Path.cwd(), Path.cwd().parent, Path.home(),
                 Path.home() / "Desktop", Path.home() / "Downloads",
                 Path.home() / "Documents"]
        seen_roots = set()
        scanned_dirs = set()
        for root in roots:
            try:
                r = root.resolve()
            except Exception:
                continue
            if r in seen_roots or not r.is_dir():
                continue
            seen_roots.add(r)
            try:
                children = [c for c in r.iterdir() if c.is_dir()]
            except Exception:
                continue

            def _norm_dir(d) -> str:
                return re.sub(r"[\s\-_]+", "", d.name.lower())

            # 第一层: 先精确匹配(完整相等), 再退化到子串匹配(仅当名字足够长避免误匹配 super-ide-build)
            for child in children:
                cn = _norm_dir(child)
                if cn == name_norm:
                    return str(child.resolve())
            for child in children:
                cn = _norm_dir(child)
                if len(name_norm) >= 4 and cn and cn.endswith(name_norm) and len(cn) - len(name_norm) <= 8:
                    return str(child.resolve())
                # 子串匹配需长度相近(长度差≤6): 避免 super-ide 误匹配 SuperIDE-GOAI-...(不同项目)
                if len(name_norm) >= 6 and cn and name_norm in cn and 0 <= len(cn) - len(name_norm) <= 6:
                    return str(child.resolve())
            # 第二层: 直接子目录的一级子目录(限制数量防扫描过深), 同样先精确后子串
            for child in children[:40]:
                try:
                    cid = child.resolve()
                except Exception:
                    continue
                if cid in scanned_dirs:
                    continue
                scanned_dirs.add(cid)
                try:
                    grand = [g for g in cid.iterdir() if g.is_dir()]
                except Exception:
                    continue
                for g in grand[:40]:
                    gn = _norm_dir(g)
                    if gn == name_norm:
                        return str(g.resolve())
                for g in grand[:40]:
                    gn = _norm_dir(g)
                    if len(name_norm) >= 4 and gn and gn.endswith(name_norm) and len(gn) - len(name_norm) <= 8:
                        return str(g.resolve())
                    if len(name_norm) >= 6 and gn and name_norm in gn and 0 <= len(gn) - len(name_norm) <= 6:
                        return str(g.resolve())
        return None

    _CN_STOPWORDS = {
        "什么", "怎么", "为什么", "多少", "哪里", "谁", "怎样", "如何",
        "还有", "也是", "还是", "或者", "以及", "关于", "对于", "由于",
        "虽然", "但是", "因为", "所以", "如果", "那么", "然后", "接着",
        "现在", "刚才", "之前", "之后", "上面", "下面", "这里", "那里",
    }

    def _extract_most_recent_entity(self, recent_text: str) -> Optional[str]:
        """从历史文本中提取最近提到的项目/文件/主题名。"""
        candidates = []
        # 英文项目/文件名
        for m in re.finditer(r"[a-zA-Z_\-][\w\-]*(?:\s+[a-zA-Z_\-][\w\-]*)*", recent_text):
            cand = m.group(0).strip()
            if len(cand) >= 2 and cand.lower() not in {"the", "this", "that", "with", "from", "for", "and", "or", "in", "on", "to", "of"}:
                candidates.append(cand)
        # 中文主题词(前面有"的")
        for m in re.finditer(r"的\s*([^\s,。！？\n]{2,20}?)(?:项目|文件夹|文件|目录|代码)?", recent_text):
            candidates.append(m.group(1).strip())

        # 去重并保持顺序，过滤常见停用词/疑问词
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = c.lower()
            if key in self._CN_STOPWORDS:
                continue
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        if not unique_candidates:
            return None

        # 优先选择像专有名词的候选（大写开头或包含数字/连字符），
        # 否则返回最近一个非停用词候选。
        for c in reversed(unique_candidates):
            if any(ch.isupper() for ch in c) or "-" in c or any(ch.isdigit() for ch in c):
                return c
        return unique_candidates[-1]

    def _append_to_history(self, user_msg: str, assistant_msg: str):
        """记录本轮对话到短期历史并持久化."""
        # 不把残缺/截断的回答写进历史(否则会污染后续所有追问)
        if self._is_truncated_answer(assistant_msg):
            self.logger.warning(f"skip appending truncated assistant reply: {assistant_msg!r}")
            return
        self.conversation_history.append({
            "user": user_msg.strip(),
            "assistant": assistant_msg.strip(),
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
        self._save_history()

        # Persist turn to SQLite session memory (best-effort, non-blocking).
        raw_mm = getattr(self, "_raw_memory_manager", None)
        if raw_mm is not None and hasattr(raw_mm, "remember_turn"):
            try:
                raw_mm.remember_turn(
                    self.session_id,
                    [
                        {"role": "user", "content": user_msg.strip()},
                        {"role": "assistant", "content": assistant_msg.strip()},
                    ],
                )
            except Exception as e:
                self.logger.debug(f"session memory persist failed: {e}")

    _TOOL_ALIASES = {
        # LLM 常写错的工具名 → 合法工具名
        "search": "web_search", "websearch": "web_search", "web-search": "web_search",
        "searh": "web_search", "serach": "web_search", "websearh": "web_search",
        "internet_search": "web_search", "google": "web_search", "duckduckgo": "web_search",
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

    def _correct_tool_name(self, name: str) -> Optional[str]:
        """工具名纠错: 尝试把 LLM 写错/臆造的工具名映射到最接近的合法工具."""
        if not name:
            return None
        stripped = name.strip().lower()
        # 1) 精确别名映射
        if stripped in self._TOOL_ALIASES:
            return self._TOOL_ALIASES[stripped]
        # 2) 前缀匹配(如 web_sear → web_search, file_op → file_ops)
        valid = set(TOOLS_REGISTRY.list_tools())
        for v in valid:
            v_l = v.lower()
            if stripped in v_l or v_l in stripped:
                return v
        # 2.5) 接近拼写错误(如 searh→search, 编辑距离小且长度相近)
        best, best_dist = None, 3
        for v in valid:
            d = self._edit_distance(stripped, v.lower())
            # 编辑距离<=2 即视为接近; 或相对距离小(<=40%)也接受
            len_ok = d <= 2 or (len(v) and abs(len(stripped) - len(v)) / max(len(v), 1) <= 0.4)
            if d < best_dist and len_ok:
                best, best_dist = v, d
        if best:
            return best
        # 3) 去除下划线/连字符后的模糊匹配
        norm = re.sub(r'[_\-\s]', '', stripped)
        for v in valid:
            if re.sub(r'[_\-\s]', '', v.lower()) == norm:
                return v
        return None

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
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

    def _parse_output_for_action(self, text: str) -> Optional[ToolCall]:
        """Parse the first actionable tool call from model output.

        Supports: [TOOL:name] args [/TOOL], <provider>name</provider>, and
        OpenCode wrapped function call formats.
        Iterates over all parsed calls and returns the first one whose tool
        is actually registered (auto-correcting misspelled tool names),
        so an invalid/nonexistent leading call (e.g.
        <|message_model|>think...) does not block a valid later call.
        """
        for action in self._parse_all_output_actions(text):
            if TOOLS_REGISTRY.get(action.tool_name):
                return action
            # 工具名纠错: LLM 可能写错名, 尝试映射到合法工具
            corrected = self._correct_tool_name(action.tool_name)
            if corrected and TOOLS_REGISTRY.get(corrected):
                from .tools import ToolCall
                self.logger.info(f"tool name corrected: {action.tool_name!r} → {corrected}")
                return ToolCall(tool_name=corrected, arguments=action.arguments)
        return None

    def _parse_all_output_actions(self, text: str) -> List[ToolCall]:
        """从模型输出中提取所有工具调用。

        委托给 policies.ToolCallParser(单一权威实现), 避免双解析器重复维护。
        支持 [TOOL:] / XML / OpenCode / JSON action+args / tool_calls 数组 /
        function-call style 等格式。
        """
        if not text:
            return []
        from .policies import ToolCallParser
        parsed = ToolCallParser.parse_all(text)
        return [ToolCall(tool_name=name, arguments=args) for name, args in parsed]


    def _extract_search_keywords(self, task: str) -> str:
        """Delegate to the shared grounding helper in research_report.py."""
        from .research_report import extract_search_keywords
        return extract_search_keywords(task)

    def _ground_search_query(self, orig_keywords: str, generated_query: str) -> str:
        """Delegate to the shared grounding helper in research_report.py."""
        from .research_report import ground_search_query
        return ground_search_query(orig_keywords, generated_query)

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用 - 支持多种调用方式，带超时保护。"""
        tool = TOOLS_REGISTRY.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                success=False, output="",
                error=f"Tool not found: {tool_call.tool_name}"
            )
        try:
            # 确保 arguments 是 dict
            args = tool_call.arguments or {}

            # WebSearch Query 锚定校正：
            #   用户可能说"搜索实在智能"，但推理模型在混乱对话历史影响下，
            #   生成的 web_search query 被"发散"成了"人工智能 2025 最新进展..."这种
            #   完全偏离用户原意的关键词。这里用 self._current_task（用户原始任务）做锚定：
            #   提取任务中"搜索/查找/搜/查一下"后面的名词短语，确保这些词在 query 里出现。
            if tool_call.tool_name == "web_search":
                orig_task = getattr(self, "_current_task", "") or ""
                orig_query = self._extract_search_keywords(orig_task)
                generated_query = str(args.get("query", "")).strip() if isinstance(args.get("query"), str) else ""
                if orig_query and generated_query:
                    grounded = self._ground_search_query(orig_query, generated_query)
                    if grounded and grounded != generated_query:
                        args = dict(args)
                        args["query"] = grounded
                        self.logger.info(f"web_search query anchored: {generated_query!r} → {grounded!r}")
                elif orig_query and not generated_query:
                    # 模型没给 query, 直接用用户关键词兜底
                    args = dict(args)
                    args["query"] = orig_query

            # file_ops 参数预处理：展开 ~，补全缺失字段
            if tool_call.tool_name == "file_ops":
                import os

                # 缺省路径: 未提供/为空时默认当前工作目录(让模型先看到自己在哪,
                # 而不是直接报错; 这能避免它跑去桌面等无关目录)
                if "path" not in args or not isinstance(args.get("path"), str) or not args["path"].strip():
                    args = dict(args)
                    args["path"] = "."

                # 展开 ~ 为用户主目录
                if isinstance(args["path"], str):
                    args["path"] = os.path.expanduser(args["path"])

                # 如果缺少 action，从 path 推断
                if "action" not in args:
                    args = dict(args)
                    args["action"] = "list" if os.path.isdir(args["path"]) else "read"

            # 通用参数修复: 常见缺省/类型错误在执行前自动补齐, 减少无效调用
            args = self._repair_tool_arguments(tool_call.tool_name, args)

            # Harness 策略门：在真实执行前评估效果意图，拒绝即短路
            if self._harness_kernel is not None:
                from .harness.effects import make_effect
                from .harness.kernel import Decision

                admission = self._harness_kernel.evaluate(
                    make_effect(tool_call.tool_name, args)
                )
                if admission.decision is not Decision.ALLOW:
                    self.logger.warning(
                        f"harness denied {tool_call.tool_name}: {admission.reason}"
                    )
                    return ToolResult(
                        success=False, output="",
                        error=f"Denied by harness policy: {admission.reason}"
                    )

            # Per-tool timeouts so a hanging tool cannot freeze the agent.
            timeout_map = {
                "web_search": 30,
                "bash_exec": 120,
                "python_exec": 30,
                "file_ops": 30,
                "project_context": 30,
                "playwright_browser": 60,
                "api_call": 20,
            }
            timeout = timeout_map.get(tool_call.tool_name, 25)

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool.execute, **args)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                return ToolResult(
                    success=False, output="",
                    error=f"Tool '{tool_call.tool_name}' timed out after {timeout}s"
                )
            finally:
                # Do not block shutdown on a hanging future.
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            import traceback
            return ToolResult(success=False, output="", error=f"Tool execution error: {str(e)}\n{traceback.format_exc()}")

    def _execute_tool_and_observe(self, action: ToolCall,
                                  stream_callback: Optional[callable] = None,
                                  suppress_content: bool = False) -> Tuple[ToolResult, str]:
        """Execute a tool, observe it in working memory, and format the final answer.

        suppress_content: 为 True 时不直接播放原始工具内容作为最终回复,
        但仍显示工具调用框(tool_call/tool_result), 由调用方播放提炼后的回答。
        """
        cache_key = f"{action.tool_name}:{json.dumps(action.arguments, sort_keys=True, ensure_ascii=False)}"
        cached = self._tool_result_cache.get(cache_key)
        if cached is not None:
            final_answer = cached.output.strip() if cached.success else f"工具调用失败: {cached.error or 'unknown error'}"
            if stream_callback:
                self._stream_tool_observation(action, cached, final_answer, stream_callback, suppress_content)
            return cached, final_answer

        if self.context_engine:
            self.context_engine.observe_tool_call(action.tool_name, action.arguments)
        tool_result = self._execute_tool(action)
        self._tool_result_cache[cache_key] = tool_result
        if self.context_engine:
            self.context_engine.observe_tool_result(
                action.tool_name,
                tool_result.output or tool_result.error or "",
                tool_result.success
            )
        if tool_result.success:
            final_answer = tool_result.output.strip()
        else:
            final_answer = f"工具调用失败: {tool_result.error or 'unknown error'}"
        if stream_callback:
            self._stream_tool_observation(action, tool_result, final_answer, stream_callback, suppress_content)
        return tool_result, final_answer

    def _stream_tool_observation(self, action: ToolCall, tool_result: ToolResult,
                                 final_answer: str, stream_callback: callable,
                                 suppress_content: bool = False) -> None:
        """Emit tool_call/tool_result events plus a compact content preview.

        suppress_content: 为 True 时不播放原始内容(避免先播列表又播总结);
        工具调用框(名称/参数/结果预览)仍会显示, 让用户看到"做了什么"。
        """
        display_text = final_answer
        try:
            if action.tool_name == "web_search" and tool_result.success:
                parsed = json.loads(tool_result.output or "[]")
                if isinstance(parsed, list):
                    lines = [f"{i+1}. {r.get('title', '')} — {r.get('url', '')}" for i, r in enumerate(parsed[:5])]
                    display_text = "\n".join([f"web_search returned {len(parsed)} results:"] + lines)
        except Exception:
            pass

        try:
            stream_callback("tool_call", f"{action.tool_name}: {action.arguments}")
            stream_callback("tool_result", final_answer if not tool_result.success else display_text)
            if not suppress_content:
                # Mark content as started so the final answer is not re-printed verbatim.
                stream_callback("content", "\n" + display_text)
        except Exception:
            pass

    def _build_tool_retry_prompt(self, task: str, raw_answer: str,
                                 action: ToolCall, tool_result: ToolResult) -> str:
        """Build a corrective prompt for one-shot tool call retry."""
        return (
            "You are a helpful assistant. The previous tool call failed. "
            "Fix the arguments and call the tool again using EXACTLY JSON format.\n\n"
            f"Task: {task}\n"
            f"Your previous response: {raw_answer[:500]}\n"
            f"Failed call: {action.tool_name}({action.arguments})\n"
            f"Error: {tool_result.error or 'unknown error'}\n\n"
            "Output only the corrected tool call, e.g.:\n"
            '[TOOL:file_ops] {"action": "list", "path": "~/Desktop"} [/TOOL]'
        )

    def _compute_reward(self, trajectory: Dict) -> float:
        """计算奖励"""
        if trajectory['success']:
            return 1.0
        else:
            # 可以基于步骤数、工具使用等给出部分奖励
            return 0.1 * (trajectory['outer_loops'] / self.config.max_outer_loops)

    def _store_experience(self, trajectory: Dict[str, Any]):
        """存储经验并提取可复用教训"""
        if not self.experience_buffer:
            return
        task_type = self._infer_task_type(trajectory['task'])
        episode_id = self.experience_buffer.add_episode(
            task=trajectory['task'],
            trajectory=trajectory,
            task_type=task_type
        )

        # 提取并存储教训,让失败/成功模式真正影响未来决策
        try:
            lessons = self._extract_lessons(trajectory)
            for lesson in lessons:
                lesson.source_episode_id = episode_id
                self.experience_buffer.add_lesson(lesson)
        except Exception as e:
            self.logger.warning(f"lesson extraction failed {e}")
            lessons = []

        self._log_to_file(
            f"experience stored | episodes: {self.experience_buffer.count():>5} | lessons: {len(lessons):>3} | episode_id: {episode_id}"
        )

    def _infer_task_type(self, task: str) -> str:
        """推断任务类型"""
        task_lower = task.lower()
        if 'weather' in task_lower:
            return 'weather_query'
        elif any(kw in task_lower for kw in ['calculate', 'math', 'compute']):
            return 'calculation'
        elif any(kw in task_lower for kw in ['file', 'read', 'write']):
            return 'file_operation'
        elif 'search' in task_lower:
            return 'web_search'
        elif 'api' in task_lower:
            return 'api_call'
        elif 'python' in task_lower or 'code' in task_lower:
            return 'code_execution'
        else:
            return 'general'

    def _should_reflect(self) -> bool:
        """检查是否应该触发反思.

        触发条件(全部满足):
        1. reflection.enabled 且经验库可用;
        2. 最近 frequency*10 条经验中失败数 >= min_failures_threshold;
        3. 距上次反思至少间隔 frequency 个 episode(冷却, 避免每轮都触发).

        注意: 旧的"episodes_completed % frequency == 0"依赖进程内计数器,
        重启即归零且 fast-path 任务不计入, 导致自动反思几乎从未触发。
        """
        if not self.config.reflection.enabled:
            return False
        if not self.experience_buffer:
            return False

        try:
            recent = self.experience_buffer.get_recent(
                n=self.config.reflection.frequency * 10,
                success_only=False,
            )
            failures = [e for e in recent if not e.trajectory.get('success', False)]
            if len(failures) < self.config.reflection.min_failures_threshold:
                return False
        except Exception as e:
            self.logger.warning(f"should_reflect check failed: {e}")
            return False

        # 冷却: 距上次反思至少 frequency 个 episode
        cooled = (self.episodes_completed - self._last_reflection_episode) >= self.config.reflection.frequency
        # 低置信度内部评估(< 0.4)时缩短冷却, 允许更早触发反思以改进
        low_conf = getattr(self, '_last_confidence', 0.5) < 0.4
        if not cooled and not low_conf:
            return False
        return True

    def _trigger_self_improvement(self):
        """触发自我改进循环(反思失败案例 + 提取策略 + 可选微调)."""
        self.logger.info("self-improvement...")
        try:
            # 1. 反思失败案例
            if not self.experience_buffer or not self.reflection_module:
                self.logger.info("self-improvement skipped (modules unavailable)")
                return
            recent_failures = self.experience_buffer.get_failures(n=10)
            if not recent_failures:
                self.logger.info("no failures")
                return

            reflections = self.reflection_module.batch_reflect(recent_failures)
            self.logger.info(f"reflections {len(reflections)}")
            # 反思洞察回灌为 lessons, 让后续任务真正受益
            self._store_reflection_lessons(recent_failures, reflections)

            # 2. 从所有成功案例中提取策略(包括反思带来的新洞察)
            if self.strategy_db:
                recent_success = self.experience_buffer.get_recent(n=50, success_only=True)
                if len(recent_success) >= 5:
                    new_strategies = self.strategy_db.update_from_experiences(
                        recent_success,
                        self.experience_buffer
                    )
                    # 记录策略改进
                    for strat in new_strategies:
                        self.logger.info(f"strategy: {strat.task_type} {strat.pattern[:40]}{'...' if len(strat.pattern) > 40 else ''}")

            # 3. (可选)生成训练数据并微调
            if self.config.self_improvement.auto_training:
                self._generate_and_train()

            self.logger.info("self-improvement done")
        except Exception as e:
            self.logger.warning(f"self-improvement failed: {e}")
        finally:
            self._last_reflection_episode = getattr(self, 'episodes_completed', 0)

    def _store_reflection_lessons(self, episodes, reflections):
        """把反思产出的通用规则/改进建议回灌为 lessons, 影响后续任务决策."""
        if not self.experience_buffer:
            return
        try:
            from .experience import Lesson
        except Exception:
            return
        stored = 0
        for ep, ref in zip(episodes, reflections):
            try:
                rule = (getattr(ref, 'generalized_rule', '') or '').strip()
                if not rule:
                    continue
                lesson = Lesson(
                    task_pattern=(getattr(ep, 'task', '') or '')[:200],
                    condition=(
                        "reflection(" +
                        ",".join(getattr(ref, 'identified_patterns', [])[:3]) +
                        ")" if getattr(ref, 'identified_patterns', None) else "reflection insight"
                    ),
                    action=rule[:300],
                    outcome=f"quality={getattr(ref, 'quality_score', 0)}",
                    success=getattr(ref, 'quality_score', 0) >= 6,
                    source_episode_id=getattr(ep, 'id', None),
                )
                self.experience_buffer.add_lesson(lesson)
                stored += 1
            except Exception as e:
                self.logger.warning(f"reflection lesson store failed: {e}")
        if stored:
            self.logger.info(f"lessons {stored} (from reflection)")

    def _generate_and_train(self):
        """生成训练数据并微调模型(placeholder)"""
        self.logger.info("training data")
        # TODO: 实现TrainingDataGenerator逻辑
        # TODO: 调用训练脚本
        pass

    # ============ 推理接口 ============

    def chat(self, task: str, **kwargs) -> str:
        """
        简化的单轮接口:输入任务,返回结果字符串。
        优先返回 trajectory.final_answer(反思/整理后的最终回答),否则回落到最后一次观察。
        """
        result = self.run(task, **kwargs)

        # 1) 优先:完整推理路径在 trajectory.final_answer 中给出了整理后的自然语言回答
        fa = result.get('final_answer') or None
        if isinstance(fa, str) and fa.strip():
            return fa.strip()

        # 2) 其次:最后一个 observation 的 output 字段
        if result.get('observations'):
            last = result['observations'][-1]
            if isinstance(last, dict):
                out = last.get('output')
                if isinstance(out, str) and out.strip():
                    return out.strip()
            elif isinstance(last, (str, int, float)):
                return str(last).strip()

        # 3) 兜底:result['answer'](simple 路径使用)或提示无结果
        ans = result.get('answer')
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
        return "任务未完成，未获得结果。"
