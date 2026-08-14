"""
Configuration loading and validation.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class ToolConfig(BaseModel):
    """Tool configuration supporting both legacy list and per-tool dict format."""

    enabled: list = Field(default_factory=list)

    web_search: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "provider": "duckduckgo",
        "max_results": 5,
        "providers": ["duckduckgo", "360", "bing", "google"],
        "quality_threshold": 0.3,
        "cache_ttl": 300,
        "use_playwright": False,
        "sequential_fallback": False,
        "max_fetch_urls": 3,
        "domain_trust": {
            "wikipedia.org": 1.0,
            "github.com": 0.95,
            "zhihu.com": 0.85,
            "baike.baidu.com": 0.8,
            "news.sina.com.cn": 0.8,
            "news.qq.com": 0.8,
            "36kr.com": 0.75,
            "techcrunch.com": 0.75,
        },
    })
    file_ops: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False, "allowed_dirs": ["./data", "./workspace"], "max_file_size": 1048576
    })
    code_exec: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False, "timeout": 10
    })
    api_call: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False, "allowed_hosts": [], "timeout": 30
    })
    bash_exec: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True, "timeout": 120, "max_timeout": 600, "default_cwd": ""
    })
    search_files: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True
    })
    glob: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True
    })
    project_context: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True, "max_depth": 3, "max_files": 100
    })
    browser: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    git: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    database: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    telegram: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "bot_token": None,
        "allowed_user_ids": [],
        "polling": True,
        "webhook_url": "",
        "config_path": "./data/telegram"
    })

    _NAME_MAP = {"code_exec": "python_exec"}

    @property
    def enabled_tools(self) -> List[str]:
        """Return normalized list of enabled internal tool names."""
        enabled = set(self.enabled)

        per_tool = {
            "web_search": self.web_search,
            "file_ops": self.file_ops,
            "code_exec": self.code_exec,
            "api_call": self.api_call,
            "bash_exec": self.bash_exec,
            "search_files": self.search_files,
            "glob": self.glob,
            "project_context": self.project_context,
            "browser": self.browser,
            "git": self.git,
            "database": self.database,
            "telegram": self.telegram,
        }

        for name, cfg in per_tool.items():
            if not isinstance(cfg, dict):
                continue
            is_enabled = cfg.get("enabled") is True
            # file_ops is considered enabled if allowed_dirs is configured
            if name == "file_ops" and cfg.get("allowed_dirs"):
                is_enabled = True
            if is_enabled:
                enabled.add(self._NAME_MAP.get(name, name))

        return list(enabled)


class ReflectionConfig(BaseModel):
    enabled: bool = True
    frequency: int = 5
    min_failures_threshold: int = 3
    thinking_loops_for_reflection: int = 16
    save_reflections: bool = True
    reflections_path: str = "./data/reflections"


class SelfImprovementConfig(BaseModel):
    enabled: bool = True
    auto_training: bool = False
    training: Dict[str, Any] = Field(default_factory=lambda: {
        "sft_epochs": 1, "learning_rate": 1e-5,
        "batch_size": 2, "gradient_accumulation_steps": 4
    })


class ExperienceConfig(BaseModel):
    storage_type: str = "chromadb"
    vector_db_path: str = "./data/experience_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_episodes: int = 10000
    auto_save: bool = True
    save_interval: int = 10


class StrategyConfig(BaseModel):
    enabled: bool = True
    db_path: str = "./data/strategies"
    min_success_rate: float = 0.7
    max_strategies_per_type: int = 50


class PlanningConfig(BaseModel):
    enabled: bool = True
    default_strategy: str = "adaptive"
    optimize_plans: bool = True
    max_subtasks: int = 10
    mcts_iterations: int = 100
    mcts_max_depth: int = 20
    mcts_exploration: float = 1.41


class ReasoningConfig(BaseModel):
    enabled: bool = True
    default_strategy: str = "react"
    multi_strategy_voting: bool = False
    strategies_to_try: List[str] = Field(default_factory=lambda: ["react", "chain_of_thought"])
    loop_controller_min_loops: int = 2
    loop_controller_max_loops: int = 16
    loop_controller_default_loops: int = 2


class MemoryConfig(BaseModel):
    enabled: bool = True
    kg_storage_path: str = "./data/kg_store"
    episodic_storage_path: str = "./data/episodic_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_episodes: int = 10000
    auto_extract_entities: bool = True
    context_compression: bool = True
    compression_max_tokens: int = 512
    file_memory_path: str = "./data/memory.md"
    user_memory_path: str = "./data/user.md"
    sqlite_session_path: str = "./data/sessions.db"
    importance_threshold: float = 0.45  # 记忆整合: 低于此重要性的交互不进长期记忆
    max_facts_per_turn: int = 5         # 每轮最多提取写入长期记忆的关键事实数


class SelfCorrectionConfig(BaseModel):
    enabled: bool = True
    low_confidence_threshold: float = 0.6
    high_error_threshold: float = 0.3
    inefficiency_threshold: float = 0.4
    intervention_window: int = 10
    auto_retraining_threshold: float = 0.6


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./logs/agent.log"
    # console 默认关闭: 日志走文件(logs/agent.log), 避免 warn/confidence 等
    # 调试信息打到 stderr 污染交互式 UI 界面。
    console: bool = False
    rich_markup: bool = True
    output_mode: str = "auto"  # "auto" | "rich" | "plain" | "json"


class HealthConfig(BaseModel):
    print_status_on_startup: bool = True
    show_install_hints: bool = True


class ExecutionConfig(BaseModel):
    use_legacy: bool = False  # fallback to old _run_traditional implementation
    default_strategy: str = "react"  # react | super_agent | chain_of_thought | verification | direct


class ResearchConfig(BaseModel):
    """Deep research mode configuration: very broad search + deep reasoning synthesis."""

    enabled: bool = True
    # Search breadth
    max_search_results_per_query: int = 30      # per provider per query
    max_search_queries: int = 12                # auto-generated query variants
    max_total_search_results: int = 200         # after deduplication/scoring
    max_urls_to_fetch: int = 50                 # webpage bodies to fetch
    max_sources_for_report: int = 50            # sources fed into final report
    # Reasoning depth
    thinking_strategy: str = "super_agent"      # react | chain_of_thought | super_agent | verification
    max_thinking_steps: int = 32                # ExecutionEngine max steps
    verification_rounds: int = 2                # verification passes on final report
    report_max_tokens: int = 8192               # report generation token budget
    report_formats: List[str] = Field(default_factory=lambda: ["md", "html"])  # 报告输出格式
    # Iterative deepening
    iterative_rounds: int = 4                   # search -> synthesize -> search again
    enable_follow_up_search: bool = True        # generate follow-up queries from gaps
    # Evidence & confidence engine
    min_new_sources_per_round: int = 3          # 每轮新增源低于此值且无新补搜时提前停止
    max_followup_queries: int = 4               # 每轮补搜查询上限
    require_citations: bool = True              # 报告要求关键论断带 [n] 引用标注
    max_sources_for_gap_analysis: int = 20      # gap 分析喂入的证据条数上限
    max_claims_to_verify: int = 8               # 关键论断抽取/核验上限
    min_support_for_high_confidence: int = 3    # 论断判"高置信"所需的最少独立来源数
    # Latent-space thinking (future: wire to OpenMythosBackend latent layers)
    enable_latent_space: bool = False


class HarnessConfig(BaseModel):
    """Opt-in wiring of the event-sourced harness into the legacy agent.

    Enabled by default; disabling routes tool calls around the capability
    kernel (policy + audit) without changing execution semantics.
    """

    enabled: bool = True
    policy: str = "safe"            # "safe" | "permissive"
    workspace_root: Optional[str] = None   # defaults to the project directory
    audit_log: bool = True          # record admissions to the agent logger
    allowlist_path: str = "./data/harness_allowlist.txt"
    prompt_injection_scan: bool = True
    # --- harness 运行时增强(智能/准确/高效) ---
    max_turns: int = 12             # 回合上限(预算熔断)
    max_seconds: Optional[float] = 600.0
    max_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_model_retries: int = 3      # 模型调用失败重试次数
    verify_final_answer: bool = True   # 最终答案 LLM 核验(准确)
    max_verification_rounds: int = 2   # 核验-修正轮数
    converge_on_stable: bool = True    # 连续两次核验结果一致则提前收敛(高效)


class ModelConfig(BaseModel):
    use_local: bool = True
    model_path: Optional[str] = None
    attention_type: str = "mla"
    dim: int = 256
    n_heads: int = 8
    max_loop_iters: int = 16
    n_experts: int = 8
    n_shared_experts: int = 1
    n_experts_per_tok: int = 2
    expert_dim: int = 64
    vocab_size: int = 32000


class AgentConfig(BaseModel):
    backend: str = Field(default="deepseek", description="Backend: openai, deepseek, openrouter, anthropic, openmythos")
    # 新增：原生官方模型注册表路径（支持 openai / anthropic 格式）
    model_registry_path: Optional[str] = Field(default="agent_project/config/models.yaml",
        description="Path to official native model registry (openai/anthropic format)")

    openai: Dict[str, Any] = Field(default_factory=lambda: {
        "api_key": None,
        "base_url": None,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "timeout": 120
    })

    deepseek: Dict[str, Any] = Field(default_factory=lambda: {
        "api_key": None,
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "timeout": 120
    })

    openrouter: Dict[str, Any] = Field(default_factory=lambda: {
        "api_key": None,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "timeout": 120
    })

    anthropic: Dict[str, Any] = Field(default_factory=lambda: {
        "api_key": None,
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 120
    })

    openmythos: Dict[str, Any] = Field(default_factory=lambda: {
        "model_path": None,
        "device": "cpu",
        "dim": 256,
        "n_heads": 8,
        "n_kv_heads": 2,
        "max_seq_len": 4096,
        "max_loops": 8,
        "prelude_layers": 2,
        "coda_layers": 2,
        "attn_type": "gqa",
        "n_experts": 8,
        "n_shared_experts": 1,
        "n_experts_per_tok": 2,
        "expert_dim": 64,
        "lora_rank": 8,
        "act_threshold": 0.95,
        "rope_theta": 500000.0,
        "dropout": 0.0
    })

    model: ModelConfig = ModelConfig()
    max_outer_loops: int = 1
    default_thinking_loops: int = 4
    max_thinking_loops: int = 16
    temperature: float = 0.7
    top_k: int = 50
    act_threshold: float = 0.7
    fast_mode: bool = True
    reflection: ReflectionConfig = ReflectionConfig()
    self_improvement: SelfImprovementConfig = SelfImprovementConfig()
    tools: ToolConfig = ToolConfig()
    experience: ExperienceConfig = ExperienceConfig()
    strategies: StrategyConfig = StrategyConfig()
    logging: LoggingConfig = LoggingConfig()
    planning: PlanningConfig = PlanningConfig()
    reasoning: ReasoningConfig = ReasoningConfig()
    memory: MemoryConfig = MemoryConfig()
    self_correction: SelfCorrectionConfig = SelfCorrectionConfig()
    harness: HarnessConfig = HarnessConfig()
    health: HealthConfig = HealthConfig()
    execution: ExecutionConfig = ExecutionConfig()
    research: ResearchConfig = ResearchConfig()
    mcp: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "servers": {}})
    display: Dict[str, Any] = Field(
        default_factory=dict,
        description="UI/CLI display and theme settings",
    )


def load_model_registry(path: str = "agent_project/config/models.yaml") -> Dict[str, Any]:
    """Load official native model registry (openai / anthropic formats)."""
    p = Path(path)
    if p.exists():
        with open(p, 'r') as f:
            return yaml.safe_load(f) or {}
    return {"models": {}}


def get_model_for_format(format_name: str, provider_id: str = "deepseek") -> Dict[str, Any]:
    """Retrieve model info by format + provider for clear selection flow."""
    registry = load_model_registry()
    fmt_key = format_name if format_name in registry.get("models", {}) else "openai_native"
    providers = registry.get("models", {}).get(fmt_key, {}).get("providers", {})
    provider = providers.get(provider_id, {})
    models = provider.get("model_list", [])
    if not models:
        return {
            "id": "default",
            "name": f"{provider_id} (default)",
            "model_name": provider_id,
            "format": format_name,
            "base_url": provider.get("base_url", ""),
            "description": "No model registered — please check models.yaml"
        }
    # Return first (default) model; user can override via config
    first = models[0]
    return {
        "id": first.get("id", provider_id),
        "name": first.get("name", provider_id),
        "model_name": first.get("model_name", provider_id),
        "format": format_name,
        "base_url": provider.get("base_url", ""),
        "api_key_env": provider.get("api_key_env", f"{provider_id.upper()}_API_KEY"),
        "max_tokens": first.get("max_tokens", 4096),
        "description": first.get("description", ""),
        "recommended_for": first.get("recommended_for", []),
    }


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively replace ${VAR} or ${VAR:-default} with environment values."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(v) for v in obj]
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

        def repl(match):
            var_name = match.group(1)
            default = match.group(2)
            return os.getenv(var_name, default if default is not None else match.group(0))

        return pattern.sub(repl, obj)
    return obj


def load_config(config_path: str = "config.yaml") -> AgentConfig:
    """Load config from YAML (supports nested structure and ${ENV_VAR} substitution)."""
    path = Path(config_path)

    if not path.exists():
        default_config = AgentConfig()
        config_dict = default_config.model_dump()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"\033[2m Created default config at {path}.\033[0m")
        return default_config

    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    raw_config = _substitute_env_vars(raw_config)

    agent_config = raw_config.get('agent', {})

    top_level = {
        'experience': raw_config.get('experience', {}),
        'strategies': raw_config.get('strategies', {}),
        'logging': raw_config.get('logging', {}),
        'tools': raw_config.get('tools', {}),
        'mcp': raw_config.get('mcp', {}),
        'display': raw_config.get('display', {}),
    }

    merged = {**agent_config, **top_level}
    # 兼容两种布局: 顶层 tools/logging/mcp/display 若为空 dict, 不覆盖 agent 内部已配置的同名段
    for key in ('tools', 'logging', 'mcp', 'experience', 'strategies', 'display'):
        if not raw_config.get(key) and key in agent_config and agent_config.get(key):
            merged[key] = agent_config[key]

    backend = merged.get('backend', 'deepseek')
    # Ensure backend-specific config section exists to avoid AttributeError
    backend_sections = ['openai', 'openmythos', 'deepseek', 'anthropic', 'openrouter']
    for section in backend_sections:
        if backend == section and section not in merged:
            merged[section] = {}

    try:
        return AgentConfig(**merged)
    except ValidationError as e:
        print(f"\033[2m Config validation error:\n{e}\033[0m")
        raise
