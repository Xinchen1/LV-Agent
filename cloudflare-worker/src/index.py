"""Cloudflare Worker that runs the LV Hermes Agent.

Exposes a POST /run endpoint that accepts:
  {"task": "your task here"}

Returns the agent's response after one turn of processing.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Add the agent project to path so we can import it
sys.path.insert(0, os.path.dirname(__file__))

from agent_project.agent import OpenMythosAgent
from agent_project.config import AgentConfig


def get_agent() -> OpenMythosAgent:
    """Lazy-load and return the agent instance using only environment variables."""
    # Read model configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

    # Build agent config from environment variables only
    agent_cfg_dict = {
        "model_registry_path": "agent_project/config/models.yaml",
        "openai": {
            "api_key": api_key or "ollama",
            "base_url": base_url or "http://localhost:11434/v1",
            "model": model or "nvidia/nemotron-3.5-lightning-30b-a3b",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "timeout": 120,
        },
        "deepseek": {
            "api_key": None,
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "timeout": 120,
        },
        "openrouter": {
            "api_key": None,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "anthropic/claude-sonnet-4",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "timeout": 120,
        },
        "anthropic": {
            "api_key": None,
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "timeout": 120,
        },
        "openmythos": {
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
            "dropout": 0.0,
        },
        "model": {
            "use_local": True,
            "model_path": None,
            "attention_type": "mla",
            "dim": 256,
            "n_heads": 8,
            "max_loop_iters": 16,
            "n_experts": 8,
            "n_shared_experts": 1,
            "n_experts_per_tok": 2,
            "expert_dim": 64,
            "vocab_size": 32000,
        },
        "max_outer_loops": 1,
        "default_thinking_loops": 2,
        "max_thinking_loops": 3,
        "temperature": 0.7,
        "top_k": 50,
        "act_threshold": 0.7,
        "fast_mode": True,
        "reflection": {
            "enabled": True,
            "frequency": 5,
            "min_failures_threshold": 3,
            "thinking_loops_for_reflection": 16,
            "save_reflections": True,
            "reflections_path": "./data/reflections",
        },
        "self_improvement": {
            "enabled": True,
            "auto_training": False,
            "training": {
                "sft_epochs": 1,
                "learning_rate": 1.0e-05,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
            },
        },
        "tools": {
            "enabled": [],
            "web_search": {
                "enabled": True,
                "provider": "duckduckgo",
                "max_results": 5,
                "providers": ["duckduckgo", "360"],
                "quality_threshold": 0.3,
                "cache_ttl": 300,
                "use_playwright": False,
                "sequential_fallback": True,
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
            },
            "file_ops": {"enabled": False, "allowed_dirs": ["./data", "./workspace"], "max_file_size": 1048576},
            "code_exec": {"enabled": False, "timeout": 10},
            "api_call": {"enabled": False, "allowed_hosts": [], "timeout": 30},
            "bash_exec": {"enabled": True, "timeout": 120, "max_timeout": 600, "default_cwd": ""},
            "search_files": {"enabled": True},
            "glob": {"enabled": True},
            "project_context": {"enabled": True, "max_depth": 3, "max_files": 100},
            "browser": {"enabled": False},
            "git": {"enabled": False},
            "database": {"enabled": False},
            "telegram": {"enabled": False, "bot_token": None, "allowed_user_ids": [], "polling": True, "webhook_url": "", "config_path": "./data/telegram"},
        },
        "experience": {
            "storage_type": "memory",
            "vector_db_path": "./data/experience_store",
            "max_episodes": 10000,
            "auto_save": True,
            "save_interval": 10,
        },
        "strategies": {"enabled": True, "db_path": "./data/strategies", "min_success_rate": 0.7, "max_strategies_per_type": 50},
        "logging": {
            "level": "INFO",
            "file": "./logs/agent.log",
            "console": False,
            "rich_markup": True,
            "output_mode": "auto",
            "min_loops": 2,
            "max_loops": 16,
            "default_loops": 4,
            "max_outer_loops": 12,
        },
        "thinking": {"min_loops": 2, "max_loops": 16, "default_loops": 4, "max_outer_loops": 12, "thinking_loops_for_reflection": 16},
        "planning": {"enabled": True, "default_strategy": "adaptive", "optimize_plans": True, "max_subtasks": 10},
        "reasoning": {"enabled": True, "default_strategy": "react", "loop_controller_min_loops": 2, "loop_controller_max_loops": 16, "loop_controller_default_loops": 2},
        "memory": {
            "enabled": True,
            "kg_storage_path": "./data/kg_store",
            "episodic_storage_path": "./data/episodic_store",
            "max_episodes": 10000,
            "auto_extract_entities": True,
            "context_compression": True,
            "compression_max_tokens": 512,
            "file_memory_path": "./data/memory.md",
            "user_memory_path": "./data/user.md",
            "sqlite_session_path": "./data/sessions.db",
            "importance_threshold": 0.45,
            "max_facts_per_turn": 5,
        },
        "self_correction": {
            "enabled": True,
            "low_confidence_threshold": 0.6,
            "high_error_threshold": 0.3,
            "inefficiency_threshold": 0.4,
            "intervention_window": 10,
            "auto_retraining_threshold": 0.6,
        },
        "harness": {
            "enabled": True,
            "policy": "safe",
            "workspace_root": None,
            "audit_log": True,
            "allowlist_path": "./data/harness_allowlist.txt",
            "prompt_injection_scan": True,
            "max_turns": 12,
            "max_seconds": 600.0,
            "max_tokens": None,
            "max_tool_calls": None,
            "max_model_retries": 3,
            "verify_final_answer": True,
            "max_verification_rounds": 2,
            "converge_on_stable": True,
        },
        "health": {
            "print_status_on_startup": True,
            "show_install_hints": True,
        },
        "execution": {"use_legacy": False, "default_strategy": "react"},
        "research": {
            "enabled": True,
            "max_search_results_per_query": 30,
            "max_search_queries": 12,
            "max_total_search_results": 200,
            "max_urls_to_fetch": 50,
            "max_sources_for_report": 50,
            "max_thinking_steps": 32,
            "verification_rounds": 2,
            "report_max_tokens": 8192,
            "report_formats": ["md", "html"],
            "iterative_rounds": 4,
            "enable_follow_up_search": True,
            "min_new_sources_per_round": 3,
            "max_followup_queries": 4,
            "require_citations": True,
            "max_sources_for_gap_analysis": 20,
            "max_claims_to_verify": 8,
            "min_support_for_high_confidence": 3,
            "enable_latent_space": False,
        },
        "mcp": {"enabled": True, "servers": {"filesystem": {"enabled": True, "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "${HOME}/Desktop"], "env": {}, "timeout": 60, "init_timeout": 120}}},
        "display": {},
        "agent": {
            "backend": "openai",
            "openai": {
                "api_key": api_key or "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",
                "base_url": base_url or "https://integrate.api.nvidia.com/v1",
                "model": model or "nvidia/nemotron-3.5-lightning-30b-a3b",
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 16384,
                "timeout": 120,
            },
        },
    }

    config = AgentConfig(**agent_cfg_dict)
    agent = OpenMythosAgent(config)
    return agent


class Response:
    """Response class for Cloudflare Workers."""

    def __init__(self, body, status=200, headers=None):
        self.body = body if isinstance(body, bytes) else body.encode()
        self.status = status
        self.headers = headers or {}
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"


class json:
    """Helper to create JSON responses."""

    def __init__(self, data, status=200, headers=None):
        self.body = json.dumps(data).encode()
        self.status = status
        self.headers = headers or {}
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"


async def on_fetch(request, env, ctx):
    """Entry point called by Cloudflare Workers runtime."""
    method = request.method
    path = request.url.path

    # Health check
    if path == "/" and method in ("GET", "HEAD"):
        return Response(
            json({"status": "ok", "message": "Hermes Agent is running"}),
            status=200,
        )

    # Run agent endpoint
    if path == "/run" and method == "POST":
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}

        task = body.get("task", "")

        if not task:
            return Response(
                json({"error": "Missing 'task' in request body"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        try:
            # Get agent instance
            agent = get_agent()

            # Run one turn of the agent
            result = agent.run(task)

            return Response(
                json({"result": result, "task": task}),
                status=200,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            import traceback
            return Response(
                json({"error": str(e), "trace": traceback.format_exc()}),
                status=500,
                headers={"Content-Type": "application/json"},
            )

    # Method not allowed
    return Response(
        json({"error": f"Method {method} not allowed"}),
        status=405,
        headers={"Content-Type": "application/json"},
    )