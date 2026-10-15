""" OpenMythos Agent - 深度思考智能体 利用OpenMythos的循环架构实现深度推理和自我改进

PEP 562 lazy exports: `import agent_project` 本体近零成本, 只在真正访问属性时
才加载对应子模块(旧版 eager 导入会把 agent+tools+wiki 全链 ~300ms 压在首帧之前,
UI 还没画 banner 就先等 import)。`from agent_project import X` 行为不变。
"""

from __future__ import annotations

import importlib
from typing import Dict

__version__ = "0.2.0"

# 属性名 -> 子模块(相对包名)
_LAZY_ATTRS: Dict[str, str] = {
    # core
    "OpenMythosAgent": ".agent",
    # tools
    "BaseTool": ".tools",
    "WebSearchTool": ".tools",
    "CalculatorTool": ".tools",
    "PythonExecTool": ".tools",
    "FileOpsTool": ".tools",
    "ApiCallTool": ".tools",
    "TOOLS_REGISTRY": ".tools",
    # experience / reflection / strategies
    "ExperienceBuffer": ".experience",
    "Experience": ".experience",
    "ReflectionModule": ".reflection",
    "StrategyDatabase": ".strategies",
    "Strategy": ".strategies",
    # config
    "AgentConfig": ".config",
    "load_config": ".config",
    # memory
    "MemoryManager": ".memory",
    "create_memory_manager": ".wiki_memory",
    # wiki memory
    "WikiPage": ".wiki_memory",
    "WikiGraph": ".wiki_memory",
    "LLMWikiManager": ".wiki_memory",
    "LLMClient": ".wiki_memory",
    "OpenAICompatibleClient": ".wiki_memory",
    "PassthroughBackendClient": ".wiki_memory",
    "llm_generate_page": ".wiki_memory",
    "llm_infer_relation": ".wiki_memory",
    "llm_merge_page": ".wiki_memory",
    "llm_extract_entities": ".wiki_memory",
    "quick_store": ".wiki_memory",
    # fast path
    "FastPathRouter": ".fast_path",
    "LocationFastPath": ".fast_path",
    "SimpleQueryClassifier": ".fast_path",
    "is_simple_query": ".fast_path",
}

__all__ = sorted(
    list(_LAZY_ATTRS)
    + ["__version__"]
)


def __getattr__(name: str):
    mod_name = _LAZY_ATTRS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # create_memory_manager 同时存在于 .memory(legacy) 与 .wiki_memory,
    # 按 __all__ 语义取 wiki_memory 版本(与旧 eager 导入一致)。
    module = importlib.import_module(mod_name, __name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value  # 缓存, 后续零开销
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
