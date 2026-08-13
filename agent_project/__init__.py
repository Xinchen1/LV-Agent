""" OpenMythos Agent - 深度思考智能体 利用OpenMythos的循环架构实现深度推理和自我改进
"""

from .agent import OpenMythosAgent
from .tools import (
BaseTool,
WebSearchTool,
CalculatorTool,
PythonExecTool,
FileOpsTool,
ApiCallTool,
TOOLS_REGISTRY,
)
from .experience import ExperienceBuffer, Experience
from .reflection import ReflectionModule
from .strategies import StrategyDatabase, Strategy
from .config import AgentConfig, load_config
from .memory import MemoryManager, create_memory_manager as _create_legacy_mm   # noqa: F401
from .wiki_memory import (   # noqa: F401
WikiPage,
WikiGraph,
LLMWikiManager,
LLMClient,
OpenAICompatibleClient,
PassthroughBackendClient,
llm_generate_page,
llm_infer_relation,
llm_merge_page,
llm_extract_entities,
create_memory_manager,
quick_store,
)

__all__ = [
"OpenMythosAgent",
"BaseTool",
"WebSearchTool",
"CalculatorTool",
"PythonExecTool",
"FileOpsTool",
"ApiCallTool",
"TOOLS_REGISTRY",
"ExperienceBuffer",
"Experience",
"ReflectionModule",
"StrategyDatabase",
"Strategy",
"AgentConfig",
"load_config",
# wiki memory
"MemoryManager",
"WikiPage",
"WikiGraph",
"LLMWikiManager",
"LLMClient",
"OpenAICompatibleClient",
"PassthroughBackendClient",
"llm_generate_page",
"llm_infer_relation",
"llm_merge_page",
"llm_extract_entities",
"create_memory_manager",
"quick_store",
]

__version__ = "0.2.0"
