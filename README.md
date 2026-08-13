<div align="center">

<img src="assets/portrait.png" width="220" height="220" alt="Lv Agent" />

# Lv Agent

**Lux Vita · 光与生命**

**by cleveris research**

[**cleveris-research.pages.dev**](https://cleveris-research.pages.dev/)

*Deep thinking, real tools. Recurrent reasoning · tool-driven · self-learning*

**Capabilities**
- Multi-turn deep reasoning for complex problems
- Real tools for live information
- Continuous self-learning, smarter over time

**Mission**
- Open-source AI at the core
- Great AI for everyone
- Intelligence for all

以开源人工智能为核心，让每一个人都能用上最好的人工智能，实现智能平权

</div>

---

LV Agent 是一个**终端原生的深度思考 AI 智能体**。它结合了循环深度推理、真实工具调用、长期记忆与自我学习，目标是让复杂的任务在终端里获得接近 Cursor CLI、Hermes Agent 与 Grok Build 的交互体验。

## Features

- **Multi-strategy reasoning**: CoT, ReAct, Self-Consistency, Verification, Zero-shot
- **Adaptive loop control**: adjusts thinking depth by task complexity
- **Task planning**: MCTS, graph-based planning, key-path analysis
- **Self-correction**: quality monitoring + automatic parameter tuning
- **Memory system**: knowledge graph, episodic memory, context compression
- **Tool ecosystem**: web search, file ops, code exec, bash, git, browser, database, telegram
- **Terminal-native UI**: status bar with context bar, tool cards, themes, deep-research live panel
- **Multi-backend**: NVIDIA NIM, OpenAI, DeepSeek, Anthropic, OpenRouter, Ollama

## Quick Start

### 1. Install

```bash
cd agent_project
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -e .
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
# or set env vars
export NIM_API_KEY="your_key_here"
```

### 3. Run

```bash
python super_agent.py        # interactive CLI (recommended)
# or
python -m agent_project --task "Your task"
```

## Usage

- **Interactive**: type tasks directly; `!command` runs shell; `@file` attaches file content
- **Deep research**: `深度研究 <topic>` or `/deep <topic>` opens a live progress panel and auto-opens the HTML report
- **Commands**: `/model`, `/theme`, `/tools`, `/plan`, `/sessions`, `/dashboard`, `/drafts`, `/compress`, `/help`
- **Shortcuts**: `Ctrl+S` draft, `Ctrl+\` dashboard, `ESC` interrupt

## Backend Configuration

```yaml
agent:
  backend: "nim"   # nim | openai | deepseek | anthropic | openrouter | ollama
  nim:
    api_key: "${NIM_API_KEY:-}"
    base_url: "https://integrate.api.nvidia.com/v1"
    model: "nvidia/nemotron-3-super-120b-a12b"
```

See `config.example.yaml` for all options.

## Architecture

```
User Input → Input Handler → Command Parser
  ├─ Slash Commands ( /model /tools ... )
  ├─ Shell Mode ( !command )
  ├─ File Ref ( @file )
  └─ Agent Loop ( Perceive → Think → Act → Observe )
       └─ Tool Execution → Renderer → ANSI Terminal
```

## Project Structure

```
agent_project/
├── super_agent.py          # Interactive CLI entry
├── agent_project/          # Core package
│   ├── agent.py            # Main Agent
│   ├── reasoning.py        # Reasoning engine
│   ├── planning.py         # Task planner
│   ├── memory.py           # Knowledge graph + memory
│   ├── self_correction.py  # Self-correction
│   ├── model_backends.py   # NIM/OpenAI/DeepSeek/Anthropic backends
│   ├── stream_adapters.py  # Terminal rendering
│   ├── ui/                 # Terminal UI (theme/status bar/cards)
│   └── tools/              # Tool ecosystem
├── tests/                  # Test suite
├── assets/portrait.png     # The boy portrait
└── config.example.yaml     # Example config
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE)

---

**Lv Agent · Lux Vita · by cleveris research**

*Open-source AI at the core — great AI for everyone, intelligence for all*
