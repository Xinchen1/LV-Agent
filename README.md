<div align="center">

<img src="assets/portrait.png" width="220" height="220" alt="Lv Agent" />

# Lv Agent

**Lux Vita · 光与生命**

**by cleveris research**

[**cleveris-research.pages.dev**](https://cleveris-research.pages.dev/)

*Deep thinking, real tools. Recurrent reasoning · tool-driven · self-learning*

**Capabilities**<br>
Multi-turn deep reasoning for complex problems<br>
Real tools for live information<br>
Continuous self-learning, smarter over time

**Mission**<br>
Open-source AI at the core<br>
Great AI for everyone<br>
Intelligence for all

以开源人工智能为核心，让每一个人都能用上最好的人工智能，实现智能平权

</div>

---

LV Agent 是一个**终端原生的深度思考 AI 智能体**。它结合了循环深度推理、真实工具调用、长期记忆与自我学习，目标是让复杂的任务在终端里获得接近 Cursor CLI、Hermes Agent 与 Grok Build 的交互体验。

## Features

**Reasoning & Planning**
- **Multi-strategy reasoning**: CoT, ReAct, Self-Consistency, Verification, Zero-shot
- **Adaptive loop control**: adjusts thinking depth by task complexity
- **Task planning**: MCTS, graph-based planning, key-path analysis
- **Self-correction**: quality monitoring + automatic parameter tuning

**Memory & Learning**
- **Knowledge graph**: structured long-term memory
- **Episodic memory**: recalls past conversations across sessions
- **Self-learning**: learns from successes/failures and improves over time
- **Context compression**: keeps long sessions within token budget

**Tools**
- **Web search & fusion**: multi-source search with scoring/reranking
- **File ops**: read / write / list / grep / analyze
- **Code execution**: python / bash / file execution with sandbox
- **Git ops**: clone / commit / push / pull
- **Browser**: Playwright automation, web page fetching
- **Database & Telegram**: structured DB queries, Telegram bot mode
- **Weather / Calculator / URL unfurl / GitHub search**

**Terminal UI**
- **Status bar** with context bar & tool cards
- **Multiple themes**
- **Deep-research live panel** with progress tracking
- **Dashboard**: `/dashboard` Ctrl+`\`

**Backends**
- NVIDIA NIM, OpenAI, DeepSeek, Anthropic, OpenRouter, Ollama
- Streaming with native reasoning-content display

## Quick Start

### 1. Install

```bash
git clone https://github.com/Xinchen1/LV-Agent.git
cd LV-Agent
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
lv                      # global launcher: type "lv" to start from any directory
# or
python super_agent.py   # interactive CLI
# or
python -m agent_project --task "Your task"
```

> **Tip**: add the repo to your `PATH` (or symlink `lv` into `/usr/local/bin`) so you can just type `lv` anywhere to launch.

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
