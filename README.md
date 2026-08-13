<div align="center">

<img src="assets/portrait.png" width="240" height="240" alt="Lv Agent" />

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
AI that is easier for everyone to use<br>
A small step toward intelligence equality

以开源人工智能为核心，希望让更多人能方便地用上人工智能，为智能平权尽一份力

</div>

---

LV Agent 是一个**终端原生的深度思考 AI 智能体**。它结合了循环深度推理、真实工具调用、长期记忆与自我学习，在终端里提供多轮推理与工具协作的交互体验。项目仍在持续改进中，期待与大家一起成长。

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
- DeepSeek, OpenAI, Anthropic, OpenRouter, Ollama
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
export DEEPSEEK_API_KEY="your_key_here"
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
  backend: "deepseek"   # deepseek | openai | openrouter | anthropic | ollama
  deepseek:
    api_key: "${DEEPSEEK_API_KEY:-}"
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"   # or "deepseek-reasoner"
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
│   ├── model_backends.py   # DeepSeek/OpenAI/Anthropic backends
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
