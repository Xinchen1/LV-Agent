<div align="center">

<img src="assets/portrait.png" width="240" height="240" alt="Lv Agent" />

# Lv Agent

**Lux Vita · 光与生命**

**by cleveris research**

[**cleveris-research.pages.dev**](https://cleveris-research.pages.dev/)

*Deep thinking, real tools. Event-sourced · multi-strategy · self-learning*

**Capabilities**<br>
Multi-turn deep reasoning with 8 strategies<br>
Event-sourced harness with policy enforcement<br>
5-layer memory system with self-learning<br>
Real tools for code, files, web, and research

**Mission**<br>
Open-source AI at the core<br>
AI that is easier for everyone to use<br>
A small step toward intelligence equality

以开源人工智能为核心，希望让更多人能方便地用上人工智能，为智能平权尽一份力

</div>

---

LV Agent 是一个**终端原生的深度思考 AI 智能体**。它采用事件溯源微内核架构（Harness），结合循环深度推理、真实工具调用、5 层记忆系统与自我学习，在终端里提供多轮推理与工具协作的交互体验。项目仍在持续改进中，期待与大家一起成长。

## Features

**Reasoning & Planning**
- **8 reasoning strategies**: CoT, ReAct, Self-Consistency, Tree-of-Thoughts, MCTS, Verification, Zero-shot, Super-Agent
- **Execution Engine**: strategy-agnostic think → act → observe loop with pluggable policies
- **Adaptive loop control**: adjusts thinking depth by task complexity (2–16 loops)
- **Self-correction**: real-time quality monitoring + automatic parameter tuning
- **Reflection**: meta-cognitive analysis of agent performance after each session

**Harness Microkernel**
- **Event-sourced journal**: append-only JSON Lines log, every state change is an immutable event
- **Policy kernel**: rule-based allow/deny/ask admission for tool effects
- **Lane scheduler**: parallel / write-serialized / global-serial execution with idempotency dedup
- **Budget ledger**: tracks tokens, wall-clock time, tool calls, and dollars with circuit breakers
- **Context assembler**: token-budget-aware message list with head/tail preservation
- **Checkpointing**: snapshot-based undo trail for mutating effects
- **Prompt guard**: rule-based injection detection (jailbreak, fence-break, tool arg injection)
- **Interactive approval**: human-in-the-loop for high-risk operations

**Memory System (5 layers)**
- **Knowledge Graph**: entity/concept/episode nodes with typed edges, embedding-based similarity search
- **Wiki Memory**: Karpathy-style page-first graph — every entity is a wiki page with LLM-generated summaries
- **SQLite Session Memory**: FTS5 full-text search over conversation history
- **File Memory**: human-readable `MEMORY.md` / `USER.md` for project and user preferences
- **Experience Buffer**: trajectory storage with vector-based retrieval, lesson extraction, and strategy learning

**Tools**
- **Web search & fusion**: multi-source search with scoring/reranking
- **File ops**: read / write / list / grep / analyze with safety policies
- **Code execution**: python_exec (sandboxed globals) / bash_exec (subprocess) with harness gating
- **Deep research**: parallel web search → article fetch → LLM report generation → HTML output
- **Git ops**: clone / commit / push / pull
- **Browser**: Playwright automation, web page fetching
- **Database & Telegram**: structured DB queries, Telegram bot mode
- **MCP integration**: Model Context Protocol client for external tool servers
- **Weather / Calculator / URL unfurl / GitHub search / PDF tool**

**Terminal UI**
- **Rich theme engine**: light / dark / minimal themes with ANSI SGR token system
- **Status bar**: context usage bar with 4-color thresholds, width-adaptive layout
- **Tool cards**: four-state cards (pending → running → success/error) with foldable output
- **Braille portrait**: PNG → braille pixel art via PIL with Otsu thresholding
- **Deep-research live panel**: real-time progress tracking with source/query/token counters

**Backends**
- DeepSeek, OpenAI (NVIDIA NIM), Anthropic, OpenRouter, Ollama
- Streaming with native reasoning-content display
- Rate-limit-aware retry with exponential backoff

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/LV-Agent.git
cd LV-Agent
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -e .
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml to set your backend and API key
```

Or set environment variables:

```bash
export DEEPSEEK_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
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
  backend: "openai"   # deepseek | openai | openrouter | anthropic | ollama
  openai:
    api_key: "${OPENAI_API_KEY:-}"
    base_url: "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM
    model: "nvidia/nemotron-3-super-120b-a12b"
  deepseek:
    api_key: "${DEEPSEEK_API_KEY:-}"
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"
```

See `config.yaml` for all options.

## Architecture

```
User Input → CLIApp (ui/app.py)
  ├─ Slash Commands ( /model /tools /theme ... )
  ├─ Shell Mode ( !command )
  ├─ File Ref ( @file )
  └─ Agent (agent.py)
       ├─ Harness Session (harness/session.py)
       │    ├─ AgentLoop (loop.py)          — state machine: THINKING → EXECUTING → OBSERVING
       │    ├─ Kernel (kernel.py)           — policy rules: ALLOW / DENY / ASK
       │    ├─ Scheduler (scheduler.py)     — async lanes: parallel / write-serialized / serial
       │    ├─ Journal (journal.py)         — append-only event log (crash-safe)
       │    ├─ Ledger (budget.py)           — resource tracking: tokens, time, calls, dollars
       │    ├─ EventBus (stream.py)         — process-wide event fan-out
       │    ├─ ContextAssembler (context.py)— token-budget-aware message assembly
       │    ├─ CheckpointMiddleware          — snapshot-based undo for mutations
       │    └─ PromptGuard                  — injection detection
       ├─ Model Backend (model_backends.py) — DeepSeek / OpenAI / Anthropic / Ollama
       ├─ Reasoning (reasoning.py)          — 8 strategies with ThoughtStep tracing
       ├─ ExecutionEngine (execution_engine.py) — strategy-agnostic think→act→observe
       ├─ Memory (5 layers)
       │    ├─ Knowledge Graph (memory.py)
       │    ├─ Wiki Memory (wiki_memory.py)
       │    ├─ SQLite Session (sqlite_memory.py)
       │    ├─ File Memory (file_memory.py)
       │    └─ Experience Buffer (experience.py)
       ├─ Reflection (reflection.py)        — meta-cognitive performance analysis
       ├─ Self-Correction (self_correction.py) — quality monitoring & adaptation
       ├─ Skills (skills.py)                — .skill.md dynamic loading
       ├─ ContextEngine (context_engine.py) — 5-layer retrieval facade
       └─ StreamAdapter (stream_adapters.py)— Rich / Plain / JSON output modes
```

## Project Structure

```
agent_project/
├── super_agent.py              # Interactive CLI entry point
├── agent_project/              # Core package
│   ├── agent.py                # Main Agent — orchestrates all subsystems
│   ├── harness/                # Event-sourced microkernel
│   │   ├── loop.py             #   State machine (THINKING→EXECUTING→OBSERVING)
│   │   ├── kernel.py           #   Policy rules (ALLOW/DENY/ASK)
│   │   ├── scheduler.py        #   Async lane scheduler with dedup cache
│   │   ├── journal.py          #   Append-only event log
│   │   ├── events.py           #   Event taxonomy (15+ event types)
│   │   ├── effects.py          #   Tool intent declarations
│   │   ├── budget.py           #   Resource tracking + circuit breakers
│   │   ├── context.py          #   Token-budget-aware message assembly
│   │   ├── bridge.py           #   Legacy backend → harness adapter
│   │   ├── runner.py           #   Legacy-compatible entry point
│   │   ├── session.py          #   Facade: journal + loop + budget + bus
│   │   ├── checkpointing.py    #   Snapshot-based undo trail
│   │   ├── approval.py         #   Human-in-the-loop for ASK decisions
│   │   ├── prompt_guard.py     #   Injection detection
│   │   ├── renderers.py        #   Event → legacy callback adapter
│   │   ├── stream.py           #   EventBus fan-out
│   │   └── errors.py           #   Typed error taxonomy
│   ├── model_backends.py       # LLM backends (OpenAI/DeepSeek/Anthropic/Ollama)
│   ├── stream_adapters.py      # Terminal rendering (Rich/Plain/JSON)
│   ├── reasoning.py            # 8 reasoning strategies
│   ├── execution_engine.py     # Strategy-agnostic think→act→observe loop
│   ├── policies.py             # Thinking policies (CoT/ReAct/SuperAgent/Verify)
│   ├── planning.py             # Knowledge graph memory
│   ├── memory.py               # Knowledge graph (nodes + edges + embeddings)
│   ├── wiki_memory.py          # Karpathy-style page-first graph memory
│   ├── sqlite_memory.py        # SQLite session memory with FTS5
│   ├── file_memory.py          # MEMORY.md / USER.md file-based memory
│   ├── experience.py           # Experience buffer with vector retrieval
│   ├── strategies.py           # Strategy extraction from experiences
│   ├── reflection.py           # Meta-cognitive reflection module
│   ├── self_correction.py      # Quality monitoring & auto-correction
│   ├── context_engine.py       # 5-layer memory retrieval facade
│   ├── config.py               # YAML config with env var override
│   ├── skills.py               # .skill.md dynamic skill loading
│   ├── policies.py             # Tool call parsing & thinking policies
│   ├── research_report.py      # Deep research pipeline
│   ├── tools/                  # Tool ecosystem (26 tools)
│   │   ├── python_exec.py      #   Sandboxed Python execution
│   │   ├── bash_exec.py        #   Shell command execution
│   │   ├── file_ops.py         #   File read/write/list/grep
│   │   ├── web_search.py       #   Web search
│   │   ├── web_fetcher.py      #   URL content fetching
│   │   ├── search_fusion.py    #   Multi-source search fusion
│   │   ├── git_ops.py          #   Git operations
│   │   ├── mcp_client.py       #   MCP protocol client
│   │   ├── telegram_bot.py     #   Telegram bot mode
│   │   └── ...                 #   +17 more tools
│   └── ui/                     # Terminal UI
│       ├── app.py              #   CLIApp — main frontend
│       ├── renderer.py         #   ANSI output engine
│       ├── themes.py           #   Light / Dark / Minimal themes
│       ├── banner.py           #   Braille portrait + brand text
│       ├── status_bar.py       #   Context usage bar
│       └── cards.py            #   Tool execution cards
├── tests/                      # 158 tests
├── assets/portrait.png         # The boy portrait
├── config.yaml                 # Active configuration
├── config.example.yaml         # Example config template
└── pyproject.toml              # Package metadata
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
