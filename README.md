# OpenMythos Agent - 深度思考智能体

**基于NVIDIA Step3.5 API + OpenMythos循环架构的世界顶级Agent**

## 🌟 核心特性

### 推理能力
- ✅ **多策略推理引擎**：CoT, ReAct, Self-Consistency, Verification, Zero-Shot
- ✅ **自适应循环控制**：根据任务复杂度动态调节思考深度
- ✅ **任务规划系统**：MCTS、Graph-based规划、关键路径分析
- ✅ **自校正机制**：实时质量监控与自动参数调整

### 记忆系统
- ✅ **知识图谱**：实体-关系存储 + 语义检索
- ✅ **长期经验存储**：Episodic Memory，支持向量相似度检索
- ✅ **上下文压缩**：提取式摘要、分层压缩，处理长文本

### 工具生态
- ✅ **9大工具**：
  1. `web_search` - 网络搜索 (DuckDuckGo/SerpAPI)
  2. `calculator` - 安全数学计算
  3. `python_exec` - 受限Python执行
  4. `file_ops` - 文件操作（沙箱）
  5. `api_call` - HTTP请求
  6. `browser` - 浏览器自动化 (Playwright)
  7. `git` - Git仓库管理
  8. `database` - SQLite查询
  9. `telegram` - Telegram Bot连接

### 后端支持
- ✅ **NIM (推荐)**：使用`stepfun-ai/step-3.5-flash`，质量极高
- ✅ **OpenMythos本地**：保留原RDT架构（需自行训练权重）
- ✅ **完全可运行**：所有模块都经过验证和测试

## 🚀 快速开始

### 1. 一键安装 (推荐)

运行自动设置脚本（会创建桌面快捷方式）：

```bash
cd agent_project
chmod +x setup_desktop.sh
./setup_desktop.sh
```

完成后，桌面上会有两个图标：
- **Start Agent** - 交互式AI助手
- **Start Telegram Bot** - Telegram连接

### 2. 手动安装

```bash
cd agent_project
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate  # Windows
pip install -e .
```

**推荐依赖（可选）**：
```bash
pip install chromadb sentence-transformers  # 向量记忆
pip install python-telegram-bot --upgrade    # Telegram支持
pip install playwright && playwright install # 浏览器自动化
```

### 3. 配置API密钥

编辑 `config.yaml`:

```yaml
agent:
  backend: "nim"
  nim:
    api_key: "nvapi-xxxxx"  # 您的API key
```

或设置环境变量：

```bash
export NIM_API_KEY="your_key_here"
```

### 3. 运行

```bash
# 交互模式
python -m agent_project

# 单任务
python -m agent_project --task "Calculate 12345 * 67890"

# 调整思考深度
python -m agent_project --task "Explain quantum computing" --loops 16
```

## 📱 Telegram Bot连接 (新功能)

使用用户提供的Bot Token快速启动：

```bash
# 方式1：使用快速启动脚本（自动使用提供的token）
# 修改 start_telegram.py 中的 token 或设置环境变量：
export TELEGRAM_BOT_TOKEN="8805378062:AAH4Ru3UumdccYvJOPoKrYs8L30ps_pzeeg"
python start_telegram.py

# 方式2：通过配置文件
# 在 config.yaml 中添加：
# tools:
#   telegram:
#     enabled: true
#     bot_token: "YOUR_TOKEN"
python -m agent_project
```

**功能**：
- 实时消息收发
- 支持文本、图片、文档
- 内联按钮交互
- 用户白名单控制

详细文档：见 [TELEGRAM_INTEGRATION.md](TELEGRAM_INTEGRATION.md)

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    OpenMythos Agent                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Planning  │  │  Reasoning  │  │   Memory     │  │
│  │     Module  │  │   Engine    │  │  Manager     │  │
│  ├─────────────┤  ├─────────────┤  ├──────────────┤  │
│  │ • MCTS      │  │ • CoT       │  │ • KG         │  │
│  │ • Graph     │  │ • ReAct     │  │ • Episodic   │  │
│  │ • Validate  │  │ • Verify    │  │ • Compression│  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
│                                                         │
│  ┌────────────────────────────────────────────┐      │
│  │        Self-Correction Module              │      │
│  │  • Quality Evaluation                     │      │
│  │  • Adaptive Controller                    │      │
│  │  • Real-time Adjustments                  │      │
│  └────────────────────────────────────────────┘      │
│                                                         │
│  ┌────────────────────────────────────────────┐      │
│  │           Tool Ecosystem (9 tools)         │      │
│  │  web_search, calculator, python_exec, ... │      │
│  └────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## 🔧 配置详解

### 推理配置
```yaml
reasoning:
  enabled: true
  default_strategy: "react"  # cot | react | self_consistency | verify | zero_shot
  multi_strategy_voting: false
  loop_controller_min_loops: 4
  loop_controller_max_loops: 32
```

### 规划配置
```yaml
planning:
  enabled: true
  default_strategy: "adaptive"  # mcts | graph | sequential | adaptive
  optimize_plans: true
  max_subtasks: 10
  mcts_iterations: 100
```

### 记忆配置
```yaml
memory:
  enabled: true
  kg_storage_path: "./data/kg_store"
  episodic_storage_path: "./data/episodic_store"
  embedding_model: "all-MiniLM-L6-v2"
  max_episodes: 10000
```

### 自校正配置
```yaml
self_correction:
  enabled: true
  low_confidence_threshold: 0.6
  high_error_threshold: 0.3
```

完整配置选项见 `config.yaml`。

## 📖 使用示例

### Python API
```python
from agent_project.agent import OpenMythosAgent
from agent_project.config import load_config

# Load config
config = load_config("config.yaml")
agent = OpenMythosAgent(config)

# Run a task
result = agent.run("Calculate fibonacci(20) and explain the pattern")
print(f"Result: {result['observations'][-1]['output']}")
print(f"Success: {result['success']}")
print(f"Loops used: {result['thinking_steps']}")
```

### 交互模式
```bash
python -m agent_project
```
然后输入任务，如：
```
Calculate 12345 * 67890
```

Agent会：
1. 深度思考（可配置的循环次数）
2. 决定是否使用工具
3. 执行工具调用
4. 综合结果并返回

### 使用Telegram Bot
```python
from agent_project.tools import create_telegram_bot
from agent_project.agent import OpenMythosAgent

agent = OpenMythosAgent(config)

async def callback(update, context):
    text = update.message.text
    result = agent.run(text)
    return result['observations'][-1]['output']

bot = create_telegram_bot(
    token="YOUR_BOT_TOKEN",
    agent_callback=callback,
    polling=True
)
bot.execute(action="start")
```

## 🧪 测试

运行验证脚本：

```bash
python final_validation.py
```

运行完整测试套件：

```bash
pytest tests/ -v
```

## 📁 项目结构

```
agent_project/
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖列表
├── validate_system.py       # 系统验证
├── final_validation.py      # 最终验证
├── start_telegram.py        # Telegram快速启动
├── TELEGRAM_INTEGRATION.md  # Telegram详细文档
├── README.md                # 本文件
├── data/                    # 运行时数据（自动生成）
│   ├── experience_store/    # 经验向量库
│   ├── strategies/          # 策略数据库
│   ├── reflections/         # 反思记录
│   ├── telegram/            # Telegram数据
│   └── logs/                # 日志文件
├── agent_project/
│   ├── agent.py             # 主Agent类
│   ├── config.py            # 配置加载
│   ├── planning.py          # 规划模块
│   ├── reasoning.py         # 推理引擎
│   ├── memory.py            # 记忆系统
│   ├── self_correction.py   # 自校正
│   ├── experience.py        # 经验存储
│   ├── reflection.py        # 反思机制
│   ├── strategies.py        # 策略管理
│   ├── model_backends.py    # 模型后端
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── web_search.py
│   │   ├── calculator.py
│   │   ├── python_exec.py
│   │   ├── file_ops.py
│   │   ├── api_call.py
│   │   ├── playwright_browser.py
│   │   ├── git_ops.py
│   │   ├── database.py
│   │   └── telegram_bot.py  # ← 新增
│   └── ...
└── tests/
    ├── test_planning.py
    └── test_agent_enhanced.py
```

## 🎯 核心特性详解

### 1. 自适应推理深度
`LoopController` 根据任务复杂度、历史相似度自动决定思考循环次数：
- 简单任务：4-6 loops
- 中等任务：8-12 loops
- 复杂任务：16-32 loops

### 2. 多策略推理引擎
支持切换不同推理策略：
- **CoT**: 标准思维链
- **ReAct**: 思考-行动-观察循环
- **Self-Consistency**: 多路径采样投票
- **Verification**: 生成-验证两阶段
- **Zero-Shot**: 直接回答

### 3. 高级任务规划
- **MCTS**: 蒙特卡洛树搜索优化执行顺序
- **Graph-based**: 拓扑排序 + 关键路径分析
- **Validation**: 自动检测依赖环、资源冲突

### 4. 知识图谱记忆
- 结构化实体-关系存储
- 语义向量检索
- 最短路径查询
- 关联推荐

### 5. 实时自校正
- 多维度质量评估：成功率、置信度、连贯性、效率
- 性能退化检测
- 自动参数调整：loop深度、temperature、策略覆盖

## ⚡ 性能优化

- **向量内存**: ChromaDB + sentence-transformers (可选)
- **并行处理**: 多工具调用可并行
- **缓存机制**: 经验检索、策略匹配
- **渐进加载**: 可选依赖延迟导入

## 🔒 安全考虑

- **工具沙箱**: python_exec 受限环境
- **路径限制**: file_ops 仅允许配置目录
- **用户白名单**: Telegram, API key验证
- **危险操作阻止**: git clean, database DDL等

## 🐛 故障排除

### 常见问题

**Q: Import errors for optional modules**  
A: 安装对应依赖，或系统会自动降级到内存模式。

**Q: Telegram bot not starting**  
A: 1) 检查bot token 2) 确保 `python-telegram-bot` 已安装 3) 检查网络连接

**Q: Memory module slow**  
A: 向量搜索需要嵌入模型，首次会下载。简单内存模式更快但无语义检索。

**Q: Agent loops too many/few**  
A: 调整 `reasoning.loop_controller_default_loops` 或使用 `--loops` 参数

## 📚 参考论文

- **Recurrent-Depth Transformers**: [Loop, Think, & Generalize](https://arxiv.org/pdf/2604.07822)
- **Parcae**: [Scaling Laws for Stable Looped Models](https://arxiv.org/abs/2604.12946)
- **ReAct**: [Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- **Self-Consistency**: [CoT自一致性](https://arxiv.org/abs/2203.11171)

## 📄 许可证

MIT License - 详见 [LICENSE](../../LICENSE)

## 🙏 致谢

- OpenMythos 项目：循环架构实现
- Anthropic Claude：研究启发
- NVIDIA NIM：推理API
- python-telegram-bot：Telegram集成

---

**🚀 Production Ready!**  
所有模块已验证可运行，支持复杂任务推理、长期记忆、实时自适应、Telegram交互。

**升级日期**: 2026-07-22

## 快速开始

### 1. 安装

```bash
cd agent_project
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -e .
```

### 2. 配置NIM API

编辑 `config.yaml`:

```yaml
agent:
  backend: "nim"  # 使用NIM
  nim:
    api_key: "nvapi-xxxxx"  # 您的API key
```

### 3. 运行

```bash
# 交互模式
python -m agent_project

# 单任务
python -m agent_project --task "Calculate 12345 * 67890"

# 调整思考深度（更深入的思考）
python -m agent_project --task "Explain quantum computing" --loops 16
```

## 架构设计

```
NIM Step3.5 API (或OpenMythos本地)
    ↓ 生成文本
Agent控制器 (保持OpenMythos架构)
    ↓ 管理n_loops
外层循环: Think → Act → Observe (最多10次)
    ↓
内层思考: 单次调用但提示词指定深度
    ↓
工具调用
    ↓
经验存储 → 反思 → 策略学习
```

**核心思想**：用NIM的Step3.5超级推理能力，替换OpenMythos的权重，但保留Agent的完整框架（工具、经验、反思、策略）。

---

## 详细文档

### 双后端配置

**NIM后端（推荐）**：
```yaml
agent:
  backend: "nim"
  nim:
    api_key: "nvapi-xxxxxxxx"
    base_url: "https://integrate.api.nvidia.com/v1"
    model: "stepfun-ai/step-3.5-flash"
    temperature: 0.7
    max_tokens: 4096
```

**OpenMythos本地（研究用途）**：
```yaml
agent:
  backend: "openmythos"
  model:
    dim: 256  # 或2048（需要更多内存）
    max_loop_iters: 8
```

### 思考深度控制

- `default_thinking_loops`: 8 - 正常任务深度
- `max_thinking_loops`: 32 - 探索/反思深度
- `reflection.thinking_loops_for_reflection`: 32 - 自我反思用超深

**在NIM后端**：深度通过system prompt指导
**在OpenMythos后端**：深度通过`model.forward(n_loops=T)`参数

---

## 工具列表

| 工具 | 用法示例 | 说明 |
|------|---------|------|
| `web_search` | `[TOOL:web_search] query="Python async"` | DuckDuckGo免费搜索 |
| `calculator` | `[TOOL:calculator] expression="2+2"` | 数学计算 |
| `python_exec` | `[TOOL:python_exec] code="print(2**10)"` | 受限Python |
| `file_ops` | `[TOOL:file_ops] action="read", path="./data/file.txt"` | 文件操作 |
| `api_call` | `[TOOL:api_call] url="https://httpbin.org/get"` | HTTP请求 |

所有工具都严格限制权限，安全考虑。

---

## 示例输出

```
🤖 You: Calculate fibonacci(10) and save to file

🤔 Thinking...

[TOOL:python_exec] code="def fib(n):\n    a,b=0,1\n    for _ in range(n):\n        a,b=b,a+b\n    return b\nprint(fib(10))" [/TOOL]

Observation: 55

Agent: Now save to file.

[TOOL:file_ops] action="write", path="./data/fib10.txt", content="55" [/TOOL]

✅ Done! Result: 55 saved to ./data/fib10.txt
```

---

## 自我改进机制

1. **经验库**：向量化存储所有交互
2. **反思**：失败案例 → 高n_loops分析 → 提取模式
3. **策略**：成功案例 → 聚类 → 生成可复用规则
4. **自适应**：检索相似历史 → 推荐思考深度

---

## 依赖

- `openai` >= 1.0.0 (NIM API)
- `chromadb` (向量存储)
- `sentence-transformers` (embedding)
- `pyyaml`, `pydantic`, `rich`, `tqdm`
- (可选) `open-mythos` (本地后端)

---

## 故障排除

**NIM API错误**：
```bash
export NIM_API_KEY="your-key"
```

**OpenMythos未找到**：
```bash
cd ../OpenMythos-main
pip install -e .
```

**ChromaDB错误**：
```bash
rm -rf data/experience_store  # 重置
```

---

## 开发路线

- [x] NIM API后端
- [x] 工具调用系统
- [x] 经验存储
- [x] 自我反思
- [x] 策略学习
- [ ] Docker沙箱化python_exec
- [ ] 可视化界面
- [ ] 多Agent协作

---

**立即开始**：
```bash
python quick_test.py  # 检查环境
python -m agent_project  # 启动
```
