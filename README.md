<div align="center">

<img width="1480" height="263" alt="screenshot" src="https://github.com/user-attachments/assets/1002e6ab-91fb-4ce6-9813-343736dc0cc9" />



</div>
# LV Agent



> 终端原生智能体框架。Deep thinking, real tools.

---

## 设计灵感

架构受图灵机通用计算模型启发，实际实现为 Harness 微内核架构。

---

## 项目简介

LV Agent 是一个**终端原生的智能体框架**，有基于 mythos 的一些思路，也是我研究机构 **Cleveris Research** 的作品。

它通过**多轮 LLM 调用 + 工具循环 + 自我修正**，实现"逐步深入思考"的过程。目前项目还很早期，有不少不足，期待与大家一起交流进步。

---

## 核心特性

### 推理与规划

| 特性 | 说明 |
|------|------|
| **多策略推理** | CoT / ReAct / Verification（Self-Consistency / MCTS 规划中） |
| **自适应循环控制** | 简单问题少轮调用，难题多轮调用 |
| **任务规划** | 支持 SEQUENTIAL / PARALLEL / HIERARCHICAL / ADAPTIVE 策略 |
| **自我修正** | 质量评估 + 自动修正 + 参数自适应调整 |

### 工具链

- **Web 搜索** — 多查询融合
- **文件操作** — 读 / 写 / grep / glob，支持「桌面上的 XXX」等跨目录文件夹浏览
- **代码执行** — Python / Bash 全终端访问（管道、重定向、`&&`/`||`、env），timeout 隔离
- **GitHub 搜索、PDF 读取、天气查询、网页抓取**
- **Telegram Bot 集成**（可独立运行）

### 记忆与上下文

- **知识图谱** — 实体-关系结构化长期记忆
- **经历记忆** — 跨会话向量相似度检索
- **记忆技能** — 从对话中提取可复用的策略（`/learn` + `/memskill`）
- **上下文压缩** — 长会话自动归纳，512 token 预算内保留核心信息

### Harness 运行时

- **事件溯源** — 执行过程可追溯、可重放
- **会话持久化** — SQLite 存储，支持历史会话选择（`/sessions`）
- **预算控制** — token 消耗 + 时间双重限制
- **工具确认** — 危险操作前请求用户批准
- **检查点** — 执行中断后可恢复
- **热插拔** — 模块生命周期管理，支持运行时动态替换与回滚

### 终端体验

- 头像像素画启动画面（Braille 渲染）
- 底部状态栏：token 占用 / 上下文进度
- 输入历史翻页 / Ctrl+S 草稿暂存 / Ctrl+\ Dashboard
- 实时流式输出 + 深色 / 浅色主题（渲染间隔 0.12s / 10fps，工具结果折叠至 3 行，流式阈值 64/128 字符）

---

## 技术架构

### API 模式（主流使用方式）

```
用户输入
    ↓
Planner 分解任务 → 分配 thinking loops
    ↓
[循环推理引擎]
    ├→ LLM 调用 （策略：CoT / ReAct / MCTS / Self-Consistency）
    ├→ 工具执行 → 观察结果 → 继续推理
    ├→ 自我修正 → 质量不达标则重答
    └→ ACT 自适应停止（达到质量阈值即退出循环）
    ↓
上下文压缩（长会话自动归纳）
    ↓
最终回答
```

### 深度研究模式

```
深度研究请求
    ↓
多角度搜索（多 query 并行）
    ↓
网页正文抓取 + 评分排序
    ↓
信息综合 + 信源评估
    ↓
生成 HTML 报告 → 自动打开浏览器
```

### 本地模型模式（实验性）

```
输入 → [Prelude] → [Recurrent Block 循环 T 次] → [Coda] → 输出

特性：
- 同一组权重循环多次，像人一样"反复思考"
- MoE 稀疏专家、LoRA 深度适配、LTI 稳定注入
- 需要本地 GPU / Metal 加速
```

---

## 快速开始

### 方式一：一键安装（推荐，全局 `lv` 命令）

```bash
git clone https://github.com/Xinchen1/LV-Agent.git
cd LV-Agent
./install.sh
source ~/.zshrc   # 或 source ~/.bashrc
lv                # 任意目录直接启动
```

### 方式二：手动运行（无需安装）

```bash
git clone https://github.com/Xinchen1/LV-Agent.git
cd LV-Agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
./lv            # 或 python super_agent.py
```

**支持后端：** DeepSeek / OpenAI / Anthropic / OpenRouter / Ollama（本地离线）。

---

## 已实现的开箱功能

### 内置命令

| 命令 | 功能 |
|------|------|
| `/deep_research` | 多角度搜索 + 自动生成 HTML 报告 |
| `!命令` | 直接执行 shell 命令 |
| `@文件` | 把文件内容注入到输入 |
| `/model` | 实时切换模型 |
| `/strategy` | 切换推理策略 |
| `/compress` | 手动压缩上下文 |
| `/learn` | 从当前对话中学习记忆技能 |
| `/memskill` | 管理已学习的策略（list / evolve / snapshot / restore） |
| `/sessions` | 浏览历史会话 |
| `/dashboard` | 打开 Agent 状态面板 |
| `/drafts` | 查看暂存的输入草稿 |

### 快捷键

| 快捷键 | 动作 |
|--------|------|
| `Ctrl+S` | 暂存当前输入草稿 |
| `Ctrl+\` | 打开 Dashboard |
| `ESC` | 中断正在运行的任务（深度研究等长任务） |
| `↑↓` | 翻页输入历史 |

---

## 未来规划

### 仍在探索的方向

- 接入更多搜索源
- 更丰富的工具（数据库 / Git 操作 / 浏览器自动化）
- 更强的推理策略（Best-of-N / 投票机制）

### 中期探索

- 记忆检索改进（混合向量 + 关键词召回）
- 多模态支持（图像理解输入）
- 插件系统（MCP 协议初步支持）

### 长期方向

- 本地循环模型与 API 模型的深度融合
- 更智能的自主规划能力

---

## 更新日志

### 2026-09-04

- **执行引擎** — 跨步重复调用强制收敛：首次重复返回缓存+换工具提示，二次重复直接下 `STOP` 指令要求输出最终答案；去重不再覆盖原始缓存；`policies` 历史回注补 `Action` 行（Thought→Action→Observation 完整链）。
- **工具能力** — `web_search` 空结果早停（失败明示，不再换词死循环）；observation 压缩为 title/url/snippet/score 四字段（约 1/3 体积）；`sequential_fallback` 默认开启。
- **意图识别** — 显式搜索动词（搜索/查一下/最新/新闻…）优先于 `python` 等裸关键词，`intent_classifier` 与 `agent.py` 双副本同修。
- **记忆与上下文** — `compress_events` 防膨胀守卫（短文本不再越压越大）+ token-aware 硬截断（中英文都守预算）。
- **打包** — `build_mac_app.sh` 补 `VERSION`/`ARCH` 变量（DMG 文件名修正）、launcher 加 shebang；Electron 客户端（`desktop/`）可打 `LV Agent-1.0.0-arm64.dmg`，首次运行自动落配置到用户目录。
- **定位快路** — 修 `_try_location_fast_path` 三处误劫持：目标名截断到第一个分句标点（整句中文不再被当文件名）+ 长度守卫；只收 `metadata.count>0` 真实命中，空结果/复合任务（定位+修改）回退正常循环，避免假完成跳过后半任务。
- **学习闭环 P1** — 任务前注入 memskill 选中技能（之前学完永不使用）；成功率按任务成败记分；技能选择加中文二元组匹配；`/memskill list` 显示 score。
- **上下文 P2** — 首轮用户需求常驻保护（压缩/溢出永不丢）；叙事摘要 Facts/Preferences/Open 三段结构；研究/分析/代码类任务工作记忆预算 x1.5。
- **工具 P3** — 任务感知子集：8 常驻核心 + 关键词加挂（天气/git/pdf/网页…），无命中回退全量；native 调用只传子集 schema，省约 2400 tokens/轮。
- **回归 P4** — `tests/` 新增 5 个离线回归文件 29 用例（意图/子集/上下文/防劫持/技能闭环），0.4s 跑完；换模型改引擎先跑分。
- **code 模式假完成** — 状态追问（可以了没/看了没/?）回注上任务继续干活；定位快路加`改为/免登`等复合守卫，两段式任务不再秒回 ok。
- **只读不总结** — 有实质内容的延续追问改走主循环（单次快路干不了多步活）；工具成功但总结为空时兜底；展示层脱敏 API key（nvapi/AIzaSy/sk…不上屏）。

### 2026-09-02

- **执行引擎** — 预编译正则热点路径；代码模式 loops 6→10、`max_tokens` 16384；分析/报告类任务保底 8192 token，避免长报告截断；假进展/敷衍判定加固（`DONE[]` 标记扫描、观察去重）。
- **推理与策略** — `intent_classifier` 支持「桌面上的 XXX」文件夹定位；`model_backends` 缓存 tiktoken encoder；`policies` 预编译 JSON 解析正则。
- **终端渲染** — `stream_adapters` 渲染间隔 0.25s→0.12s、刷新 10fps、spinner 回收 0.05s；工具结果默认折叠至 3 行，流式阈值 64/128 字符；链接/数字/路径行内高亮。
- **工具能力** — `bash_exec` 说明扩展为“全终端访问”，仅拦截 `curl|bash`/`wget|bash`/`chmod -R 777`，放行 `sudo` 与受限 `find`，满足真实工程任务。
- **其它** — `agent.py` 清理冗余 import、`!` 命令提示统一为 `terminal.token` 风格。

---

## 链接

- **GitHub：** https://github.com/Xinchen1/LV-Agent
- **推荐后端：** DeepSeek Chat / DeepSeek Reasoner（开箱即用）
- **本地离线：** Ollama + `qwen2.5-coder:7b`（断网可用）

---

## 修复状态

最近修复了 lv agent 中 SelfEvolutionController 初始化问题：

- 修复文件：agent_project/agent.py (+17/-3 行)
- 在 `_build_harness_kernel()` 中添加了 HotSwapKernel 包装
- 在 agent `__init__` 后添加了 EvolutionController 初始化
- verified force_observe() 在每个 turn 后运行
- 自动 promote/rollback 决策: 启用

系统当前状态: 就绪 Ready

> 项目在成长，期待与各位交流。

---

## About

LV Agent is a self-contained AI agent system built around three principles: intelligent planning, native tool use, and long-horizon memory. It operates against OpenAI-compatible APIs or runs fully offline with local LLMs.

### Core features

- Reasoning-first workflow with automatic re-planning (code mode 10 loops, 16k tokens for analysis)
- Native tool integration: web search, file system, full shell access, and MCP servers
- Holographic memory for durable facts and recent context
- Terminal-first UX with streaming and rich rendering (0.12s interval, 10fps, folded tool output)

### Modes

- **API** (recommended) — OpenRouter or OpenAI-compatible endpoints
- **Local** (experimental) — Ollama + `qwen2.5-coder:7b`, no internet required

### Links

- **Repository:** https://github.com/Xinchen1/LV-Agent
- **Recommended backend:** DeepSeek Chat / DeepSeek Reasoner
- **Local backend:** Ollama + `qwen2.5-coder:7b`

---

> Thank you for your interest.
