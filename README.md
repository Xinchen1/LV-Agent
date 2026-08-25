<div align="center">

<img src="assets/screenshot.png" width="800" alt="LV Agent GitHub" />

</div>

# LV Agent

> 终端原生智能体框架。Deep thinking, real tools.

---

## 设计灵感

> 架构受图灵机通用计算模型启发，实际实现为 Harness 微内核架构

图灵机是设计隐喻而非直接实现：磁带概念映射为上下文窗口管理，状态转移映射为 Agent 状态管理。项目包含 `turing_machine` 工具用于教育演示。

<div align="center">
  <em>图灵机设计灵感：概念性架构隐喻</em>
</div>

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
- **文件操作** — 读 / 写 / grep / glob
- **代码执行** — Python / Bash，timeout 隔离
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

### 终端体验

- 头像像素画启动画面（Braille 渲染）
- 底部状态栏：token 占用 / 上下文进度
- 输入历史翻页 / Ctrl+S 草稿暂存 / Ctrl+\ Dashboard
- 实时流式输出 + 深色 / 浅色主题

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

## 链接

- **GitHub：** https://github.com/Xinchen1/LV-Agent
- **推荐后端：** DeepSeek Chat / DeepSeek Reasoner（开箱即用）
- **本地离线：** Ollama + `qwen2.5-coder:7b`（断网可用）

---

> 项目在成长，期待与各位交流。
