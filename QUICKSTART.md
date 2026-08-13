# OpenMythos Agent + Step3.5-flash
# Quick Start Guide

## 1️⃣ 安装 (1分钟)

```bash
cd agent_project
chmod +x install.sh
./install.sh
```

安装脚本会自动：
- ✅ 创建虚拟环境
- ✅ 安装所有依赖（openai, chromadb等）
- ✅ 可选安装OpenMythos本地（不用也可以）
- ✅ 创建数据目录

## 2️⃣ 配置 (30秒)

编辑 `config.yaml`：

```yaml
agent:
  backend: "nim"  # ← 确认是nim
  nim:
    api_key: "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"  # 您的key
```

**或**设置环境变量（更安全）：
```bash
export NIM_API_KEY="nvapi-xxxxx"
```

## 3️⃣ 测试 (30秒)

```bash
# 激活虚拟环境（如果还没激活）
source .venv/bin/activate

# 快速检查
python quick_test.py

# 应该看到: ✓ All files present!
```

## 4️⃣ 运行！

### 方式A: 交互模式（推荐新手）

```bash
python -m agent_project
```

然后输入任务：
```
You: Calculate 12345 * 67890
Agent: [思考...] [TOOL:calculator] expression="12345 * 67890" [/TOOL]
Result: 838102050
```

### 方式B: 单次任务

```bash
python -m agent_project --task "What's the weather in Tokyo?" --loops 8
```

### 方式C: 深度思考任务

```bash
python -m agent_project --task "Explain quantum mechanics" --loops 32
```

### 方式D: 运行演示套件

```bash
python demo.py
```

---

## 🎯 常用命令

```bash
# 列出所有工具
python -m agent_project --list-tools

# 运行测试
python -m agent_project --run-tests

# 指定不同的思考深度
python -m agent_project --task "analyze this code" --loops 16

# 探索模式（随机深度）
python -m agent_project --task "find information" --mode exploration
```

---

## 📊 思考深度建议

| n_loops | 使用场景 | 预期耗时 |
|---------|---------|---------|
| 4-8 (默认) | 简单计算、查询 | 5-15秒 |
| 12-16 | 中等复杂分析 | 10-30秒 |
| 20-32 | 深度思考、学术问题 | 20-60秒 |
| 32+ | 自我反思模式 | 60+秒 |

**成本提示**：n_loops越大，prompt越深，token消耗越多。

---

## 🔧 配置文件说明

`config.yaml` 控制所有行为：

### 切换后端

```yaml
agent:
  backend: "nim"          # 或 "openmythos"
```

### 调整Agent行为

```yaml
agent:
  max_outer_loops: 10         # 最多多少次思考-行动循环
  default_thinking_loops: 8   # 默认思考深度
  max_thinking_loops: 32      # 最大深度
  temperature: 0.7            # 随机性 (0.1-1.0)
```

### 启用/禁用工具

```yaml
tools:
  enabled:
    - web_search    # 只保留需要的
    - calculator
    # - python_exec  # 注释掉禁用
```

### 开启自我改进

```yaml
reflection:
  enabled: true
  frequency: 5  # 每5次任务反思一次

self_improvement:
  enabled: true
  auto_training: false  # 设为true可自动微调（实验性）
```

---

## 🗂️ 项目结构

```
agent_project/
├── config.yaml              # 配置文件（您需要编辑这个）
├── README.md                # 完整文档
├── quick_test.py            # 快速验证
├── demo.py                  # 演示套件
├── install.sh               # 一键安装脚本
├── .env.example             # 环境变量示例
│
├── agent_project/
│   ├── __init__.py
│   ├── __main__.py         # 主入口: python -m agent_project
│   ├── agent.py            # 核心Agent类
│   ├── config.py           # 配置加载
│   ├── model_backends.py   # ⭐ NIM + OpenMythos后端
│   ├── experience.py       # 经验存储
│   ├── reflection.py       # 自我反思
│   ├── strategies.py       # 策略学习
│   └── tools/              # 5个工具
│
├── data/                    # 运行时数据（自动生成）
│   ├── experience_store/   # 向量数据库
│   ├── strategies/         # 策略文件
│   ├── reflections/        # 反思记录
│   └── workspace/          # 文件操作目录
│
└── logs/                   # 日志文件
```

---

## 🐛 故障排除

### 问题: ImportError: No module named 'openai'
```bash
pip install openai>=1.0.0
```

### 问题: NIM API错误
```
1. 检查config.yaml中的api_key是否正确
2. 或设置环境变量: export NIM_API_KEY="your-key"
3. 确认网络可以访问 https://integrate.api.nvidia.com
```

### 问题: ChromaDB错误
```bash
rm -rf data/experience_store
python -m agent_project  # 会自动重建
```

### 问题: 工具不工作
```bash
# 检查config.yaml
tools:
  enabled:
    - web_search   # 确保这个工具在列表中
```

### 问题: 推理太慢
```yaml
# config.yaml 调整:
agent:
  default_thinking_loops: 4   # 降低深度
  temperature: 0.9           # 提高温度减少采样
```

### 问题: Mac M1/M2/M3安装torch失败
```bash
# 我们其实不需要torch（用NIM后端）
# 如果非要OpenMythos本地：
pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

---

## 💡 使用技巧

### 1. 让Agent更准确
```bash
# 降低temperature，更确定
python -m agent_project --task "factual question" --loops 12

# 减少随机性
# config.yaml: temperature: 0.3
```

### 2. 让Agent更有创意
```bash
# 提高temperature
--loops 8 --temperature 0.9
```

### 3. 复杂任务分步
```bash
# 先分析
python -m agent_project --task "analyze this algorithm" --loops 16

# 再实现
python -m agent_project --task "implement the algorithm in Python" --loops 12
```

### 4. 查看Agent在想什么
```bash
# 日志会显示:
# - 每次思考内容（截断）
# - 工具调用
# - 观察结果
# 查看详细日志: tail -f logs/agent.log
```

---

## 🎓 示例任务

```bash
# 1. 数学计算
python -m agent_project --task "Calculate 12345 * 67890 + 11111"

# 2. 文件操作
python -m agent_project --task "Create a file ./data/hello.txt with content 'Hello World'"

# 3. 搜索查询
python -m agent_project --task "Search for latest news about AI" --loops 12

# 4. 代码执行
python -m agent_project --task "Write Python to calculate primes under 100" --loops 16

# 5. API调用
python -m agent_project --task "GET https://jsonplaceholder.typicode.com/posts/1" --loops 8
```

---

## 📈 监控与调试

### 查看经验库大小
```bash
python -c "
from agent_project.experience import ExperienceBuffer
from agent_project.config import load_config
buf = ExperienceBuffer(load_config())
print(f'Total episodes: {buf.count()}')
"
```

### 清空经验库（重新开始）
```bash
rm -rf data/experience_store
```

### 查看最近的经验
```bash
python -c "
from agent_project.experience import ExperienceBuffer
from agent_project.config import load_config
buf = ExperienceBuffer(load_config())
for exp in buf.get_recent(3):
    print(f'Task: {exp.task[:50]}...')
    print(f'Success: {exp.trajectory[\"success\"]}')
    print()
"
```

---

## 🎉 您准备好了！

**现在您拥有：**
- ✅ 最强的推理模型（Step3.5-flash）
- ✅ OpenMythos深度思考机制
- ✅ 完整Agent能力（工具+经验+反思+策略）
- ✅ 本地CPU运行
- ✅ 一键安装

**下一步：**
1. 运行 `./install.sh`
2. 编辑 `config.yaml` 填入您的NIM API key
3. 运行 `python -m agent_project`
4. 输入一个复杂任务，观察深度思考！

**有任何问题，随时问我！** 🚀
