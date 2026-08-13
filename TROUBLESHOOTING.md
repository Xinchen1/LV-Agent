# 🛠️ 故障排除与修复指南

## 常见问题及解决方案

### 问题1：双击启动器提示 "No module named agent_project"

**原因**：虚拟环境依赖未正确安装

**修复**：
```bash
cd ~/Downloads/OpenMythos-main/agent_project
# 删除旧的虚拟环境
rm -rf .venv
# 重新创建（启动器会自动安装依赖）
./StartAgent.command
```

或者手动安装：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai pyyaml pydantic requests tiktoken rich
```

---

### 问题2：pip install 报错 (TOML/poetry errors)

**原因**：`pyproject.toml` 配置问题

**修复**：我已经更新了 `pyproject.toml` 移除了复杂的poetry配置。现在可以直接使用简单的requirements。

```bash
# 使用简化依赖列表
pip install -r requirements-minimal.txt
```

或只安装核心依赖：
```bash
pip install openai pyyaml pydantic requests tiktoken rich
```

---

### 问题3：导入失败 "ModuleNotFoundError: No module named X"

**常见缺失模块**：
- `yaml` → `pip install pyyaml`
- `requests` → `pip install requests`
- `pydantic` → `pip install pydantic`
- `openai` → `pip install openai`
- `tiktoken` → `pip install tiktoken`

**快速修复**：
```bash
cd ~/Downloads/OpenMythos-main/agent_project
source .venv/bin/activate
pip install pyyaml requests tiktoken rich --upgrade
```

---

### 问题4：API key 无效

**检查步骤**：
1. 确认密钥格式：应以 `nvapi-` 开头
2. 验证密钥有效：登录 https://api.nvidia.com
3. 确认有Google Gemma 4模型访问权限
4. 编辑 `config.yaml`，确保密钥正确粘贴：
   ```yaml
   agent:
     nim:
       api_key: "nvapi-你的密钥"
   ```

---

### 问题5：模型未找到 (Model not found)

**原因**：
- 模型名称拼写错误
- 账户无该模型权限

**修复**：
1. 检查配置：
   ```yaml
   agent:
     nim:
       model: "google/gemma-4-31b-it"  # 正确名称
   ```
2. 在NVIDIA NIM控制台确认模型已启用
3. 尝试其他可用模型：
   - `google/gemma-4-31b-it` (推荐)
   - `stepfun-ai/step-3.5-flash`
   - `meta/llama-3.1-405b-instruct`

---

### 问题6：Telegram Bot 不响应

**检查**：
1. Bot Token是否正确：
   ```yaml
   tools:
     telegram:
       bot_token: "你的token"
   ```
2. 安装依赖：
   ```bash
   pip install python-telegram-bot --upgrade
   ```
3. 查看日志：
   ```bash
   tail -f data/logs/agent.log
   ```

---

### 问题7：桌面双击没反应

**原因**：macOS安全限制

**修复**：
1. 右键点击文件 → "打开"
2. 在弹出窗口点"打开"
3. 或终端执行：
   ```bash
   chmod +x ~/Desktop/StartAgent.command
   ```

---

### 问题8：内存不足

Gemma 4-31B需要约8-16GB RAM。

**解决**：
1. 关闭其他应用
2. 减少最大输出：
   ```yaml
   agent:
     nim:
       max_tokens: 4096  # 减少
   ```
3. 降低思考深度：
   ```yaml
   reasoning:
     loop_controller_default_loops: 4
   ```

---

## 🚀 快速修复脚本

运行以下命令自动修复常见问题：

```bash
cd ~/Downloads/OpenMythos-main/agent_project

# 1. 修复虚拟环境
rm -rf .venv
python3 -m venv .venv

# 2. 安装核心依赖
source .venv/bin/activate
pip install --upgrade pip
pip install openai pyyaml pydantic requests tiktoken rich --no-cache-dir

# 3. 验证系统
python final_validation.py

# 4. 启动
./quick_start.sh
```

---

## 📊 依赖清单

### 必需依赖（运行核心功能）
- ✅ openai (>=1.0.0) - NIM API客户端
- ✅ pyyaml (>=6.0) - 配置解析
- ✅ pydantic (>=2.5.0) - 配置验证
- ✅ requests (>=2.31.0) - HTTP请求
- ✅ tiktoken (>=0.5.0) - Token计数
- ✅ rich (>=13.7.0) - 彩色输出
- ✅ numpy (>=1.24.0) - 数值计算
- ✅ tqdm (>=4.66.0) - 进度条

### 可选依赖（增强功能）
- 🔵 chromadb + sentence-transformers - 向量记忆（否则用简单内存）
- 🔵 python-telegram-bot - Telegram Bot支持
- 🔵 playwright - 浏览器自动化
- 🔵 torch - PyTorch（OpenMythos本地后端需要）

---

## ✅ 验证安装

运行：
```bash
python final_validation.py
```

期望输出：
```
✅ SUCCESS: System fully validated!
🎯 System Capabilities:
   • Advanced Planning (MCTS, Graph-based)
   • Multi-Strategy Reasoning
   • Knowledge Graph + Episodic Memory
   • Self-Correction & Quality Control
   • 9 Tools
   • NVIDIA NIM API connection
```

如果看到 ✅，说明一切正常！

---

## 📝 配置文件检查

确保 `config.yaml` 包含：

```yaml
agent:
  backend: "nim"
  nim:
    api_key: "nvapi-..."  # 您的密钥
    model: "google/gemma-4-31b-it"
    base_url: "https://integrate.api.nvidia.com/v1"
```

---

## 🔄 重置步骤

如果问题持续，可以重置：

```bash
cd ~/Downloads/OpenMythos-main/agent_project

# 1. 备份数据（可选）
cp -r data data.backup

# 2. 删除虚拟环境
rm -rf .venv

# 3. 清理配置（保留API密钥）
git checkout -- config.yaml  # 如果config.yaml被破坏

# 4. 重新开始
./StartAgent.command
```

---

## 💬 获取帮助

1. **查看日志**：
   ```bash
   tail -f data/logs/agent.log
   ```

2. **运行验证**：
   ```bash
   python final_validation.py
   ```

3. **测试导入**：
   ```bash
   python -c "import agent_project.agent; print('OK')"
   ```

---

## 🎉 成功指标

当看到以下输出，表示系统正常：

```bash
$ ./StartAgent.command
╔═══════════════════════════════════════════════════════════════╗
║   🚀 OpenMythos Agent + NVIDIA NIM                          ║
╚═══════════════════════════════════════════════════════════════╝

✓ Configuration found
✓ System validation passed

╔═══════════════════════════════════════════════════════╗
║           🚀 Starting Agent...                      ║
╚═══════════════════════════════════════════════════════╝
```

然后输入任务测试：
```
You: Calculate fibonacci(10)
Agent: 55
```

---

**系统已修复并提供完整支持！**
