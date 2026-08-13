#!/bin/bash
# OpenMythos Agent + DeepSeek 一键安装脚本
# 支持: macOS / Linux / Windows (WSL)

set -e  # 遇到错误退出

echo "🚀 OpenMythos Agent + Step3.5 Installation"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python
echo -e "${YELLOW}[1/7] Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi
python3 --version
echo -e "${GREEN}✓ Python found${NC}"
echo ""

# 检查重要目录
echo -e "${YELLOW}[2/7] Checking project structure...${NC}"
if [ ! -d "../OpenMythos-main" ]; then
    echo -e "${YELLOW}⚠ OpenMythos not found in parent directory${NC}"
    echo "   You can still use cloud backends (DeepSeek/OpenAI) without it."
    echo "   To use local OpenMythos:"
    echo "     git clone https://github.com/The-Swarm-Corporation/OpenMythos.git ../OpenMythos-main"
fi
echo -e "${GREEN}✓ Structure OK${NC}"
echo ""

# 虚拟环境
echo -e "${YELLOW}[3/7] Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    PY_LAUNCHER="/opt/homebrew/bin/python3"
    [ -x "$PY_LAUNCHER" ] || PY_LAUNCHER="/usr/bin/python3"
    [ -x "$PY_LAUNCHER" ] || PY_LAUNCHER="python3"
    "$PY_LAUNCHER" -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# 升级pip
echo -e "${YELLOW}[4/7] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# 安装依赖
echo -e "${YELLOW}[5/7] Installing dependencies...${NC}"
pip install \
    "openai>=1.0.0" \
    "chromadb>=0.4.22" \
    "sentence-transformers>=2.2.2" \
    "pyyaml>=6.0" \
    "pydantic>=2.5.0" \
    "rich>=13.7.0" \
    "tqdm>=4.66.0" \
    "python-dotenv>=1.0.0" \
    --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# 安装OpenMythos（如果存在且需要）
if [ -f "../OpenMythos-main/setup.py" ] || [ -f "../OpenMythos-main/pyproject.toml" ]; then
    echo -e "${YELLOW}[6/7] Installing OpenMythos (optional)...${NC}"
    if pip install -e "../OpenMythos-main" --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ OpenMythos installed${NC}"
    else
        echo -e "${YELLOW}⚠ OpenMythos install skipped (may fail on some systems)${NC}"
        echo "  You can still use cloud backends (DeepSeek/OpenAI) without it."
    fi
else
    echo -e "${YELLOW}⚠ OpenMythos not found, skipping${NC}"
fi
echo ""

# 安装当前项目
echo -e "${YELLOW}[7/7] Installing OpenMythos Agent...${NC}"
pip install -e . --quiet
echo -e "${GREEN}✓ Agent installed${NC}"
echo ""

# 创建数据目录
echo -e "${YELLOW}[8/8] Creating data directories...${NC}"
mkdir -p data/experience_store data/strategies data/reflections data/workspace logs
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo -e "${YELLOW}⚠ Config file not found, will be created on first run${NC}"
fi

echo "=========================================="
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "📝 Next steps:"
echo "   1. Edit config.yaml:"
echo "      agent:"
echo "        backend: \"deepseek\""
echo "        deepseek:"
echo "          api_key: \"your-key-here\""
echo ""
echo "   2. Test the installation:"
echo "      python quick_test.py"
echo ""
echo "   3. Run the agent:"
echo "      python -m agent_project"
echo ""
echo "   4. Or try a single task:"
echo "      python -m agent_project --task \"Calculate 12345 * 67890\""
echo ""
echo "📚 Documentation:"
echo "   - README.md - full guide"
echo "   - config.yaml - all settings"
echo "   - demo.py - run demo suite"
echo ""
echo "🎯 Need help? Check the README or ask for assistance!"
echo ""
