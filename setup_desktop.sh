#!/bin/bash
# ============================================
# 🚀 Quick Setup - One Command to Rule All
# This script sets up desktop shortcuts and verifies everything
# ============================================

cd "$(dirname "$0")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🚀 OpenMythos Agent - Quick Setup                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# Step 1: Check Python
echo "Step 1/6: Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "  ${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "  ${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.12+ from python.org"
    exit 1
fi

# Step 2: Setup virtual environment
echo ""
echo "Step 2/6: Setting up virtual environment..."
if [ -d ".venv" ]; then
    echo -e "  ${YELLOW}⚠ Existing venv found${NC}"
    read -p "  Recreate? (y/N): " recreate
    if [[ $recreate == "y" ]]; then
        rm -rf .venv
        python3 -m venv .venv
        echo -e "  ${GREEN}✓ Created new venv${NC}"
    else
        echo -e "  ${BLUE}ℹ Using existing venv${NC}"
    fi
else
    python3 -m venv .venv
    echo -e "  ${GREEN}✓ Created venv${NC}"
fi

source .venv/bin/activate

# Step 3: Install dependencies
echo ""
echo "Step 3/6: Installing dependencies..."
pip install -e . --quiet
echo -e "  ${GREEN}✓ Core dependencies installed${NC}"

# Optional dependencies
echo ""
read -p "Install optional dependencies? (y/N): " install_optional
if [[ $install_optional == "y" ]]; then
    echo "  Installing: chromadb, sentence-transformers, python-telegram-bot, playwright..."
    pip install chromadb sentence-transformers --quiet
    pip install python-telegram-bot --upgrade --quiet
    pip install playwright --quiet && playwright install --quiet
    echo -e "  ${GREEN}✓ Optional dependencies installed${NC}"
fi

# Step 4: Validate system
echo ""
echo "Step 4/6: Validating system..."
python final_validation.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ System validation passed${NC}"
else
    echo -e "  ${YELLOW}⚠ Some validation checks failed${NC}"
    echo "  Run 'python final_validation.py' for details"
fi

# Step 5: Check configuration
echo ""
echo "Step 5/6: Checking configuration..."
if grep -qE "api_key: (sk-|\\\${DEEPSEEK_API_KEY)" config.yaml 2>/dev/null || grep -q "DEEPSEEK_API_KEY" config.yaml 2>/dev/null; then
    echo -e "  ${GREEN}✓ DeepSeek API key seems set${NC}"
else
    echo -e "  ${YELLOW}⚠ DeepSeek API key not found in config.yaml${NC}"
    echo "  Please edit config.yaml and set agent.deepseek.api_key"
fi

if grep -q "bot_token: \".+\"" config.yaml 2>/dev/null; then
    echo -e "  ${GREEN}✓ Telegram bot token configured${NC}"
else
    echo -e "  ${YELLOW}⚠ Telegram bot token not set (optional)${NC}"
    echo "  Edit config.yaml under tools.telegram.bot_token"
fi

# Step 6: Desktop shortcuts
echo ""
echo "Step 6/6: Desktop shortcuts..."
case "$(uname -s)" in
    Darwin*)
        DESKTOP="$HOME/Desktop"
        cp "Start Agent.command" "$DESKTOP/"
        cp "Start Telegram Bot.command" "$DESKTOP/"
        chmod +x "$DESKTOP/Start Agent.command"
        chmod +x "$DESKTOP/Start Telegram Bot.command"
        echo -e "  ${GREEN}✓ Created 2 shortcuts on Desktop${NC}"
        echo "  - Start Agent.command"
        echo "  - Start Telegram Bot.command"
        ;;
    Linux*)
        DESKTOP="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
        if [ ! -d "$DESKTOP" ]; then
            mkdir -p "$DESKTOP"
        fi
        cp start_agent.sh "$DESKTOP/"
        cp start_telegram.sh "$DESKTOP/"
        chmod +x "$DESKTOP/start_agent.sh" "$DESKTOP/start_telegram.sh"
        echo -e "  ${GREEN}✓ Created 2 shortcuts on Desktop${NC}"
        echo "  - start_agent.sh"
        echo "  - start_telegram.sh"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo -e "  ${BLUE}ℹ Windows: Copy start_agent.bat to Desktop manually${NC}"
        echo "  or run: cp start_agent.bat %USERPROFILE%\Desktop"
        ;;
    *)
        echo -e "  ${YELLOW}⚠ Unknown OS, please copy launchers manually${NC}"
        ;;
esac

# Summary
echo ""
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}               ✅ Setup Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "What's next:"
echo ""
echo "1. Edit config.yaml and set:"
echo "   - agent.deepseek.api_key (get from https://platform.deepseek.com)"
echo "   - tools.telegram.bot_token (optional, from @BotFather)"
echo ""
echo "2. Launch from Desktop:"
echo "   - Start Agent.command (macOS) or start_agent.sh (Linux)"
echo "   - Start Telegram Bot.command to connect to Telegram"
echo ""
echo "3. Or from terminal:"
echo "   python -m agent_project              # Interactive"
echo "   python start_telegram.py            # Telegram Bot"
echo ""
echo "Need help? See README.md and TELEGRAM_INTEGRATION.md"
echo ""
