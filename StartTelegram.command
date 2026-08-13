#!/bin/bash
# ============================================
# 📱 Desktop Launcher for Telegram Bot
# One-click start for Telegram interface
# ============================================

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$SCRIPT_PATH"
cd "$PROJECT_PATH"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   📱 OpenMythos Telegram Bot                                 ║
║   Powered by Google Gemma 4-31B / Local LLM                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Use virtual environment from project root (../.venv)
VENV_PYTHON="../.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../.venv
    echo "📦 Installing dependencies..."
    if [ -f "requirements-minimal.txt" ]; then
        ../.venv/bin/pip install -r requirements-minimal.txt --quiet
    else
        ../.venv/bin/pip install openai pyyaml pydantic requests tiktoken rich --quiet
    fi
    echo -e "${GREEN}✓ Setup complete${NC}"
    echo ""
else
    echo "✓ Using virtual environment at ../.venv"
fi

# Check Telegram token
TELEGRAM_TOKEN=$($VENV_PYTHON -c "
import yaml
try:
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    token = cfg.get('tools', {}).get('telegram', {}).get('bot_token', '')
    print(token or '')
except:
    print('')
")

if [ -z "$TELEGRAM_TOKEN" ]; then
    echo ""
    echo -e "${YELLOW}⚠ Telegram bot token not configured${NC}"
    echo ""
    echo "Please set your bot token:"
    echo "  1. Message @BotFather on Telegram"
    echo "  2. Create a new bot and get the token"
    echo "  3. Edit config.yaml and add:"
    echo "      tools.telegram.bot_token: \"YOUR_TOKEN\""
    echo ""
    read -p "Press Enter after configuring token to continue..."
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           📱 Starting Telegram Bot...                ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Bot will start polling for messages..."
echo "Press Ctrl+C to stop"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start Telegram bot
../.venv/bin/python start_telegram.py

exit_code=$?
echo ""
echo -e "${YELLOW}Bot stopped (exit code: $exit_code)${NC}"
echo ""