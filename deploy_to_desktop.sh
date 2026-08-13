#!/bin/bash
# ============================================
# 📦 Package for Desktop - 复制所有启动脚本到桌面
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
║   📦 Desktop Deployment                                      ║
║   Copy launchers to your desktop                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# Determine desktop path
if [[ "$OSTYPE" == "darwin"* ]]; then
    DESKTOP_PATH="$HOME/Desktop"
    OS_NAME="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    DESKTOP_PATH="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
    OS_NAME="Linux"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Create desktop directory if not exists
mkdir -p "$DESKTOP_PATH"

echo "Operating System: $OS_NAME"
echo "Desktop path: $DESKTOP_PATH"
echo ""

# Files to copy based on OS
if [[ "$OS_NAME" == "macOS" ]]; then
    FILES=(
        "Start Agent.command"
        "Start Telegram Bot.command"
        "quick_start.sh"
        "quick_config.py"
    )
    EXTRA_DESC="macOS .command files are ready to double-click"
elif [[ "$OS_NAME" == "Linux" ]]; then
    # Create .desktop files for Linux
    echo "Creating .desktop shortcuts..."
    
    # Agent launcher
    cat > "$DESKTOP_PATH/openmythos-agent.desktop" << 'DESKTOP'
[Desktop Entry]
Name=OpenMythos Agent
Comment=AI Agent with Deep Reasoning
Exec=/path/to/agent_project/start_agent.sh
Icon=utilities-terminal
Terminal=true
Type=Application
Categories=Utility;ArtificialIntelligence;
DESKTOP
    
    # Telegram launcher
    cat > "$DESKTOP_PATH/openmythos-telegram.desktop" << 'DESKTOP'
[Desktop Entry]
Name=OpenMythos Telegram
Comment=Telegram Bot for OpenMythos
Exec=/path/to/agent_project/start_telegram.sh
Icon=utilities-terminal
Terminal=true
Type=Application
Categories=Utility;InstantMessaging;
DESKTOP
    
    # Fix paths in .desktop files
    sed -i "s|/path/to/agent_project|$(pwd)|g" "$DESKTOP_PATH/openmythos-agent.desktop"
    sed -i "s|/path/to/agent_project|$(pwd)|g" "$DESKTOP_PATH/openmythos-telegram.desktop"
    
    # Make executable
    chmod +x "$DESKTOP_PATH/openmythos-agent.desktop"
    chmod +x "$DESKTOP_PATH/openmythos-telegram.desktop"
    
    FILES=(
        "start_agent.sh"
        "start_telegram.sh"
        "quick_start.sh"
        "quick_config.py"
    )
    EXTRA_DESC=".desktop files created and configured"
else
    FILES=()
    EXTRA_DESC=""
fi

# Copy files
echo "Copying launchers to desktop..."
copied=0

# macOS .command files
if [[ "$OS_NAME" == "macOS" ]]; then
    for file in "Start Agent.command" "Start Telegram Bot.command" "quick_start.sh" "quick_config.py"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$DESKTOP_PATH/"
            chmod +x "$DESKTOP_PATH/$file" 2>/dev/null || true
            echo "  ✓ $file"
            ((copied++))
        else
            echo "  ⚠ $file not found (skipped)"
        fi
    done
else
    # Linux files
    for file in "start_agent.sh" "start_telegram.sh" "quick_start.sh" "quick_config.py"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$DESKTOP_PATH/"
            chmod +x "$DESKTOP_PATH/$file" 2>/dev/null || true
            echo "  ✓ $file"
            ((copied++))
        else
            echo "  ⚠ $file not found (skipped)"
        fi
    done
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✅ Deployment Complete!                         ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Copied $copied file(s) to your desktop:"
echo "  Location: $DESKTOP_PATH"
echo ""
echo "Next steps:"
echo "  1) Double-click the launcher icon on your desktop"
echo "  2) First run will configure your API key"
echo "  3) Enjoy your AI assistant!"
echo ""
echo "Files on desktop:"
for file in "${FILES[@]}"; do
    if [[ -f "$DESKTOP_PATH/$file" ]]; then
        echo "  • $file"
    fi
done
echo ""
echo "To reconfigure later:"
echo "  • Run quick_config.py from desktop or terminal"
echo ""
echo "═══════════════════════════════════════════════════════════════"
