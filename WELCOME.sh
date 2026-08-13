#!/bin/bash
# ============================================
# 🎉 Welcome! 首次使用快速引导
# ============================================

cd "$(dirname "$0")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🎉 OpenMythos Agent + Google Gemma 4-31B                  ║
║                                                               ║
║   🌟 世界顶级AI智能体现在已就绪！                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 您桌面上有以下文件："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🚀 Start Agent.command"
echo "     交互式AI助手（双击启动）"
echo ""
echo "  📱 Start Telegram Bot.command"
echo "     Telegram机器人（双击启动）"
echo ""
echo "  ⚡ quick_start.sh"
echo "     一键配置并启动（命令行）"
echo ""
echo "  ⚙️  quick_config.py"
echo "     配置NIM API密钥"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 3步开始使用："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1️⃣  配置API密钥"
echo "     运行：./quick_config.py"
echo "     或：双击 quick_config.py"
echo ""
echo "  2️⃣  验证系统"
echo "     运行：python final_validation.py"
echo ""
echo "  3️⃣  启动Agent"
echo "     双击：Start Agent.command"
echo "     或：./quick_start.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📖 文档："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  • DESKTOP_README.md - 桌面使用说明"
echo "  • QUICKSTART_G4.md  - 快速入门指南"
echo "  • NIM_SETUP.md      - NVIDIA NIM完整设置"
echo "  • TELEGRAM_INTEGRATION.md - Telegram详细文档"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 示例对话（启动后输入）："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  You: Calculate fibonacci(30) and explain the pattern"
echo "  Agent: [深度思考...] 832040"
echo ""
echo "  You: Search for latest AI news"
echo "  Agent: [使用web_search...]"
echo ""
echo "  You: Write a Python quick sort"
echo "  Agent: [生成代码...]"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔑 获取NVIDIA NIM API密钥："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. 访问 https://api.nvidia.com"
echo "  2. 登录 → API Keys → Create Key"
echo "  3. 复制key（格式：nvapi-xxxxx）"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✨ 现在就开始吧！双击 Start Agent.command${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Wait for user
read -p "Press Enter to exit..."
