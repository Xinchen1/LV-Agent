#!/bin/bash
# ============================================
# 🤖 OpenMythos Agent - Linux/Mac Launcher
# Simple shell script to start the agent
# ============================================

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════╗"
echo "║    🤖 OpenMythos Agent                   ║"
echo "╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start agent with all args passed through
python -m agent_project "$@"
