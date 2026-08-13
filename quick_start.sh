#!/bin/bash
# Quick terminal starter - uses direct venv interpreter

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🚀 OpenMythos Agent                                        ║
║   Powered by Google Gemma 4-31B / Local LLM                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# Ensure venv exists and has dependencies
VENV_PYTHON="../.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚙️  Initializing virtual environment..."
    python3 -m venv ../.venv
    echo "📦 Installing core dependencies..."
    if [ -f "requirements-minimal.txt" ]; then
        ../.venv/bin/pip install -r requirements-minimal.txt --quiet
    else
        ../.venv/bin/pip install openai pyyaml pydantic requests tiktoken rich --quiet
    fi
    echo -e "${GREEN}✓ System ready${NC}"
else
    echo "✓ Using virtual environment at ../.venv"
fi

# Check config
if [ ! -f "config.yaml" ]; then
    echo -e "${YELLOW}⚠ No config.yaml found${NC}"
    echo "Run: python quick_config.py"
    exit 1
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           🚀 Starting Agent...                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Start agent with direct venv interpreter
../.venv/bin/python -m agent_project "$@"