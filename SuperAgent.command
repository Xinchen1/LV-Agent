#!/bin/bash
# Super Agent CLI - Desktop Launcher
# Usage: double-click or run ./SuperAgent.command from repo root

set -e

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_PATH"

DIM='\033[2m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

VENV_PYTHON="./.venv/bin/python"
REQUIREMENTS="requirements-minimal.txt"

# Auto-create venv and install deps if missing
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${DIM}venv created${NC}"
fi

if [ ! -f "$REQUIREMENTS" ]; then
    REQUIREMENTS="requirements.txt"
fi

# Upgrade deps only when requirements are newer than marker
MARKER="./.venv/.installed_requirements"
if [ "$REQUIREMENTS" -nt "$MARKER" ] 2>/dev/null || [ ! -f "$MARKER" ]; then
    echo -e "${DIM}installing dependencies...${NC}"
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet
    "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS" --quiet
    touch "$MARKER"
    echo -e "${DIM}dependencies ready${NC}"
fi

# Build Rust file_ops if not already built
RUST_BIN="./rust_file_ops/target/release/rust_file_ops"
if [ -d "rust_file_ops" ] && [ ! -f "$RUST_BIN" ]; then
    if command -v cargo &> /dev/null; then
        echo -e "${DIM}building rust_file_ops...${NC}"
        (cd rust_file_ops && cargo build --release --quiet)
        echo -e "${DIM}rust_file_ops ready${NC}"
    else
        echo -e "${YELLOW}warn: cargo not found, using Python fallback for file_ops${NC}"
    fi
fi

# Create config from example if missing
if [ ! -f "config.yaml" ]; then
    if [ -f "config.example.yaml" ]; then
        echo -e "${YELLOW}config.yaml not found, copying from config.example.yaml${NC}"
        cp config.example.yaml config.yaml
        echo -e "${RED}IMPORTANT: edit config.yaml and set your API key (or use DEEPSEEK_API_KEY env var)${NC}"
    else
        echo -e "${RED}no config.yaml or config.example.yaml found${NC}"
        exit 1
    fi
fi

# Validate config and Python imports before launching
if ! "$VENV_PYTHON" -c "import agent_project" 2>/dev/null; then
    echo -e "${RED}agent_project package not importable${NC}"
    echo "Try: ./setup.sh"
    exit 1
fi

# Launch Super Agent
"$VENV_PYTHON" super_agent.py "$@"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo -e "\n${DIM}session ended normally${NC}"
else
    echo -e "\n${DIM}session ended with code $exit_code${NC}"
fi
