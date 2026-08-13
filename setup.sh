#!/bin/bash
# OpenMythos Agent - One-click setup
# Works on macOS / Linux / WSL. Windows users should use setup.bat or WSL.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="./.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQUIREMENTS="requirements-minimal.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    REQUIREMENTS="requirements.txt"
fi

echo "OpenMythos Agent setup"
echo "======================"
echo ""

# 1. Python check
echo "[1/6] Checking Python..."
if ! command -v "$PYTHON" &> /dev/null; then
    echo "error: Python not found. Install Python 3.10+ and try again."
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "  Python $PY_VERSION"

# 2. Virtual environment
echo "[2/6] Preparing virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  created .venv"
else
    echo "  .venv exists"
fi

# 3. Dependencies
echo "[3/6] Installing dependencies from $REQUIREMENTS..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS" --quiet
echo "  dependencies installed"

# 4. Rust file_ops (optional but recommended)
echo "[4/6] Building optional Rust file_ops..."
if command -v cargo &> /dev/null && [ -d "rust_file_ops" ]; then
    (cd rust_file_ops && cargo build --release --quiet)
    echo "  rust_file_ops built"
else
    echo "  skipped (cargo not installed or rust_file_ops missing)"
fi

# 5. Runtime directories
echo "[5/6] Creating runtime directories..."
mkdir -p data/checkpoints data/episodic_store data/experience_store \
         data/fast_read_cache data/kg_store data/reflections data/strategies \
         data/telegram logs
echo "  directories ready"

# 6. Config
echo "[6/6] Checking configuration..."
if [ ! -f "config.yaml" ]; then
    if [ -f "config.example.yaml" ]; then
        cp config.example.yaml config.yaml
        echo "  created config.yaml from example"
        echo ""
        echo "IMPORTANT: edit config.yaml and set your API key, or set env var:"
        echo "  export NIM_API_KEY=your_key_here"
    else
        echo "  warning: no config.example.yaml found"
    fi
else
    echo "  config.yaml exists"
fi

echo ""
echo "Setup complete."
echo ""
echo "Start the agent:"
echo "  ./SuperAgent.command"
echo "Or:"
echo "  $VENV_PYTHON super_agent.py"
