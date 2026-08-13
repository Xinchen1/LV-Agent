#!/bin/bash
# ============================================
# 🚀 Desktop Launcher for OpenMythos Agent
# One-click start with automatic setup + omniroute
# Supports: CLI mode & API server mode
# ============================================

# Resolve project path (script location)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$SCRIPT_PATH"

cd "$PROJECT_PATH"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Banner
clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║ 🚀 OpenMythos Agent + DeepSeek V4                           ║
║ Powered by OpenMythos + Local LLM                            ║
║                                                               ║
║ ═════════════════════════════════════════════════════════ ║
║ Mode Selection:                                               ║
║ 1) Interactive CLI (default)                                  ║
║ 2) API Server (REST API)                                      ║
║ ═════════════════════════════════════════════════════════ ║
╚═══════════════════════════════════════════════════════════════╝
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
  echo "⚙️ First-time setup: creating virtual environment..."
  python3 -m venv ../.venv
  echo "📦 Installing dependencies..."
  if [ -f "requirements-minimal.txt" ]; then
    ../.venv/bin/pip install -r requirements-minimal.txt --quiet
  else
    ../.venv/bin/pip install openai pyyaml pydantic requests tiktoken rich fastapi uvicorn --quiet
  fi
  echo -e "${GREEN}✓ Setup complete${NC}"
  echo ""
else
  echo "✓ Using virtual environment at ../.venv"
fi

# Verify config exists
if [ ! -f "config.yaml" ]; then
 echo -e "${YELLOW}⚠ No config.yaml found${NC}"
 echo "Creating a starter config for NVIDIA NIM."
 "$VENV_PYTHON" - <<'PYCONF'
import yaml
from pathlib import Path
cfg = {
  'agent': {
    'backend': 'nim',
    'nim': {
      'api_key': '',
      'base_url': 'https://integrate.api.nvidia.com/v1',
      'model': 'google/gemma-4-31b-it',
      'temperature': 0.7,
      'max_tokens': 4096,
    },
    'default_thinking_loops': 8,
    'max_thinking_loops': 32,
    'planning': {'enabled': True},
    'reasoning': {'enabled': True},
    'memory': {'enabled': True},
    'self_correction': {'enabled': True},
  }
}
Path('config.yaml').write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
PYCONF
 echo -e "${GREEN}✓ Created starter config.yaml${NC}"
fi

# Check if API server file exists
if [ ! -f "agent_api.py" ]; then
  echo -e "${RED}❌ agent_api.py not found${NC}"
  echo "Please ensure the API server file is present."
  exit 1
fi

# Start omniroute (DeepSeek V4 API server)
echo ""
echo "Starting DeepSeek V4 server via omniroute..."

# Check if omniroute is installed
if ! command -v omniroute &> /dev/null; then
  echo -e "${YELLOW}⚠ omniroute not found in PATH${NC}"
  echo "You can start it manually in another terminal:"
  echo " cd <your-omniroute-directory>"
  echo " source .env && omniroute"
  echo ""
  read -p "Continue anyway? (y/N): " continue_anyway
  if [[ "$continue_anyway" != "y" ]]; then
    exit 1
  fi
  OMNIROUTE_PID=""
else
  # Start omniroute in background from the correct directory
  OMNIROUTE_DIR="${OMNIROUTE_DIR:-}"
  if [ -z "$OMNIROUTE_DIR" ] || [ ! -d "$OMNIROUTE_DIR" ]; then
    echo -e "${YELLOW}⚠ omniroute directory not found: $OMNIROUTE_DIR${NC}"
    OMNIROUTE_PID=""
  else
    echo " → cd $OMNIROUTE_DIR"
    echo " → source .env"
    echo " → omniroute (background)"

    # Start omniroute in background
    (
      cd "$OMNIROUTE_DIR" && \
      source .env && \
      omniroute \
    ) 2>/dev/null &

    OMNIROUTE_PID=$!
    echo " ✓ omniroute started (PID: $OMNIROUTE_PID)"

    # Wait for server to be ready
    echo -n " → Waiting for server to start"
    for i in {1..30}; do
      if curl -s http://localhost:20128/v1/models >/dev/null 2>&1; then
        echo " ✓"
        echo " ✓ Server is ready on http://localhost:20128"
        break
      fi
      echo -n "."
      sleep 1
      if [ $i -eq 30 ]; then
        echo ""
        echo -e " ${YELLOW}⚠ Server didn't respond within 30 seconds${NC}"
        echo " You can continue and try again later."
        read -p "Continue anyway? (y/N): " continue_anyway
        if [[ "$continue_anyway" != "y" ]]; then
          kill $OMNIROUTE_PID 2>/dev/null
          exit 1
        fi
      fi
    done
  fi
fi

# Check backend configuration
BACKEND=$("$VENV_PYTHON" -c "
import yaml
try:
  with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
    print(cfg.get('agent', {}).get('backend', 'openai'))
except Exception:
  print('openai')
" 2>/dev/null)

# Ensure NIM API is configured when backend is nim
if [ "$BACKEND" = "nim" ]; then
  NIM_KEY=$("$VENV_PYTHON" -c "
import yaml
try:
  with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
    print(cfg.get('agent', {}).get('nim', {}).get('api_key', ''))
except Exception:
  print('')
" 2>/dev/null)
  NIM_MODEL=$("$VENV_PYTHON" -c "
import yaml
try:
  with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
    print((cfg.get('agent', {}).get('nim', {}).get('model')) or '')
except Exception:
  print('')
" 2>/dev/null)

  if [ -z "$NIM_KEY" ] || [ -z "$NIM_MODEL" ]; then
    echo -e "${YELLOW}⚠ NIM backend selected but API key or model is missing${NC}"
    echo ""
    echo "Please provide your NVIDIA NIM API key:"
    echo " 1) Paste from clipboard (macOS: cmd+V)"
    echo " 2) Type manually"
    echo " 3) Use NIM_API_KEY environment variable"
    echo ""
    read -p "Choice (1-3) or enter key directly: " nim_choice

    if [[ "$nim_choice" == "1" ]]; then
      echo "Paste your API key and press Enter:"
      read -r nim_key_input
    elif [[ "$nim_choice" == "2" ]]; then
      read -s -p "Enter your API key: " nim_key_input
      echo ""
    elif [[ "$nim_choice" == "3" ]]; then
      nim_key_input="${NIM_API_KEY}"
    else
      nim_key_input="$nim_choice"
    fi

    if [ -z "$nim_key_input" ]; then
      echo -e "${RED}❌ No API key provided${NC}"
      echo "Falling back to OpenAI backend for this session."
      BACKEND="openai"
    else
      echo ""
      echo "Select model to use:"
      echo " 1) google/gemma-4-31b-it (Recommended)"
      echo " 2) stepfun-ai/step-3.5-flash"
      echo " 3) stepfun-ai/step-3.7"
      echo " 4) meta/llama-3.1-405b-instruct"
      echo " 5) Custom..."
      read -p "Choice [1-5]: " model_choice

      case "$model_choice" in
        2) nim_model_input="stepfun-ai/step-3.5-flash" ;;
        3) nim_model_input="stepfun-ai/step-3.7" ;;
        4) nim_model_input="meta/llama-3.1-405b-instruct" ;;
        5) read -p "Enter model name: " nim_model_input ;;
        *) nim_model_input="google/gemma-4-31b-it" ;;
      esac
tmpfile=$(mktemp)
cat > "$tmpfile" <<'PYCONF'
import sys, yaml
from pathlib import Path
key, model = sys.argv[1], sys.argv[2]
path = Path('config.yaml')
cfg = yaml.safe_load(path.read_text()) or {}
if 'agent' not in cfg:
    cfg['agent'] = {}
if 'nim' not in cfg['agent']:
    cfg['agent']['nim'] = {}
cfg['agent']['nim']['api_key'] = key
cfg['agent']['nim']['model'] = model
cfg['agent']['nim']['base_url'] = cfg['agent']['nim'].get('base_url', 'https://integrate.api.nvidia.com/v1')
cfg['agent']['backend'] = 'nim'
path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
PYCONF
"$VENV_PYTHON" "$tmpfile" "$nim_key_input" "$nim_model_input" >/dev/null 2>&1
rm -f "$tmpfile"

NIM_KEY="$nim_key_input"
NIM_MODEL="$nim_model_input"
      NIM_KEY="$nim_key_input"
      NIM_MODEL="$nim_model_input"
      echo -e "${GREEN}✓ NIM configuration saved${NC}"
      echo " API Key: ********${nim_key_input: -4}"
      echo " Model: $nim_model_input"
    fi
  fi
fi

# Validate config
echo ""
echo "Validating configuration..."
if ! "$VENV_PYTHON" final_validation.py > /dev/null 2>&1; then
  echo -e "${YELLOW}⚠ Some validation checks failed${NC}"
  echo " Run 'python final_validation.py' for details"
  read -p "Continue anyway? (y/N): " continue_anyway
  if [[ "$continue_anyway" != "y" ]]; then
    kill $OMNIROUTE_PID 2>/dev/null
    exit 1
  fi
fi

# Ask for mode selection
echo ""
echo "───────────────────────────────────────────────────────────────"
echo "Select mode:"
echo " 1) Interactive CLI (chat with agent)"
echo " 2) API Server (REST endpoint)"
echo "───────────────────────────────────────────────────────────────"
read -p "Choose [1-2] (default: 1): " MODE_CHOICE
MODE_CHOICE=${MODE_CHOICE:-1}

# Final launch
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"

if [ "$MODE_CHOICE" = "2" ]; then
  # API Server mode
  echo -e "${GREEN}║ 🚀 Starting Agent API Server... ║${NC}"
  echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo "Backend: $BACKEND"
  echo "API URL: http://localhost:8000"
  echo "Docs: http://localhost:8000/docs"
  echo "Press Ctrl+C to stop"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  # Start API server
  "$VENV_PYTHON" agent_api.py --host 0.0.0.0 --port 8000
  AGENT_EXIT_CODE=$?
else
  # CLI mode (default)
  echo -e "${GREEN}║ 🚀 Starting Agent... ║${NC}"
  echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo "Backend: $BACKEND"
  if [ "$BACKEND" = "nim" ]; then
    MODEL=$("$VENV_PYTHON" -c "
import yaml
try:
  with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
    print(cfg.get('agent', {}).get('nim', {}).get('model', 'unknown'))
except Exception:
  print('unknown')
" 2>/dev/null)
    echo "Model: $MODEL (via NIM)"
  elif [ "$BACKEND" = "openai" ]; then
    MODEL=$("$VENV_PYTHON" -c "
import yaml
try:
  with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
    print(cfg.get('agent', {}).get('openai', {}).get('model', 'unknown'))
except Exception:
  print('unknown')
" 2>/dev/null)
    echo "Model: $MODEL (local endpoint)"
  else
    echo "Model: OpenMythos (local)"
  fi
  echo "Mode: Interactive CLI"
  echo "Press Ctrl+C to stop"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  # Start CLI agent
  "$VENV_PYTHON" super_agent.py "$@"
  AGENT_EXIT_CODE=$?
fi

# Cleanup: Stop omniroute if we started it
if [ -n "$OMNIROUTE_PID" ]; then
  echo ""
  echo "Stopping omniroute (PID: $OMNIROUTE_PID)..."
  kill $OMNIROUTE_PID 2>/dev/null
  sleep 1
  if kill -0 $OMNIROUTE_PID 2>/dev/null; then
    echo " → Force killing..."
    kill -9 $OMNIROUTE_PID 2>/dev/null
  fi
  echo " ✓ Stopped"
fi

# Exit code
echo ""
if [ $AGENT_EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}✓ Session ended normally${NC}"
else
  echo -e "${YELLOW}⚠ Session ended with code $AGENT_EXIT_CODE${NC}"
fi
echo ""