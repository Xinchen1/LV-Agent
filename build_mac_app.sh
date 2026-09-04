#!/bin/bash
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research




# Build LV Agent as a macOS .app bundle
# This creates "LV Agent.app" that opens in Terminal and runs the agent.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="LV Agent"
APP_BUNDLE="$PROJECT_DIR/dist/${APP_NAME}.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

VERSION="0.1.0"
ARCH="$(uname -m 2>/dev/null || echo 'universal')"

echo "=== Building ${APP_NAME}.app ==="

# Clean previous build
rm -rf "$PROJECT_DIR/dist"
mkdir -p "$MACOS_DIR" "$RESOURCES" "$CONTENTS/Frameworks"

# ── 1. Copy Python project into Resources ──
echo "[1/5] Copying project files..."
mkdir -p "$RESOURCES/agent_project"
cp "$PROJECT_DIR/super_agent.py" "$RESOURCES/"
cp "$PROJECT_DIR/config.yaml" "$RESOURCES/"
cp "$PROJECT_DIR/config.example.yaml" "$RESOURCES/"
cp "$PROJECT_DIR/requirements-core.txt" "$RESOURCES/"
cp "$PROJECT_DIR/requirements-minimal.txt" "$RESOURCES/" 2>/dev/null || true

# Copy agent_project package
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' \
    "$PROJECT_DIR/agent_project/" "$RESOURCES/agent_project/"

# Copy assets
cp -R "$PROJECT_DIR/assets" "$RESOURCES/"

# Copy data directory (empty placeholder for runtime data)
mkdir -p "$RESOURCES/data"

# Copy .env if exists (secrets)
if [ -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env" "$RESOURCES/"
fi

# Copy rust binary if available
RUST_BIN="$PROJECT_DIR/rust_file_ops/target/release/rust_file_ops"
if [ -x "$RUST_BIN" ]; then
    mkdir -p "$RESOURCES/rust_file_ops"
    cp "$RUST_BIN" "$RESOURCES/rust_file_ops/"
fi

# ── 2. Create a self-contained venv inside the app ──
echo "[2/5] Creating virtual environment inside app bundle..."
VENV_DIR="$RESOURCES/.venv"
PY_LAUNCHER="/opt/homebrew/bin/python3"
[ -x "$PY_LAUNCHER" ] || PY_LAUNCHER="/usr/bin/python3"
[ -x "$PY_LAUNCHER" ] || PY_LAUNCHER="python3"

"$PY_LAUNCHER" -m venv "$VENV_DIR" --clear

# Install core dependencies
echo "[2b/5] Installing dependencies..."
VENV_PYTHON="$VENV_DIR/bin/python"
PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
"$VENV_PYTHON" -m pip install -r "$RESOURCES/requirements-core.txt" $PIP_MIRROR --quiet
"$VENV_PYTHON" -m pip install pillow --quiet

# ── 3. Create the launcher script ──
echo "[3/5] Creating launcher script..."
cat > "$MACOS_DIR/$APP_NAME" << 'LAUNCHER_EOF'
#!/bin/bash
# LV Agent - macOS App Launcher

# Resolve the .app bundle path
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES="$APP_DIR/Resources"

# The app runs from the user's home directory
cd "$HOME"

# Environment setup
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CHROMADB_TELEMETRY_DISABLED=1
unset PYTHONHOME PYTHONPATH
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

VENV_PYTHON="$RESOURCES/.venv/bin/python"
SUPER_AGENT="$RESOURCES/super_agent.py"

# Create runtime data directory in user's home
mkdir -p "$HOME/.lv_agent/data" "$HOME/.lv_agent/logs"

# Load .env secrets from app bundle (or user's copy)
if [ -f "$RESOURCES/.env" ]; then
    while IFS= read -r line; do
        key=$(echo "$line" | cut -d'=' -f1 | xargs)
        val=$(echo "$line" | cut -d'=' -f2- | xargs)
        case "$key" in
            NIM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY|DEEPSEEK_API_KEY|SERPAPI_KEY|TELEGRAM_BOT_TOKEN)
                export "$key=$val"
                ;;
        esac
    done < "$RESOURCES/.env"
fi

# Also check user's home for .env override
if [ -f "$HOME/.lv_agent/.env" ]; then
    while IFS= read -r line; do
        key=$(echo "$line" | cut -d'=' -f1 | xargs)
        val=$(echo "$line" | cut -d'=' -f2- | xargs)
        case "$key" in
            NIM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY|DEEPSEEK_API_KEY|SERPAPI_KEY|TELEGRAM_BOT_TOKEN)
                export "$key=$val"
                ;;
        esac
    done < "$HOME/.lv_agent/.env"
fi

# Open Terminal if not already in one
if [ -z "$TERM_PROGRAM" ] || [ "$TERM_PROGRAM" != "Apple_Terminal" ] && [ "$TERM_PROGRAM" != "iTerm.app" ] && [ "$TERM_PROGRAM" != "vscode" ]; then
    osascript -e "tell application \"Terminal\" to activate" 2>/dev/null || true
fi

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║         LV Agent v0.1.0              ║"
echo "  ║   Terminal-Native AI Agent           ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

exec "$VENV_PYTHON" "$SUPER_AGENT" "$@"
LAUNCHER_EOF

chmod +x "$MACOS_DIR/$APP_NAME"

# ── 4. Create Info.plist ──
echo "[4/5] Creating Info.plist..."
cat > "$CONTENTS/Info.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>com.cleveris.lv-agent</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>LVA </string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
PLIST_EOF

# ── 5. Generate a simple app icon (optional, uses portrait if available) ──
echo "[5/5] Setting up icon..."
# Create a basic .icns placeholder using sips if portrait exists
if command -v sips >/dev/null 2>&1 && [ -f "$RESOURCES/assets/portrait.png" ]; then
    # Convert PNG to icns using macOS tools
    ICONSET_DIR="$RESOURCES/AppIcon.iconset"
    mkdir -p "$ICONSET_DIR"

    # Generate required sizes
    for size in 16 32 64 128 256 512; do
        sips -z $size $size "$RESOURCES/assets/portrait.png" --out "$ICONSET_DIR/icon_${size}x${size}.png" >/dev/null 2>&1
        DOUBLE=$((size * 2))
        if [ $DOUBLE -le 1024 ]; then
            sips -z $DOUBLE $DOUBLE "$RESOURCES/assets/portrait.png" --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" >/dev/null 2>&1
        fi
    done

    if command -v iconutil >/dev/null 2>&1; then
        iconutil -c icns "$ICONSET_DIR" -o "$RESOURCES/AppIcon.icns" 2>/dev/null || true
        rm -rf "$ICONSET_DIR"
    fi
fi

# ── Done ──
echo ""
echo "=== Build complete! ==="
echo "  App: $APP_BUNDLE"
echo ""
echo "To use:"
echo "  open \"$APP_BUNDLE\""
echo ""
echo "Or copy to /Applications:"
echo "  cp -R \"$APP_BUNDLE\" /Applications/"
echo ""

# ── DMG Generation ──
if command -v hdiutil >/dev/null 2>&1; then
    echo "  Creating DMG..."
    DMG_PATH="$PROJECT_DIR/dist/LV-Agent-macOS-${ARCH}-${VERSION}.dmg"
    DMG_TMP="$PROJECT_DIR/dist/.dmg_tmp"
    rm -rf "$DMG_TMP" "$DMG_PATH"
    mkdir -p "$DMG_TMP"
    cp -R "$APP_BUNDLE" "$DMG_TMP/"
    # Add Applications symlink for drag-install UX
    ln -s /Applications "$DMG_TMP/Applications" 2>/dev/null || true
    hdiutil create -volname "LV Agent ${VERSION}" -srcfolder "$DMG_TMP" -ov -format UDZO "$DMG_PATH" 2>&1 | tail -5 && echo "  DMG: $DMG_PATH" || echo "  DMG creation failed (non-fatal)"
    rm -rf "$DMG_TMP"
else
    echo "  hdiutil not found, skipping DMG."
fi

# Size report
echo ""
echo "=== Build complete! ==="
du -sh "$APP_BUNDLE" 2>/dev/null || true
[ -f "$DMG_PATH" ] && du -sh "$DMG_PATH" 2>/dev/null || true
[ -f "$ZIP_PATH" ] && du -sh "$ZIP_PATH" 2>/dev/null || true
echo "  App: $APP_BUNDLE"
