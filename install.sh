# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research




# LV Agent 一键安装脚本
# 用法: ./install.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/bin"

echo "🚀 LV Agent 安装程序"
echo "项目目录: $PROJECT_DIR"
echo "安装目录: $INSTALL_DIR"

# 创建 ~/.local/bin
mkdir -p "$HOME/.local/bin"

# 创建软链接
ln -sf "$PROJECT_DIR/lv" "$HOME/.local/bin/lv"
chmod +x "$PROJECT_DIR/lv"

echo "✅ 已创建软链接: $INSTALL_DIR/lv -> $PROJECT_DIR/lv"

# 检查 PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "⚠️  $HOME/.local/bin 不在 PATH 中"
    SHELL_RC=""
    if [[ "$SHELL" == */zsh ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ "$SHELL" == */bash ]]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_RC" ]; then
        if ! grep -q '\.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo "已添加 PATH 到 $SHELL_RC"
        fi
        echo "请执行: source $SHELL_RC"
    else
        echo "请手动将 $HOME/.local/bin 加入 PATH"
    fi
else
    echo "✅ $HOME/.local/bin 已在 PATH 中"
fi

echo ""
echo "🎉 安装完成！"
echo "重启终端或执行: source ~/.zshrc (或 ~/.bashrc)"
echo "然后在任意目录输入: lv"