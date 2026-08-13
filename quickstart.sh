#!/bin/bash
# 快速开始检查清单

echo "📋 OpenMythos Agent Quick Start Checklist"
echo "========================================="
echo ""

# 1. 环境检查
echo "✅ Step 1: Environment Check"
echo "   - Python 3.10+"
python3 --version
echo ""

# 2. 目录检查
echo "✅ Step 2: Directory Structure"
if [ -f "pyproject.toml" ]; then
    echo "   ✓ pyproject.toml exists"
else
    echo "   ✗ pyproject.toml missing"
fi

if [ -f "config.yaml" ]; then
    echo "   ✓ config.yaml exists"
else
    echo "   ⚠ config.yaml will be created on first run"
fi
echo ""

# 3. 依赖检查
echo "✅ Step 3: Dependencies"
pip list | grep -E "torch|chromadb|sentence-transformers" || echo "   Some dependencies missing"
echo ""

# 4. OpenMythos
echo "✅ Step 4: OpenMythos Installation"
python3 -c "import open_mythos; print('   ✓ OpenMythos is installed')" 2>/dev/null || echo "   ✗ OpenMythos not found - run: pip install -e ../OpenMythos-main"
echo ""

# 5. 数据目录
echo "✅ Step 5: Create data directories"
mkdir -p data/experience_store data/strategies data/reflections data/workspace logs
echo "   ✓ Directories created"
echo ""

# 6. 运行测试
echo "✅ Step 6: Run quick test"
echo "   python quick_test.py"
echo ""

echo "========================================="
echo "🎉 Setup complete! Ready to run:"
echo "   python -m agent_project          # Interactive"
echo "   python demo.py                   # Demos"
echo ""
