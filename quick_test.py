#!/usr/bin/env python3
"""
简易快速测试脚本
不依赖完整依赖，仅验证代码结构
"""

import sys
from pathlib import Path

print("🔍 Quick sanity check...\n")

# 检查文件结构
expected_files = [
    "agent_project/__init__.py",
    "agent_project/agent.py",
    "agent_project/config.py",
    "agent_project/tools/__init__.py",
    "agent_project/tools/web_search.py",
    "agent_project/tools/calculator.py",
    "agent_project/tools/python_exec.py",
    "agent_project/tools/file_ops.py",
    "agent_project/tools/api_call.py",
    "agent_project/experience.py",
    "agent_project/reflection.py",
    "agent_project/strategies.py",
    "agent_project/__main__.py",
    "pyproject.toml",
    "config.yaml",
    "README.md",
]

all_exist = True
for f in expected_files:
    if Path(f).exists():
        print(f"✓ {f}")
    else:
        print(f"✗ {f} MISSING")
        all_exist = False

if all_exist:
    print("\n✅ All files present!")
else:
    print("\n❌ Some files missing!")
    sys.exit(1)

# 尝试导入
print("\n📦 Checking imports...")
try:
    import yaml
    print("✓ PyYAML")
except ImportError:
    print("✗ PyYAML not installed")

try:
    import chromadb
    print("✓ ChromaDB")
except ImportError:
    print("✗ ChromaDB not installed (will fail at runtime)")

try:
    import sentence_transformers
    print("✓ sentence-transformers")
except ImportError:
    print("✗ sentence-transformers not installed (will fail at runtime)")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed")

# 检查OpenMythos
try:
    from open_mythos.main import OpenMythos, MythosConfig
    print("✓ OpenMythos")
except ImportError:
    print("✗ OpenMythos not found (run: pip install -e ../OpenMythos-main)")

print("\n✅ Sanity check complete!")
print("\nNext steps:")
print("  1. pip install -e .")
print("  2. Ensure OpenMythos is installed")
print("  3. python -m agent_project --run-tests")
