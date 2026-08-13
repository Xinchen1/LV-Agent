#!/usr/bin/env python3
"""
Lv Super Agent - 主入口
本地CPU环境下的深度思考智能体
"""

import sys
from pathlib import Path

# 添加父目录到 path，方便导入 OpenMythos
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_project.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
