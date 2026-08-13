# 项目结构说明

agent_project/
├── pyproject.toml              # Python包配置
├── requirements.txt            # 依赖列表（备用）
├── config.yaml                 # 主配置文件（运行前可编辑）
├── .env.example                # 环境变量示例
├── README.md                   # 使用指南
├── quick_test.py               # 快速检查脚本
├── demo.py                     # 示例任务演示
│
├── agent_project/              # 主包
│   ├── __init__.py             # 包入口
│   ├── __main__.py             # `python -m agent_project`入口
│   ├── agent.py                # OpenMythosAgent核心类
│   ├── config.py               # 配置加载和验证
│   ├── experience.py           # ExperienceBuffer
│   ├── reflection.py           # ReflectionModule
│   ├── strategies.py           # StrategyDatabase
│   │
│   └── tools/                  # 工具模块
│       ├── __init__.py         # 工具注册表
│       ├── web_search.py       # 网络搜索
│       ├── calculator.py       # 计算器
│       ├── python_exec.py      # Python代码执行
│       ├── file_ops.py         # 文件操作
│       └── api_call.py         # HTTP API调用
│
├── data/                       # 运行时数据（自动生成）
│   ├── experience_store/       # 向量数据库（ChromaDB）
│   ├── strategies/             # 策略文件
│   ├── reflections/            # 反思记录
│   └── workspace/              # 文件工具工作目录
│
├── logs/                       # 日志（自动生成）
│   └── agent.log
│
└── notebooks/                  # （可选）Jupyter示例
    └── demo.ipynb
