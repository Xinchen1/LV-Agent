"""
简单测试数据 - 用于验证Agent基本功能
"""

TEST_TASKS = [
    {
        "task": "Calculate 123 * 456",
        "expected_tool": "calculator",
        "expected_output_contains": "56088"
    },
    {
        "task": "What is 2^10?",
        "expected_tool": "calculator",
        "expected_output_contains": "1024"
    },
    {
        "task": "List files in ./data directory",
        "expected_tool": "file_ops",
        "expected_output_contains": None  # just check it's a list
    },
    {
        "task": "Write 'Hello World' to ./data/test.txt",
        "expected_tool": "file_ops",
        "expected_output_contains": "Written"
    },
    {
        "task": "Read the file ./data/test.txt",
        "expected_tool": "file_ops",
        "expected_output_contains": "Hello World"
    },
]

# 可以用Python生成的样本代码
SAMPLE_CODE_TASKS = [
    "Write Python code to calculate fibonacci(10)",
    "Create a dictionary of squares from 1 to 10",
    "Sort a list of random numbers"
]
