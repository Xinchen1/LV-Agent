"""
Python Exec Tool - 安全Python代码执行
注意：这是简化版本。生产环境应使用Docker沙箱或RestrictedPython
"""

import sys
import io
import re
import contextlib
from typing import Dict, Any, List, Optional, Tuple
from . import BaseTool, ToolResult, TOOLS_REGISTRY, get_harness_kernel


def _harness_check(tool_name: str, arguments: dict) -> Tuple[bool, Optional[str]]:
    """Return (allowed, reason_or_none) after harness policy evaluation."""
    kernel = get_harness_kernel()
    if kernel is None:
        return True, None
    try:
        from ..harness.effects import make_effect
        effect = make_effect(tool_name, arguments)
        admission = kernel.evaluate(effect)
        from ..harness.kernel import Decision
        if admission.decision == Decision.ALLOW:
            return True, None
        if admission.decision == Decision.DENY:
            return False, f"Harness denied: {admission.reason}"
        granted = kernel.ask(effect, admission.reason) if kernel.ask else False
        if granted:
            return True, None
        return False, f"Harness approval denied: {admission.reason}"
    except Exception as e:
        return False, f"Harness check failed: {e}"


class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "Execute Python code and return output. Useful for data processing, calculations, testing ideas."

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must be a complete, self-contained script."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 10, max: 60)",
                "default": 10,
                "maximum": 60
            }
        },
        "required": ["code"]
    }

    # 安全模块白名单
    ALLOWED_MODULES = {
        'math', 'json', 'datetime', 're', 'collections',
        'random', 'statistics', 'itertools', 'functools',
        'decimal', 'fractions', 'numbers', 'os', 'pathlib',
        'typing', 'inspect', 'textwrap', 'hashlib', 'glob'
    }

    def __init__(self, timeout: int = 10, allowed_modules: List[str] = None):
        self.timeout = timeout
        self.allowed_modules = set(allowed_modules or self.ALLOWED_MODULES)

    @staticmethod
    def _normalize_code(code: str) -> str:
        """修复模型输出中常见的缩进/换行问题。"""
        if not code or "\n" in code:
            return code

        # 单行代码但包含冒号分隔的嵌套语句（常见模型输出问题）
        # 例如: "import os for root, dirs, files in os.walk(...):  for f in files:  ..."
        if ":" not in code:
            return code

        # 先把 import 语句独立成行
        lines = []
        rest = code.strip()
        while rest.startswith("import ") or rest.startswith("from "):
            # 找到 import 语句结束位置（下一个 import/from 或第一个 for/if/while/def/class）
            next_kw = re.search(r"\b(?:import|from|for|if|while|def|class|with|try)\b", rest[7 if rest.startswith("import ") else 5:])
            if next_kw:
                stmt_end = 7 + next_kw.start() if rest.startswith("import ") else 5 + next_kw.start()
                lines.append(rest[:stmt_end].strip())
                rest = rest[stmt_end:].strip()
            else:
                lines.append(rest)
                rest = ""
                break

        if not rest:
            return "\n".join(lines)

        # 按冒号+空格拆分嵌套语句
        segments = [s.strip() for s in re.split(r":\s+", rest) if s.strip()]
        if len(segments) <= 1:
            return code

        # 最后一段是最终执行体，前面的都是带冒号的控制头
        body = segments[-1]
        headers = segments[:-1]

        for i, header in enumerate(headers):
            lines.append("    " * i + header + ":")
        lines.append("    " * len(headers) + body)

        normalized = "\n".join(lines)
        # 简单语法校验，失败则回退原输入
        try:
            import ast
            ast.parse(normalized)
            return normalized
        except SyntaxError:
            return code

    def execute(self, code: str, timeout: int = None) -> ToolResult:
        timeout = timeout or self.timeout
        code = self._normalize_code(code)

        # 安全检查：禁止危险操作
        if self._contains_dangerous_code(code):
            return ToolResult(
                success=False,
                output="",
                error="Code contains dangerous operations (subprocess, eval, exec, etc.)"
            )

        # Harness policy gate
        allowed, reason = _harness_check(self.name, {"code": code})
        if not allowed:
            return ToolResult(success=False, output="", error=reason)

        try:
            # 重定向stdout和stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # 创建受限的全局命名空间
                builtins_dict = {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'any': any,
                    'all': all,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'sorted': sorted,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'int': int,
                    'float': float,
                    'str': str,
                    'bool': bool,
                    'Exception': Exception,
                    '__import__': __import__,
                    'open': open,
                }

                globals_dict = {'__builtins__': builtins_dict}

                # 只导入白名单模块
                for module_name in self.allowed_modules:
                    try:
                        module = __import__(module_name)
                        globals_dict[module_name] = module
                    except ImportError:
                        pass

                # 执行代码（这里简化，不实现真正的超时）
                exec(code, globals_dict)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            output = stdout
            if stderr:
                output += f"\n[STDERR]\n{stderr}"

            return ToolResult(
                success=True,
                output=output or "(no output)",
                metadata={"executed_lines": len(code.split('\n'))}
            )

        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)
            hint = ""
            if err_type == "SyntaxError":
                hint = "\nHint: use file_ops verify for syntax checks, or pass code as JSON {\"code\": \"...\"}"
            snippet = code[:200].replace("\n", " ")
            return ToolResult(
                success=False,
                output="",
                error=f"{err_type}: {err_msg}{hint}\nCode snippet: {snippet}"
            )

    def _contains_dangerous_code(self, code: str) -> bool:
        """检查代码中是否包含危险操作"""
        dangerous_patterns = [
            'import subprocess',
            'eval(', 'exec(', 'compile(',
            'quit()', 'exit()',
            'os.system', 'os.popen', 'os.spawn',
            'subprocess.run', 'subprocess.call', 'subprocess.Popen',
        ]
        code_lower = code.lower()
        return any(pattern in code_lower for pattern in dangerous_patterns)


TOOLS_REGISTRY.register(PythonExecTool())
