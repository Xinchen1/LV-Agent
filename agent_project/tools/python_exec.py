"""
Python Exec Tool - 受限Python代码执行 (最小可用安全沙箱)

安全模型 (Karpathy 极简):
  1. 代码在**独立子进程**中运行 → 真正的超时(超时即杀进程) + 进程级隔离,
     模型无法用死循环卡死 Agent, 也无法逃逸到 Agent 主进程。
  2. 子进程内只暴露一个**裁剪过的 __builtins__**: 没有 __import__ / open / eval / exec,
     从根上堵死 __import__('os')、open('/etc/shadow') 这类绕过。
  3. 模块白名单**不含 os/pathlib/glob** —— 文件操作本就属于 file_ops 工具的职责,
     不在这里给文件系统入口。
  4. 执行前做一次 **AST 审计**, 对明显危险写法(危险 import / 危险属性调用)直接拒绝,
     给出清晰错误, 不必起进程。AST 审计只是快速前置过滤; 真正的安全由 1+2 兜底。
"""

import ast as _ast
import io
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
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


# 暴露给执行代码的受限内置(刻意不含 __import__ / open / eval / exec / input / breakpoint)
_SAFE_BUILTIN_NAMES = [
    'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'any', 'all',
    'sum', 'min', 'max', 'sorted', 'list', 'dict', 'tuple', 'set', 'int', 'float',
    'str', 'bool', 'complex', 'abs', 'round', 'pow', 'divmod', 'ord', 'chr',
    'repr', 'format', 'isinstance', 'issubclass', 'hasattr', 'getattr', 'iter',
    'next', 'reversed', 'slice', 'bytes', 'bytearray', 'frozenset',
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'ZeroDivisionError', 'AttributeError', 'RuntimeError', 'StopIteration',
]

# 预加载的模块白名单 —— 不含任何能碰文件系统的模块(os/pathlib/glob/shutil)
_ALLOWED_MODULES = {
    'math', 'json', 'datetime', 're', 'collections',
    'random', 'statistics', 'itertools', 'functools',
    'decimal', 'fractions', 'numbers', 'typing', 'inspect',
    'textwrap', 'hashlib', 'string', 'heapq', 'bisect', 'base64',
}

# AST 审计: 禁止直接 import 的危险模块(首段)
_DANGEROUS_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'ctypes', 'ctypeslib',
    'multiprocessing', 'threading', 'builtins', 'importlib', 'pdb', 'code',
    'signal', 'resource', 'pty', 'fcntl', 'msvcrt', 'gc', 'pathlib', 'glob',
}
# 禁止调用的危险内置名
_DANGEROUS_NAMES = {'__import__', 'open', 'eval', 'exec', 'compile', 'quit', 'exit', 'input', 'breakpoint'}
# 禁止通过属性访问调用的危险方法(如 os.system / os.popen / os.remove / os.kill)
_DANGEROUS_ATTRS = {
    'system', 'popen', 'spawnl', 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
    'spawn', 'execv', 'execl', 'execve', 'execlp', 'execlpe', 'execvp', 'execvpe',
    'kill', 'killpg', 'remove', 'unlink', 'rmdir', 'rename', 'replace',
    'chmod', 'chown', 'mknod', 'mkfifo', 'truncate', 'chroot', 'chdir',
    'putenv', 'setuid', 'setgid', 'seteuid', 'setegid', 'symlink', 'unlink',
}

# 子进程内运行的受限执行包装器(固定模板, 用户代码经 stdin 传入)
# 占位符 {WL}/{SB} 用 .replace() 做字面替换: .replace 不做模板语法解析,
# 即使白名单/内置名字符串里含有 { / } 也不会被误当作占位符, 比 str.format 安全。
_WRAPPER_TEMPLATE = r'''
import sys, io, contextlib, traceback
code = sys.stdin.read()
gwhitelist = {WL}
import builtins as _b
_safe_names = {SB}
safe_builtins = {n: getattr(_b, n) for n in _safe_names if hasattr(_b, n)}
_allowed_imports = set(gwhitelist)
def _restricted_import(name, *a, **k):
    if name.split('.')[0] not in _allowed_imports:
        raise ImportError("import of '%s' is not allowed by the sandbox" % name)
    return __import__(name, *a, **k)
safe_builtins['__import__'] = _restricted_import
g = {"__builtins__": safe_builtins}
for _m in gwhitelist:
    try:
        g[_m] = __import__(_m)
    except Exception:
        pass
out = io.StringIO(); err = io.StringIO()
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        exec(code, g)
except BaseException:
    err.write(traceback.format_exc())
    sys.stdout.write(out.getvalue())
    se = err.getvalue()
    if se:
        sys.stdout.write("\n[STDERR]\n" + se)
    sys.exit(1)
sys.stdout.write(out.getvalue())
se = err.getvalue()
if se:
    sys.stdout.write("\n[STDERR]\n" + se)
'''


class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "Execute Python code and return output. Useful for data processing, calculations, testing ideas."

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must be a complete, self-contained script.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 10, max: 60)",
                "default": 10,
                "maximum": 60,
            },
        },
        "required": ["code"],
    }

    ALLOWED_MODULES = set(_ALLOWED_MODULES)

    def __init__(self, timeout: int = 10, allowed_modules: List[str] = None):
        self.timeout = timeout
        self.allowed_modules = set(allowed_modules or self.ALLOWED_MODULES)

    @staticmethod
    def _normalize_code(code: str) -> str:
        """修复模型输出中常见的缩进/换行问题。"""
        if not code or "\n" in code:
            return code

        if ":" not in code:
            return code

        lines = []
        rest = code.strip()
        while rest.startswith("import ") or rest.startswith("from "):
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

        segments = [s.strip() for s in re.split(r":\s+", rest) if s.strip()]
        if len(segments) <= 1:
            return code

        body = segments[-1]
        headers = segments[:-1]

        for i, header in enumerate(headers):
            lines.append("    " * i + header + ":")
        lines.append("    " * len(headers) + body)

        normalized = "\n".join(lines)
        try:
            _ast.parse(normalized)
            return normalized
        except SyntaxError:
            return code

    @staticmethod
    def _audit_code(code: str) -> Optional[str]:
        """AST 审计: 返回拒绝原因字符串; 通过则返回 None(SyntaxError 不算危险, 放行给执行层报错)。"""
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return None
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in _DANGEROUS_IMPORTS:
                        return f"import of unsafe module '{alias.name}'"
            elif isinstance(node, _ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in _DANGEROUS_IMPORTS:
                    return f"import of unsafe module '{node.module}'"
            elif isinstance(node, _ast.Call):
                f = node.func
                name = f.attr if isinstance(f, _ast.Attribute) else (f.id if isinstance(f, _ast.Name) else "")
                if name in _DANGEROUS_NAMES:
                    return f"call to disallowed builtin '{name}'"
                if isinstance(f, _ast.Attribute) and f.attr in _DANGEROUS_ATTRS:
                    return f"call to dangerous method '{f.attr}'"
                # 拦截 getattr(__builtins__/builtins/globals, 'xxx') 这类内省逃逸
                if name == "getattr" and node.args:
                    a0 = node.args[0]
                    if isinstance(a0, _ast.Name) and a0.id in ("__builtins__", "builtins", "globals"):
                        return "attempt to introspect builtins via getattr"
        return None

    def _build_wrapper(self) -> str:
        wl = repr(sorted(self.allowed_modules))
        sb = repr(_SAFE_BUILTIN_NAMES)
        return _WRAPPER_TEMPLATE.replace("{WL}", wl).replace("{SB}", sb)

    def execute(self, code: str, timeout: int = None) -> ToolResult:
        timeout = max(1, min(int(timeout or self.timeout), 60))
        code = self._normalize_code(code)

        # 1) AST 前置审计 —— 明显危险写法直接拒绝, 给出清晰错误
        block = self._audit_code(code)
        if block:
            return ToolResult(
                success=False,
                output="",
                error=f"Code rejected by security audit: {block}",
            )

        # 2) Harness 策略门
        allowed, reason = _harness_check(self.name, {"code": code})
        if not allowed:
            return ToolResult(success=False, output="", error=reason)

        # 3) 隔离子进程执行 —— 真正的超时 + 进程级隔离
        wrapper = self._build_wrapper()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONPATH": "", "PYTHONHOME": ""},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout}s (hard limit reached; process killed).",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution failed to start: {type(e).__name__}: {e}",
            )

        output = proc.stdout or ""
        if proc.returncode != 0:
            err = proc.stderr or ""
            return ToolResult(
                success=False,
                output=output or "(no output)",
                error=err or f"exit code {proc.returncode}",
            )

        if not output.strip():
            output = "(no output)"
        return ToolResult(
            success=True,
            output=output,
            metadata={"executed_lines": len(code.split("\n"))},
        )


TOOLS_REGISTRY.register(PythonExecTool())
