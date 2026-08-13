"""
Calculator Tool - 安全数学计算
"""

import ast
import math
from typing import Dict, Any
from . import BaseTool, ToolResult, TOOLS_REGISTRY


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e, etc."

    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
            }
        },
        "required": ["expression"]
    }

    # 安全允许的数学函数和常量
    SAFE_NAMES = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    SAFE_NAMES.update({
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    })

    def execute(self, expression: str) -> ToolResult:
        try:
            # 清理表达式
            expr = expression.strip()

            # 使用ast解析，确保安全
            tree = ast.parse(expr, mode='eval')

            # 验证只包含允许的操作
            self._validate_ast(tree)

            # 安全求值
            result = eval(compile(tree, '', 'eval'), {"__builtins__": {}}, self.SAFE_NAMES)

            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": expression, "result": result}
            )

        except ZeroDivisionError:
            return ToolResult(success=False, output="", error="Division by zero")
        except (SyntaxError, NameError, TypeError, ValueError) as e:
            return ToolResult(success=False, output="", error=f"Invalid expression: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Calculation error: {str(e)}")

    def _validate_ast(self, tree):
        """验证AST只包含安全节点"""
        allowed_nodes = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.USub, ast.UAdd, ast.Call, ast.Module
        }

        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Disallowed operation: {type(node).__name__}")

            # 检查函数调用是否安全
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.SAFE_NAMES:
                        raise ValueError(f"Function '{node.func.id}' is not allowed")


TOOLS_REGISTRY.register(CalculatorTool())
