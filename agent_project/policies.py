"""
Thinking policies for the unified ExecutionEngine.

Each policy decides:
  - first prompt given the task/context
  - how to parse model output into reasoning / tool_calls / final_answer
  - how to build the next prompt given observations

This replaces the policy logic that previously lived inside ReasoningEngine
(_reason_react, _reason_super, _reason_cot, _reason_verify, _reason_zero_shot)
and inside OpenMythosAgent._run_traditional.
"""

from __future__ import annotations

import json
import os
import re
import ast
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .execution_engine import ExecutionContext, PolicyOutput, ToolCallRequest
from .tools import TOOLS_REGISTRY


def _cwd() -> str:
    """当前工作目录(喂给模型, 避免它瞎猜绝对路径)."""
    try:
        return os.getcwd()
    except Exception:
        return "."


# ---------------------------------------------------------------------------
# Shared parsing helpers (consolidated from reasoning.py + agent.py)
# ---------------------------------------------------------------------------

class ToolCallParser:
    """Robust parser for [TOOL:name] / XML / JSON / function-call formats."""

    FILE_OPS_ACTIONS = {
        "read", "multi_read", "fast_read", "write", "list", "exists",
        "analyze", "grep", "diff", "backup", "find", "apply_diff", "verify", "open",
    }

    @classmethod
    def parse_all(cls, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        calls: List[Tuple[str, Dict[str, Any]]] = []
        if not text:
            return calls

        registry = TOOLS_REGISTRY

        def strip_tool_markers(s: str) -> str:
            return re.sub(r"\[/TOOL\]|</tool_call>|</function>", "", s, flags=re.IGNORECASE).strip()

        def add_call(tool_name: str, args: Dict[str, Any]):
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                return
            args = cls._sanitize_parsed_args(args)
            # 工具名大小写归一化: 模型可能输出 BASH_EXEC / Bash_Exec, registry 用小写
            tool = registry.get(tool_name)
            if tool is None:
                tool = registry.get(tool_name.lower())
            if tool is None:
                tool = registry.get(tool_name.strip().lower())
            if tool is None:
                return
            # 统一使用注册名, 后续 file_ops/python_exec 特判也用注册名
            tool_name = tool.name if hasattr(tool, "name") else tool_name
            if tool_name == "file_ops":
                action = args.get("action")
                for key in list(args.keys()):
                    if key in cls.FILE_OPS_ACTIONS and key != "action":
                        if args[key] is True or (isinstance(args[key], str) and args[key].strip().lower() == action):
                            del args[key]
                if isinstance(args.get("path"), str):
                    args["path"] = cls._sanitize_file_ops_path(args["path"])
                if isinstance(args.get("paths"), list):
                    args["paths"] = [cls._sanitize_file_ops_path(p) for p in args["paths"]]
            if tool_name == "python_exec":
                code = args.get("code", "")
                if not isinstance(code, str) or not code.strip():
                    return
            schema = getattr(tool, "parameters", {}) or {}
            required = schema.get("required", []) if isinstance(schema, dict) else []
            for key in required:
                val = args.get(key)
                if val is None or (isinstance(val, str) and not val.strip()):
                    return
            calls.append((tool_name, args))

        # Format 1: [TOOL:name] args [/TOOL]
        for match in re.finditer(r"\[TOOL:([^\]]+)\](.*?)\[/TOOL\]", text, re.DOTALL | re.IGNORECASE):
            tool_name = match.group(1).strip()
            args_str = strip_tool_markers(match.group(2))
            if tool_name == "python_exec":
                add_call(tool_name, cls._extract_python_exec_code(args_str))
            else:
                add_call(tool_name, cls._parse_args(args_str))

        # Format 1b: [TOOL:name] args without closing tag
        # 允许行首列表标记(如 "- [TOOL:...]" / "1. [TOOL:...]" / "* [TOOL:...]")
        valid_lower = set(t.lower() for t in registry.list_tools())
        for match in re.finditer(r"^[ \t]*(?:[-*•]|\d+[.)]|>)?[ \t]*\[TOOL:([^\]]+)\][ \t]*(.*?)[ \t]*$", text, re.MULTILINE | re.IGNORECASE):
            tool_name = match.group(1).strip()
            args_str = strip_tool_markers(match.group(2))
            if tool_name.lower() not in valid_lower:
                continue
            if tool_name.lower() == "python_exec":
                add_call(tool_name, cls._extract_python_exec_code(args_str))
            else:
                add_call(tool_name, cls._parse_args(args_str))

        # Format 2: XML <tool_call>
        for match in re.finditer(
            r"<tool_call>\s*<function=(.+?)>(.*?)</function>\s*</tool_call>", text, re.DOTALL | re.IGNORECASE
        ):
            tool_name = match.group(1).strip().strip('"').strip("'")
            inner = strip_tool_markers(match.group(2))
            args = {}
            for param_match in re.finditer(r"<parameter=(.+?)>(.*?)</parameter>", inner, re.DOTALL | re.IGNORECASE):
                key = param_match.group(1).strip().strip('"').strip("'")
                value = param_match.group(2).strip()
                args[key] = value
            add_call(tool_name, args)

        # Format 3: python_exec(code=...)
        for match in re.finditer(r"^[ \t]*python_exec\s*\((.*?)\)[ \t]*$", text, re.MULTILINE | re.DOTALL | re.IGNORECASE):
            args_str = strip_tool_markers(match.group(1))
            add_call("python_exec", cls._extract_python_exec_code(args_str))

        # Format 4: OpenCode wrapped calls
        for match in re.finditer(
            r"<\|message_model\|>\s*([\w_]+)\s*<\|content_invoke_tool_json\|>\s*(.*?)\s*<\|end_message\|>",
            text, re.DOTALL
        ):
            tool_name = match.group(1).strip()
            payload_text = match.group(2).strip()
            payload = None
            first_brace = payload_text.find("{")
            if first_brace >= 0:
                for start in range(first_brace, len(payload_text)):
                    candidate = payload_text[start:]
                    if not candidate.startswith("{"):
                        continue
                    try:
                        payload = json.loads(candidate)
                        if isinstance(payload, dict):
                            break
                    except Exception:
                        continue
            if not isinstance(payload, dict):
                continue
            args = payload.get("args", {})
            if isinstance(args, dict):
                if tool_name == "python_exec":
                    add_call(tool_name, cls._extract_python_exec_code(args.get("code", "")))
                else:
                    add_call(tool_name, args)

        # Format 5: JSON {"action": [...]}
        first_brace = text.find("{")
        if first_brace >= 0:
            for start in range(first_brace, len(text)):
                candidate = text[start:]
                if not candidate.startswith("{"):
                    continue
                try:
                    payload = cls._normalize_json_keys(json.loads(candidate))
                    if not isinstance(payload, dict):
                        continue
                    actions = payload.get("action") or payload.get("actions") or payload.get("tool_calls")
                    if isinstance(actions, list):
                        for action in actions:
                            if not isinstance(action, dict):
                                continue
                            tool_name = action.get("name") or action.get("tool") or action.get("action")
                            args = action.get("args") or action.get("arguments") or action.get("parameters", {})
                            if tool_name == "python_exec":
                                add_call(tool_name, cls._extract_python_exec_code(args.get("code", "")))
                            else:
                                add_call(tool_name, args)
                    elif isinstance(actions, dict):
                        tool_name = actions.get("name") or actions.get("tool") or actions.get("action")
                        args = actions.get("args") or actions.get("arguments") or actions.get("parameters", {})
                        if tool_name:
                            if tool_name == "python_exec":
                                add_call(tool_name, cls._extract_python_exec_code(args.get("code", "")))
                            else:
                                add_call(tool_name, args)
                    break
                except Exception:
                    continue

        # Format 5b: JSON {"action": "tool_name", "args": {...}} (action 是字符串, args 同级)
        # 模型常输出 {thoughts/action/args} 或 {reasoning/action/arguments} 单工具 JSON
        if not calls:
            first_brace = text.find("{")
            if first_brace >= 0:
                for start in range(first_brace, len(text)):
                    candidate = text[start:]
                    if not candidate.startswith("{"):
                        continue
                    try:
                        payload = cls._normalize_json_keys(json.loads(candidate))
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    action_name = payload.get("action") or payload.get("tool") or payload.get("name")
                    if not isinstance(action_name, str):
                        continue
                    args = payload.get("args") or payload.get("arguments") or payload.get("parameters", {})
                    if not isinstance(args, dict):
                        args = {}
                    if action_name == "python_exec":
                        add_call(action_name, cls._extract_python_exec_code(args.get("code", "")))
                    else:
                        add_call(action_name, args)
                    break

        # Format 6: function-call style: tool_name(key="value", key2="value2")
        # 用平衡括号匹配, 支持嵌套括号(如 python_exec("shutil.rmtree(p)"))
        if not calls:
            func_pattern = re.compile(r"(?<!\w)([a-zA-Z_][\w_]*)\s*\(", re.DOTALL)
            for m in func_pattern.finditer(text):
                tool_name = m.group(1).strip()
                if registry.get(tool_name) is None and registry.get(tool_name.lower()) is None:
                    continue
                start = m.end() - 1
                depth = 0
                i = start
                while i < len(text):
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                if depth != 0:
                    continue
                args_str = text[start + 1:i].strip()
                if args_str:
                    single_str = (args_str.startswith('"') and args_str.endswith('"')) or \
                                 (args_str.startswith("'") and args_str.endswith("'"))
                    tool_lower = tool_name.lower()
                    if single_str and tool_lower in ("bash_exec", "python_exec", "run_code"):
                        inner = args_str[1:-1]
                        inner = inner.replace('\\"', '"').replace("\\'", "'")
                        arguments = {("command" if tool_lower == "bash_exec" else "code"): inner}
                    else:
                        arguments = cls._parse_args(args_str)
                else:
                    arguments = {}
                add_call(tool_name, arguments)

        seen = set()
        unique = []
        for name, args in calls:
            try:
                key = f"{name}:{cls._format_args(args)}"
                if key not in seen:
                    seen.add(key)
                    unique.append((name, args))
            except TypeError:
                continue
        return unique

    @classmethod
    def _extract_python_exec_code(cls, args_str: str) -> Dict[str, str]:
        if not args_str:
            return {"code": ""}
        cleaned = re.sub(r"\[/TOOL\]|</tool_call>|</function>|</parameter>", "", args_str, flags=re.IGNORECASE).strip()
        for candidate in [cleaned, args_str]:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "code" in parsed:
                    return {"code": str(parsed["code"])}
            except Exception:
                pass
        m = re.search(r"\{\s*\"code\"\s*:\s*\".*?\"\s*\}", cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return {"code": str(parsed["code"])}
            except Exception:
                pass
        args = cls._parse_args(cleaned)
        if "code" in args:
            code = str(args["code"])
            try:
                ast.parse(code)
                return {"code": code}
            except SyntaxError:
                fixed = cls._extract_balanced_code_value(cleaned)
                if fixed is not None:
                    return {"code": fixed}
                return {"code": code}
        fixed = cls._extract_balanced_code_value(cleaned)
        if fixed is not None:
            return {"code": fixed}
        m = re.match(r"^[\w_]+\s*\((\{.*\})\)\s*$", cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict) and "code" in parsed:
                    return {"code": str(parsed["code"])}
            except Exception:
                pass
        m = re.match(r"^[\w_]+\s*\((.*)\)\s*$", cleaned, re.DOTALL)
        if m:
            args = cls._parse_args(m.group(1))
            if "code" in args:
                return {"code": str(args["code"])}
        code = cleaned
        if code.startswith("```"):
            lines = code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines).strip()
        return {"code": code}

    @classmethod
    def _extract_balanced_code_value(cls, s: str) -> Optional[str]:
        idx = s.find("code=")
        if idx < 0:
            return None
        start = idx + len("code=")
        if start >= len(s):
            return None
        quote = s[start]
        if quote not in ('"', "'"):
            return None
        i = start + 1
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s):
                i += 2
            elif s[i] == quote:
                return s[start + 1 : i]
            else:
                i += 1
        return None

    @classmethod
    def _parse_args(cls, args_str: str) -> Dict[str, Any]:
        import re
        args: Dict[str, Any] = {}
        if not args_str:
            return args
        # 注意: 中文全角引号(“ ” ‘ ’)在 JSON 字符串内是合法字符, 不应替换成半角(会破坏 JSON)。
        # 先尝试原样解析(最标准), 再尝试反引号替换版本(模型有时用 ` 代替 JSON 引号).
        candidates = [args_str, "{" + args_str + "}"]
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return cls._sanitize_parsed_args(parsed)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return cls._sanitize_parsed_args(parsed)
            except (SyntaxError, ValueError):
                continue

        # 反引号替换版本: 模型有时用 ` 包 JSON 键值(尤其键). 仅当原样解析失败时尝试.
        quote_normalized = args_str.replace("`", '"')
        candidates2 = [quote_normalized, "{" + quote_normalized + "}"]
        for candidate in candidates2:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return cls._sanitize_parsed_args(parsed)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return cls._sanitize_parsed_args(parsed)
            except (SyntaxError, ValueError):
                continue

        # 健壮兜底: 提取最外层 JSON 对象, 逐位置尝试解析
        # 模型可能在 content 里内嵌未转义的引号/中文引号, 导致整块 JSON 解析失败。
        # 这里尝试找到所有 '{' 起始的候选子串, 用 json.loads 逐个试, 取第一个能完整解析的。
        first_brace = quote_normalized.find("{")
        if first_brace >= 0:
            for start in range(first_brace, len(quote_normalized)):
                candidate = quote_normalized[start:]
                if not candidate.startswith("{"):
                    continue
                # 用 json.JSONDecoder.raw_decode 从指定位置解析, 支持前置内容
                try:
                    from json import JSONDecoder
                    parsed, _end = JSONDecoder().raw_decode(candidate)
                    if isinstance(parsed, dict):
                        return cls._sanitize_parsed_args(parsed)
                except Exception:
                    pass
                # 也试 ast.literal_eval
                try:
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, dict):
                        return cls._sanitize_parsed_args(parsed)
                except Exception:
                    continue

        normalized = re.sub(r"\s+", " ", quote_normalized).strip()
        try:
            tokens = shlex.split(normalized)
            for token in tokens:
                token = token.strip().rstrip(",;")
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
                    v = v[1:-1]
                args[k] = v
            if args:
                return cls._sanitize_parsed_args(args)
        except ValueError:
            pass
        pairs = cls._split_key_value_pairs(normalized)
        for pair in pairs:
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
                v = v[1:-1]
            args[k] = v
        return cls._sanitize_parsed_args(args)

    @classmethod
    def _split_key_value_pairs(cls, s: str) -> List[str]:
        pairs = []
        current = []
        in_quote = None
        for ch in s:
            if ch in ('"', "'"):
                if in_quote is None:
                    in_quote = ch
                elif in_quote == ch:
                    in_quote = None
                current.append(ch)
            elif ch in ",;\n" and in_quote is None:
                pairs.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            pairs.append("".join(current))
        return pairs

    @classmethod
    def _sanitize_parsed_args(cls, parsed: Dict[Any, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in parsed.items():
            if not isinstance(k, (str, int, float, bool)):
                continue
            result[str(k)] = cls._sanitize_value(v)
        return result

    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        if isinstance(value, set):
            return sorted([cls._sanitize_value(v) for v in value], key=str)
        if isinstance(value, tuple):
            return [cls._sanitize_value(v) for v in value]
        if isinstance(value, list):
            return [cls._sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): cls._sanitize_value(v) for k, v in value.items() if isinstance(k, (str, int, float, bool))}
        return value

    @classmethod
    def _normalize_json_keys(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k).lower(): cls._normalize_json_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._normalize_json_keys(v) for v in obj]
        return obj

    @classmethod
    def _sanitize_file_ops_path(cls, path: Any) -> Any:
        if not isinstance(path, str):
            return path
        path = path.strip()
        path = re.sub(r"[\u200b\u200c\u200d\ufeff\x00-\x08\x0b-\x1f]", "", path)
        path = re.sub(r"(?<=[a-z0-9_\-])\s+(?=[a-z0-9_\-])", "", path)
        return path

    @classmethod
    def _format_args(cls, args: Dict[str, Any]) -> str:
        if not args:
            return ""
        if not isinstance(args, dict):
            return str(args)
        parts = []
        for k, v in args.items():
            if isinstance(v, set):
                v = sorted(v, key=str)
            parts.append(f'{k}="{v}"')
        return ", ".join(parts)

    @classmethod
    def strip_tool_calls(cls, text: str) -> str:
        text = re.sub(r"\[TOOL:[^\]]+\].*?\[/TOOL\]", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^[ \t]*\[TOOL:[^\]]+\][ \t]*.*?[ \t]*$", "", text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<tool_call>\s*<function=.+?>.*?</function>\s*</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"<\|message_model\|>\s*[\w_]+\s*<\|content_invoke_tool_json\|>.*?<\|end_message\|>",
            "", text, flags=re.DOTALL
        )
        first_brace = text.find("{")
        if first_brace >= 0:
            for start in range(first_brace, len(text)):
                candidate = text[start:]
                if not candidate.startswith("{"):
                    continue
                try:
                    payload = cls._normalize_json_keys(json.loads(candidate))
                    if isinstance(payload, dict) and ("action" in payload or "actions" in payload or "tool_calls" in payload):
                        text = text[:start] + json.dumps({k: v for k, v in payload.items() if k not in ("action", "actions", "tool_calls")})
                        break
                except Exception:
                    continue
        return text.strip()


# ---------------------------------------------------------------------------
# Base policy
# ---------------------------------------------------------------------------

class ThinkingPolicy(ABC):
    name = "base"

    @abstractmethod
    def first_prompt(self, ctx: ExecutionContext) -> str:
        ...

    @abstractmethod
    def parse_output(self, output: str, ctx: ExecutionContext) -> PolicyOutput:
        ...

    def next_prompt(self, ctx: ExecutionContext, last_output: str) -> Optional[str]:
        return None


# ---------------------------------------------------------------------------
# ReAct policy
# ---------------------------------------------------------------------------

class ReActPolicy(ThinkingPolicy):
    name = "react"

    def __init__(self, include_available_tools: bool = True):
        self.include_available_tools = include_available_tools

    def first_prompt(self, ctx: ExecutionContext) -> str:
        return self._build_react_prompt(ctx.task, ctx.available_tools, ctx.code_mode, ctx.extra_context)

    def parse_output(self, output: str, ctx: ExecutionContext) -> PolicyOutput:
        return self._route_react_output(output)

    def next_prompt(self, ctx: ExecutionContext, last_output: str) -> Optional[str]:
        base = self._build_react_prompt(ctx.task, ctx.available_tools, ctx.code_mode, ctx.extra_context)
        history = self._format_react_history(ctx.steps)
        return base + "\n" + history + "\nAgent:"

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_react_prompt(
        self,
        task: str,
        tools: Optional[Dict[str, Any]],
        code_mode: bool = False,
        extra_context: str = "",
    ) -> str:
        base = f"""You are an AI that can reason, use tools, and remembers past conversations.

Task: {task}

Working directory: {_cwd()}
(Use paths RELATIVE to the working directory above for file_ops. Never invent absolute paths.)

CRITICAL: Put all your internal reasoning inside <think>...</think> tags. The user will see this thinking process in real time. Keep the actual Thought: lines brief; the detailed reasoning goes into <think>...</think>.

{f'''You are in CODE MODE. This task is a software engineering task. Prioritize speed, precision, and verification.
Engineering workflow (Claude Code intelligence + Cursor speed):
  1. UNDERSTAND: Use project_context first. Then read target files with file_ops read and line_numbers=true so you can see exact line numbers and text. Never guess file contents.
  2. PLAN: Decide the minimal set of files to change. Prefer many small, correct edits over one giant rewrite. For multi-file changes, list the files and edit order before starting.
  3. IMPLEMENT: Use file_ops apply_diff with search/replace blocks for every code change. For new files use file_ops write. Do NOT use python_exec to create, modify, or overwrite files; use python_exec only for running scripts, tests, or computations. NEVER output a whole file in Final Answer unless asked.
  4. VERIFY: After every apply_diff or write, the system automatically runs file_ops verify. If verification fails, read the error, fix with apply_diff, and re-verify. Then run bash_exec with the appropriate test/linter (python -m py_compile, pytest, npm test, cargo check, etc.). Repeat the test-fix loop until all checks pass.
  5. REVIEW: Before finalizing, re-read the changed file(s) briefly with line_numbers=true to confirm the diff applied exactly as intended.
Do not explain code in Final Answer unless asked; just report what changed and the verification/test result.'''
if code_mode else 'Follow the ReAct pattern exactly. Each step must be ONE of:'}

Follow the ReAct pattern exactly. Each step must be ONE of:

1) Thought + Action (when you need a tool):
<think>[your detailed reasoning about what to do]</think>
Thought: [brief summary]
Action: [TOOL:tool_name] {{"arg": "value"}} [/TOOL]

2) Thought + Final Answer (when you have enough information):
<think>[brief reasoning]</think>
Thought: [brief summary]
Final Answer: [your complete final answer in the same language as the task]

Rules:
- SMALL STEPS, FAST ITERATION: Break the task into the smallest verifiable units. Each response advances by ONE concrete micro-step.
- SELF-VERIFY EACH STEP: Inside <think>, first ask "What is the next smallest step?", then reason, then ask "Is this step sound?".
- Use tools ONLY when you need external or current information. Greetings and simple math do NOT require tools.
- After you see an Observation, analyze it and then either act again or give Final Answer.
- You may output ONE independent tool call per response, or MULTIPLE tool calls if they are independent.
- ALWAYS use this exact tool syntax: [TOOL:tool_name] {{"arg1": "value1"}} [/TOOL]. Arguments MUST be valid JSON.
- For python_exec, put raw multi-line code between the tags, or use JSON: [TOOL:python_exec] {{"code": "..."}} [/TOOL].
- For file_ops, always use JSON: [TOOL:file_ops] {{"action": "read", "path": "./utils.py"}} [/TOOL].
- DO NOT repeat the same tool call with the same arguments.
- DO NOT invent tools not in the Available tools list.
- Final Answer must be plain text, not a tool call.
- LOCAL FILES: If the task asks about files/articles/documents in the current directory, FIRST use file_ops to list, then read. Do NOT search the web before checking local files.
- LOCATE FIRST: For finding projects/folders/files by name, use bash_exec `find` or search_files directly.
- OPEN FILES: When the user says 'open' / '打开' a file, folder, or report, immediately use file_ops with action='open' and the exact path. This launches it with the default system application (Preview, Finder, browser, etc.). Do NOT just say it is saved; open it.
- PARTIAL READS: Use file_ops with offset and limit when only a range is needed.
- PROJECT ANALYSIS: For analyzing a project, FIRST use project_context, then read key files.
- PRECISE CODE EDITS: Use file_ops apply_diff with search/replace blocks. Read target files with line_numbers=true first.
- NEVER use file_ops write on an existing file unless asked to create/overwrite.
- NEVER use python_exec to create, modify, or overwrite files.
- VERIFY WITH THE RIGHT TOOL: Use file_ops verify for syntax checks, not python_exec.
- IF A TOOL FAILS, pivot immediately; do NOT retry the exact same failed call more than once.
- If you see 'SYSTEM STOP: You already executed...', do NOT repeat that tool call.
"""
        if extra_context:
            base += f"\n\n{extra_context}\n"
        if self.include_available_tools and tools:
            base += "\nAvailable tools:\n"
            for name, desc in tools.items():
                base += f"- {name}: {desc}\n"
        base += "\nStart:\n"
        return base

    def _format_react_history(self, steps: List[Any]) -> str:
        lines = []
        for step in steps:
            if step.reasoning:
                lines.append(f"Thought: {step.reasoning[:300]}")
            for obs in step.observations:
                lines.append(f"Observation: {obs[:2000]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Output routing
    # ------------------------------------------------------------------

    def _route_react_output(self, output: str) -> PolicyOutput:
        final_markers = [
            "FINAL ANSWER:", "Final Answer:", "最终答案：", "最终答案:",
            "总结：", "结论：", "综上所述", "最终建议",
        ]
        tool_calls = ToolCallParser.parse_all(output)

        # Heuristic: markdown report without explicit final marker
        if not tool_calls and len(output) > 300 and (output.count("## ") + output.count("### ")) >= 2:
            cleaned = self._strip_think_tags(output.strip())
            return PolicyOutput(reasoning="", final_answer=cleaned, done=True)

        # Locate tool calls / final marker boundaries to extract reasoning
        first_tool_pos = None
        if tool_calls:
            m = re.search(r"\[TOOL:[^\]]+\].*?\[/TOOL\]", output, re.DOTALL)
            if m:
                first_tool_pos = m.start()
            else:
                call_str = f"{tool_calls[0][0]}({ToolCallParser._format_args(tool_calls[0][1])})"
                first_tool_pos = output.find(call_str)

        final_pos = None
        final_marker = None
        for marker in final_markers:
            idx = output.find(marker)
            if idx != -1 and (final_pos is None or idx < final_pos):
                final_pos = idx
                final_marker = marker

        boundary_candidates = [p for p in [first_tool_pos, final_pos] if p is not None]
        boundary = min(boundary_candidates) if boundary_candidates else None

        # Try JSON thought extraction
        json_thought = None
        first_brace = output.find("{")
        if first_brace >= 0:
            for start in range(first_brace, len(output)):
                candidate = output[start:]
                if not candidate.startswith("{"):
                    continue
                try:
                    payload = ToolCallParser._normalize_json_keys(json.loads(candidate))
                    if isinstance(payload, dict):
                        json_thought = payload.get("thought") or payload.get("reasoning") or payload.get("content")
                        break
                except Exception:
                    continue

        if json_thought is not None:
            thought = str(json_thought).strip()
        elif boundary is not None:
            thought = output[:boundary].strip()
        else:
            thought = output.strip()

        thought = self._strip_think_tags(thought)

        if tool_calls:
            requests = [ToolCallRequest(tool_name=n, arguments=a) for n, a in tool_calls]
            return PolicyOutput(reasoning=thought, tool_calls=requests)

        if final_pos is not None and final_marker:
            final_text = output.split(final_marker, 1)[1].strip()
            final_text = self._strip_think_tags(final_text)
            return PolicyOutput(reasoning=thought, final_answer=final_text, done=True)

        # JSON 里带 final_answer/answer 字段: 提取它, 而不是把整个 JSON 当答案
        if first_brace >= 0:
            for start in range(first_brace, len(output)):
                candidate = output[start:]
                if not candidate.startswith("{"):
                    continue
                try:
                    payload = ToolCallParser._normalize_json_keys(json.loads(candidate))
                    if isinstance(payload, dict):
                        fa = payload.get("final_answer") or payload.get("final answer") or payload.get("answer")
                        if isinstance(fa, str) and fa.strip():
                            return PolicyOutput(reasoning=thought, final_answer=fa.strip(), done=True)
                        break
                except Exception:
                    continue

        # Fallback: plain answer
        stripped = self._strip_think_tags(output.strip())
        if stripped:
            return PolicyOutput(reasoning=thought, final_answer=stripped, done=True)
        return PolicyOutput(reasoning=thought)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Thought:") or stripped.startswith("Action:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Super Agent policy (adaptive meta-loop)
# ---------------------------------------------------------------------------

class SuperAgentPolicy(ThinkingPolicy):
    name = "super_agent"

    def __init__(self, model_backend: Any):
        self.model = model_backend
        self.inner = ReActPolicy()

    def first_prompt(self, ctx: ExecutionContext) -> str:
        return self.inner.first_prompt(ctx)

    def parse_output(self, output: str, ctx: ExecutionContext) -> PolicyOutput:
        return self.inner.parse_output(output, ctx)

    def next_prompt(self, ctx: ExecutionContext, last_output: str) -> Optional[str]:
        return self.inner.next_prompt(ctx, last_output)

    def run_meta_loop(
        self,
        engine,
        ctx: ExecutionContext,
    ) -> Any:
        """Called by ExecutionEngine variants that support meta-looping."""
        max_attempts = 3
        best_trace = None
        best_score = -1.0
        current_task = ctx.task

        executed_calls: Dict[str, str] = {}
        call_counts: Dict[str, int] = {}

        for attempt in range(max_attempts):
            sub_ctx = ExecutionContext(
                task=current_task,
                available_tools=ctx.available_tools,
                config=ctx.config,
                max_steps=ctx.max_steps,
                stream_callback=ctx.stream_callback,
                token_callback=ctx.token_callback,
                code_mode=ctx.code_mode,
                extra_context=ctx.extra_context,
                history_context=ctx.history_context,
            )
            sub_ctx.executed_calls = dict(executed_calls)
            sub_ctx.call_counts = dict(call_counts)

            trace = engine.run(self.inner, sub_ctx)

            # Merge session dedup state so future attempts avoid repeated calls
            executed_calls.update(sub_ctx.executed_calls)
            for k, v in sub_ctx.call_counts.items():
                call_counts[k] = call_counts.get(k, 0) + v

            score = trace.quality_score
            if score > best_score:
                best_score = score
                best_trace = trace

            if score >= 0.85 or attempt == max_attempts - 1:
                break

            if not self._has_actionable_errors(trace.observations):
                break

            reflection = self._reflect_and_replan(trace, ctx)
            current_task = ctx.task + "\n\n## Reflection & Revised Plan (attempt " + str(attempt + 2) + "):\n" + reflection

        return best_trace

    def _has_actionable_errors(self, observations: List[str]) -> bool:
        markers = [
            "tool error", "tool execution error", "verification failed",
            "search block not found", "file does not exist", "syntax error",
            "cannot access local variable", "name '", "is not defined",
        ]
        # 注意: "system stop: you already executed" (去重) 是良性提示,
        # 模型应换工具继续, 不算可重试的错误, 避免无谓的 reflection 重试。
        for obs in observations[-10:]:
            obs_lower = obs.lower()
            if any(m in obs_lower for m in markers):
                return True
        return False

    def _reflect_and_replan(self, trace: Any, ctx: ExecutionContext) -> str:
        recent = trace.observations[-6:] if len(trace.observations) > 6 else trace.observations
        lines = ["Task: " + ctx.task, "", "Previous tool results:"]
        for i, obs in enumerate(recent, 1):
            lines.append(f"{i}. {obs[:800]}{'...' if len(obs) > 800 else ''}")
        lines.extend([
            "",
            "Critique the previous attempt in one sentence, then provide a 3-5 step revised plan.",
            "The revised plan must avoid repeating failed tool calls and fix any syntax/path/argument errors.",
            "Revised plan:",
        ])
        prompt = "\n".join(lines)
        reflection = self.model.generate(prompt, n_loops=1, temperature=0.4, max_tokens=2048)
        cleaned = ReActPolicy._strip_think_tags(reflection or "")
        if ctx.stream_callback:
            ctx.stream_callback("tool_result", "[reflection] " + cleaned[:300])
        return cleaned


# ---------------------------------------------------------------------------
# Chain-of-Thought policy
# ---------------------------------------------------------------------------

class CoTPolicy(ThinkingPolicy):
    name = "chain_of_thought"

    def first_prompt(self, ctx: ExecutionContext) -> str:
        base = f"""You are a deep reasoning AI. Solve the following problem step by step.

Task: {ctx.task}

CRITICAL: Put all your internal reasoning inside <think>...</think> tags.

Think through this systematically:
1. Analyze what is being asked
2. Identify key information needed
3. Plan your reasoning steps
4. Execute reasoning, showing your work
5. Arrive at a final answer

When you need external tools, use format: [TOOL:name] arguments [/TOOL]

After your thinking, provide the final answer in plain text (no more <think> tags).

LOCAL FILES: If the task asks about files, articles, documents, or contents in the current directory, FIRST use file_ops to list and read. Do NOT ask the user for content. Do NOT search the web before checking local files.
OPEN FILES: When the user says 'open' / '打开' a file, folder, or report, use file_ops with action='open' and the exact path.
"""
        if ctx.available_tools:
            base += "\nAvailable tools:\n"
            for name, desc in ctx.available_tools.items():
                base += f"- {name}: {desc}\n"
            base += "\nWhen using a tool, output EXACTLY: [TOOL:name] {{\"arg\": \"value\"}} [/TOOL]\n"
        if ctx.extra_context:
            base += f"\n\n{ctx.extra_context}\n"
        return base + "\nStart:\n"

    def parse_output(self, output: str, ctx: ExecutionContext) -> PolicyOutput:
        tool_calls = ToolCallParser.parse_all(output)
        if tool_calls:
            reasoning = self._extract_reasoning(output)
            requests = [ToolCallRequest(tool_name=n, arguments=a) for n, a in tool_calls]
            return PolicyOutput(reasoning=reasoning, tool_calls=requests)
        cleaned = ReActPolicy._strip_think_tags(output.strip())
        return PolicyOutput(reasoning="", final_answer=cleaned, done=True)

    def next_prompt(self, ctx: ExecutionContext, last_output: str) -> Optional[str]:
        base = self.first_prompt(ctx)
        history_lines = []
        for step in ctx.steps:
            if step.reasoning:
                history_lines.append(f"Step {step.step_number}: {step.reasoning[:500]}")
            for obs in step.observations:
                history_lines.append(f"Observation: {obs[:2000]}")
        history = "\n".join(history_lines)
        return base + "\n" + history + "\nContinue:"

    @staticmethod
    def _extract_reasoning(output: str) -> str:
        m = re.search(r"<think>(.*?)</think>", output, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        first_tool = output.find("[TOOL:")
        if first_tool != -1:
            return output[:first_tool].strip()
        return ""


# ---------------------------------------------------------------------------
# Verification policy
# ---------------------------------------------------------------------------

class VerifyPolicy(ThinkingPolicy):
    name = "verification"

    def __init__(self):
        self.inner = CoTPolicy()

    def first_prompt(self, ctx: ExecutionContext) -> str:
        return self.inner.first_prompt(ctx)

    def parse_output(self, output: str, ctx: ExecutionContext) -> PolicyOutput:
        return self.inner.parse_output(output, ctx)

    def next_prompt(self, ctx: ExecutionContext, last_output: str) -> Optional[str]:
        return self.inner.next_prompt(ctx, last_output)

    def verify(self, answer: str, ctx: ExecutionContext) -> str:
        prompt = f"""Task: {ctx.task}

Proposed answer: {answer}

Verify this answer step by step.
Check for: correctness, completeness, edge cases, logical consistency.

If the answer is correct, output: "VERIFIED: <reasoned answer>"
If incorrect, output: "REJECTED: <corrected answer>"

Reasoning:"""
        verification = self.model.generate(prompt, n_loops=1, temperature=0.3, max_tokens=1024)
        if "VERIFIED:" in verification:
            return verification.split("VERIFIED:", 1)[1].strip()
        if "REJECTED:" in verification:
            return verification.split("REJECTED:", 1)[1].strip()
        return answer


# ---------------------------------------------------------------------------
# Direct policy (single-shot, replaces legacy _run_traditional when no tools)
# ---------------------------------------------------------------------------

class DirectPolicy(ThinkingPolicy):
    name = "direct"

    def first_prompt(self, ctx: ExecutionContext) -> str:
        base = f"""You are Lv Super Agent, a helpful assistant.

Task: {ctx.task}

Think briefly inside <think>...</think> tags, then respond directly and concisely.
If the task requires a tool, use [TOOL:name] {{"arg": "value"}} [/TOOL].
When the user asks to open a file/report/folder (e.g. '打开', 'open'), use [TOOL:file_ops] {{"action": "open", "path": "<exact path>"}} [/TOOL].

{ctx.extra_context}

Start:
"""
        if ctx.available_tools:
            base += "\nAvailable tools:\n"
            for name, desc in ctx.available_tools.items():
                base += f"- {name}: {desc}\n"
        return base

    def parse_output(self, output: str, ctx: ExecutionContext) -> PolicyOutput:
        tool_calls = ToolCallParser.parse_all(output)
        if tool_calls:
            requests = [ToolCallRequest(tool_name=n, arguments=a) for n, a in tool_calls]
            reasoning = CoTPolicy._extract_reasoning(output)
            return PolicyOutput(reasoning=reasoning, tool_calls=requests)
        cleaned = ReActPolicy._strip_think_tags(output.strip())
        final_markers = ["FINAL ANSWER:", "Final Answer:", "最终答案：", "最终答案:"]
        for marker in final_markers:
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker):].strip()
                break
        return PolicyOutput(reasoning="", final_answer=cleaned, done=True)

    def next_prompt(self, ctx: ExecutionContext, last_output: str) -> Optional[str]:
        base = self.first_prompt(ctx)
        history_lines = []
        for step in ctx.steps:
            if step.reasoning:
                history_lines.append(f"Thought: {step.reasoning[:300]}")
            for obs in step.observations:
                history_lines.append(f"Observation: {obs[:2000]}")
        return base + "\n" + "\n".join(history_lines) + "\nContinue:"
