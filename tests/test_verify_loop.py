"""验证闭环测试: apply_diff/write 自动语法验证 + 失败定位 + python fallback."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

from agent_project.tools.file_ops import FileOpsTool


def make_tool():
    return FileOpsTool()


def test_apply_diff_auto_verify_ok():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "good.py")
    Path(py).write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    tool = make_tool()
    diff = (
        "<<<<<<< SEARCH\ndef hello():\n    return 'hi'\n"
        "=======\ndef hello():\n    return 'hello world'\n>>>>>>> REPLACE"
    )
    r = tool.execute(action="apply_diff", path=py, diff=diff)
    assert r.success, f"正常 diff 应成功: {r.error}"
    assert "syntax OK" in r.output.lower() or "Verify" not in r.output, r.output


def test_apply_diff_detects_syntax_error():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "bad.py")
    Path(py).write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    tool = make_tool()
    diff = (
        "<<<<<<< SEARCH\ndef hello():\n    return 'hi'\n"
        "=======\ndef broken(:\n    return 'x'\n>>>>>>> REPLACE"
    )
    r = tool.execute(action="apply_diff", path=py, diff=diff)
    assert r.success, "diff 应应用成功(语法错误作为 warning)"
    assert "SYNTAX VERIFICATION FAILED" in r.output.upper(), r.output


def test_apply_diff_not_found_has_hint():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "x.py")
    Path(py).write_text("import os\n\ndef main():\n    return os.getcwd()\n", encoding="utf-8")
    tool = make_tool()
    diff = (
        "<<<<<<< SEARCH\nthis line does not exist at all\n"
        "=======\nreplacement\n>>>>>>> REPLACE"
    )
    r = tool.execute(action="apply_diff", path=py, diff=diff)
    assert not r.success, "不匹配的 block 应失败"
    assert "not found" in (r.error or "").lower(), r.error
    assert "whitespace" in (r.error or "").lower(), "应有修复提示"


def test_write_auto_verify():
    td = tempfile.mkdtemp()
    py = os.path.join(td, "new.py")
    tool = make_tool()
    r = tool.execute(action="write", path=py, content="def broken(:\n    pass\n")
    assert r.success
    assert "SYNTAX VERIFICATION FAILED" in r.output.upper(), r.output


def test_verify_action():
    td = tempfile.mkdtemp()
    good = os.path.join(td, "good.py")
    bad = os.path.join(td, "bad.py")
    Path(good).write_text("print(1)\n", encoding="utf-8")
    Path(bad).write_text("def broken(:\n", encoding="utf-8")
    tool = make_tool()
    rg = tool.execute(action="verify", path=good)
    assert rg.success, rg.error
    rb = tool.execute(action="verify", path=bad)
    assert not rb.success, "坏代码 verify 应失败"
    assert rb.error and "error" in rb.error.lower() or "invalid" in (rb.error or "").lower()


def test_python_fallback_verify_syntax():
    """后缀比较 bug 回归: .py 文件必须被正确识别."""
    tool = make_tool()
    td = tempfile.mkdtemp()
    bad = os.path.join(td, "bad.py")
    r = tool._python_verify_syntax(Path(bad), "def broken(:\n    pass\n")
    assert r, "坏 Python 应返回语法错误, 而不是空(后缀 lstrip bug)"
    good = os.path.join(td, "good.py")
    r2 = tool._python_verify_syntax(Path(good), "print(1)\n")
    assert r2 == "", "好代码应返回空"
