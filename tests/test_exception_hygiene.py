"""吞异常治理基线锁: 静默 except 块只许减不许增(无网络, 纯 AST).

背景: 全仓 ~160 个全空 except 块(pass/continue), 多为刻意 fallback
(解析回退/控制流 StopIteration/尽力循环), 少量是真吞错.
逐个改风险大于收益, 故先锁基线防新增; 存量按文件逐步认领清理.

判定标准与治理时一致: except 体内无任何语句(除 docstring 外),
或仅 pass/continue, 即视为"静默".
"""
import ast
from pathlib import Path

# 基线: 2026-09-10 实测 160(已含 contextual_search_hook 裸 except 收窄为 OSError).
# 只许减不许增; 如刻意新增 fallback, 同步更新本数并说明理由.
BASELINE_SILENT = 160

ROOT = Path(__file__).resolve().parent.parent / "agent_project"


def count_silent(root: Path = ROOT):
    silent = []
    for f in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    silent.append((str(f), node.lineno, "BARE"))
                    continue
                body = [n for n in node.body
                        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
                if not body or (len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Continue))):
                    silent.append((str(f), node.lineno, "silent"))
    return silent


def test_no_bare_except():
    silent = count_silent()
    bare = [(f, l) for f, l, k in silent if k == "BARE"]
    assert not bare, f"裸 except 必须具名化, 发现: {bare}"


def test_silent_except_not_growing():
    silent = count_silent()
    n = len(silent)
    assert n <= BASELINE_SILENT, (
        f"静默 except 从 {BASELINE_SILENT} 涨到 {n}, 新增必须打日志或写理由并更新基线"
    )
