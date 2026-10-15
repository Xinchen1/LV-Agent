#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: AGPL-3.0-only
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research

"""性能热点回归: 锁死冷启动/意图分类/健康检查上的历史坑.

- MCP bridge 探测曾用 `--version` 起守护进程永不退出 -> 每次启动白等 3s
- 直出快路曾无条件跑 LLM 意图分类, 结果 conf 0.75 又被 0.85 门槛拒掉
- package __init__ 曾 eager 导入全链(~300ms)压在首帧之前
- 后端首调曾同步打 /models 探活(最多 5s)拉长首 token
"""

import subprocess
import sys
import time
import types
import logging
import shutil
import os
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "agent_project")

# 有别的测试会 chdir: 所有相对路径一律锚到仓库根(不改全局 cwd, 免影响他人)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from agent_project.cache import MemoCache
from agent_project.agent import OpenMythosAgent


# ----------------------------------------------------------------------
# 1. rust_mcp_bridge 探测必须零子进程、微秒级
# ----------------------------------------------------------------------

def test_mcp_bridge_probe_is_fast_and_cached():
    from agent_project.tools.mcp_client import _find_rust_mcp_bridge, _RUST_BRIDGE_PROBE

    t0 = time.perf_counter()
    first = _find_rust_mcp_bridge()
    elapsed = time.perf_counter() - t0
    # 旧实现要跑 3s 子进程超时; 现在必须是纯文件头读取
    assert elapsed < 0.25, f"bridge probe took {elapsed*1000:.0f}ms (应为微秒级, 不允许子进程)"

    t0 = time.perf_counter()
    second = _find_rust_mcp_bridge()
    assert time.perf_counter() - t0 < 0.01
    assert first == second
    # 结果被进程级缓存(第二次不再探测)
    from agent_project.tools import mcp_client as MC
    assert MC._RUST_BRIDGE_PROBE is not MC._BRIDGE_PROBE_NOT_SET

    binary = PROJECT_ROOT / "rust_file_ops" / "target" / "release" / "rust_mcp_bridge"
    if binary.exists() and os.access(binary, os.X_OK):
        assert first is not None and first.endswith("rust_mcp_bridge"), f"本机二进制存在时必须返回路径, 实际 {first}"


def test_binary_arch_check_no_subprocess():
    from agent_project.tools.mcp_client import _binary_arch_compatible
    from pathlib import Path

    # 本机二进制: 必须兼容(否则会退回 Python 客户端)
    local = PROJECT_ROOT / "rust_file_ops" / "target" / "release" / "rust_mcp_bridge"
    if local.exists():
        assert _binary_arch_compatible(local) is True
    # 不存在的文件
    assert _binary_arch_compatible(Path("/nonexistent/bridge_xyz")) is False
    # 非 ELF/Mach-O 文本(如 shebang 脚本): 保守放行
    script = Path("/tmp/_perf_fake_bridge.sh")
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    try:
        assert _binary_arch_compatible(script) is True
    finally:
        script.unlink(missing_ok=True)


def test_mcp_bridge_probe_never_spawns_daemon():
    """探测阶段绝不允许启动守护进程(会占 socket + 挂起)."""
    from agent_project.tools import mcp_client as MC

    calls = {"n": 0}
    real_run = MC.subprocess.run

    def counting_run(*a, **kw):
        calls["n"] += 1
        return real_run(*a, **kw)

    MC.subprocess.run = counting_run
    try:
        MC._RUST_BRIDGE_PROBE = MC._BRIDGE_PROBE_NOT_SET
        MC._find_rust_mcp_bridge()
    finally:
        MC.subprocess.run = real_run
        MC._RUST_BRIDGE_PROBE = MC._BRIDGE_PROBE_NOT_SET  # 还原缓存
    assert calls["n"] == 0, "探测阶段不允许 subprocess.run"


# ----------------------------------------------------------------------
# 2. MCP 服务器 defer 启动(不阻塞主线程)
# ----------------------------------------------------------------------

def test_mcp_add_server_defer_is_lazy():
    from agent_project.tools import mcp_client as MC

    class FakeBridge:
        def __init__(self):
            self.added = []

        def add_server(self, name, command, args, **kw):
            self.added.append(name)
            return {"success": True, "metadata": {"tools": []}}

    mgr = MC.McpManager.__new__(MC.McpManager)
    mgr.servers = {}
    mgr._registered_tool_names = {}
    mgr._bridge = FakeBridge()
    mgr._bridge_configs = {}
    mgr._pending_bridge = {}
    mgr._use_bridge = True
    mgr._logger = logging.getLogger("test_mcp")

    cmd = sys.executable  # 一定存在的 runtime
    cfg = {"command": cmd, "args": ["-V"]}

    r = mgr._add_server_internal("fake", cfg, defer=True)
    assert r is None
    assert "fake" in mgr._pending_bridge, "defer 必须只登记"
    assert not mgr._bridge_configs, "defer 不允许立刻启动"
    assert mgr._bridge.added == []

    mgr.start_pending_servers()
    assert "fake" in mgr._bridge_configs, "start_pending 后才真正启动"
    assert mgr._bridge.added == ["fake"]


def test_mcp_missing_runtime_skipped():
    """npx/uvx 未安装时直接跳过, 不 spawn 不报错."""
    from agent_project.tools import mcp_client as MC

    mgr = MC.McpManager.__new__(MC.McpManager)
    mgr.servers = {}
    mgr._registered_tool_names = {}
    mgr._bridge = None
    mgr._bridge_configs = {}
    mgr._pending_bridge = {}
    mgr._use_bridge = False
    mgr._logger = logging.getLogger("test_mcp")

    assert mgr._add_server_internal("fs", {"command": "definitely_missing_bin_xyz"}) is None
    assert "fs" not in mgr.servers


# ----------------------------------------------------------------------
# 3. package __init__ 懒加载
# ----------------------------------------------------------------------

def test_package_import_is_lazy():
    code = (
        "import sys, agent_project; "
        "assert 'agent_project.agent' not in sys.modules, 'package init 不许 eager 导入 agent'; "
        "assert 'agent_project.tools' not in sys.modules; "
        "assert agent_project.__version__; "
        "from agent_project import OpenMythosAgent, TOOLS_REGISTRY; "
        "assert 'agent_project.agent' in sys.modules"
    )
    p = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
    )
    assert p.returncode == 0, p.stderr


# ----------------------------------------------------------------------
# 4. 意图分类: 快路与主路不许白烧 LLM 回合
# ----------------------------------------------------------------------

class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        return '{"tool": "web_search", "args": {"query": "x"}, "reason": "r"}'


def make_agent():
    a = object.__new__(OpenMythosAgent)
    a._method_cache = MemoCache()
    a._code_mode_override = False
    a._current_task = ""
    a.logger = logging.getLogger("test_perf")
    a.backend = FakeBackend()
    return a


def test_direct_intent_path_never_calls_llm():
    """直出快路只跑规则层: 规则未命中直接返回 None, 不许发 LLM 意图请求."""
    a = make_agent()
    # 规则未命中、长度 >=5 的任务(旧实现会触发 LLM 意图分类)
    assert a._try_direct_intent_path("给我讲一个关于海底两万里的故事吧") is None
    assert a.backend.calls == 0, f"快路不该调 LLM, 实际 {a.backend.calls} 次"


def test_classify_intent_allow_llm_flag():
    a = make_agent()
    # allow_llm=False: 规则未命中 -> None, 0 次 LLM
    assert a._classify_intent("给我找点AI方面的资料", allow_llm=False) is None
    assert a.backend.calls == 0
    # allow_llm=True: LLM 兜底生效
    r = a._classify_intent("给我找点AI方面的资料", allow_llm=True)
    assert r is not None and a.backend.calls == 1


def test_resolve_fast_action_skips_llm_when_model_action_valid():
    """模型已给出有效工具 + 规则未命中 -> 不许再花一次 LLM 意图分析."""
    a = make_agent()
    # 模型给了合法工具调用(weather 在注册表里, 参数满足 schema)
    answer = '[TOOL:weather]\n{"location": "beijing"}\n[/TOOL]'
    assert a._parse_output_for_action(answer) is not None, "前置: 解析器必须认这个工具调用"
    # 规则层: 这句话含"天气"会命中 weather 规则 -> 改用规则也命不中的措辞
    action, ambiguous, mem = a._resolve_fast_action("帮我看看那边的状况如何了呢", answer)
    assert action is not None and action.tool_name == "weather"
    assert a.backend.calls == 0, f"模型已有有效动作时不许调 LLM 意图分类, 实际 {a.backend.calls} 次"


# ----------------------------------------------------------------------
# 5. 后端健康检查: 首调不许同步探活
# ----------------------------------------------------------------------

def test_ensure_connection_first_call_does_not_probe():
    import types
    from agent_project.model_backends import OpenAIBackend

    b = OpenAIBackend.__new__(OpenAIBackend)
    b.base_url = "https://example.invalid/v1"
    b._client = object()  # 模拟已有健康 client
    b._last_health_check = time.time()  # 与 __init__ 一致: 新 client 即健康
    b._health_check_interval = 300

    b._ensure_connection()  # 刚初始化过 -> 直接短路
    assert b._client is not None, "刚初始化的 client 不许被探活重建"

    # 超过间隔后探活: 收到 HTTP 响应(4xx) 不算连接坏, 不许丢弃连接池
    calls = {"n": 0}

    class HTTPStatusError(Exception):
        status_code = 404

    class FakeResp:
        def raise_for_status(self):
            calls["n"] += 1
            raise HTTPStatusError("404 Client Error")

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.get = lambda *a, **kw: FakeResp()
    prev = sys.modules.get("httpx")
    sys.modules["httpx"] = fake_httpx
    b._last_health_check = 0
    try:
        b._ensure_connection()
    finally:
        if prev is None:
            sys.modules.pop("httpx", None)
        else:
            sys.modules["httpx"] = prev
    assert calls["n"] == 1, "前置: 探活确实发生了"
    assert b._client is not None, "HTTP 层错误(链路通)不许重建 client"


def test_retry_discards_dead_connection_pool():
    """连接层失败(无 HTTP 状态)重试时必须丢弃连接池换新连接."""
    import re as _re

    with open(PROJECT_ROOT / "agent_project" / "model_backends.py", encoding="utf-8") as f:
        src = f.read()
    assert _re.search(
        r"if self\._should_retry\(e\):(?:(?!def ).)*?if status is None:\s+self\._client = None",
        src,
        _re.DOTALL,
    ), "重试分支缺少死连接丢弃逻辑(status is None -> self._client = None)"


# ----------------------------------------------------------------------
# 6. 后端预热: warm_up 失败静默, 成功才刷健康时间戳
# ----------------------------------------------------------------------

def test_warm_up_silent_on_failure():
    from agent_project.model_backends import OpenAIBackend

    b = OpenAIBackend.__new__(OpenAIBackend)
    b._last_health_check = 123.0

    class BoomModels:
        def list(self, **kw):
            raise ConnectionError("refused")

    class BoomClient:
        models = BoomModels()

    b._client = BoomClient()
    b.warm_up()  # 不许抛异常
    assert b._last_health_check == 123.0, "失败不许刷新健康时间戳(留给真实请求探活)"
    assert b._client is not None


def test_warm_up_success_refreshes_health():
    from agent_project.model_backends import OpenAIBackend

    b = OpenAIBackend.__new__(OpenAIBackend)
    b._last_health_check = 123.0

    class OkModels:
        def list(self, **kw):
            return []

    class OkClient:
        models = OkModels()

    b._client = OkClient()
    b.warm_up()
    assert b._last_health_check > 123.0, "成功必须刷新时间戳, 避免首请求再同步探活"


def test_agent_init_starts_warmup_thread_in_background():
    """init 不许同步等 warm_up(只许起 daemon 线程), 且 pytest 下必须跳过."""
    import ast

    with open(PROJECT_ROOT / "agent_project" / "agent.py", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    init_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    init_src = ast.get_source_segment(src, init_fn)
    assert "warm_up" in init_src, "agent init 必须触发后端预热"
    assert "daemon=True" in init_src, "预热线程必须是 daemon(不阻塞退出)"
    assert '"pytest" not in sys.modules' in init_src, "pytest 环境必须跳过预热"
    # init 里不允许同步调用 warm_up(必须包在线程 target 里)
    assert "self.backend.warm_up()" not in init_src, "warm_up 不许在主线程同步调用"


# ----------------------------------------------------------------------
# 7. MemoCache 递归死锁(线上"73s 0tk 永久转圈"事故根因)
# ----------------------------------------------------------------------

def test_memo_cache_allows_reentrant_factory():
    """factory 内再次 get_or_set 不许死锁(不可重入 Lock 曾导致 UI 永久卡死)."""
    import threading
    from agent_project.cache import MemoCache

    c = MemoCache()
    hits = []

    def inner_factory():
        hits.append("inner")
        return "INNER"

    def outer_factory():
        # 模拟 _decide_thinking_loops -> _is_simple_query 的嵌套 _cache_get
        inner = c.get_or_set(("ns_b", 1), inner_factory)
        hits.append("outer")
        return f"OUTER+{inner}"

    result = {}

    def run():
        result["v"] = c.get_or_set(("ns_a", 1), outer_factory)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "MemoCache.get_or_set 递归调用死锁(factory 内再取缓存)"
    assert result.get("v") == "OUTER+INNER"
    assert hits == ["inner", "outer"]


def test_memo_cache_still_caches_across_calls():
    """加锁改 RLock 后, 记忆化语义必须保持: factory 只跑一次."""
    from agent_project.cache import MemoCache

    c = MemoCache()
    n = {"calls": 0}

    def f():
        n["calls"] += 1
        return 42

    assert c.get_or_set(("ns", 1), f) == 42
    assert c.get_or_set(("ns", 1), f) == 42
    assert n["calls"] == 1, "重复 key 必须命中缓存, 不许重复执行 factory"


def test_decide_thinking_loops_no_nameerror():
    """task_lower 曾先用后定义 -> NameError; 且该函数内嵌套 _cache_get(死锁点)."""
    a = make_agent()
    a.config = types.SimpleNamespace(
        max_thinking_loops=8,
        default_thinking_loops=3,
        reflection=types.SimpleNamespace(thinking_loops_for_reflection=4),
    )
    a.strategy_db = None
    a._pending_correction = None

    # 非 simple / 非 complex 的长任务: 旧实现走到 long_signals 就 NameError
    n = a._compute_decide_thinking_loops(
        "帮我整理一下整个项目仓库的目录结构和所有历史提交记录做一份完整的梳理", "production"
    )
    assert isinstance(n, int) and 1 <= n <= 8
