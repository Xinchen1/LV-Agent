"""AuxiliaryClient 单测(无网络): 委托/计数/热插/总结切流."""
from agent_project.auxiliary_client import AuxiliaryClient


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, n_loops=1, temperature=0.7, max_tokens=512,
                 token_callback=None, **kw):
        self.calls.append((prompt[:20], temperature, max_tokens))
        return "side answer"


def test_generate_delegates_and_counts():
    b = FakeBackend()
    aux = AuxiliaryClient(backend=b)
    assert aux.generate("hello") == "side answer"
    assert b.calls[0][1:] == (0.4, 1024), "默认低温短输出"
    assert aux.stats() == {"side_calls": 1, "side_failures": 0}


def test_failure_counted_and_reraised():
    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("down")
    aux = AuxiliaryClient(backend=Boom())
    try:
        aux.generate("x")
    except RuntimeError:
        pass
    else:
        raise AssertionError("应透出异常由调用方降级")
    assert aux.stats()["side_failures"] == 1


def test_no_backend_raises():
    aux = AuxiliaryClient(backend=None)
    try:
        aux.generate("x")
    except RuntimeError:
        return
    raise AssertionError("未 attach 应 RuntimeError")


def test_attach_hot_swaps_backend():
    b1, b2 = FakeBackend(), FakeBackend()
    aux = AuxiliaryClient(backend=b1)
    aux.generate("a")
    aux.attach(b2)
    aux.generate("b")
    assert len(b1.calls) == 1 and len(b2.calls) == 1


def test_summarize_uses_auxiliary_not_main():
    from agent_project.agent import OpenMythosAgent
    main, side = FakeBackend(), FakeBackend()
    a = OpenMythosAgent.__new__(OpenMythosAgent)
    a.backend = main
    from agent_project.auxiliary_client import AuxiliaryClient as AC
    a._auxiliary_client = AC(backend=side)
    import logging
    a.logger = logging.getLogger("smoke")
    a._clean_fast_answer = lambda t: t
    action = type("A", (), {"tool_name": "file_ops", "arguments": {"action": "list"}})()
    raw = "line1\nline2\nline3\nline4\nline5" * 100  # >400 触发 LLM 总结
    out = a._summarize_tool_answer("列目录", action, raw)
    assert out == "side answer", out
    assert len(side.calls) == 1, "总结应走 side"
    assert len(main.calls) == 0, "总结不应占用主 backend"
    assert a.auxiliary.stats()["side_calls"] == 1
