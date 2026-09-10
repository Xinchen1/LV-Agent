"""MemoryProvider 单测: ABC 约束 + file 真盘读写 + wiki 假体 + fanout."""
import pytest
from agent_project.file_memory import FileMemoryManager
from agent_project.memory_provider import (
    FileMemoryAdapter,
    MemoryProvider,
    WikiMemoryAdapter,
    fanout_recall,
)


def test_abc_not_instantiable():
    with pytest.raises(TypeError):
        MemoryProvider()


class FakeWiki:
    def __init__(self):
        self.saved = []

    def remember(self, text, context=None, auto_link=True):
        self.saved.append(text)
        return ["p1"]

    def search(self, query, k=5):
        return [{"content": "wiki hit"}]


def test_wiki_adapter_roundtrip():
    w = FakeWiki()
    ad = WikiMemoryAdapter(w)
    assert isinstance(ad, MemoryProvider)
    assert ad.remember("hello", context="t") == ["p1"]
    assert ad.recall("q") == ["wiki hit"]


def test_wiki_recall_failure_returns_empty():
    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("down")
    assert WikiMemoryAdapter(Boom()).recall("q") == []


def test_file_adapter_roundtrip(tmp_path):
    store = FileMemoryManager(
        memory_path=str(tmp_path / "memory.md"),
        user_path=str(tmp_path / "user.md"),
        project_root=str(tmp_path),
    )
    ad = FileMemoryAdapter(store)
    assert isinstance(ad, MemoryProvider)
    assert ad.remember("偏好深色模式", context="用户偏好") == ["用户偏好"]
    hits = ad.recall("深色")
    assert hits and any("深色" in h for h in hits)


def test_fanout_dedups_and_tolerates_failure(tmp_path):
    store = FileMemoryManager(
        memory_path=str(tmp_path / "m.md"),
        user_path=str(tmp_path / "u.md"),
        project_root=str(tmp_path),
    )
    a = FileMemoryAdapter(store)
    a.remember("alpha 内容", context="t")
    b = WikiMemoryAdapter(FakeWiki())

    class Boom:
        def recall(self, *a, **k):
            raise RuntimeError("down")
    out = fanout_recall([a, b, Boom()], "alpha", k=5)
    assert any("alpha" in h for h in out)
    assert len(out) == len(set(out)), "应去重"
