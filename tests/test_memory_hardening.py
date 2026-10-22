"""回归测试: 记忆子系统加固 (dedup / 原子写 / 线程安全 / 检索评分 / 截断)."""

import json
import os
import re
import threading
from pathlib import Path

import pytest

from agent_project.memory import (
    EpisodicMemory,
    KnowledgeGraph,
    MemoryEdge,
    MemoryNode,
    _atomic_write_text,
    _coverage_similarity,
    _dedup_key,
    _text_tokens,
)


@pytest.fixture
def tmp_store(tmp_path):
    return EpisodicMemory(storage_path=str(tmp_path / "episodes"))


# ---------------------------------------------------------------- helpers

def test_atomic_write_text_swaps_atomically(tmp_path):
    target = tmp_path / "store.json"
    _atomic_write_text(target, json.dumps({"a": 1}))
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    # no leftover tmp files
    assert [p.name for p in target.parent.iterdir()] == ["store.json"]


def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "store.json"
    _atomic_write_text(target, "v1")
    _atomic_write_text(target, "v2-longer-than-v1")
    assert target.read_text(encoding="utf-8") == "v2-longer-than-v1"


# ---------------------------------------------------------------- dedup

def test_store_exact_duplicate_does_not_create_new_episode(tmp_store):
    content = "Task: 用中文回答\nOutcome: 好的，以后都用中文回复。"
    first = tmp_store.store(content, "偏好", task_type="interaction", importance=0.6)
    second = tmp_store.store(content, "偏好", task_type="interaction", importance=0.7)

    assert first == second
    assert len(tmp_store.episodes) == 1
    ep = tmp_store.episodes[first]
    assert ep.importance == 0.7  # importance ratchets up
    assert ep.metadata["duplicate_hits"] == 1


def test_store_near_duplicate_is_deduped(tmp_store):
    base = "用户偏好: 用中文回答，风格保持简洁。以后遵循此设置。"
    first = tmp_store.store(base, "偏好", task_type="interaction", importance=0.5)
    # same content + trailing whitespace noise, different case
    noise = " " + base.upper().replace(" ", "  ") + " \n"
    second = tmp_store.store(noise, "偏好", task_type="interaction", importance=0.5)

    assert first == second
    assert len(tmp_store.episodes) == 1


def test_store_different_content_is_kept(tmp_store):
    a = tmp_store.store("完成 GitHub 搜索功能", "搜索", task_type="tool", importance=0.4)
    b = tmp_store.store("完成天气预报解析", "天气", task_type="tool", importance=0.4)
    assert a != b
    assert len(tmp_store.episodes) == 2


def test_dedup_index_rebuilt_on_load(tmp_path):
    store = EpisodicMemory(storage_path=str(tmp_path / "episodes"))
    cid = store.store("记忆内容甲乙丙丁", "摘要", task_type="interaction", importance=0.5)
    dup = store.store("记忆内容甲乙丙丁", "摘要", task_type="interaction", importance=0.5)

    fresh = EpisodicMemory(storage_path=str(tmp_path / "episodes"))
    assert cid in fresh.episodes
    assert len(fresh.episodes) == 1
    # dedup still active across process boundaries
    again = fresh.store("记忆内容甲乙丙丁", "摘要", task_type="interaction", importance=0.9)
    assert again == cid
    assert dup == cid


# ---------------------------------------------------------------- retrieval

def test_retrieve_similar_scores_by_token_overlap(tmp_store):
    tmp_store.store(
        "Task: 分析 GitHub 仓库结构\nOutcome: 仓库包含 python 模块和测试目录",
        "仓库分析", task_type="analysis", importance=0.5,
    )
    tmp_store.store(
        "Task: 天气预报\nOutcome: 北京明天晴，15 度",
        "天气", task_type="tool", importance=0.5,
    )
    tmp_store.store(
        "Task: 帮我写周报\nOutcome: 周报已生成，包含本完成事项",
        "周报", task_type="writing", importance=0.5,
    )

    hits, scores = zip(
        *tmp_store.retrieve_similar("GitHub 仓库分析 结果", min_similarity=0.1, k=5)
    )
    assert "仓库分析" in [h.summary for h in hits]
    best = hits[0]
    assert best.summary == "仓库分析" or scores[0] >= scores[1]
    # unrelated query should rank 天气/周报 far lower than 仓库分析
    top_summary = hits[0].summary
    assert top_summary == "仓库分析"


def test_retrieve_similar_substring_still_works(tmp_store):
    tmp_store.store("融资 7500万美元", "LightX 融资", task_type="fact", importance=0.5)
    hits = tmp_store.retrieve_similar("融资", min_similarity=0.6)
    assert any(h.summary == "LightX 融资" for h, _ in hits)


def test_retrieve_similar_respects_min_similarity(tmp_store):
    tmp_store.store("毫无关联的天气数据", "天气", task_type="tool", importance=0.5)
    assert tmp_store.retrieve_similar("量子计算论文综述", min_similarity=0.6) == []


def test_retrieve_recent_and_by_id_under_lock(tmp_store):
    variants = [("苹果香蕉出售了", "水果摘要"),
                ("汽车飞机出发了", "交通摘要"),
                ("冬夏春秋轮换了", "季节摘要")]
    ids = [
        tmp_store.store(text, summary, task_type="t", importance=0.5)
        for text, summary in variants
    ]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert tmp_store.retrieve_by_id(ids[1]).summary == "交通摘要"
    recent = tmp_store.retrieve_recent(2)
    assert len(recent) == 2


# ---------------------------------------------------------------- concurrency & persistence

def test_concurrent_store_threads_keep_store_consistent(tmp_path):
    store = EpisodicMemory(storage_path=str(tmp_path / "episodes"), max_episodes=10000)

    def worker(tid):
        for i in range(15):
            store.store(f"任务{tid}的第{i}次执行记录内容", f"摘要{tid}-{i}", task_type="t", importance=0.5)
            store.save()

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.episodes) >= 3
    # disk file is valid JSON, no corruption from concurrent atomic writes
    data = json.loads((Path(tmp_path / "episodes") / "episodes.json").read_text(encoding="utf-8"))
    assert len(data) == len(store.episodes)


def test_episodic_save_never_leaves_tmp_files(tmp_store):
    for i in range(5):
        tmp_store.store(f"内容{i}甲乙丙", f"摘要{i}", task_type="t", importance=0.5)
    leftovers = [
        p.name for p in Path(tmp_store.storage_path).iterdir()
        if p.name.endswith(".tmp")
    ]
    assert leftovers == []


def test_knowledge_graph_thread_safety_and_atomic_save(tmp_path):
    kg = KnowledgeGraph(storage_path=str(tmp_path / "kg"))

    def worker(tid):
        edges = 0
        for i in range(20):
            nid = kg.add_node(MemoryNode(label=f"概念{tid}", properties={"i": i}))
            if i > 0:
                kg.add_edge(MemoryEdge(source_id=nid, target_id=nid, relation="relates"))
                edges += 1
        return edges

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    kg.save()
    data = json.loads((Path(tmp_path / "kg") / "graph.json").read_text(encoding="utf-8"))
    assert len(data["nodes"]) == 80
    assert len(data["edges"]) == 76  # 4 threads x 19 edges

    fresh = KnowledgeGraph(storage_path=str(tmp_path / "kg"))
    fresh.load()
    assert len(fresh.nodes) == 80
    assert len(fresh.edges) == 76


# ---------------------------------------------------------------- helpers themselves

def test_text_tokens_cjk_bigrams_and_ascii():
    tokens = _text_tokens("GitHub 仓库分析")
    assert "github" in tokens
    assert "仓库" in tokens
    assert "分析" in tokens


def test_coverage_similarity_basic():
    q = _text_tokens("GitHub 仓库")
    assert _coverage_similarity(q, "这是一个 GitHub 仓库本体") > 0.5
    assert _coverage_similarity(q, "完全无关内容") == 0.0
    assert _coverage_similarity(set(), "anything") == 0.0


def test_dedup_key_ignores_case_and_whitespace():
    assert _dedup_key("Hello   World") == _dedup_key("hello world")
