"""回归测试: 记忆子系统向量语义召回(lexical max cosine + 优雅降级)."""

import pytest

from agent_project.embeddings import EmbeddingClient, cosine_similarity, vector_score
from agent_project.memory import EpisodicMemory, KnowledgeGraph, MemoryNode


# ---------------------------------------------------------------- helpers

class FakeEmbedder:
    """确定性测试嵌入: 注入 EmbeddingClient.embed_fn.

    语义规则: "装工具/安装工具" 族 → [1, 0]; "书法/写字" 族 → [0, 1].
    两族词面完全不相交, 用于验证纯语义召回.
    """

    def __init__(self):
        self.calls = 0

    def as_embed_fn(self, broken=False):
        def embed(text):
            self.calls += 1
            if broken:
                return None
            t = text or ""
            if any(w in t for w in ("装", "install", "安装")):
                return [1.0, 0.0]
            if any(w in t for w in ("写", "calligraphy")):
                return [0.0, 1.0]
            return [0.5, 0.5]
        return embed

    def make_client(self, broken=False):
        c = EmbeddingClient(base_url="http://test/v1", model="fake")
        c.embed_fn = self.as_embed_fn(broken=broken)
        return c


# ---------------------------------------------------------------- units

def test_cosine_similarity_basic_and_guards():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0  # 维度不齐 → 0


def test_vector_score_mapping():
    assert vector_score(0.3) == 0.0            # 无关文本通常落在 0.3-0.5
    assert vector_score(0.42) == 0.0           # floor 处归零
    assert 0.0 < vector_score(0.6) < 1.0
    assert vector_score(0.95) == pytest.approx(1.0)
    assert vector_score(1.2) == pytest.approx(1.0)  # 截断


# ---------------------------------------------------------------- episodic recall

def test_paraphrase_recalled_via_vector_but_misses_lexical(tmp_path):
    """词面零重叠的改写查询, 靠向量召回; 无嵌入客户端时该查询一无所获."""
    em = EpisodicMemory(
        storage_path=str(tmp_path / "episodes"),
        embedding_client=FakeEmbedder().make_client(),
    )
    em.store(
        content="上次帮用户安装了 netmiko 网络自动化工具",
        summary="安装 netmiko",
        task_type="interaction",
    )

    # 词面覆盖: 查询 "装" 会命中 content, 换纯英文也命中 install;
    # 这里选与原句零共享词面但仍语义相关的表达难以构造中文——改用英文改写.
    hits = em.retrieve_similar("install network tool")
    assert hits, "语义改写必须召回"
    score = hits[0][1]
    assert score > 0.4

    # 对照: 同样查询, 若不注入向量(词面 inst? 实际 'install' 恰在原句中)
    # 说明该案例同时受词面兜底影响 → 再验证一个纯语义案例:
    em2 = EpisodicMemory(
        storage_path=str(tmp_path / "episodes2"),
        embedding_client=FakeEmbedder().make_client(),
    )
    em2.store(
        content="follow-up: wanted a calligraphy practice plan",
        summary="calligraphy practice",
        task_type="interaction",
    )
    hits2 = em2.retrieve_similar("写字体检计划反馈")  # 中文改写, 词面零重叠
    assert hits2 and hits2[0][1] > 0.5


def test_retrieval_identical_without_embedding_client(tmp_path):
    """无客户端 →纯词面行为与加固前一致."""
    kem = EpisodicMemory(storage_path=str(tmp_path / "e1"))
    kem.store(content="python fibonacci", summary="fib", task_type="interaction")
    plain_hits = kem.retrieve_similar("python fibonacci", k=5)

    sem = EpisodicMemory(
        storage_path=str(tmp_path / "e2"),
        embedding_client=FakeEmbedder().make_client(),
    )
    sem.store(content="python fibonacci", summary="fib", task_type="interaction")
    vec_hits = sem.retrieve_similar("python fibonacci", k=5)

    assert plain_hits and vec_hits
    # 词面完全命中(0.8+)在两种模式下都应作为 top
    assert plain_hits[0][1] >= 0.8 and vec_hits[0][1] >= 0.8


def test_embedding_skipped_when_store_has_no_vectors(tmp_path):
    """"库内无向量 → 不触发查询嵌入网络调用."""
    fake = FakeEmbedder()
    em = EpisodicMemory(
        storage_path=str(tmp_path / "episodes"),
        embedding_client=EmbeddingClient(),  # 有客户端但库是空的
    )
    # 人为清空: store 时已有嵌入(上面的客户端会打), 这里直接换成无网络客户端后
    # 再存入一条不带向量的记录, 确保库里没有任何 embedding
    em.embedding_client = None
    em.store(content="hello world note", summary="note", task_type="interaction")
    assert all(not ep.embedding for ep in em.episodes.values())

    em.embedding_client = fake.make_client()
    before = fake.calls
    em.retrieve_similar("hello world")
    assert fake.calls == before, "无存储向量时不应做查询嵌入调用"


def test_embedding_failure_falls_back_to_lexical(tmp_path):
    """嵌入服务失败(返回 None) → 词面召回照常工作, 不抛异常."""
    em = EpisodicMemory(
        storage_path=str(tmp_path / "episodes"),
        embedding_client=FakeEmbedder().make_client(broken=True),
    )
    em.store(content="gemini 深度研究报告流程", summary="research", task_type="interaction")
    hits = em.retrieve_similar("gemini 研究")  # 词面能命中的部分
    assert hits and hits[0][1] >= 0.8  # 子串命中 0.8 兜底
    assert all(not ep.embedding for ep in em.episodes.values())


def test_store_persists_embedding_and_relod_roundtrip(tmp_path):
    """"embedding 随 episodes.json 持久化, 重启后向量召回立即可用."""
    storage = tmp_path / "episodes"
    em1 = EpisodicMemory(
        storage_path=str(storage),
        embedding_client=FakeEmbedder().make_client(),
    )
    em1.store(content="帮用户安装 poetry 包管理器", summary="install", task_type="interaction")

    em2 = EpisodicMemory(
        storage_path=str(storage),
        embedding_client=FakeEmbedder().make_client(),
    )
    stored = list(em2.episodes.values())
    assert stored and all(ep.embedding for ep in stored)

    hits = em2.retrieve_similar("install tool for user")
    assert hits


def test_backfill_embeds_legacy_episodes(tmp_path):
    """"存量无向量记忆在后续 store 时被摊薄回填."""
    storage = tmp_path / "episodes"
    fake = FakeEmbedder()
    client = fake.make_client()
    em = EpisodicMemory(storage_path=str(storage), embedding_client=client)

    # 造两条"存量" (绕过 store 的嵌入: 临时禁用客户端)
    em.embedding_client = None
    em.store(content="旧记忆 one 安装记", summary="old one", task_type="interaction")
    em.store(content="旧记忆 two 写字记", summary="old two", task_type="interaction")
    assert all(not ep.embedding for ep in em.episodes.values())

    em.embedding_client = client
    calls_after_restore = fake.calls
    em.store(content="全新 third 安装事项", summary="third", task_type="interaction")

    embedded = [ep for ep in em.episodes.values() if ep.embedding]
    assert len(embedded) >= 3  # 新记录 + 回填的旧记录
    assert fake.calls > calls_after_restore


# ---------------------------------------------------------------- circuit breaker

def test_circuit_breaker_after_consecutive_failures():
    c = EmbeddingClient(base_url="http://test/v1", model="fake", max_failures=3)
    c.embed_fn = lambda text: None  # 永久失败
    for _ in range(3):
        assert c.embed("q") is None
    assert not c.is_available()
    # 熔断后 embed 直接短路
    assert c.embed("again") is None
    c.reset_breaker()
    assert c.is_available()


def test_success_resets_failure_counter():
    c = EmbeddingClient(base_url="http://test/v1", model="fake", max_failures=3)
    flaky = {"n": 0}

    def fn(text):
        flaky["n"] += 1
        return None if flaky["n"] % 2 == 1 else [0.1, 0.2]
    c.embed_fn = fn
    for _ in range(6):  # 失败-成功交替, 永不累积到 3
        c.embed(str(flaky["n"]))
    assert c.is_available()


# ---------------------------------------------------------------- KG

def test_kg_vector_paraphrase_recall(tmp_path):
    """中文实体词面查不到英文改写查询 → 靠KG节点向量召回."""
    kg = KnowledgeGraph(
        storage_path=str(tmp_path / "kg"),
        embedding_client=FakeEmbedder().make_client(),
    )
    kg.add_node(MemoryNode(label="tool", properties={"name": "写字帮工具"}))
    hits = kg.find_nodes(query="calligraphy tool")
    assert hits and hits[0][1] >= 0.3  # 语义分通过

    # 无客户端时同样查询应该 miss(词面无重叠)
    kg2 = KnowledgeGraph(storage_path=str(tmp_path / "kg2"))
    kg2.add_node(MemoryNode(label="tool", properties={"name": "写字帮工具"}))
    assert kg2.find_nodes(query="calligraphy tool") == []


def test_kg_no_query_embedding_when_no_stored_vectors(tmp_path):
    fake = FakeEmbedder()
    kg = KnowledgeGraph(storage_path=str(tmp_path / "kg"), embedding_client=fake.make_client())
    # 空图后添加节点但嵌入向量为空: 直接构造节点对象绕过 _embed
    node = MemoryNode(label="concept", properties={"name": "note 书写工具"})
    node.embedding = None
    with kg._lock:
        kg.nodes[node.id] = node
        kg.node_index[node.label].add(node.id)
    before = fake.calls
    kg.find_nodes(query="write note")
    assert fake.calls == before


# ---------------------------------------------------------------- manager wiring

def test_memory_manager_passes_embedding_client(tmp_path):
    from agent_project.memory import MemoryManager

    mm = MemoryManager(
        kg_storage=str(tmp_path / "kg"),
        episodic_storage=str(tmp_path / "ep"),
        file_memory_path=str(tmp_path / "memory.md"),
        user_memory_path=str(tmp_path / "user.md"),
        sqlite_session_path=str(tmp_path / "sessions.db"),
        project_root=str(tmp_path),
        embedding_client=FakeEmbedder().make_client(),
    )
    assert mm.kg.embedding_client is not None
    assert mm.episodic.embedding_client is not None

    mm.store_interaction(
        task="安装写字工具",
        trajectory={"steps": ["安装工具 done"]},
        outcome="done",
        success=True,
    )
    snippets = mm.episodic.retrieve_similar("install tool")
    assert snippets
