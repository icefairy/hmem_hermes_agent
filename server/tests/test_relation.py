"""reason / contradict 组合推理模块单元测试。

覆盖: reason 的 AND 语义与排序、contradict 的对检索。
用内嵌临时库，不碰生产数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections.abc import Generator

import pytest

from engine.holographic import encode_atom, encode_text, similarity  # noqa: F401
from engine.relation import contradict, reason
from engine.store import _HRR_TABLE, HybridMemoryStore

DIM = 256  # 测试用小维度


@pytest.fixture()
def store(tmp_path: Path) -> Generator[HybridMemoryStore, None, None]:
    db = HybridMemoryStore(db_path=str(tmp_path / "test.db"), embedding_dim=DIM)
    db.initialize()
    db.add_memory("docker 部署 镜像 推送 是一套完整流程", memory_type="experience")
    db.add_memory("docker push 推送到 HTTP registry 配置", memory_type="experience")
    db.add_memory("mizar 事件驱动架构 主循环 设计", memory_type="observation")
    db.add_memory("发明专利 撰写 需要 技术交底书", memory_type="observation")
    yield db
    db.close()


def test_reason_returns_all_and_sorts(store: HybridMemoryStore):
    """reason 应返回全部记忆按组合分降序。"""
    results = reason(store, ["docker"], limit=10)
    assert len(results) == 4  # 所有记忆都会有一次打分
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # docker 记忆应排最前
    assert "docker" in results[0]["content"]


def test_reason_and_semantics(store: HybridMemoryStore):
    """多实体 AND：同时命中所有实体的记忆分数高于只命中部分。"""
    both = reason(store, ["docker", "push"], limit=10)
    top = both[0]
    assert "docker" in top["content"] and "push" in top["content"]
    # 每个实体都有明细相似度
    assert set(top["entity_sims"].keys()) == {"docker", "push"}
    # AND 分数 = min 分量 -> 不大于任一单实体分
    assert top["score"] <= top["entity_sims"]["docker"]
    assert top["score"] <= top["entity_sims"]["push"]


def test_reason_empty_entities(store: HybridMemoryStore):
    assert reason(store, []) == []


def test_reason_entity_sims_communicate(store: HybridMemoryStore):
    """相同实体检索两次结果一致（确定性）。"""
    a = [r["id"] for r in reason(store, ["docker"], limit=10)]
    b = [r["id"] for r in reason(store, ["docker"], limit=10)]
    assert a == b


def test_contradict_low_similarity_pairs(store: HybridMemoryStore):
    """contradict 应返回相似度<=threshold 的对。"""
    results = contradict(store, threshold=0.5, limit=10)
    assert len(results) >= 1
    assert all(r["similarity"] <= 0.5 for r in results)
    # 每个对都有双方 id 与内容摘要
    for r in results:
        assert r["memory_a"] != r["memory_b"]
        assert r["content_a"] and r["content_b"]


def test_contradict_threshold_filters(store: HybridMemoryStore):
    """threshold 收紧后返回更少（或同样多的）对。"""
    loose = contradict(store, threshold=0.9, limit=50)
    strict = contradict(store, threshold=0.05, limit=50)
    assert len(strict) <= len(loose)


def test_relation_uses_hrr_vectors(store: HybridMemoryStore):
    """reason 依赖 hrr_memories 表：无向量时返回空。"""
    # 清掉 hrr 表 -> reason 返回空
    store._conn.execute(f"DELETE FROM {_HRR_TABLE}")
    store._conn.commit()
    assert reason(store, ["docker"], limit=10) == []
    assert contradict(store, limit=10) == []
