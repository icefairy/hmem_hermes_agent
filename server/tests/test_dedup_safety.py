"""去重安全回归测试 —— 2026-09-12 课程笔记被吞事故。

事故：merge_similar 把同类型（尤指 insight）相似记忆聚类后，只保留最长的一条
并将其余条目**硬删除**，且不合并内容；同时 pinned/no_forget 保护标记被无视。
文始道《卜筮正术》第003讲（#12175，2280 字）写入 46 秒后被吞。

本文件锁定三条不变量：
  1. 去重永不硬删除任何成员——被合并者只归档（可逆）
  2. 同簇所有成员的内容必须并入保留条目，内容零丢失
  3. pinned / no_forget 记忆永不参与合并，也永不可被 delete_memory 删除

用内嵌临时库 + 手工向量，不碰生产数据与真实 API。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections.abc import Generator

import pytest

from engine.dedup import merge_similar
from engine.store import HybridMemoryStore

DIM = 256
# 手工向量：cos(v1, v2) ≈ 0.914 > 0.85 阈值 → 必被判为同一簇
V_A = [1.0] + [0.0] * (DIM - 1)
V_B = [0.9, 0.4] + [0.0] * (DIM - 2)


@pytest.fixture()
def store(tmp_path: Path) -> Generator[HybridMemoryStore, None, None]:
    s = HybridMemoryStore(db_path=str(tmp_path / "t.db"), embedding_dim=DIM)
    s.initialize()
    yield s
    s.close()


def _identical_cluster(store: HybridMemoryStore, n: int = 5, mtype: str = "insight") -> list[int]:
    """n 条内容相同、向量相同 → 相似度 1.0，必同簇。"""
    return [
        store.add_memory(f"identical note {i}", embedding=V_A, memory_type=mtype)
        for i in range(n)
    ]


# -- 不变量 1: 永不硬删除 ---------------------------------------------------


def test_members_are_archived_not_deleted(store: HybridMemoryStore):
    ids = _identical_cluster(store, 5)
    r = merge_similar(store, None, "insight", threshold=0.85)

    assert r["merged_count"] == 4, f"应有 4 条被并入, 实际 {r['merged_count']}"
    # 事故核心：这些 id 以前直接从库里消失
    for mid in ids:
        m = store.get_memory(mid)
        assert m is not None, f"记忆 {mid} 被硬删除——只允许归档"
        assert m["content"], f"记忆 {mid} 内容被清空"


def test_no_hard_delete_reported(store: HybridMemoryStore):
    _identical_cluster(store, 3)
    r = merge_similar(store, None, "insight", threshold=0.85)
    assert r["deleted_ids"] == [], (
        f"去重不允许硬删除, 却报了 deleted_ids={r['deleted_ids']}"
    )
    assert len(r["superseded_ids"]) == 2, f"应记录被归档的 id, 实际 {r['superseded_ids']}"


def test_archived_members_are_recoverable(store: HybridMemoryStore):
    ids = _identical_cluster(store, 4)
    merge_similar(store, None, "insight", threshold=0.85)

    archived = store.list_archived(limit=100)
    archived_ids = {a["id"] for a in archived}
    # 至少 3 条被归档，且都能通过 unarchive 还原
    assert len(archived_ids & set(ids)) == 3, f"归档集={archived_ids}"
    for mid in archived_ids:
        assert store.unarchive_memory(mid), f"归档 {mid} 应可恢复"


# -- 不变量 2: 内容零丢失 ---------------------------------------------------


def test_similar_notes_merge_content_without_loss(store: HybridMemoryStore):
    """两条相似但不同的笔记：短的那条内容必须并入留存条目，不能蒸发。

    注：此处故意用短文（<500 字）——长文走 0.92 严阈值，不保证在 0.85 下合并。
    本测试关注的是「合并时内容不得丢失」，与阈值无关。
    """
    long_id = store.add_memory("甲" * 300, embedding=V_A, memory_type="insight")
    short_id = store.add_memory("UNIQUE_MARKER_B 的关键结论", embedding=V_B,
                                memory_type="insight")

    r = merge_similar(store, None, "insight", threshold=0.85)
    assert r["merged_count"] == 1, "两条应被判为同簇"

    keeper = store.get_memory(long_id)
    assert keeper is not None, "最长的条目应作为保留条目"
    assert "UNIQUE_MARKER_B 的关键结论" in keeper["content"], (
        "被合并条目的内容必须并入保留条目——这正是 003 讲丢失的原因"
    )
    # 被合并者自身内容也不变
    assert "UNIQUE_MARKER_B" in store.get_memory(short_id)["content"]


def test_every_cluster_member_content_survives_anywhere(store: HybridMemoryStore):
    """核心不变量：合并后，每条原记忆的内容仍能在库中找到。"""
    notes = {
        store.add_memory(f"note-{i} 独有结论{i}", embedding=V_A, memory_type="insight")
        for i in range(6)
    }
    merge_similar(store, None, "insight", threshold=0.85)

    blob = "\n".join(
        m["content"] for m in store.list_memories(memory_type="insight", limit=999)
    )
    for i in range(6):
        assert f"独有结论{i}" in blob, f"note-{i} 的内容在合并后消失"


# -- 不变量 3: pinned / no_forget 让路 ---------------------------------------


def test_pinned_memory_never_enters_merge(store: HybridMemoryStore):
    pinned_id = store.add_memory("pinned 长文" + "P" * 600, embedding=V_A,
                                 memory_type="insight")
    store.set_pinned(pinned_id, True)
    other_ids = [
        store.add_memory(f"ordinary {i}", embedding=V_A, memory_type="insight")
        for i in range(3)
    ]

    r = merge_similar(store, None, "insight", threshold=0.85)

    assert store.get_memory(pinned_id) is not None, "pinned 记忆不可被去重删除"
    assert store.get_memory(pinned_id)["pinned"] == 1, "pinned 标记不应被改动"
    assert store.get_memory(pinned_id)["archived"] == 0, "pinned 记忆不应被归档"
    assert pinned_id not in r["superseded_ids"], "pinned 不应出现在被合并名单"
    for mid in other_ids:
        assert store.get_memory(mid) is not None, f"{mid} 也应只归档不删除"


def test_no_forget_memory_never_enters_merge(store: HybridMemoryStore):
    nf_id = store.add_memory("no_forget 长文" + "N" * 600, embedding=V_A,
                             memory_type="insight")
    store.set_no_forget(nf_id, True)
    store.add_memory("ordinary", embedding=V_A, memory_type="insight")

    r = merge_similar(store, None, "insight", threshold=0.85)

    assert store.get_memory(nf_id)["no_forget"] == 1
    assert store.get_memory(nf_id)["archived"] == 0
    assert nf_id not in r["superseded_ids"]


def test_delete_memory_refuses_pinned(store: HybridMemoryStore):
    """底层删除也必须尊重 pinned —— 防未来新代码路径再次绕过。"""
    mid = store.add_memory("protected forever", embedding=V_A,
                           memory_type="observation")
    store.set_pinned(mid, True)

    assert store.delete_memory(mid) is False, "pinned 记忆不可被 delete_memory 删除"
    assert store.get_memory(mid) is not None, "pinned 记忆应仍在库中"

    store.set_pinned(mid, False)
    assert store.delete_memory(mid) is True, "解除 pinned 后应可正常删除"


def test_delete_memory_refuses_no_forget(store: HybridMemoryStore):
    mid = store.add_memory("self identity", embedding=V_A,
                           memory_type="observation")
    store.set_no_forget(mid, True)

    assert store.delete_memory(mid) is False, "no_forget 记忆不可被删除"
    assert store.get_memory(mid) is not None


# -- 长文保护：长文档用更严阈值 ---------------------------------------------
# 长文（课程/经典精读笔记）嵌入天然趋同：同为"卜筮/易经"主题的不同讲次
# 很容易越过 0.80/0.85 被判为重复。这是 003 讲被吞的同源隐患。


def test_long_notes_use_stricter_threshold(store: HybridMemoryStore):
    """两篇 600 字长文，cos≈0.914：普通阈值(0.85)会合并，长文阈值(0.92)不合并。"""
    a = store.add_memory("讲次甲 " + "甲" * 600, embedding=V_A, memory_type="insight")
    b = store.add_memory("讲次乙 " + "乙" * 600, embedding=V_B, memory_type="insight")

    r = merge_similar(store, None, "insight", threshold=0.85)

    assert r["merged_count"] == 0, (
        f"两篇不同讲次的长文不应被合并 (cos≈0.914 < 0.92)，实际合并了 {r['merged_count']} 条"
    )
    assert store.get_memory(a)["archived"] == 0
    assert store.get_memory(b)["archived"] == 0


def test_short_notes_still_use_normal_threshold(store: HybridMemoryStore):
    """同样的 cos≈0.914，短文仍按传入阈值合并——不因长文规则而收紧短文行为。"""
    a = store.add_memory("短文甲", embedding=V_A, memory_type="insight")
    b = store.add_memory("短文乙", embedding=V_B, memory_type="insight")

    r = merge_similar(store, None, "insight", threshold=0.85)

    assert r["merged_count"] == 1, "短文应仍按 0.85 阈值合并"
    keeper = store.get_memory(a) if store.get_memory(a)["archived"] == 0 else store.get_memory(b)
    assert "短文甲" in keeper["content"] and "短文乙" in keeper["content"], (
        "短文合并同样不得丢内容"
    )


def test_long_threshold_applies_when_one_side_is_long(store: HybridMemoryStore):
    """长文 + 短文：只要有一方是长文就走严阈值，防长文被短文拖进簇。"""
    store.add_memory("短笔记", embedding=V_A, memory_type="insight")
    store.add_memory("讲次长文 " + "长" * 600, embedding=V_B, memory_type="insight")

    r = merge_similar(store, None, "insight", threshold=0.80)

    assert r["merged_count"] == 0, (
        "长短混搭且一方 >=500 字时应用 0.92 严阈值，不应合并"
    )


def test_near_identical_long_notes_still_merge(store: HybridMemoryStore):
    """严阈值不等于"长文永不合并"：几乎相同的长文仍应合并（且内容相加）。"""
    store.add_memory("同一讲 " + "X" * 600, embedding=V_A, memory_type="insight")
    store.add_memory("同一讲 " + "X" * 600, embedding=V_A, memory_type="insight")

    r = merge_similar(store, None, "insight", threshold=0.80)

    assert r["merged_count"] == 1, f"完全相同(cos=1.0)的长文仍应合并, 实际 {r}"
    assert len(r["superseded_ids"]) == 1


# -- 绕过路径封堵：保护必须下沉到 store 层，不能只挂在 merge_similar -------------
# 2026-09-12 复核发现：保护原先只在 merge_similar 的前置过滤里，属补丁式防护。
# 以下测试锁定「任何删除/归档路径都过同一道关」。


def test_delete_by_doc_id_skips_pinned(store: HybridMemoryStore):
    """级联删文档时跳过 pinned chunk —— 否则带 doc_id 的笔记可被整篇硬删除。"""
    a = store.add_memory("chunk-A", embedding=V_A, memory_type="knowledge",
                         doc_id="d1", chunk_index=0)
    b = store.add_memory("chunk-B", embedding=V_A, memory_type="knowledge",
                         doc_id="d1", chunk_index=1)
    c = store.add_memory("chunk-C", embedding=V_A, memory_type="knowledge",
                         doc_id="d1", chunk_index=2)
    store.set_pinned(b, True)

    removed = store.delete_by_doc_id("d1")

    assert removed == 2, f"应只删 2 个未受保护 chunk，实际 {removed}"
    assert store.get_memory(a) is None, "未受保护 chunk 应被删除"
    assert store.get_memory(c) is None, "未受保护 chunk 应被删除"
    assert store.get_memory(b) is not None, "pinned chunk 不可被级联删除"
    assert store.get_memory(b)["pinned"] == 1


def test_delete_by_doc_id_refuses_when_all_protected(store: HybridMemoryStore):
    """整篇文档都受保护时，级联删除应等于无操作。"""
    ids = [store.add_memory(f"p{i}", embedding=V_A, memory_type="knowledge",
                            doc_id="d2", chunk_index=i) for i in range(3)]
    for mid in ids:
        store.set_pinned(mid, True)

    assert store.delete_by_doc_id("d2") == 0, "全受保护时应删 0 条"
    for mid in ids:
        assert store.get_memory(mid) is not None, "整篇文档必须完好"
    assert store.count_memories() == 3


def test_delete_by_doc_id_skips_no_forget(store: HybridMemoryStore):
    """no_forget 同样受保护（原实现只查 pinned）。"""
    a = store.add_memory("nf", embedding=V_A, memory_type="knowledge",
                         doc_id="d3", chunk_index=0)
    store.set_no_forget(a, True)
    assert store.delete_by_doc_id("d3") == 0
    assert store.get_memory(a) is not None


def test_archive_memory_refuses_no_forget(store: HybridMemoryStore):
    """归档也必须尊重 no_forget —— 原实现只查 pinned，SELF 层可被归档。"""
    mid = store.add_memory("self identity", embedding=V_A,
                           memory_type="self_identity")
    store.set_no_forget(mid, True)

    assert store.archive_memory(mid) is False, "no_forget 记忆不可被归档"
    assert store.get_memory(mid)["archived"] == 0

    store.set_no_forget(mid, False)
    assert store.archive_memory(mid) is True, "解除后应可归档"


def test_merge_similar_does_not_record_refused_archive(store: HybridMemoryStore):
    """merge_similar 不得把「归档被拒」的 id 记入 superseded_ids。

    archive_memory 拒绝时返回 False（不抛异常），若不检查返回值会误报已合并。
    """
    keeper = store.add_memory("长文" + "K" * 700, embedding=V_A, memory_type="insight")
    # 同簇但受保护的成员：不应被归档，也不该出现在 superseded_ids
    shielded = store.add_memory("同簇短条", embedding=V_A, memory_type="insight")
    store.set_pinned(shielded, True)

    r = merge_similar(store, None, "insight", threshold=0.85)

    assert shielded not in r["superseded_ids"], (
        "受保护条目被归档却不能出现在 superseded_ids"
    )
    assert store.get_memory(shielded)["archived"] == 0, "受保护条目不应被归档"
    assert store.get_memory(keeper) is not None
