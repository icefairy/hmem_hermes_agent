"""POST /memories 溯源字段（doc_*）回归测试。

2026-09-12 需求：文档型长文（课程/经典精读笔记）应能带上 doc_id 等溯源字段，
以获得与知识库文档同等的待遇：
  - 检索不随时间衰减（retriever 对 doc_id 非空则 time_weight=1.0）
  - 不被主动遗忘顾问建议归档

同时**不要求**改成 memory_type=knowledge —— 保持 insight 类型才能继续参与
reflect 提炼，这是与「整篇 hmem_doc_import 分块入库」的关键区别。

用 TestClient + 临时数据目录，不碰生产库。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = tempfile.mkdtemp(prefix="hmem_docmeta_test_")
os.environ["HMEM_DATA_DIR"] = _tmp
os.environ["HMEM_API_KEY"] = "test-key"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

with TestClient(app) as _client:
    client = _client

API_KEY = "test-key"
NS = "docmeta"


def _auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def _write(**kw) -> dict:
    body = {"content": kw.pop("content", "test content"), "namespace": NS, **kw}
    r = client.post("/api/v1/memories", json=body, headers=_auth())
    assert r.status_code == 200, r.text
    return r.json()


def test_doc_fields_persisted():
    """doc_id / doc_title / chunk_index / doc_category / doc_tags 应落库。"""
    res = _write(
        content="【卜筮正术·第002讲】正文……",
        memory_type="insight",
        doc_id="bushi-zhengshu-002",
        doc_title="卜筮正术·第002讲",
        chunk_index=0,
        doc_category="文始道",
        doc_tags=["卜筮", "易学"],
    )
    mid = res["memory_id"]

    r = client.get(f"/api/v1/memories/{mid}?namespace={NS}", headers=_auth())
    m = r.json()
    assert m["doc_id"] == "bushi-zhengshu-002"
    assert m["doc_title"] == "卜筮正术·第002讲"
    assert m["doc_category"] == "文始道"
    assert "卜筮" in (m["doc_tags"] or "")


def test_doc_id_does_not_force_knowledge_type():
    """带 doc_id 时 memory_type 仍应保持调用方指定的值（默认不改成 knowledge）。"""
    res = _write(content="带溯源的经验", memory_type="insight",
                 doc_id="some-doc")
    mid = res["memory_id"]
    m = client.get(f"/api/v1/memories/{mid}?namespace={NS}", headers=_auth()).json()
    assert m["memory_type"] == "insight", (
        "doc_id 是为了抗衰减/防遗忘，不应把类型改成 knowledge——"
        "改了会退出 reflect 提炼管线"
    )


def test_doc_id_shields_from_time_decay():
    """带 doc_id 的条目 time_weight 不衰减；同样陈旧的普通条目会衰减。"""
    from engine.retriever import HybridRetriever
    from engine.store import HybridMemoryStore

    settings = app.state.settings
    store = HybridMemoryStore(db_path=f"{settings.db_root}/{NS}.db",
                              embedding_dim=settings.embedding_dim)
    store.initialize()
    retriever = HybridRetriever(store=store, embedding_client=None)

    old_ts = "2020-01-01 00:00:00"  # 极旧，普通条目必然衰减
    plain = {
        "id": 1, "content": "x", "memory_type": "insight",
        "created_at": old_ts, "fts_rank": -1.0,
    }
    shielded = dict(plain, doc_id="keep-doc")

    s_plain = retriever._compute_score(plain)
    s_shield = retriever._compute_score(shielded)
    store.close()

    assert s_shield > s_plain, (
        f"带 doc_id 应免于时间衰减: doc_id={s_shield:.4f} 普通={s_plain:.4f}"
    )
    # 反向验证：doc_id 不是万能加分——去掉陈旧因子后两者应一致
    fresh = dict(plain, created_at=None)
    assert abs(retriever._compute_score(fresh) - s_shield) < 1e-9, (
        "doc_id 的作用应仅为豁免时间衰减，不应额外抬分"
    )


def test_doc_id_excluded_from_forget_advisor():
    """带 doc_id 的条目不应出现在遗忘建议候选中（对照：同样陈旧的普通条目会出现）。"""
    from engine.store import HybridMemoryStore

    plain_res = _write(content="陈旧低重要度普通条目", memory_type="insight")
    doc_res = _write(content="陈旧低重要度文档条目", memory_type="insight",
                     doc_id="keep-me-doc")

    settings = app.state.settings
    store = HybridMemoryStore(db_path=f"{settings.db_root}/{NS}.db",
                              embedding_dim=settings.embedding_dim)
    store.initialize()
    # 把两者都变陈旧且低重要度
    for mid in (plain_res["memory_id"], doc_res["memory_id"]):
        store._conn.execute(
            "UPDATE memories SET created_at=?, last_hit_at='', importance=0.05 WHERE id=?",
            ("2020-01-01 00:00:00", mid),
        )
    store._conn.commit()

    adv = store.forget_advisor(threshold_days=1, min_importance=0.5)
    cand_ids = {c["id"] for c in adv.get("candidates", [])}
    store.close()

    assert plain_res["memory_id"] in cand_ids, (
        f"对照组（无 doc_id）应出现在遗忘候选: {cand_ids}"
    )
    assert doc_res["memory_id"] not in cand_ids, (
        "带 doc_id 的条目不应被建议遗忘——否则本条测试无法证伪"
    )


def test_doc_fields_default_empty_for_plain_memory():
    """不带 doc_* 的普通写入应保持原行为（字段为空）。"""
    res = _write(content="普通记忆", memory_type="observation")
    mid = res["memory_id"]
    m = client.get(f"/api/v1/memories/{mid}?namespace={NS}", headers=_auth()).json()
    assert not m.get("doc_id"), "普通写入不应凭空产生 doc_id"
