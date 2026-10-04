"""文本归一化 + 存量回填测试（Engram tokenizer compression 借鉴）。"""

from __future__ import annotations

import os
import tempfile

from engine.normalize import normalize
from engine.store import HybridMemoryStore


def test_normalize_fullwidth_halfwidth():
    assert normalize("ＭＣＰ（测试）") == "mcp(测试)"
    assert normalize("ＨＥＬＬＯ") == "hello"


def test_normalize_case_and_whitespace():
    assert normalize("  Math   Test  ") == "math test"
    assert normalize("MCP") == normalize("mcp")


def test_normalize_circled_digit():
    assert normalize("⑦") == "7"
    assert normalize("①") == "1"


def test_normalize_empty_safe():
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def _fresh_store() -> HybridMemoryStore:
    d = tempfile.mkdtemp()
    s = HybridMemoryStore(db_path=os.path.join(d, "t.db"), embedding_dim=8)
    s.initialize()
    return s


def test_fts_finds_fullwidth_via_halfwidth_query():
    """写入全角，用半角查询应能命中（归一化生效）。"""
    s = _fresh_store()
    try:
        s.add_memory(content="部署手册提到 ＭＣＰ 的 ＨＴＴＰ 配置", memory_type="observation")
        hits = s.search_fts("mcp http", limit=5)
        assert len(hits) == 1, f"归一化后应命中，实际 {len(hits)} 条"
    finally:
        s.close()


def test_fts_finds_mixed_case():
    s = _fresh_store()
    try:
        s.add_memory(content="Kubernetes Deployment 指南", memory_type="observation")
        assert len(s.search_fts("kubernetes", limit=5)) == 1
        assert len(s.search_fts("KUBERNETES", limit=5)) == 1
    finally:
        s.close()


def test_rebuild_tokenization_is_idempotent():
    """存量回填：第一次改、第二次无改动（幂等）。"""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.db")
    s = HybridMemoryStore(db_path=path, embedding_dim=8)
    s.initialize()
    try:
        # 直接写入一条「未归一化」的 content_jieba 模拟旧数据
        s.add_memory(content="ＭＣＰ 测试", memory_type="observation")
        mid = s._conn.execute("SELECT id FROM memories").fetchone()[0]
        s._conn.execute(
            "UPDATE memories SET content_jieba = ? WHERE id = ?",
            ("ＭＣＰ 测试", mid),
        )
        s._conn.commit()
        n1 = s.rebuild_tokenization()
        assert n1 >= 1
        n2 = s.rebuild_tokenization()
        assert n2 == 0, "第二次应无改动（幂等）"
        # 归一化后 FTS 命中
        assert len(s.search_fts("mcp", limit=5)) == 1
    finally:
        s.close()
