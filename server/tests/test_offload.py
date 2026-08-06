"""上下文卸载模块单元测试。

覆盖: put→get 100% 找回、自动摘要生成、node_id 自动生成、
session 索引、软删除/硬删除、Mermaid 画布、upsert 覆盖更新。
用临时目录隔离数据，不碰生产库。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 保证能 import server 包（测试从 server/ 目录跑）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

# 必须先设置环境变量再 import app
_tmp = tempfile.mkdtemp(prefix="hmem_offload_test_")
os.environ["HMEM_DATA_DIR"] = _tmp
os.environ["HMEM_API_KEY"] = "test-key"
os.environ["HMEM_DEBUG"] = "1"

from main import app  # noqa: E402

API_KEY = "test-key"

# 用上下文管理器触发 lifespan（app.state.settings 在 lifespan 中初始化）
with TestClient(app) as _client:
    client = _client


def _auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture(autouse=True)
def _cleanup():
    """每个测试后清理数据目录，保证隔离。"""
    yield
    for p in Path(_tmp).glob("*"):
        if p.is_dir():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# ─── put → get 找回链路 ────────────────────────────────────────────────────


def test_put_then_get_roundtrip():
    """原文 100% 可找回（含多行、中文、特殊字符）。"""
    content = "第一条完整原文\n第二行\n{\"json\": true}\n<tag>&特殊字符</tag>"
    r = client.post(
        "/api/v1/offload/put",
        json={"session_key": "s1", "node_id": "step1", "content": content,
              "summary": "摘要", "content_type": "tool_result"},
        headers=_auth(),
    )
    assert r.status_code == 200, r.text
    put = r.json()
    assert put["node_id"] == "step1"
    assert put["summary"] == "摘要"

    r = client.get("/api/v1/offload/get", params={"session_key": "s1", "node_id": "step1"}, headers=_auth())
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["content"] == content
    assert got["refs_path"] == "refs/step1.md"
    # refs 文件确实落盘
    ref_file = Path(_tmp) / "offload" / "default" / "s1" / "refs" / "step1.md"
    assert ref_file.read_text(encoding="utf-8") == content


def test_put_auto_summary():
    """不传 summary 时自动生成一行摘要（去换行限长）。"""
    long_content = "\n".join(f"line{i} 内容" for i in range(30))
    r = client.post(
        "/api/v1/offload/put",
        json={"session_key": "s2", "content": long_content},
        headers=_auth(),
    )
    assert r.status_code == 200
    put = r.json()
    assert put["summary"]
    assert "\n" not in put["summary"]
    assert len(put["summary"]) <= 120
    # node_id 自动生成
    assert put["node_id"].startswith("node_")


def test_put_auto_node_id_unique():
    """自动 node_id 两次不同。"""
    a = client.post("/api/v1/offload/put", json={"session_key": "s3", "content": "x1"}, headers=_auth()).json()
    b = client.post("/api/v1/offload/put", json={"session_key": "s3", "content": "x2"}, headers=_auth()).json()
    assert a["node_id"] != b["node_id"]


def test_put_upsert_same_node():
    """同 session 同 node_id 重复 put = 覆盖更新，记录数不增。"""
    for i in range(2):
        r = client.post(
            "/api/v1/offload/put",
            json={"session_key": "s4", "node_id": "n1", "content": f"v{i}", "summary": f"sum{i}"},
            headers=_auth(),
        )
        assert r.status_code == 200
    idx = client.get("/api/v1/offload/session/s4", headers=_auth()).json()
    assert idx["count"] == 1
    got = client.get("/api/v1/offload/get", params={"session_key": "s4", "node_id": "n1"}, headers=_auth()).json()
    assert got["content"] == "v1"
    assert got["summary"] == "sum1"


def test_get_missing_record_404():
    r = client.get("/api/v1/offload/get", params={"session_key": "s5", "node_id": "nope"}, headers=_auth())
    assert r.status_code == 404


# ─── session 空间 ──────────────────────────────────────────────────────────


def test_session_create_idempotent():
    a = client.post("/api/v1/offload/session", json={"session_key": "sess-a"}, headers=_auth())
    assert a.status_code == 200
    assert a.json()["session_key"] == "sess-a"
    assert Path(a.json()["refs_dir"]).is_dir()
    b = client.post("/api/v1/offload/session", json={"session_key": "sess-a"}, headers=_auth())
    assert b.json()["record_count"] == a.json()["record_count"]


def test_session_index_lists_summaries():
    client.post("/api/v1/offload/put", json={"session_key": "s6", "node_id": "a", "content": "AAA 内容", "summary": "sum-a"}, headers=_auth())
    client.post("/api/v1/offload/put", json={"session_key": "s6", "node_id": "b", "content": "BBB 内容", "summary": "sum-b"}, headers=_auth())
    idx = client.get("/api/v1/offload/session/s6", headers=_auth()).json()
    assert idx["count"] == 2
    assert [r["node_id"] for r in idx["records"]] == ["a", "b"]
    assert all("content" not in r for r in idx["records"])  # 索引不含原文


# ─── 清理：软删除 / 硬删除 ────────────────────────────────────────────────


def test_delete_session_soft_keeps_data():
    client.post("/api/v1/offload/session", json={"session_key": "s7"}, headers=_auth())
    client.post("/api/v1/offload/put", json={"session_key": "s7", "node_id": "a", "content": "data-a"}, headers=_auth())
    # 软删除
    r = client.delete("/api/v1/offload/session/s7", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "soft"
    assert body["count"] == 1
    # 索引默认看不到已删记录
    idx = client.get("/api/v1/offload/session/s7", headers=_auth()).json()
    assert idx["count"] == 0
    # include_deleted 能看到
    idx = client.get("/api/v1/offload/session/s7", params={"include_deleted": "true"}, headers=_auth()).json()
    assert idx["count"] == 1
    # get 默认 404（软删除不可见）
    assert client.get("/api/v1/offload/get", params={"session_key": "s7", "node_id": "a"}, headers=_auth()).status_code == 404
    # refs 目录移入 .trash，增量数据保留
    trash = Path(_tmp) / "offload" / "default" / ".trash"
    assert any(trash.iterdir())


def test_delete_session_hard_removes_all():
    client.post("/api/v1/offload/put", json={"session_key": "s8", "node_id": "a", "content": "data-a"}, headers=_auth())
    r = client.delete("/api/v1/offload/session/s8", params={"hard": "true"}, headers=_auth())
    assert r.status_code == 200
    assert r.json()["mode"] == "hard"
    idx = client.get("/api/v1/offload/session/s8", params={"include_deleted": "true"}, headers=_auth()).json()
    assert idx["count"] == 0
    session_dir = Path(_tmp) / "offload" / "default" / "s8"
    assert not session_dir.exists()


# ─── Mermaid 画布 ─────────────────────────────────────────────────────────


def test_canvas_generation_with_deps():
    client.post("/api/v1/offload/put", json={"session_key": "s9", "node_id": "parse", "content": "parse 全文", "summary": "解析输入", "meta": {}}, headers=_auth())
    client.post("/api/v1/offload/put", json={"session_key": "s9", "node_id": "analyze", "content": "analyze 全文", "summary": "分析结果", "meta": {"depends_on": "parse"}}, headers=_auth())
    r = client.get("/api/v1/offload/canvas/s9", headers=_auth())
    assert r.status_code == 200
    mermaid = r.json()["mermaid"]
    assert "flowchart TD" in mermaid
    assert 'parse["解析输入"]' in mermaid
    assert "parse --> analyze" in mermaid


def test_canvas_empty():
    r = client.get("/api/v1/offload/canvas/s-empty", headers=_auth())
    assert r.status_code == 200
    assert "empty session" in r.json()["mermaid"]


# ─── namespace 隔离 ───────────────────────────────────────────────────────


def test_namespace_isolation():
    client.post("/api/v1/offload/put", json={"session_key": "ns1", "node_id": "a", "content": "default-内容", "namespace": "default"}, headers=_auth())
    client.post("/api/v1/offload/put", json={"session_key": "ns1", "node_id": "a", "content": "other-内容", "namespace": "other"}, headers=_auth())
    d = client.get("/api/v1/offload/get", params={"session_key": "ns1", "node_id": "a"}, headers=_auth()).json()
    o = client.get("/api/v1/offload/get", params={"session_key": "ns1", "node_id": "a", "namespace": "other"}, headers=_auth()).json()
    assert d["content"] == "default-内容"
    assert o["content"] == "other-内容"
    # 文件目录按 namespace 分离
    assert (Path(_tmp) / "offload" / "default" / "ns1" / "refs" / "a.md").exists()
    assert (Path(_tmp) / "offload" / "other" / "ns1" / "refs" / "a.md").exists()


# ─── auth ─────────────────────────────────────────────────────────────────


def test_auth_required():
    r = client.post("/api/v1/offload/put", json={"session_key": "x", "content": "y"})
    assert r.status_code == 401
    r = client.get("/api/v1/offload/session/x")
    assert r.status_code == 401


# ─── 路径穿越防护 ─────────────────────────────────────────────────────────


def test_path_traversal_sanitized():
    r = client.post(
        "/api/v1/offload/put",
        json={"session_key": "../../etc", "node_id": "../../passwd", "content": "evil"},
        headers=_auth(),
    )
    assert r.status_code == 200
    # 清洗后不会逃出 offload 根目录
    put = r.json()
    assert "/" not in put["session_key"]
    assert "/" not in put["node_id"]
