"""知识库文档路由 — 文档级导入 / 级联删除 / 列表。

每份文档按 chunk 写入 memories 表，用 doc_id 关联；chunk 同时走
FTS5 + 向量 + HRR，可被混合检索命中；结果携带 source 溯源字段。

用法:
  POST   /api/v1/documents              导入纯文本文档（自动分块 + 向量化）
  GET    /api/v1/documents?namespace=x  列出该库所有文档及 chunk 数
  GET    /api/v1/documents/{doc_id}?namespace=x  取文档明细（含所有 chunk）
  DELETE /api/v1/documents/{doc_id}?namespace=x  级联删除整篇文档
"""

from __future__ import annotations

import json
import logging
import re

from engine.embeddings import EmbeddingClient
from engine.store import HybridMemoryStore
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

# 默认分块参数（按中文字符/词近似）
_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_OVERLAP = 80

# 单片字符硬上限：与 store 层超长记忆分片阈值一致。
# 上游 embedding/rerank 上下文 8192 token，超过会导致整个请求 400。
_MAX_CHUNK_CHARS = 3000

# 单次 embedding 请求的最大片数（大文档分批，避免一次请求过大）
_EMBED_BATCH_SIZE = 32


class DocumentRequest(BaseModel):
    content: str
    doc_id: str | None = None
    title: str = ""
    uri: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    overlap: int = _DEFAULT_OVERLAP
    namespace: str | None = None


class _DocRouterHelpers:
    @staticmethod
    def make_store(req: Request, namespace: str) -> HybridMemoryStore:
        settings = req.app.state.settings
        store = HybridMemoryStore(
            db_path=f"{settings.db_root}/{namespace}.db",
            embedding_dim=settings.embedding_dim,
        )
        store.initialize()
        return store

    @staticmethod
    def make_embedding_client(req: Request) -> EmbeddingClient | None:
        settings = req.app.state.settings
        if not (settings.embedding_base_url and settings.embedding_api_key):
            return None
        return EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )


def _split_chunks(content: str, chunk_size: int, overlap: int) -> list[str]:
    """按词切分贪心拼块，保留 overlap 词重叠，防止语义断裂。

    切好词后按原始文本偏移做子串切片，保持原文原样（不带空格）。

    上限保护（2026-09-17）：chunk_size 是「词数」且由调用方传入，原先无上限
    校验——传大值或碰到无词边界的文本（如纯符号）会产出超长 chunk，
    写入后又会变成 rerank 整体 400 的根源。这里把单片硬限在
    _MAX_CHUNK_CHARS 字符内（与超长记忆分片阈值一致）。
    """
    text = (content or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        chunk_size = _DEFAULT_CHUNK_SIZE
    # 词数上限：中文字符数 ≤ 词数，按最坏情况（1 字符/词）反推，
    # 保证单片字符数不超 _MAX_CHUNK_CHARS。
    chunk_size = min(chunk_size, _MAX_CHUNK_CHARS)
    if overlap < 0 or overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)
    # 按词切分（中英文混合，jieba 词级），记录每个词的起止偏移
    words = list(re.finditer(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+|[^\w\s]", text))
    if not words:
        # 无词边界：按字符定长切片，不能整篇返回（否则产生超长 chunk）
        step = max(1, chunk_size - overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]
    tokens: list[tuple[int, int]] = [(m.start(), m.end()) for m in words]  # (start, end) 偏移
    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        seg = tokens[i : i + chunk_size]
        if not seg:
            continue
        s = seg[0][0]
        e = seg[-1][1]
        piece = text[s:e].strip()
        if piece:
            chunks.append(piece)
    if not chunks:
        chunks = [text]
    # 兵：仍超限的片（极端情况，如单"词"异常长）再按字符硬切
    out: list[str] = []
    for c in chunks:
        if len(c) <= _MAX_CHUNK_CHARS:
            out.append(c)
        else:
            out.extend(
                c[i : i + _MAX_CHUNK_CHARS]
                for i in range(0, len(c), _MAX_CHUNK_CHARS)
            )
    return out


@router.post("/documents")
async def import_document(req: Request, body: DocumentRequest):
    """导入一篇纯文本文档，按 chunk 分块写入知识库 namespace。

    doc_id 缺省时自动生成（基于标题/uuid）。返回 chunk 数、嵌入状态。
    """
    if not body.content or not body.content.strip():
        raise HTTPException(400, "content is required")

    namespace = body.namespace or "default"
    store = _DocRouterHelpers.make_store(req, namespace)
    embedding_client = _DocRouterHelpers.make_embedding_client(req)
    try:
        doc_id = (body.doc_id or "").strip()
        if not doc_id:
            import uuid

            base = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", body.title.strip() or "doc")
            doc_id = f"{base}-{uuid.uuid4().hex[:8]}"
        doc_id = doc_id.strip()
        category = (body.category or "").strip()
        tags = ",".join(t.strip() for t in (body.tags or []) if t.strip())

        chunks = _split_chunks(body.content, body.chunk_size, body.overlap)
        if not chunks:
            raise HTTPException(400, "content produced no chunks")

        # 同 doc_id 覆盖写入：先清旧块
        store.delete_by_doc_id(doc_id)

        # 批量向量化：分批调用，避免一次请求太大（大文档可能有上千片，
        # 上游对单次请求的批量与总 token 都有限制）。
        embeddings: list[list[float] | None] = []
        if embedding_client:
            for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
                batch = chunks[i : i + _EMBED_BATCH_SIZE]
                part = embedding_client.embed_batch(batch)
                if not part or len(part) != len(batch):
                    part = [None] * len(batch)
                embeddings.extend(part)
        else:
            embeddings = [None] * len(chunks)
        written = 0
        embedded_count = 0
        meta = json.dumps({"kind": "knowledge_doc", "doc_id": doc_id}, ensure_ascii=False)
        for idx, chunk in enumerate(chunks):
            mid = store.add_memory(
                content=chunk,
                embedding=embeddings[idx] if embeddings and embeddings[idx] else None,
                memory_type="knowledge",
                mem_metadata=meta,
                compute_hrr=True,
                doc_id=doc_id,
                doc_uri=body.uri,
                doc_title=body.title,
                chunk_index=idx,
                doc_category=category,
                doc_tags=tags,
            )
            if mid:
                written += 1
                if embeddings and embeddings[idx]:
                    embedded_count += 1
            else:
                logger.warning("document chunk write failed: %s#%d", doc_id, idx)

        store.add_log(
            action="导入知识库文档",
            status="success",
            count=written,
            detail=f"doc_id={doc_id} chunks={written}",
            namespace=namespace,
        )
        # 知识库文档库关闭自动反思（保真，不被 LLM 蒸馏）
        try:
            create = (
                "CREATE TABLE IF NOT EXISTS _config "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )
            store._conn.execute(create)
            store._conn.execute(
                "INSERT OR IGNORE INTO _config(key, value) VALUES (?, ?)",
                ("reflect_config", '{"auto_reflect": false}'),
            )
            store._conn.commit()
        except Exception as e:
            logger.debug("disable reflect for docs ns failed: %s", e)

        return {
            "doc_id": doc_id,
            "title": body.title,
            "uri": body.uri,
            "category": category,
            "tags": (body.tags or []),
            "namespace": namespace,
            "chunks": written,
            "embedded": embedded_count,
            "vector_dim": store.embedding_dim,
        }
    finally:
        if embedding_client:
            embedding_client.close()
        store.close()


@router.get("/documents")
async def list_documents(
    req: Request,
    namespace: str | None = None,
):
    namespace = namespace or "default"
    store = _DocRouterHelpers.make_store(req, namespace)
    try:
        docs = store.list_documents()
        return {"documents": docs, "count": len(docs), "namespace": namespace}
    finally:
        store.close()


@router.get("/documents/{doc_id}")
async def get_document(
    req: Request,
    doc_id: str,
    namespace: str | None = None,
):
    namespace = namespace or "default"
    store = _DocRouterHelpers.make_store(req, namespace)
    try:
        docs = store.list_documents()
        match = [d for d in docs if d["doc_id"] == doc_id]
        if not match:
            raise HTTPException(404, "document not found")
        # 取该文档全部 chunk
        rows = store.list_memories(limit=10000, offset=0)
        chunks = [
            {
                "chunk_index": m.get("chunk_index", 0),
                "content": m["content"],
                "memory_id": m["id"],
                "created_at": m.get("created_at"),
            }
            for m in rows
            if m.get("doc_id") == doc_id
        ]
        chunks.sort(key=lambda c: c["chunk_index"])
        return {**match[0], "namespace": namespace, "chunks": chunks}
    finally:
        store.close()


@router.delete("/documents/{doc_id}")
async def delete_document(
    req: Request,
    doc_id: str,
    namespace: str | None = None,
):
    namespace = namespace or "default"
    store = _DocRouterHelpers.make_store(req, namespace)
    try:
        removed = store.delete_by_doc_id(doc_id)
        if removed == 0:
            raise HTTPException(404, "document not found")
        store.add_log(
            action="删除知识库文档",
            status="success",
            count=removed,
            detail=f"doc_id={doc_id} chunks={removed}",
            namespace=namespace,
        )
        return {"deleted": True, "doc_id": doc_id, "chunks_removed": removed}
    finally:
        store.close()

