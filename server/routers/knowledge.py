"""知识库资源路由 — knowledge 条目的增删查 + 分类统计 + 知识库(库级)管理。

语义上把知识库角色收敛到一组一等公民接口:

  knowledge（单条条目，等价 memories 中 memory_type=knowledge）:
    POST   /api/v1/knowledge                新增单条知识（可归属 doc_id/category/tags）
    GET    /api/v1/knowledge                列表（category / doc_id / tags 过滤 + 分页）
    GET    /api/v1/knowledge/categories     分类汇总（各分类条目数 / 文档数 / 标签）
    GET    /api/v1/knowledge/{memory_id}    单条详情
    DELETE /api/v1/knowledge/{memory_id}    删除单条（仅 knowledge 类型）
    注: 文档级别批量导入 / 级联删除仍走 /api/v1/documents，文档 CRUD 归 归它。

  knowledge-bases（库级，对应独立 namespace + db 文件）:
    POST   /api/v1/knowledge-bases          创建知识库（空 namespace 落库，含 kmount 约定）
    GET    /api/v1/knowledge-bases          列出所有知识库（含条目 / 文档 / 分类统计）
    DELETE /api/v1/knowledge-bases/{ns}     删除整个知识库（删除 db 文件）
"""

from __future__ import annotations

import glob
import json
import logging
import os

from engine.embeddings import EmbeddingClient
from engine.store import HybridMemoryStore
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])


def _get_store(req: Request, namespace: str) -> HybridMemoryStore:
    settings = req.app.state.settings
    store = HybridMemoryStore(
        db_path=f"{settings.db_root}/{namespace}.db",
        embedding_dim=settings.embedding_dim,
    )
    store.initialize()
    return store


def _embedding_client(req: Request) -> EmbeddingClient | None:
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


def _sanitize_ns(name: str) -> str:
    """知识库(库名)白名单：仅字母数字、中划线、下划线、点。防路径穿越。"""
    if not name:
        raise HTTPException(400, "namespace is required")
    if not all(c.isalnum() or c in "-_." for c in name) or name.startswith("."):
        raise HTTPException(400, f"invalid namespace: {name!r}")
    return name


class KnowledgeWrite(BaseModel):
    content: str
    namespace: str = "default"
    doc_id: str | None = None
    title: str = ""
    uri: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    mem_metadata: dict = Field(default_factory=dict)


class KnowledgeBaseCreate(BaseModel):
    namespace: str
    title: str = ""
    description: str = ""


@router.post("/knowledge")
async def create_knowledge(req: Request, body: KnowledgeWrite):
    """新增一条知识条目。doc_id 用于归组到某篇文档(可选)。"""
    if not body.content or not body.content.strip():
        raise HTTPException(400, "content is required")
    ns = _sanitize_ns(body.namespace)
    emb = _embedding_client(req)
    store = _get_store(req, ns)
    try:
        embedding = emb.embed(body.content) if emb else None
        meta = dict(body.mem_metadata or {})
        meta.update({"kind": "knowledge_doc"})
        if body.doc_id:
            meta["doc_id"] = body.doc_id
        tags = ",".join(t.strip() for t in (body.tags or []) if t.strip())
        mid = store.add_memory(
            content=body.content,
            embedding=embedding,
            memory_type="knowledge",
            mem_metadata=json.dumps(meta, ensure_ascii=False),
            compute_hrr=True,
            doc_id=(body.doc_id or "").strip(),
            doc_uri=body.uri,
            doc_title=body.title,
            doc_category=(body.category or "").strip(),
            doc_tags=tags,
        )
        if mid is None:
            raise HTTPException(500, "failed to store knowledge")
        store.add_log(
            action="新增知识", status="success", count=1, namespace=ns,
            detail=f"id={mid} category={body.category!r}",
        )
        return {"memory_id": mid, "namespace": ns}
    finally:
        if emb:
            emb.close()
        store.close()


@router.get("/knowledge")
async def list_knowledge(
    req: Request,
    namespace: str = "default",
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    doc_id: str | None = None,
    tags: str | None = None,
):
    """列出知识库条目，支持分类 / 文档 / 标签过滤。"""
    ns = _sanitize_ns(namespace)
    store = _get_store(req, ns)
    try:
        rows = store.list_memories(
            limit=min(limit, 200),
            offset=offset,
            memory_type="knowledge",
            category=category,
            doc_id=doc_id,
            tags=tags,
        )
        return {
            "namespace": ns,
            "count": len(rows),
            "results": rows,
        }
    finally:
        store.close()


@router.get("/knowledge/categories")
async def knowledge_categories(req: Request, namespace: str = "default"):
    """知识库分类汇总：每个分类的条目数 / 文档数 / 标签集合。"""
    ns = _sanitize_ns(namespace)
    store = _get_store(req, ns)
    try:
        return {
            "namespace": ns,
            "count": store.count_memories(memory_type="knowledge"),
            "categories": store.category_stats(),
        }
    finally:
        store.close()


@router.get("/knowledge/{memory_id}")
async def get_knowledge(req: Request, memory_id: int, namespace: str = "default"):
    ns = _sanitize_ns(namespace)
    store = _get_store(req, ns)
    try:
        m = store.get_memory(memory_id)
        if not m or m.get("memory_type") != "knowledge":
            raise HTTPException(404, "knowledge not found")
        return m
    finally:
        store.close()


@router.delete("/knowledge/{memory_id}")
async def delete_knowledge(req: Request, memory_id: int, namespace: str = "default"):
    ns = _sanitize_ns(namespace)
    store = _get_store(req, ns)
    try:
        m = store.get_memory(memory_id)
        if not m or m.get("memory_type") != "knowledge":
            raise HTTPException(404, "knowledge not found")
        ok = store.delete_memory(memory_id)
        store.add_log(
            action="删除知识", status="success", count=1, namespace=ns,
            detail=f"id={memory_id}",
        )
        return {"deleted": ok, "memory_id": memory_id}
    finally:
        store.close()


# ---- knowledge-bases (库级管理) ----

@router.post("/knowledge-bases")
async def create_knowledge_base(req: Request, body: KnowledgeBaseCreate):
    """创建/激活一个知识库(即独立 namespace)。重复创建幂等。"""
    ns = _sanitize_ns(body.namespace)
    settings = req.app.state.settings
    store = _get_store(req, ns)
    try:
        store.add_log(
            action="创建知识库", status="success", count=0,
            detail=f"title={body.title!r}", namespace=ns,
        )
        return {
            "namespace": ns,
            "title": body.title,
            "description": body.description,
            "db_path": f"{settings.db_root}/{ns}.db",
            "created": True,
        }
    finally:
        store.close()


@router.get("/knowledge-bases")
async def list_knowledge_bases(req: Request):
    """列出 db_root 下所有独立(非记忆 role)知识库。

    约定: 带 `_kb_meta` 日志/标记的库即知识库; 同时会汇总每个库的
    条目数 / 文档数 / 分类数。
    """
    settings = req.app.state.settings
    dbs = sorted(glob.glob(os.path.join(settings.db_root, "*.db")))
    out = []
    for fp in dbs:
        if os.path.basename(fp).startswith((".", "_")):
            continue
        ns = os.path.splitext(os.path.basename(fp))[0]
        store = _get_store(req, ns)
        try:
            entries = store.count_memories(memory_type="knowledge")
            docs = store.count_documents()
            cats = store.category_stats()
            out.append(
                {
                    "namespace": ns,
                    "entries": entries,
                    "documents": docs,
                    "categories": len(cats),
                    "category_list": [c["category"] for c in cats],
                }
            )
        except Exception as e:
            logger.debug("kb ls %s failed: %s", ns, e)
        finally:
            store.close()
    out.sort(key=lambda x: x["namespace"])
    return {"knowledge_bases": out, "count": len(out)}


@router.delete("/knowledge-bases/{namespace}")
async def delete_knowledge_base(req: Request, namespace: str):
    """删除一个知识库(删除其 db 文件)。默认不删物理文件，可带 hard=true 强删。"""
    ns = _sanitize_ns(namespace)
    settings = req.app.state.settings
    db_path = os.path.join(settings.db_root, f"{ns}.db")
    resolved = os.path.realpath(db_path)
    root = os.path.realpath(settings.db_root)
    if not resolved.startswith(root + os.sep) and resolved != root:
        raise HTTPException(400, "invalid namespace path")
    hard = req.query_params.get("hard", "").lower() in ("1", "true", "yes")
    store = _get_store(req, ns)
    try:
        store.add_log(action="删除知识库", status="success", count=0, namespace=ns,
                      detail=f"hard={hard}")
    finally:
        store.close()
    if hard and os.path.exists(resolved):
        try:
            os.remove(resolved)
        except OSError as e:
            logger.warning("delete kb %s file failed: %s", ns, e)
            raise HTTPException(500, f"failed to remove db file: {e}") from e
    return {
        "deleted": True,
        "namespace": ns,
        "hard": hard,
        "db_removed": hard and os.path.exists(resolved),
    }
