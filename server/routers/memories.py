"""记忆 CRUD 路由。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient
from engine.retriever import HybridRetriever

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memories"])


def _get_store_for_namespace(req: Request, namespace: str) -> HybridMemoryStore:
    """从 request 的 settings 中拼接 db_path 并创建 store。"""
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


class WriteRequest(BaseModel):
    content: str
    namespace: str = "default"
    mem_action: str = ""
    mem_context: dict = {}
    mem_outcome: dict = {}
    mem_metadata: dict = {}


class UpdateRequest(BaseModel):
    content: str


@router.post("/memories")
async def write_memory(req: Request, body: WriteRequest):
    settings = req.app.state.settings

    embedding_client = None
    if settings.embedding_base_url and settings.embedding_api_key:
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    embedding = None
    if embedding_client:
        embedding = embedding_client.embed(body.content)

    store = _get_store_for_namespace(req, body.namespace)
    try:
        memory_id = store.add_memory(
            content=body.content,
            embedding=embedding,
            mem_action=body.mem_action or None,
            mem_context=json.dumps(body.mem_context) if body.mem_context else None,
            mem_outcome=json.dumps(body.mem_outcome) if body.mem_outcome else None,
            mem_metadata=json.dumps(body.mem_metadata) if body.mem_metadata else None,
        )
        if memory_id is None:
            raise HTTPException(500, "Failed to store memory")
        return {
            "memory_id": memory_id,
            "content": body.content[:100],
            "namespace": body.namespace,
            "embedded": embedding is not None,
        }
    finally:
        store.close()


@router.get("/memories")
async def list_memories(
    req: Request,
    namespace: str | None = None,
    limit: int = 50,
    offset: int = 0,
    memory_type: str | None = None,
):
    namespace = namespace or "default"
    store = _get_store_for_namespace(req, namespace)
    try:
        results = store.list_memories(
            limit=min(limit, 200),
            offset=offset,
            memory_type=memory_type,
        )
        return {"results": results, "count": len(results)}
    finally:
        store.close()


@router.get("/memories/{memory_id}")
async def get_memory(
    req: Request,
    memory_id: int,
    namespace: str | None = None,
):
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    store = _get_store_for_namespace(req, namespace)
    try:
        memory = store.get_memory(memory_id)
        if not memory:
            raise HTTPException(404, "Memory not found")
        return memory
    finally:
        store.close()


@router.delete("/memories/{memory_id}")
async def delete_memory(
    req: Request,
    memory_id: int,
    namespace: str | None = None,
):
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    store = _get_store_for_namespace(req, namespace)
    try:
        ok = store.delete_memory(memory_id)
        return {"deleted": ok, "memory_id": memory_id}
    finally:
        store.close()