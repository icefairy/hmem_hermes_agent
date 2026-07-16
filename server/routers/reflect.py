"""反思引擎路由。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException

from ..engine.store import HybridMemoryStore
from ..engine.embeddings import EmbeddingClient
from ..engine.retriever import HybridRetriever
from ..engine.reflect import ReflectEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reflect"])


@router.post("/reflect")
async def trigger_reflect(req: Request, namespace: str | None = None):
    """手动触发一次反思循环。"""
    namespace = namespace or "default"
    settings = req.app.state.settings

    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()

    embedding_client = None
    if settings.embedding_base_url and settings.embedding_api_key:
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    retriever = HybridRetriever(
        store=store,
        embedding_client=embedding_client,
        keyword_weight=0.4,
        vector_weight=0.6,
    )

    reflect_engine = ReflectEngine(
        store=store,
        retriever=retriever,
        embedding_client=embedding_client,
        min_experiences=settings.reflect_min_experiences,
        reflection_interval=settings.reflect_interval,
    )

    if not reflect_engine:
        raise HTTPException(400, "Reflect engine not configured (need LLM client)")

    try:
        result = await reflect_engine.run_once()
        return {"reflect_count": len(result.get("models", [])), "status": "ok", **result}
    except Exception as e:
        logger.exception("Reflect failed")
        raise HTTPException(500, str(e))
    finally:
        store.close()