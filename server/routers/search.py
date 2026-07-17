"""混合检索路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient
from engine.retriever import HybridRetriever

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    namespace: str | None = None
    use_rerank: bool = True


@router.post("/search")
async def search(req: Request, body: SearchRequest):
    namespace = body.namespace or "default"
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

    try:
        results = retriever.search(
            query=body.query,
            limit=min(body.limit, 50),
            use_rerank=body.use_rerank,
        )
        # 去掉 namespace 字段（分库后无意义）
        for r in results:
            r.pop("namespace", None)
        return {"results": results, "count": len(results)}
    finally:
        store.close()