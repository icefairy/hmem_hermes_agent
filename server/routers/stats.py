"""统计路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def stats(req: Request):
    store = req.app.state.store
    embedding_client = req.app.state.embedding_client

    total = store.count_memories()
    return {
        "total_memories": total,
        "embedding_enabled": embedding_client is not None,
        "by_type": store.count_by_type(),
        "by_space": store.count_by_space(),
    }