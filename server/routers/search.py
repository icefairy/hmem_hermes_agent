"""混合检索路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    agent_space: str | None = None
    use_rerank: bool = True


@router.post("/search")
async def search(req: Request, body: SearchRequest):
    retriever = req.app.state.retriever

    results = retriever.search(
        query=body.query,
        agent_space=body.agent_space,
        limit=min(body.limit, 50),
        use_rerank=body.use_rerank,
    )
    return {"results": results, "count": len(results)}