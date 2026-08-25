"""关系推理路由 — reason / contradict 组合检索。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from engine.relation import contradict, reason
from engine.store import HybridMemoryStore

router = APIRouter(tags=["relation"])


class ReasonRequest(BaseModel):
    entities: list[str]
    limit: int = 10
    namespace: str | None = None


class ContradictRequest(BaseModel):
    threshold: float = 0.25
    limit: int = 10
    namespace: str | None = None


def _get_store(settings: Any, namespace: str) -> HybridMemoryStore:
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


@router.post("/reason")
async def reason_endpoint(req: Request, body: ReasonRequest):
    """多实体组合记忆检索（AND 语义）。

    body: {"entities": ["docker", "镜像"], "limit": 10}
    返回同时涉及所有实体的记忆，按组合相似度降序。
    """
    ns = body.namespace or "default"
    store = _get_store(req.app.state.settings, ns)
    try:
        results = reason(store, body.entities, limit=min(body.limit, 50))
        return {"count": len(results), "results": results}
    finally:
        store.close()


@router.post("/contradict")
async def contradict_endpoint(req: Request, body: ContradictRequest):
    """找出可能相互矛盾的记忆对。

    body: {"threshold": 0.25, "limit": 10}
    返回相位相似度 <= threshold 的记忆对（低相似 = 可能矛盾/重复主题）。
    """
    ns = body.namespace or "default"
    store = _get_store(req.app.state.settings, ns)
    try:
        results = contradict(store, threshold=body.threshold, limit=min(body.limit, 50))
        return {"count": len(results), "results": results}
    finally:
        store.close()
