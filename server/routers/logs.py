"""操作日志路由。"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Request, HTTPException

from config import Settings
from engine.store import HybridMemoryStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_store_for_namespace(req: Request, namespace: str) -> HybridMemoryStore:
    settings: Settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


@router.get("/logs")
async def list_logs(
    req: Request,
    namespace: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    namespace = namespace or "default"
    store = _get_store_for_namespace(req, namespace)
    try:
        logs = store.list_logs(limit=limit, offset=offset, namespace=namespace)
        return {"logs": logs, "count": len(logs)}
    finally:
        store.close()