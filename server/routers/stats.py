"""统计路由。"""

from __future__ import annotations

import glob
import os

from fastapi import APIRouter, Request, HTTPException

from engine.store import HybridMemoryStore

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def stats(req: Request, namespace: str | None = None):
    namespace = namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"

    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        total = store.count_memories()
        return {
            "total_memories": total,
            "embedding_enabled": bool(settings.embedding_base_url and settings.embedding_api_key),
            "by_type": store.count_by_type(),
            "namespace": namespace,
        }
    finally:
        store.close()


@router.get("/namespaces")
async def list_namespaces(req: Request):
    """扫描 db_root 下所有 *.db 文件，返回可用 namespace 列表。"""
    settings = req.app.state.settings
    pattern = os.path.join(settings.db_root, "*.db")
    files = sorted(glob.glob(pattern))
    namespaces = []
    for fp in files:
        ns = os.path.splitext(os.path.basename(fp))[0]
        store = HybridMemoryStore(db_path=fp, embedding_dim=settings.embedding_dim)
        store.initialize()
        try:
            total = store.count_memories()
            namespaces.append({"namespace": ns, "total_memories": total})
        finally:
            store.close()
    return {"namespaces": namespaces}