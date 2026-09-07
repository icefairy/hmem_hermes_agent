"""自我认知路由（借鉴 dsh-memory P0 cognition / self_reliability）。"""

from __future__ import annotations

import logging

from engine.store import HybridMemoryStore
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cognition"])


def _get_store(req: Request, namespace: str) -> HybridMemoryStore:
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


@router.get("/cognition")
async def get_cognition(req: Request, namespace: str = "default"):
    """返回当前记忆库的自我认知快照：可靠性状态、命中分布、重要性分层、图谱健康度、改进建议。"""
    store = _get_store(req, namespace)
    try:
        report = store.cognition_report()
        report["namespace"] = namespace
        return report
    finally:
        store.close()
