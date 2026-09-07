# pyright: reportMissingImports=false
"""记忆生命周期路由：重要性评分 + 主动遗忘（借鉴 dsh-memory）。

提供：
  - forget_advisor：主动遗忘建议（长期未命中 + 低重要性 + 非 pinned 记忆）
  - archive / unarchive：可逆归档（主动遗忘）
  - set importance / set pinned：重要性管理（只升不降）与永久保留

v5 新增。
"""

from __future__ import annotations

import logging

from engine.store import HybridMemoryStore
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["lifecycle"])


def _get_store_for_namespace(req: Request, namespace: str) -> HybridMemoryStore:
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


class ForgetAdvisorRequest(BaseModel):
    namespace: str = "default"
    threshold_days: int = Field(default=60, ge=1, description="闲置阈值（天）")
    min_importance: float = Field(default=0.4, ge=0.0, le=1.0, description="重要性下限")
    max_candidates: int = Field(default=50, ge=1, le=200)


class ImportanceRequest(BaseModel):
    namespace: str = "default"
    importance: float = Field(ge=0.0, le=1.0)


class PinRequest(BaseModel):
    namespace: str = "default"
    pinned: bool = True


class ArchiveRequest(BaseModel):
    namespace: str = "default"


@router.post("/lifecycle/forget-advisor")
async def forget_advisor(req: Request, body: ForgetAdvisorRequest):
    """主动遗忘顾问：返回"闲置且低重要性且未标记保留"的记忆候选，建议归档。

    只读统计，不实际执行归档——上层（Agent / 夜间任务）决定要不要 archive。
    借鉴 dsh-memory 的 forget_advisor（未使用记忆可逆归档）。
    """
    store = _get_store_for_namespace(req, body.namespace)
    try:
        result = store.forget_advisor(
            threshold_days=body.threshold_days,
            min_importance=body.min_importance,
            max_candidates=body.max_candidates,
        )
        result["namespace"] = body.namespace
        return result
    finally:
        store.close()


@router.get("/lifecycle/archived")
async def list_archived(
    req: Request,
    namespace: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """列出已归档（主动遗忘）的记忆。"""
    namespace = namespace or "default"
    store = _get_store_for_namespace(req, namespace)
    try:
        results = store.list_archived(limit=min(limit, 200), offset=offset)
        return {"results": results, "count": len(results), "namespace": namespace}
    finally:
        store.close()


@router.post("/memories/{memory_id}/archive")
async def archive_memory(req: Request, memory_id: int, body: ArchiveRequest):
    """归档一条记忆（可逆主动遗忘）。pinned 记忆不可归档。"""
    store = _get_store_for_namespace(req, body.namespace)
    try:
        ok = store.archive_memory(memory_id)
        if not ok:
            # 可能不存在，或已被 pinned
            m = store.get_memory(memory_id)
            if not m:
                raise HTTPException(404, "Memory not found")
            raise HTTPException(400, "Memory is pinned and cannot be archived")
        store.add_log(
            action="归档记忆", status="success", detail=f"id: {memory_id}", namespace=body.namespace
        )
        return {"archived": True, "memory_id": memory_id, "namespace": body.namespace}
    finally:
        store.close()


@router.post("/memories/{memory_id}/unarchive")
async def unarchive_memory(req: Request, memory_id: int, body: ArchiveRequest):
    """从归档恢复一条记忆。"""
    store = _get_store_for_namespace(req, body.namespace)
    try:
        ok = store.unarchive_memory(memory_id)
        if not ok:
            m = store.get_memory(memory_id)
            if not m:
                raise HTTPException(404, "Memory not found")
        store.add_log(
            action="恢复记忆", status="success", detail=f"id: {memory_id}", namespace=body.namespace
        )
        return {"archived": False, "memory_id": memory_id, "namespace": body.namespace}
    finally:
        store.close()


@router.post("/memories/{memory_id}/importance")
async def set_importance(req: Request, memory_id: int, body: ImportanceRequest):
    """设置记忆重要性（0~1）。只升不降：新值低于当前值时忽略。"""
    store = _get_store_for_namespace(req, body.namespace)
    try:
        ok = store.set_importance(memory_id, body.importance)
        if not ok:
            m = store.get_memory(memory_id)
            if not m:
                raise HTTPException(404, "Memory not found")
        # 返回实际存储值（可能未被更新，因为是单调降权被拒绝）
        m = store.get_memory(memory_id)
        return {
            "memory_id": memory_id,
            "importance": m["importance"] if m else body.importance,
            "namespace": body.namespace,
            "note": "importance is monotonic non-decreasing",
        }
    finally:
        store.close()


@router.post("/memories/{memory_id}/pinned")
async def set_pinned(req: Request, memory_id: int, body: PinRequest):
    """标记 / 取消标记为永久保留（SELF 层 / 协议锚点，不可遗忘）。"""
    store = _get_store_for_namespace(req, body.namespace)
    try:
        ok = store.set_pinned(memory_id, body.pinned)
        if not ok:
            m = store.get_memory(memory_id)
            if not m:
                raise HTTPException(404, "Memory not found")
        store.add_log(
            action="设置永久保留" if body.pinned else "取消永久保留",
            status="success",
            detail=f"id: {memory_id}",
            namespace=body.namespace,
        )
        return {
            "memory_id": memory_id,
            "pinned": body.pinned,
            "namespace": body.namespace,
        }
    finally:
        store.close()
