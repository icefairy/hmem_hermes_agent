"""心智模型路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from engine.store import HybridMemoryStore

router = APIRouter(tags=["mental-models"])


@router.get("/mental-models")
async def list_mental_models(req: Request, namespace: str | None = None, limit: int = 50):
    """列出所有高级记忆（insight + mental_model）。"""
    namespace = namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"

    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        # 合并 insight 和 mental_model 两种类型
        results_a = store.list_memories(
            memory_type="insight",
            limit=min(limit, 200),
        )
        results_b = store.list_memories(
            memory_type="mental_model",
            limit=min(limit, 200),
        )
        results = results_a + results_b
        results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return {"results": results[:limit], "count": len(results[:limit])}
    finally:
        store.close()


@router.get("/mental-models/{model_id}")
async def get_mental_model(
    req: Request,
    model_id: int,
    namespace: str | None = None,
):
    """获取单个心智模型详情，包含支撑证据（子记忆）。"""
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"

    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        model = store.get_memory(model_id)
        if not model:
            return {"error": "not_found"}
        # 获取关联的子记忆
        children = store.get_child_memories(model_id)
        model["supporting_experiences"] = children
        return model
    finally:
        store.close()