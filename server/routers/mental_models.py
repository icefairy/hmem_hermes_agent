"""心智模型路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["mental-models"])


@router.get("/mental-models")
async def list_mental_models(req: Request, namespace: str | None = None, limit: int = 50):
    """列出所有心智模型（reflection 产物）。"""
    store = req.app.state.store
    results = store.list_memories(
        namespace=namespace,
        memory_type="mental_model",
        limit=min(limit, 200),
    )
    return {"results": results, "count": len(results)}


@router.get("/mental-models/{model_id}")
async def get_mental_model(req: Request, model_id: int):
    """获取单个心智模型详情，包含支撑证据（子记忆）。"""
    store = req.app.state.store
    model = store.get_memory(model_id)
    if not model:
        return {"error": "not_found"}
    # 获取关联的子记忆
    children = store.get_child_memories(model_id)
    model["supporting_experiences"] = children
    return model