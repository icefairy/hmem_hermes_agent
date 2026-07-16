"""记忆关系图谱路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..engine.store import HybridMemoryStore

router = APIRouter(tags=["graph"])


@router.get("/graph")
async def get_graph(
    req: Request,
    namespace: str | None = None,
    limit: int = 200,
):
    """返回记忆关系图数据（nodes + edges），用于前端力导向图渲染。"""
    namespace = namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"

    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        graph_data = store.get_graph(namespace=None, limit=min(limit, 500))
        # 去掉 nodes 中的 agent_space 字段（分库后无意义）
        for node in graph_data.get("nodes", []):
            node.pop("agent_space", None)
            node.pop("namespace", None)
        return graph_data
    finally:
        store.close()