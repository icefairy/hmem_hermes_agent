"""记忆关系图谱路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["graph"])


@router.get("/graph")
async def get_graph(
    req: Request,
    agent_space: str | None = None,
    limit: int = 200,
):
    """返回记忆关系图数据（nodes + edges），用于前端力导向图渲染。"""
    store = req.app.state.store
    return store.get_graph(agent_space=agent_space, limit=min(limit, 500))