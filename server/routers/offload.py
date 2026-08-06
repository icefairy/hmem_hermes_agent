"""上下文卸载路由 — 会话/任务级 Context Offloading API。

对应 TencentDB Agent Memory 的三层卸载，落地为:
  1. POST /offload/session         创建/获取会话卸载空间
  2. POST /offload/put             卸载一条内容（原文入 refs 文件，摘要入 SQLite）
  3. GET  /offload/get             按 node_id 找回完整原文
  4. GET  /offload/session/{key}   会话索引（全部摘要 + node_id，供注入上下文）
  5. GET  /offload/canvas/{key}    生成 Mermaid 任务画布
  6. DELETE /offload/session/{key} 清理会话（默认软删除，保留增量数据）

auth 复用全局 AuthMiddleware；namespace 兼容现有分库机制。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from engine.offload import OffloadStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["offload"])


def _get_offload_store(req: Request, namespace: str) -> OffloadStore:
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = OffloadStore(db_path=db_path, data_root=settings.db_root, namespace=namespace)
    store.initialize()
    return store


class SessionRequest(BaseModel):
    session_key: str
    namespace: str = "default"


class PutRequest(BaseModel):
    session_key: str
    content: str
    node_id: str | None = None
    summary: str | None = None
    content_type: str = "text"
    meta: dict = {}
    namespace: str = "default"


@router.post("/offload/session")
async def create_session(req: Request, body: SessionRequest):
    """创建/获取会话卸载空间（幂等）。"""
    if not body.session_key or not body.session_key.strip():
        raise HTTPException(400, "session_key is required")
    store = _get_offload_store(req, body.namespace)
    try:
        return store.ensure_session(body.session_key)
    finally:
        store.close()


@router.post("/offload/put")
async def offload_put(req: Request, body: PutRequest):
    """卸载一条内容：原文存 refs 文件，摘要存 SQLite 索引。

    同 session_key + node_id 重复 put = 覆盖更新。
    summary 不传时自动生成一行摘要（去换行限长）。
    """
    if not body.session_key or not body.session_key.strip():
        raise HTTPException(400, "session_key is required")
    if not body.content or not body.content.strip():
        raise HTTPException(400, "content is required")
    store = _get_offload_store(req, body.namespace)
    try:
        result = store.put(
            session_key=body.session_key,
            content=body.content,
            node_id=body.node_id,
            summary=body.summary,
            content_type=body.content_type,
            meta=body.meta,
        )
        if result is None:
            raise HTTPException(500, "Failed to offload content")
        return result
    finally:
        store.close()


@router.get("/offload/get")
async def offload_get(
    req: Request,
    session_key: str,
    node_id: str,
    namespace: str = "default",
):
    """按 node_id 找回完整原文（100% 可找回链路）。"""
    if not session_key or not node_id:
        raise HTTPException(400, "session_key and node_id are required")
    store = _get_offload_store(req, namespace)
    try:
        record = store.get(session_key, node_id)
        if record is None:
            raise HTTPException(404, "Offload record not found")
        return record
    finally:
        store.close()


@router.get("/offload/session/{session_key}")
async def offload_session_index(
    req: Request,
    session_key: str,
    namespace: str = "default",
    include_deleted: bool = False,
):
    """返回会话索引：全部摘要 + node_id 列表（不含原文，适合注入上下文）。"""
    store = _get_offload_store(req, namespace)
    try:
        return store.session_index(session_key, include_deleted=include_deleted)
    finally:
        store.close()


@router.get("/offload/canvas/{session_key}")
async def offload_canvas(
    req: Request,
    session_key: str,
    namespace: str = "default",
):
    """生成 Mermaid 任务画布文本（节点=动作带 node_id，边=依赖/时间序）。"""
    store = _get_offload_store(req, namespace)
    try:
        return {"session_key": session_key, "namespace": namespace, "mermaid": store.canvas_mermaid(session_key)}
    finally:
        store.close()


@router.delete("/offload/session/{session_key}")
async def offload_delete_session(
    req: Request,
    session_key: str,
    namespace: str = "default",
    hard: bool = False,
):
    """清理会话。默认软删除（deleted_at + refs 移入 .trash，保留增量数据）；
    hard=true 物理删除。"""
    store = _get_offload_store(req, namespace)
    try:
        return store.delete_session(session_key, hard=hard)
    finally:
        store.close()
