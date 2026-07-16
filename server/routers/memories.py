"""记忆 CRUD 路由。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memories"])


class WriteRequest(BaseModel):
    content: str
    agent_space: str = "default"
    mem_action: str = ""
    mem_context: dict = {}
    mem_outcome: dict = {}
    mem_metadata: dict = {}


class UpdateRequest(BaseModel):
    content: str


@router.post("/memories")
async def write_memory(req: Request, body: WriteRequest):
    store = req.app.state.store
    embedding_client = req.app.state.embedding_client

    embedding = None
    if embedding_client:
        embedding = embedding_client.embed(body.content)

    memory_id = store.add_memory(
        content=body.content,
        agent_space=body.agent_space,
        embedding=embedding,
        mem_action=body.mem_action or None,
        mem_context=json.dumps(body.mem_context) if body.mem_context else None,
        mem_outcome=json.dumps(body.mem_outcome) if body.mem_outcome else None,
        mem_metadata=json.dumps(body.mem_metadata) if body.mem_metadata else None,
    )
    if memory_id is None:
        raise HTTPException(500, "Failed to store memory")
    return {
        "memory_id": memory_id,
        "content": body.content[:100],
        "agent_space": body.agent_space,
        "embedded": embedding is not None,
    }


@router.get("/memories")
async def list_memories(
    req: Request,
    agent_space: str | None = None,
    limit: int = 50,
    offset: int = 0,
    memory_type: str | None = None,
):
    store = req.app.state.store
    results = store.list_memories(
        agent_space=agent_space,
        limit=min(limit, 200),
        offset=offset,
        memory_type=memory_type,
    )
    return {"results": results, "count": len(results)}


@router.get("/memories/{memory_id}")
async def get_memory(req: Request, memory_id: int):
    store = req.app.state.store
    memory = store.get_memory(memory_id)
    if not memory:
        raise HTTPException(404, "Memory not found")
    return memory


@router.delete("/memories/{memory_id}")
async def delete_memory(req: Request, memory_id: int):
    store = req.app.state.store
    ok = store.delete_memory(memory_id)
    return {"deleted": ok, "memory_id": memory_id}