"""记忆 CRUD 路由。"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient
from engine.retriever import HybridRetriever
from engine.reflect import ReflectEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memories"])


def _get_store_for_namespace(req: Request, namespace: str) -> HybridMemoryStore:
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


def _load_reflect_config(store: HybridMemoryStore, fallback: "Settings") -> dict:
    """从 DB 加载用户设置的 reflect 配置，fallback 到环境变量默认值。"""
    try:
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS _config (key TEXT PRIMARY KEY, value TEXT)"
        )
        row = store._conn.execute(
            "SELECT value FROM _config WHERE key = 'reflect_config'"
        ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return {}


def _is_night_time(night_start: str, night_end: str) -> bool:
    """检查当前时间是否在夜间时段内。"""
    if not night_start or not night_end:
        return False  # 未配置时段 = 不限制
    import datetime
    now = datetime.datetime.now()
    start_parts = night_start.split(":")
    end_parts = night_end.split(":")
    try:
        start_min = int(start_parts[0]) * 60 + int(start_parts[1])
        end_min = int(end_parts[0]) * 60 + int(end_parts[1])
        now_min = now.hour * 60 + now.minute
        if start_min <= end_min:
            return start_min <= now_min <= end_min
        else:  # 跨天，如 22:00 - 08:00
            return now_min >= start_min or now_min <= end_min
    except (ValueError, IndexError):
        return False


class WriteRequest(BaseModel):
    content: str
    namespace: str = "default"
    memory_type: str = "observation"
    mem_action: str = ""
    mem_context: dict = {}
    mem_outcome: dict = {}
    mem_metadata: dict = {}


class UpdateRequest(BaseModel):
    content: str


class Settings:
    """占位 — 实际运行时用 req.app.state.settings"""


@router.post("/memories")
async def write_memory(req: Request, body: WriteRequest):
    settings = req.app.state.settings

    embedding_client = None
    if settings.embedding_base_url and settings.embedding_api_key:
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    embedding = None
    if embedding_client:
        embedding = embedding_client.embed(body.content)

    store = _get_store_for_namespace(req, body.namespace)
    try:
        memory_id = store.add_memory(
            content=body.content,
            embedding=embedding,
            memory_type=body.memory_type,
            mem_action=body.mem_action or None,
            mem_context=json.dumps(body.mem_context) if body.mem_context else None,
            mem_outcome=json.dumps(body.mem_outcome) if body.mem_outcome else None,
            mem_metadata=json.dumps(body.mem_metadata) if body.mem_metadata else None,
        )
        if memory_id is None:
            raise HTTPException(500, "Failed to store memory")

        # 记录写入日志
        store.add_log(action="写入记忆", status="success", detail=f"类型: {body.memory_type}", namespace=body.namespace)

        # ─── 写入后后台自动触发 Reflect ───
        auto_reflected = False
        user_config = _load_reflect_config(store, settings)

        auto_enabled = user_config.get("auto_reflect", True)
        user_interval = int(user_config.get("interval_seconds", settings.reflect_interval))
        user_min_obs = int(user_config.get("min_observations", settings.reflect_min_observations))
        user_min_exp = int(user_config.get("min_experiences", settings.reflect_min_experiences))
        user_min_ins = int(user_config.get("min_insights", settings.reflect_min_insights))
        night_start = user_config.get("night_start", "")
        night_end = user_config.get("night_end", "")

        if auto_enabled:
            if night_start and night_end and not _is_night_time(night_start, night_end):
                logger.info("Skip auto-reflect: outside night window %s-%s", night_start, night_end)
            else:
                try:
                    llm_complete = None
                    if settings.embedding_base_url and settings.embedding_api_key:
                        from routers.reflect import _make_llm_complete
                        llm_complete = _make_llm_complete(
                            settings.embedding_base_url,
                            settings.embedding_api_key,
                            model=settings.reflect_model or "deepseek-v4-flash",
                        )

                    retriever = HybridRetriever(
                        store=store,
                        embedding_client=embedding_client,
                        keyword_weight=0.4,
                        vector_weight=0.6,
                    )

                    reflect_engine = ReflectEngine(
                        store=store,
                        retriever=retriever,
                        embedding_client=embedding_client,
                        min_experiences=user_min_exp,
                        min_observations=user_min_obs,
                        min_insights=user_min_ins,
                        reflection_interval=user_interval,
                        llm_complete=llm_complete,
                    )

                    if reflect_engine.should_reflect():
                        # 后台异步触发反思，不阻塞写入响应
                        async def _run_reflect_in_bg():
                            ns = body.namespace
                            s = store
                            try:
                                result = await reflect_engine.run_once()
                                stage = result.get("stage")
                                if stage:
                                    logger.info("Auto-reflect stage %s triggered after memory write (background)", stage)
                                    counts = {k: v for k, v in result.get("counts", {}).items() if v}
                                    detail = f"阶段: {stage}"
                                    if counts:
                                        detail += ", " + ", ".join(f"{k}={v}" for k, v in counts.items())
                                    s.add_log(action="自动反思", status="success",
                                               count=stage, detail=detail, namespace=ns)
                            except Exception as e:
                                logger.warning("Background auto-reflect failed (non-fatal): %s", e)

                        asyncio.create_task(_run_reflect_in_bg())
                        auto_reflected = True
                except Exception as e:
                    logger.warning("Auto-reflect after write failed (non-fatal): %s", e)

        return {
            "memory_id": memory_id,
            "content": body.content[:100],
            "namespace": body.namespace,
            "embedded": embedding is not None,
            "auto_reflected": auto_reflected,
        }
    finally:
        store.close()


@router.get("/memories")
async def list_memories(
    req: Request,
    namespace: str | None = None,
    limit: int = 50,
    offset: int = 0,
    memory_type: str | None = None,
):
    namespace = namespace or "default"
    store = _get_store_for_namespace(req, namespace)
    try:
        results = store.list_memories(
            limit=min(limit, 200),
            offset=offset,
            memory_type=memory_type,
        )
        return {"results": results, "count": len(results)}
    finally:
        store.close()


@router.get("/memories/{memory_id}")
async def get_memory(
    req: Request,
    memory_id: int,
    namespace: str | None = None,
):
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    store = _get_store_for_namespace(req, namespace)
    try:
        memory = store.get_memory(memory_id)
        if not memory:
            raise HTTPException(404, "Memory not found")
        return memory
    finally:
        store.close()


@router.delete("/memories/{memory_id}")
async def delete_memory(
    req: Request,
    memory_id: int,
    namespace: str | None = None,
):
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    store = _get_store_for_namespace(req, namespace)
    try:
        ok = store.delete_memory(memory_id)
        return {"deleted": ok, "memory_id": memory_id}
    finally:
        store.close()