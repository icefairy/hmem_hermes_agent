"""反思引擎路由。"""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, Request, HTTPException

from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient
from engine.retriever import HybridRetriever
from engine.reflect import ReflectEngine, LlmCompleteFn

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reflect"])


def _make_llm_complete(b_url: str, api_key: str, model: str = "deepseek-v4-flash") -> LlmCompleteFn:
    """创建 LLM 聊天完成回调（通过 OneAPI 兼容接口）。"""
    async def complete(messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=60.0) as c:
            resp = await c.post(
                f"{b_url.rstrip('/')}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    return complete


@router.post("/reflect")
async def trigger_reflect(req: Request, namespace: str | None = None):
    """手动触发一次反思循环。"""
    namespace = namespace or "default"
    settings = req.app.state.settings

    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()

    embedding_client = None
    if settings.embedding_base_url and settings.embedding_api_key:
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    retriever = HybridRetriever(
        store=store,
        embedding_client=embedding_client,
        keyword_weight=0.4,
        vector_weight=0.6,
    )

    # 注入 LLM 回调（复用 embedding 的 base_url）
    llm_complete = None
    if settings.embedding_base_url and settings.embedding_api_key:
        llm_complete = _make_llm_complete(
            settings.embedding_base_url,
            settings.embedding_api_key,
            model=settings.reflect_model or "deepseek-v4-flash",
        )

    reflect_engine = ReflectEngine(
        store=store,
        retriever=retriever,
        embedding_client=embedding_client,
        min_experiences=settings.reflect_min_experiences,
        min_observations=settings.reflect_min_observations,
        min_insights=settings.reflect_min_insights,
        reflection_interval=settings.reflect_interval,
        llm_complete=llm_complete,
    )

    try:
        result = await reflect_engine.run_once()
        return {"reflect_count": len(result.get("models", [])), "status": "ok", **result}
    except Exception as e:
        logger.exception("Reflect failed")
        raise HTTPException(500, str(e))
    finally:
        store.close()