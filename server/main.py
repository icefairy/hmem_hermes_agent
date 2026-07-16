"""
HMEM Server — FastAPI 入口

提供 REST API（记忆读写/检索/反思/图谱）+ Vue3 SPA 静态文件。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .config import Settings
from .middleware import AuthMiddleware
from .routers import memories, search, stats, graph, reflect, mental_models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：初始化/关闭引擎。"""
    settings = Settings()
    app.state.settings = settings

    # 初始化引擎
    from .engine.store import HybridMemoryStore
    from .engine.embeddings import EmbeddingClient
    from .engine.retriever import HybridRetriever
    from .engine.reflect import ReflectEngine

    store = HybridMemoryStore(
        db_path=settings.db_path,
        embedding_dim=settings.embedding_dim,
    )
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

    reflect_engine = ReflectEngine(
        store=store,
        retriever=retriever,
        embedding_client=embedding_client,
        min_experiences=settings.reflect_min_experiences,
        reflection_interval=settings.reflect_interval,
    )

    app.state.store = store
    app.state.retriever = retriever
    app.state.embedding_client = embedding_client
    app.state.reflect_engine = reflect_engine

    logger.info("HMEM Server started: db=%s embed=%s", settings.db_path, embedding_client is not None)

    yield

    # 关闭
    store.close()
    if embedding_client:
        embedding_client.close()


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title="HMEM · 混合记忆系统",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    # 认证中间件
    if settings.api_key:
        app.add_middleware(AuthMiddleware, api_key=settings.api_key)

    # 注册路由
    app.include_router(memories.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(stats.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(reflect.router, prefix="/api/v1")
    app.include_router(mental_models.router, prefix="/api/v1")

    # 探活
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    # 挂载 WebUI 静态文件
    webui_dir = os.path.join(os.path.dirname(__file__), "webui", "dist")
    if os.path.isdir(webui_dir):
        app.mount("/", StaticFiles(directory=webui_dir, html=True), name="webui")

    return app


app = create_app()