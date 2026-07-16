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
from .engine.store import HybridMemoryStore

logger = logging.getLogger(__name__)


def get_store(db_path: str, embedding_dim: int = 1024) -> HybridMemoryStore:
    """根据 db_path 创建并初始化一个新的 HybridMemoryStore 实例。"""
    store = HybridMemoryStore(db_path=db_path, embedding_dim=embedding_dim)
    store.initialize()
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：加载配置，启动时无需预初始化 store（按 namespace 动态创建）。"""
    settings = Settings()
    app.state.settings = settings
    logger.info("HMEM Server started: db_root=%s embed=%s", settings.db_root, bool(settings.embedding_base_url and settings.embedding_api_key))
    yield


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