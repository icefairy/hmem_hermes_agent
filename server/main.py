"""HMEM Server — FastAPI 入口"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from config import Settings
from middleware import AuthMiddleware
from routers import memories, search, stats, graph, reflect, mental_models, settings as settings_router, logs
from engine.store import HybridMemoryStore

logger = logging.getLogger(__name__)


def _load_spa_html() -> str:
    """从本地静态目录加载 SPA HTML。"""
    static_dir = os.path.join(os.path.dirname(__file__), "webui", "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    # Fallback minimal page
    return "<html><body><h1>HMEM</h1><p>SPA index.html not found</p></body></html>"


def get_store(db_path: str, embedding_dim: int = 1024) -> HybridMemoryStore:
    store = HybridMemoryStore(db_path=db_path, embedding_dim=embedding_dim)
    store.initialize()
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    logger.info("HMEM Server started: db_root=%s embed=%s", settings.db_root, bool(settings.embedding_base_url))
    yield


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title="HMEM · 混合记忆系统",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    if settings.api_key:
        app.add_middleware(AuthMiddleware, api_key=settings.api_key)

    app.include_router(memories.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(stats.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(reflect.router, prefix="/api/v1")
    app.include_router(mental_models.router, prefix="/api/v1")
    app.include_router(settings_router.router, prefix="/api/v1")
    app.include_router(logs.router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.3.0"}

    webui_dist = os.path.join(os.path.dirname(__file__), "webui", "dist")
    webui_static = os.path.join(os.path.dirname(__file__), "webui", "static")
    if os.path.isdir(webui_dist):
        app.mount("/", StaticFiles(directory=webui_dist, html=True), name="webui")
    else:
        if os.path.isdir(webui_static):
            app.mount("/static", StaticFiles(directory=webui_static), name="static")

        @app.get("/")
        async def root():
            return HTMLResponse(content=_load_spa_html())

    return app


app = create_app()