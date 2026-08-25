"""HMEM Server — FastAPI 入口"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import Settings
from engine.store import HybridMemoryStore
from middleware import AuthMiddleware
from routers import (
    backup as backup_router,
)
from routers import (
    graph,
    logs,
    memories,
    mental_models,
    reflect,
    search,
    stats,
)
from routers import (
    offload as offload_router,
)
from routers import (
    settings as settings_router,
)
from routers.relation import router as relation_router

logger = logging.getLogger(__name__)


def _load_spa_html() -> str:
    """从本地静态目录加载 SPA HTML。"""
    static_dir = os.path.join(os.path.dirname(__file__), "webui", "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            logger.warning(
                "failed to read index.html at %s", index_path
            )  # Fallback minimal page
    return "<html><body><h1>HMEM</h1><p>SPA index.html not found</p></body></html>"


def get_store(db_path: str, embedding_dim: int = 1024) -> HybridMemoryStore:
    store = HybridMemoryStore(db_path=db_path, embedding_dim=embedding_dim)
    store.initialize()
    return store


_REFLECT_CHECK_INTERVAL = 60  # 调度器每 60s 检查一次各 namespace


async def _reflect_scheduler(app: FastAPI) -> None:
    """后台定时器：定期对满足门槛的 namespace 触发一轮 reflect（不依赖写入路径）。

    与写入后触发的 auto-reflect 互补：
      - 写入后触发是即时路径（写入多时能跟上）；
      - 这里是保底路径（写入少/间隔长时也能积累产出），
        让 observation→experience→insight 图谱边持续生成。
    """
    while True:
        try:
            await _reflect_once_for_all(app)
        except Exception as e:
            logger.warning("reflect scheduler round failed: %s", e)
        await asyncio.sleep(_REFLECT_CHECK_INTERVAL)


async def _reflect_once_for_all(app: FastAPI) -> None:
    """遍历 db_root 下所有 namespace 库，对每个可跑的跑一轮 reflect。"""
    import glob as _glob

    from engine.embeddings import EmbeddingClient
    from engine.reflect import ReflectEngine
    from engine.retriever import HybridRetriever
    from routers.memories import _load_reflect_config
    from routers.reflect import _make_llm_complete

    settings = app.state.settings
    dbs = _glob.glob(os.path.join(settings.db_root, "*.db"))
    # 跳过 sqlite 临时/副作用文件
    dbs = [d for d in dbs if not os.path.basename(d).startswith(('.', '_'))]
    if not dbs:
        return

    embedding_client = None
    if settings.embedding_base_url and settings.embedding_api_key:
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    for db_path in dbs:
        ns = os.path.splitext(os.path.basename(db_path))[0]
        try:
            store = get_store(db_path, settings.embedding_dim)
        except Exception as e:
            logger.warning("reflect scheduler: open %s failed: %s", db_path, e)
            continue
        try:
            cfg = _load_reflect_config(store, settings)
            if not cfg.get("auto_reflect", True):
                continue
            llm_complete = None
            if embedding_client is not None:
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
            engine = ReflectEngine(
                store=store,
                retriever=retriever,
                embedding_client=embedding_client,
                min_experiences=int(
                    cfg.get("min_experiences", settings.reflect_min_experiences)
                ),
                min_observations=int(
                    cfg.get("min_observations", settings.reflect_min_observations)
                ),
                min_insights=int(
                    cfg.get("min_insights", settings.reflect_min_insights)
                ),
                reflection_interval=int(
                    cfg.get("interval_seconds", settings.reflect_interval)
                ),
                llm_complete=llm_complete,
            )
            if not engine.should_reflect():
                continue
            result = await engine.run_once()
            stage = result.get("stage")
            if stage:
                counts = {k: v for k, v in result.get("counts", {}).items() if v}
                detail = f"阶段: {stage}"
                if counts:
                    detail += ", " + ", ".join(f"{k}={v}" for k, v in counts.items())
                store.add_log(
                    action="定时反思",
                    status="success",
                    count=stage,
                    detail=detail,
                    namespace=ns,
                )
                logger.info("reflect scheduler: ns=%s %s", ns, detail)
        except Exception as e:
            logger.warning("reflect scheduler: ns=%s failed: %s", ns, e)
        finally:
            with suppress(Exception):
                store.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    # 启动后台备份调度器
    app.state._backup_task = asyncio.create_task(
        backup_router._backup_scheduler(app),
        name="backup-scheduler",
    )
    # 启动后台 reflect 调度器（独立保底路径，不依赖写入触发）
    app.state._reflect_task = asyncio.create_task(
        _reflect_scheduler(app),
        name="reflect-scheduler",
    )
    logger.info(
        "HMEM Server started: db_root=%s embed=%s",
        settings.db_root,
        bool(settings.embedding_base_url),
    )
    yield
    # 清理
    app.state._backup_task.cancel()
    app.state._reflect_task.cancel()


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
    app.include_router(backup_router.router, prefix="/api/v1")
    if settings.offload_enabled:
        app.include_router(offload_router.router, prefix="/api/v1")
    app.include_router(relation_router, prefix="/api/v1")

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
