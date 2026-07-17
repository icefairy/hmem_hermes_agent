"""设置路由 — 持久化 namespace 级别的 Reflect 配置。"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])


class ReflectSettings(BaseModel):
    auto_reflect: bool = True
    interval_seconds: int = 60
    min_observations: int = 3
    min_experiences: int = 5
    min_insights: int = 2
    night_start: str = ""  # e.g. "22:00"
    night_end: str = ""    # e.g. "08:00"


def _get_store(namespace: str, settings: any):
    """根据 namespace 创建 store。"""
    from engine.store import HybridMemoryStore
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


def _set_reflect_config(store, config: dict):
    """将 Reflect 配置序列化存入 store 的 KV 表。"""
    store._conn.execute(
        "CREATE TABLE IF NOT EXISTS _config (key TEXT PRIMARY KEY, value TEXT)"
    )
    store._conn.execute(
        "INSERT OR REPLACE INTO _config (key, value) VALUES ('reflect_config', ?)",
        (json.dumps(config, ensure_ascii=False),),
    )
    store._conn.commit()


def _get_reflect_config(store) -> dict:
    """从 store 的 KV 表加载 Reflect 配置。"""
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


@router.get("/settings")
async def get_settings(req: Request, namespace: str | None = None):
    ns = namespace or "default"
    settings = req.app.state.settings
    store = _get_store(ns, settings)
    try:
        saved = _get_reflect_config(store)
        return ReflectSettings(
            auto_reflect=saved.get("auto_reflect", True),
            interval_seconds=saved.get("interval_seconds", settings.reflect_interval),
            min_observations=saved.get("min_observations", settings.reflect_min_observations),
            min_experiences=saved.get("min_experiences", settings.reflect_min_experiences),
            min_insights=saved.get("min_insights", settings.reflect_min_insights),
            night_start=saved.get("night_start", ""),
            night_end=saved.get("night_end", ""),
        ).model_dump()
    finally:
        store.close()


@router.put("/settings")
async def update_settings(req: Request, body: ReflectSettings, namespace: str | None = None):
    ns = namespace or "default"
    settings = req.app.state.settings
    store = _get_store(ns, settings)
    try:
        _set_reflect_config(store, body.model_dump())
        return {"status": "ok", "namespace": ns}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        store.close()