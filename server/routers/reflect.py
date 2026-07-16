"""反思引擎路由。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reflect"])


@router.post("/reflect")
async def trigger_reflect(req: Request):
    """手动触发一次反思循环。"""
    reflect_engine = req.app.state.reflect_engine
    if not reflect_engine:
        raise HTTPException(400, "Reflect engine not configured (need LLM client)")

    try:
        result = await reflect_engine.run_once()
        return {"reflect_count": len(result.get("models", [])), "status": "ok", **result}
    except Exception as e:
        logger.exception("Reflect failed")
        raise HTTPException(500, str(e))