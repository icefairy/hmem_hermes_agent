"""认证中间件。"""

from __future__ import annotations

from typing import Set

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class AuthMiddleware(BaseHTTPMiddleware):
    """检查 Authorization: Bearer *** 头。支持多个有效 key。"""

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        # 同时支持 change-me（WebUI 默认）和配置的真实 key
        self._valid_keys: Set[str] = {"change-me"}
        if api_key and api_key != "change-me":
            self._valid_keys.add(api_key)

    async def dispatch(self, request: Request, call_next):
        # 放行健康检查、静态资源、SPA 首页
        if request.url.path in ("/health", "/") or request.url.path.startswith("/assets/") or request.url.path.startswith("/static/"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] in self._valid_keys:
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"error": "invalid_api_key", "message": "Invalid or missing API key"},
        )