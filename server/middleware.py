"""认证中间件。"""

from __future__ import annotations

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class AuthMiddleware(BaseHTTPMiddleware):
    """检查 Authorization: Bearer <api_key> 头。"""

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # 放行健康检查和静态资源
        if request.url.path in ("/health",) or request.url.path.startswith("/assets/"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] == self._api_key:
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"error": "invalid_api_key", "message": "Invalid or missing API key"},
        )