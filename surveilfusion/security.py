import hmac

from fastapi import WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

API_KEY_HEADER = "x-surveilfusion-key"
PUBLIC_HTTP_PATHS = {
    "/",
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.api_key or is_public_path(request.url.path):
            return await call_next(request)
        if is_authorized(request.headers, self.api_key):
            return await call_next(request)
        return JSONResponse(
            {"detail": "Missing or invalid SurveilFusion API key."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


def is_public_path(path: str) -> bool:
    return path in PUBLIC_HTTP_PATHS or path.startswith("/static/")


def is_authorized(headers, api_key: str) -> bool:
    provided = headers.get(API_KEY_HEADER) or _bearer_token(headers.get("authorization", ""))
    return bool(provided) and hmac.compare_digest(provided, api_key)


async def require_websocket_key(websocket: WebSocket, api_key: str | None) -> bool:
    if not api_key:
        return True
    if is_authorized(websocket.headers, api_key):
        return True
    await websocket.close(code=1008, reason="Missing or invalid SurveilFusion API key.")
    return False


def _bearer_token(value: str) -> str | None:
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()
