"""FastAPI application entry point."""

import re
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.query import router as query_router
from app.config import get_settings
from app.models import HealthResponse, ReadinessResponse
from app.request_context import request_id
from app.retrieval import RetrievalError, create_qdrant_client
from app.retrieval_runtime import resolve_retrieval_runtime, validate_frozen_runtime

settings = get_settings()


class UTF8JSONResponse(JSONResponse):
    """Declare UTF-8 explicitly for legacy clients such as Windows PowerShell 5.1."""

    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    default_response_class=UTF8JSONResponse,
)
app.include_router(query_router, prefix=settings.api_prefix)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a safe correlation ID without logging request bodies or credentials."""

    requested_id = request.headers.get("X-Request-ID", "")
    correlation_id = requested_id if _REQUEST_ID_PATTERN.fullmatch(requested_id) else uuid4().hex
    token = request_id.set(correlation_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response
    finally:
        request_id.reset(token)


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the service health status."""

    return HealthResponse(
        status="ok",
        service="industrial-rag",
        version=settings.app_version,
    )


@app.get(f"{settings.api_prefix}/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    """Check frozen Qdrant collection identity without loading any model or provider."""

    try:
        runtime_settings, contract = resolve_retrieval_runtime(settings)
        client = create_qdrant_client(runtime_settings)
        validate_frozen_runtime(
            client,
            collection_names=(
                runtime_settings.qdrant_collection,
                runtime_settings.qdrant_hybrid_collection,
            ),
            contract=contract,
        )
    except RetrievalError:
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "retrieval_not_ready",
                    "message": "Retrieval is unavailable.",
                }
            },
        )
    return ReadinessResponse(status="ok", service="industrial-rag", version=settings.app_version)
