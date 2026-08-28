"""FastAPI application factory and HTTP-only runtime behavior."""

from __future__ import annotations

import re
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    QueryServiceProvider,
    ReadinessChecker,
    Settings,
    build_readiness_checker,
    get_query_service,
    get_settings,
)
from app.api.health import create_health_router
from app.api.query import router as query_router
from app.request_context import request_id

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class UTF8JSONResponse(JSONResponse):
    """Declare UTF-8 explicitly for legacy clients such as Windows PowerShell 5.1."""

    media_type = "application/json; charset=utf-8"


def create_app(
    settings: Settings | None = None,
    *,
    query_service_provider: QueryServiceProvider | None = None,
    readiness_checker: ReadinessChecker | None = None,
) -> FastAPI:
    """Create one FastAPI adapter over injected query and readiness dependencies."""

    effective_settings = settings if settings is not None else get_settings()
    effective_readiness = (
        readiness_checker
        if readiness_checker is not None
        else build_readiness_checker(effective_settings)
    )
    application = FastAPI(
        title=effective_settings.app_name,
        version=effective_settings.app_version,
        default_response_class=UTF8JSONResponse,
    )
    application.include_router(query_router, prefix=effective_settings.api_prefix)
    application.include_router(
        create_health_router(
            effective_settings,
            readiness_checker=effective_readiness,
        ),
        prefix=effective_settings.api_prefix,
    )
    if settings is not None:
        application.dependency_overrides[get_settings] = lambda: effective_settings
    if query_service_provider is not None:
        application.dependency_overrides[get_query_service] = query_service_provider

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Attach a safe correlation ID without logging request bodies or credentials."""

        requested_id = request.headers.get("X-Request-ID", "")
        correlation_id = (
            requested_id if _REQUEST_ID_PATTERN.fullmatch(requested_id) else uuid4().hex
        )
        token = request_id.set(correlation_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = correlation_id
            return response
        finally:
            request_id.reset(token)

    return application
