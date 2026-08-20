"""Optional bearer-token protection for the query endpoint."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings


def require_query_auth(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require a configured bearer token only when API auth is explicitly enabled."""

    if not settings.api_auth_enabled:
        return
    configured_key = settings.api_auth_key
    if configured_key is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "auth_not_configured",
                "message": "Query authentication is unavailable.",
            },
        )
    scheme, _, supplied_key = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        supplied_key, configured_key.get_secret_value()
    ):
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "Query authentication failed."},
            headers={"WWW-Authenticate": "Bearer"},
        )
