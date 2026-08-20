"""Request-scoped metadata safe to include in structured operational logs."""

from __future__ import annotations

from contextvars import ContextVar

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
