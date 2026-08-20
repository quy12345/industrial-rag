"""Pure presentation helpers for the Streamlit demo."""

from __future__ import annotations

from typing import Any

from app.models import QueryResponse
from ui.api_client import RAGAPIError

MAX_HISTORY_TURNS = 20


def append_history(
    history: list[dict[str, Any]],
    *,
    question: str,
    response: QueryResponse | None = None,
    error: RAGAPIError | None = None,
) -> list[dict[str, Any]]:
    """Append one display-only turn and keep a bounded per-session history."""

    turn = {
        "question": question,
        "response": None if response is None else response.model_dump(mode="json"),
        "error": None
        if error is None
        else {"code": error.code, "message": error.message},
    }
    return [*history, turn][-MAX_HISTORY_TURNS:]


def page_label(page_numbers: list[int]) -> str:
    """Return a stable human-readable page label."""

    pages = sorted(set(page_numbers))
    return ", ".join(str(page) for page in pages) if pages else "unknown"
