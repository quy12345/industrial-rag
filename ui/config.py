"""Environment and document options for the Phase 7 Streamlit demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

INSTALLATION_DOCUMENT_ID = "atv320-installation-manual-en-nve41289-09-c181b4d7f11b"
PROGRAMMING_DOCUMENT_ID = "atv320-programming-manual-en-nve41295-06-f5e9bb48167a"

DOCUMENT_OPTIONS: dict[str, str | None] = {
    "Tất cả tài liệu ATV320": None,
    "Installation Manual": INSTALLATION_DOCUMENT_ID,
    "Programming Manual": PROGRAMMING_DOCUMENT_ID,
}

DEMO_QUESTIONS = (
    "Tài liệu Installation Manual hỗ trợ những công việc nào?",
    "How do I navigate and configure the ATV320 display menus?",
    "Khi bảo trì ATV320 cần chú ý những nguyên tắc an toàn nào?",
)


@dataclass(frozen=True)
class UISettings:
    """Server-side UI settings; secrets are never sent to the browser."""

    api_url: str
    timeout_seconds: float
    auth_token: str | None

    @classmethod
    def from_environment(cls) -> UISettings:
        api_url = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1").strip().rstrip("/")
        parsed = urlparse(api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RAG_API_URL must be an absolute HTTP(S) URL.")
        try:
            timeout_seconds = float(os.getenv("RAG_API_TIMEOUT_SECONDS", "180"))
        except ValueError as exc:
            raise ValueError("RAG_API_TIMEOUT_SECONDS must be numeric.") from exc
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("RAG_API_TIMEOUT_SECONDS must be between 1 and 300.")
        auth_token = os.getenv("RAG_API_AUTH_TOKEN", "").strip() or None
        return cls(
            api_url=api_url,
            timeout_seconds=timeout_seconds,
            auth_token=auth_token,
        )
