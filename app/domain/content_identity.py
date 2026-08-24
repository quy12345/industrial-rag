"""Dependency-free identity helpers for equivalent evidence content."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_evidence_content(text: str) -> str:
    """Normalize case and whitespace without removing technical punctuation."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def evidence_content_fingerprint(text: str) -> str:
    """Return a stable hash used only for exact-normalized content equivalence."""

    return hashlib.sha256(normalize_evidence_content(text).encode("utf-8")).hexdigest()
