"""Compatibility facade for evidence-content identity policies."""

from app.domain.content_identity import (
    evidence_content_fingerprint as evidence_content_fingerprint,
)
from app.domain.content_identity import normalize_evidence_content as normalize_evidence_content

__all__ = ["evidence_content_fingerprint", "normalize_evidence_content"]
