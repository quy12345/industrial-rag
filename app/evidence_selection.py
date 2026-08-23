"""Compatibility exports for deterministic evidence-selection policy."""

from app.domain.evidence import EvidenceDuplicateGroup as EvidenceDuplicateGroup
from app.domain.evidence import EvidenceSelection as EvidenceSelection
from app.domain.evidence import EvidenceSelectionError as EvidenceSelectionError
from app.domain.evidence import select_evidence_candidates as select_evidence_candidates
from app.domain.evidence import (
    select_evidence_candidates_for_role as select_evidence_candidates_for_role,
)
