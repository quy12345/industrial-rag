"""Offline tests for Phase 7.4.1 bounded role-aware retrieval optimisation."""

from __future__ import annotations

import pytest

from app.models import RetrievalCandidate
from app.phase7_optimization import (
    Phase7FusionProfile,
    Phase7OptimizationError,
    apply_role_aware_rank_fusion,
    infer_query_role,
    select_coverage_preserving_candidates,
)
from scripts.calibrate_phase7_weighted_fusion import _select_pareto_profiles


def _candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
    role: str = "installation",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=f"manual-{role}",
        filename="manual.pdf",
        text=chunk_id,
        page_numbers=[1],
        headings=[],
        content_type="text",
        metadata={"document_role": role},
        score=0.0,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def _profile(**updates: object) -> Phase7FusionProfile:
    values: dict[str, object] = {
        "name": "test-profile",
        "rrf_k": 60,
        "dense_weight": 1.0,
        "sparse_weight": 1.25,
        "fusion_role_multiplier": 0.1,
        "dense_reserve": 1,
        "sparse_reserve": 2,
        "max_candidates": 3,
    }
    values.update(updates)
    return Phase7FusionProfile(**values)  # type: ignore[arg-type]


def test_query_role_is_bilingual_boundary_safe_and_confidence_aware() -> None:
    installation = infer_query_role("How should the protective equipment be installed?")
    assert installation.role == "installation"
    assert installation.confidence == "strong"
    assert "installation" in installation.cue_ids
    assert infer_query_role("Can lam gi de tranh truc dong co quay?").role == "installation"
    programming = infer_query_role("Phim MODE chuyen giua cac nhom menu nao?")
    assert programming.role == "programming"
    assert programming.confidence == "strong"
    assert infer_query_role("The model was tested.").role == "neutral"
    assert infer_query_role("Which MODE parameter controls power protection?").role == "neutral"


def test_selector_preserves_sparse_tail_evidence_inside_fixed_budget() -> None:
    dense = [_candidate("dense-1", dense_rank=1), _candidate("dense-2", dense_rank=2)]
    sparse = [
        _candidate("sparse-1", sparse_rank=1),
        _candidate("sparse-2", sparse_rank=2),
        _candidate("evidence-24", sparse_rank=24),
    ]
    result = select_coverage_preserving_candidates(
        dense,
        sparse,
        profile=_profile(dense_reserve=0, sparse_reserve=3),
        query_role="neutral",
    )
    assert len(result) == 3
    assert "evidence-24" in [candidate.chunk_id for candidate in result]


def test_role_prior_is_rank_only_and_preserves_cross_encoder_score() -> None:
    wrong = _candidate("wrong", sparse_rank=1, role="programming").model_copy(
        update={"rerank_rank": 1, "rerank_score": 99.0, "score": 99.0}
    )
    evidence = _candidate("evidence", sparse_rank=2, role="installation").model_copy(
        update={"rerank_rank": 2, "rerank_score": 1.0, "score": 1.0}
    )
    result = apply_role_aware_rank_fusion(
        [wrong, evidence],
        query_role="installation",
        role_multiplier=0.2,
        rank_offset=10,
        confidence="strong",
    )
    assert [candidate.chunk_id for candidate in result] == ["evidence", "wrong"]
    assert result[0].rerank_score == 1.0
    assert result[0].metadata["cross_encoder_rank"] == 2
    assert result[0].metadata["role_prior_score"] > 0
    assert result[0].score != result[0].rerank_score


def test_weak_and_neutral_roles_do_not_receive_strong_only_prior() -> None:
    candidates = [_candidate("a", sparse_rank=1).model_copy(update={"rerank_rank": 1})]
    assert (
        apply_role_aware_rank_fusion(
            candidates,
            query_role="installation",
            role_multiplier=0.1,
            confidence="weak",
            confidence_mode="strong_only",
        )
        == candidates
    )
    assert (
        apply_role_aware_rank_fusion(candidates, query_role="neutral", role_multiplier=0.1)
        == candidates
    )


def test_selector_fails_instead_of_silently_truncating_mandatory_reserves() -> None:
    dense = [_candidate(f"d{index}", dense_rank=index + 1) for index in range(2)]
    sparse = [_candidate(f"s{index}", sparse_rank=index + 1) for index in range(2)]
    with pytest.raises(Phase7OptimizationError, match="reserves"):
        select_coverage_preserving_candidates(
            dense,
            sparse,
            profile=_profile(dense_reserve=2, sparse_reserve=2, max_candidates=3),
            query_role="neutral",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"rrf_k": 0},
        {"dense_weight": 0.0},
        {"fusion_role_multiplier": 0.3},
        {"post_rerank_role_multiplier": 0.6},
        {"post_rerank_rank_offset": 0},
        {"dense_reserve": -1},
    ],
)
def test_profile_rejects_invalid_bounds(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _profile(**updates)


def test_pareto_selection_prefers_recall_then_contamination_then_simplicity() -> None:
    summaries = {
        "high-contamination": {
            "valid": True,
            "candidate_recall": 1.0,
            "wrong_document_top1_rate": 0.25,
            "wrong_document_candidate_rate_at_5": 0.3,
            "profile": {"sparse_weight": 1.5, "fusion_role_multiplier": 0.1, "rrf_k": 60},
        },
        "winner": {
            "valid": True,
            "candidate_recall": 1.0,
            "wrong_document_top1_rate": 0.0,
            "wrong_document_candidate_rate_at_5": 0.1,
            "profile": {"sparse_weight": 1.0, "fusion_role_multiplier": 0.0, "rrf_k": 60},
        },
        "lower-recall": {
            "valid": True,
            "candidate_recall": 11 / 12,
            "wrong_document_top1_rate": 0.0,
            "wrong_document_candidate_rate_at_5": 0.0,
            "profile": {"sparse_weight": 1.0, "fusion_role_multiplier": 0.0, "rrf_k": 20},
        },
    }
    assert [row["name"] for row in _select_pareto_profiles(summaries)] == [
        "winner",
        "high-contamination",
        "lower-recall",
    ]
