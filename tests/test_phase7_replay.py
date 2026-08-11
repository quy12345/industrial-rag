"""Offline validation for sanitized Phase 7 reranker snapshots."""

from __future__ import annotations

import pytest

from app.phase7_replay import Phase7ReplayError, replay_role_prior, snapshot_candidates_to_retrieval


def _row(chunk_id: str, rank: int, score: float, role: str = "installation") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": f"manual-{role}",
        "document_role": role,
        "cross_encoder_rank": rank,
        "rerank_score": score,
        "query_identifier_match_count": 0,
        "bracketed_label_code_pair_count": 0,
    }


def test_snapshot_replay_validates_and_reorders_without_raw_evidence() -> None:
    rows = [_row("wrong", 1, 5.0, "programming"), _row("right", 2, 1.0)]
    replayed = replay_role_prior(
        rows,
        query_role="installation",
        confidence="strong",
        role_multiplier=0.2,
        rank_offset=10,
        confidence_mode="strong_only",
    )
    assert [candidate.chunk_id for candidate in replayed] == ["right", "wrong"]
    assert replayed[0].text == ""
    assert replayed[0].rerank_score == 1.0


@pytest.mark.parametrize(
    "rows",
    [
        [_row("a", 1, 1.0), _row("a", 2, 2.0)],
        [_row("a", 1, 1.0), _row("b", 1, 2.0)],
        [_row("a", 2, 1.0)],
        [_row("a", 1, float("nan"))],
        [{"chunk_id": "a"}],
    ],
)
def test_snapshot_replay_rejects_malformed_candidate_records(rows: list[dict[str, object]]) -> None:
    with pytest.raises(Phase7ReplayError):
        snapshot_candidates_to_retrieval(rows)
