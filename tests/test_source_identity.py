"""Offline characterization for canonical Phase 7 E2E source identity v2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.evaluation import evaluate_phase7_e2e

REPOSITORY_ROOT = Path(__file__).parents[1]

CANONICAL_SOURCE_IDENTITY_PATHS = {
    "citation_policy": Path("app/domain/citations.py"),
    "dataset_contract": Path("evaluation/phase7_dataset.py"),
    "dense_search_adapter": Path("app/infrastructure/qdrant/dense.py"),
    "evaluator": Path("evaluation/e2e.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_phase7_e2e.py"),
    "evidence_policy": Path("app/domain/evidence.py"),
    "fusion_policy": Path("app/domain/policies/fusion.py"),
    "generation_adapter": Path("app/infrastructure/generation/langchain_structured.py"),
    "generation_prompt_policy": Path("app/application/generation_prompt.py"),
    "list_completeness_policy": Path("app/domain/policies/list_completeness.py"),
    "query_analysis_policy": Path("app/domain/policies/query_analysis.py"),
    "query_role_policy": Path("app/domain/policies/query_roles.py"),
    "query_service": Path("app/application/query_service.py"),
    "reranker_adapter": Path("app/infrastructure/models/reranker.py"),
    "reranking_service": Path("app/application/reranking_service.py"),
    "retrieval_metrics": Path("evaluation/retrieval.py"),
    "retrieval_composition": Path("app/composition/retrieval.py"),
    "retrieval_contract": Path("app/domain/retrieval_contracts.py"),
    "search_adapters": Path("app/infrastructure/qdrant/search.py"),
    "sparse_search_adapter": Path("app/infrastructure/qdrant/hybrid.py"),
}


def test_e2e_source_identity_v2_hashes_canonical_behavior_owners() -> None:
    assert evaluate_phase7_e2e.SOURCE_IDENTITY_VERSION == 2
    assert evaluate_phase7_e2e.SOURCE_IDENTITY_PATHS == CANONICAL_SOURCE_IDENTITY_PATHS

    expected_files = {
        name: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for name, path in CANONICAL_SOURCE_IDENTITY_PATHS.items()
    }

    assert evaluate_phase7_e2e._source_identity() == {
        "version": 2,
        "files": expected_files,
        "system_prompt_sha256": (
            "bee13049c510701f72259a760fc9bab29e80e20b62f35ca27b13e1dff8f8fc93"
        ),
    }


def test_v1_checkpoint_fails_closed_under_source_identity_v2(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy-v1-checkpoint.jsonl"
    legacy_identity = {
        "source_identity": {
            "prompt": "legacy",
            "evaluator": "legacy",
        }
    }
    evaluate_phase7_e2e._write_checkpoint(checkpoint, legacy_identity, [])

    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_phase7_e2e._load_checkpoint(
            checkpoint,
            {"source_identity": evaluate_phase7_e2e._source_identity()},
        )
