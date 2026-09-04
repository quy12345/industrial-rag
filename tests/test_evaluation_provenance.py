"""Offline characterization for canonical E2E source identity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.evaluation import evaluate_e2e

REPOSITORY_ROOT = Path(__file__).parents[1]

CANONICAL_SOURCE_IDENTITY_PATHS = {
    "dataset_contract": Path("evaluation/dataset.py"),
    "evaluator": Path("evaluation/e2e.py"),
    "semantic_evaluator": Path("evaluation/semantic.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_e2e.py"),
    "generation_prompt": Path("app/application/generation_prompt.py"),
    "query_service": Path("app/application/query_service.py"),
    "retrieval_composition": Path("app/composition/retrieval.py"),
    "retrieval_contract": Path("app/domain/retrieval_contracts.py"),
}


def test_e2e_source_identity_v11_hashes_canonical_behavior_owners() -> None:
    assert evaluate_e2e.SOURCE_IDENTITY_VERSION == 11
    assert evaluate_e2e.SOURCE_IDENTITY_PATHS == CANONICAL_SOURCE_IDENTITY_PATHS

    expected_files = {
        name: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for name, path in CANONICAL_SOURCE_IDENTITY_PATHS.items()
    }

    assert evaluate_e2e._source_identity() == {
        "version": 11,
        "files": expected_files,
        "system_prompt_sha256": (
            "bee13049c510701f72259a760fc9bab29e80e20b62f35ca27b13e1dff8f8fc93"
        ),
    }


def test_legacy_checkpoint_fails_closed_under_source_identity_v8(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy-v10-checkpoint.jsonl"
    legacy_identity = {
        "source_identity": {
            "prompt": "legacy",
            "evaluator": "legacy",
        }
    }
    evaluate_e2e._write_checkpoint(checkpoint, legacy_identity, [])

    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_e2e._load_checkpoint(
            checkpoint,
            {"source_identity": evaluate_e2e._source_identity()},
        )
