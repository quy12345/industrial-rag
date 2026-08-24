"""Offline characterization for the legacy Phase 7 E2E source identity."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import evaluate_phase7_e2e

REPOSITORY_ROOT = Path(__file__).parents[1]

PINNED_GIT_BLOBS = {
    Path("scripts/evaluate_phase7_e2e.py"): "7ce4dc9c180fd675eb89e5c06844766b664c6b69",
    Path("app/evaluation_e2e.py"): "b8be722d43bc34c8bec00dfc2574d0a6341ab738",
}

LEGACY_SOURCE_IDENTITY_PATHS = {
    "prompt": Path("app/generation.py"),
    "evaluator": Path("app/evaluation_e2e.py"),
    "evidence_selector": Path("app/evidence_selection.py"),
    "retrieval_runtime": Path("app/retrieval_runtime.py"),
    "reranking": Path("app/reranking.py"),
    "phase7_optimization": Path("app/phase7_optimization.py"),
    "query_service": Path("app/query_service.py"),
    "citations": Path("app/citations.py"),
    "query_expansion": Path("app/query_expansion.py"),
}


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    git_blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(git_blob, usedforsecurity=False).hexdigest()


def test_exact_e2e_source_anchor_blobs_are_unchanged() -> None:
    actual = {
        path: _git_blob_sha1(REPOSITORY_ROOT / path) for path in PINNED_GIT_BLOBS
    }

    assert actual == PINNED_GIT_BLOBS


def test_legacy_e2e_source_identity_tracks_the_documented_files() -> None:
    actual = evaluate_phase7_e2e._source_identity()
    expected = {
        name: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for name, path in LEGACY_SOURCE_IDENTITY_PATHS.items()
    }
    expected["system_prompt_sha256"] = (
        "bee13049c510701f72259a760fc9bab29e80e20b62f35ca27b13e1dff8f8fc93"
    )

    assert actual == expected
