"""Generate fail-closed Phase 7 calibration/held-out readiness evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.phase7 import write_json_atomic


def main() -> int:
    args = _parse_args()
    rank_ablation = _read(args.rank_ablation)
    manifest = _read(args.manifest)
    diagnostic = _read_optional(args.diagnostic)
    stability = _read_optional(args.stability)
    payload = build_readiness(
        rank_ablation=rank_ablation,
        manifest=manifest,
        diagnostic=diagnostic,
        stability=stability,
        sources={
            "rank_ablation": _source(args.rank_ablation),
            "manifest": _source(args.manifest),
            "diagnostic": _optional_source(args.diagnostic),
            "stability": _optional_source(args.stability),
        },
    )
    write_json_atomic(args.output, payload)
    print(f"Phase 7 closure readiness {payload['status']}: {args.output}")
    return 0 if payload["status"] == "PASS_TECHNICAL" else 2


def build_readiness(
    *,
    rank_ablation: dict[str, Any],
    manifest: dict[str, Any],
    diagnostic: dict[str, Any] | None,
    stability: dict[str, Any] | None,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine technical gates while preserving the historical governance block."""

    quality = rank_ablation.get("quality", {})
    quality_gates = quality.get("gates", {})
    diagnostic_complete = (
        diagnostic is not None
        and diagnostic.get("run_identity", {}).get("provider_attempts") == 3
        and len(diagnostic.get("attempts", [])) == 3
    )
    diagnostic_positive = diagnostic_complete and all(
        attempt.get("status") == "completed"
        and attempt.get("deterministic_fact_match") is True
        and all(
            fact.get("polarity") == "positive"
            for fact in attempt.get("fact_results", [])
        )
        for attempt in diagnostic.get("attempts", [])
    )
    stability_pass = bool(
        stability is not None
        and stability.get("quality_gates", {}).get("overall_pass") is True
    )
    technical_gates = {
        "rank_ablation_provider_free": rank_ablation.get("provider_calls") == 0
        and rank_ablation.get("held_out_queries_executed") == 0,
        "candidate_recall_12_of_12": quality_gates.get("candidate_recall_12_of_12") is True,
        "hit_at_5_at_least_11_of_12": quality_gates.get(
            "hit_rate_at_5_at_least_11_of_12"
        )
        is True,
        "mrr_at_5_at_least_0_875": quality_gates.get("mrr_at_5_at_least_0_875") is True,
        "calibration_010_actual_evidence_top5": quality_gates.get(
            "calibration_010_rank_at_most_5"
        )
        is True,
        "calibration_005_three_attempt_diagnostic_complete": diagnostic_complete,
        "calibration_005_positive_in_all_attempts": diagnostic_positive,
        "three_run_worst_case_stability_pass": stability_pass,
        "active_calibration_hash_present": _valid_hash(
            manifest.get("calibration_dataset_sha256")
        ),
        "sealed_heldout_hash_present_without_dataset_read": _valid_hash(
            manifest.get("test_dataset_sha256")
        ),
    }
    technical_pass = all(technical_gates.values())
    governance = {
        "historical_heldout_content_was_mirrored_in_tracked_documentation": True,
        "historical_calibration_cli_loaded_both_splits": True,
        "statistically_unseen_claim_permitted": False,
        "resolution_required": (
            "Create a new access-controlled held-out set or explicitly downgrade the current "
            "set from unseen-final-test status. Git history is not rewritten."
        ),
    }
    status = "BLOCKED_GOVERNANCE"
    return {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "technical_pass": technical_pass,
        "technical_gates": technical_gates,
        "governance": governance,
        "provider_calls_by_generator": 0,
        "held_out_dataset_reads_by_generator": 0,
        "sources": sources or {},
        "source_code": {
            "evaluator_sha256": _sha256(Path("app/evaluation_e2e.py")),
            "runtime_sha256": _sha256(Path("app/retrieval_runtime.py")),
            "evidence_selector_sha256": _sha256(Path("app/evidence_selection.py")),
            "calibration_cli_sha256": _sha256(Path("scripts/evaluate_phase7_e2e.py")),
        },
        "next_action": (
            "Resolve technical calibration gates, then resolve held-out governance with a new "
            "access-controlled set or an explicit reporting downgrade."
        ),
        "sanitization": {
            "question": "excluded",
            "answer": "excluded",
            "evidence": "excluded",
            "held_out_content": "not_read",
        },
    }


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read readiness input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Readiness input is not an object: {path}")
    return value


def _read_optional(path: Path) -> dict[str, Any] | None:
    return _read(path) if path.exists() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path).replace("\\", "/"), "sha256": _sha256(path)}


def _optional_source(path: Path) -> dict[str, str] | None:
    return _source(path) if path.exists() else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-ablation",
        type=Path,
        default=Path("artifacts/metrics/phase-7-relation-list-ablation-v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/metrics/phase-7-evaluation-manifest-v3.json"),
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-005-diagnostic-v1.json"),
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-stability-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-heldout-readiness-v2.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
