"""Freeze provider-free Phase 7.4.1/7.5 evidence before any data egress.

This verifier reads only sanitized artifacts and package source.  It never
constructs a model, calls Qdrant/provider, or executes held-out questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.phase7_optimization import PHASE7_CALIBRATION_FUSION_PROFILE
from scripts.evaluate_phase7_retrieval_closure import _per_language


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure",
        type=Path,
        default=Path("artifacts/metrics/phase-7-contamination-closure-v3.json"),
    )
    parser.add_argument(
        "--cpu",
        type=Path,
        default=Path("artifacts/metrics/phase-7-cpu-reranker-ablation-v1.json"),
    )
    parser.add_argument(
        "--fact-readiness",
        type=Path,
        default=Path("artifacts/metrics/phase-7-fact-evaluator-readiness-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-runtime-readiness-v1.json"),
    )
    args = parser.parse_args()
    closure = _read(args.closure)
    cpu = _read(args.cpu)
    facts = _read(args.fact_readiness)
    gates = _gates(closure, cpu, facts)
    status = "PROVIDER_APPROVAL_REQUIRED" if all(gates.values()) else "PARTIAL"
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "runtime_profile": asdict(PHASE7_CALIBRATION_FUSION_PROFILE),
        "runtime_source_sha256": _sha256(Path("app/retrieval_runtime.py")),
        "evaluator_source_sha256": _sha256(Path("app/evaluation_e2e.py")),
        "query_expansion_source_sha256": _sha256(Path("app/query_expansion.py")),
        "sources": {
            "closure": {
                "path": str(args.closure).replace("\\", "/"),
                "sha256": _sha256(args.closure),
            },
            "cpu": {"path": str(args.cpu).replace("\\", "/"), "sha256": _sha256(args.cpu)},
            "fact_readiness": {
                "path": str(args.fact_readiness).replace("\\", "/"),
                "sha256": _sha256(args.fact_readiness),
            },
        },
        "quality_gates": gates,
        "next_action": (
            "obtain separate approval before sending calibration questions and excerpts to provider"
            if status == "PROVIDER_APPROVAL_REQUIRED"
            else "resolve failing provider-free gate; do not run provider or held-out"
        ),
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }
    _write(args.output, payload)
    print(f"Phase 7 runtime readiness {status}: {args.output}")
    return 0 if status == "PROVIDER_APPROVAL_REQUIRED" else 2


def _gates(closure: dict[str, Any], cpu: dict[str, Any], facts: dict[str, Any]) -> dict[str, bool]:
    closure_rows = closure.get("per_query")
    if not isinstance(closure_rows, list):
        raise ValueError("Closure artifact is missing per-query metrics.")
    overall = closure.get("overall", {})
    per_language = _per_language(closure_rows)
    selected = cpu.get("quality", {}).get("selected_config")
    cpu_result = next(
        (
            result
            for result in cpu.get("results", [])
            if result.get("valid") and result.get("config") == selected
        ),
        None,
    )
    cpu_quality = cpu_result.get("quality", {}) if isinstance(cpu_result, dict) else {}
    return {
        "closure_provider_free": closure.get("provider_calls") == 0
        and closure.get("held_out_queries_executed") == 0,
        "closure_candidate_recall_12_of_12": overall.get("candidate_recall") == 1.0,
        "closure_hit_at_5_11_of_12": overall.get("hit_rate_at_5", 0.0) >= 11 / 12,
        "closure_mrr_at_5_0_875": overall.get("mrr_at_5", 0.0) >= 0.875,
        "closure_wrong_document_top1_zero": overall.get("wrong_document_top1_rate") == 0.0,
        "closure_wrong_document_top5_at_most_0_15": (
            overall.get("wrong_document_candidate_rate_at_5", 1.0) <= 0.15
        ),
        "closure_english_hit_6_of_6": per_language["en"]["hit_rate_at_5"] == 1.0,
        "closure_vietnamese_hit_5_of_6": per_language["vi"]["hit_rate_at_5"] >= 5 / 6,
        "closure_010_rank_at_most_6": _rank(closure_rows, "phase7_calibration_010") <= 6,
        "cpu_full_stage": cpu.get("stage") == "full",
        "cpu_provider_free": (
            cpu.get("provider_calls") == 0 and cpu.get("held_out_queries_executed") == 0
        ),
        "cpu_selected_quality": bool(cpu_result)
        and cpu_quality.get("candidate_recall") == 1.0
        and cpu_quality.get("hit_rate_at_5", 0.0) >= 11 / 12
        and cpu_quality.get("mrr_at_5", 0.0) >= 0.875
        and cpu_quality.get("wrong_document_top1_rate") == 0.0
        and cpu_quality.get("wrong_document_candidate_rate_at_5", 1.0) <= 0.15,
        "cpu_target_or_non_regression": bool(cpu.get("quality", {}).get("latency_target_met")),
        "typed_fact_draft_not_active": facts.get("status") == "HUMAN_REVIEW_REQUIRED"
        and facts.get("activation") == "not_active"
        and facts.get("ground_truth_preserved") is True,
    }


def _rank(rows: list[dict[str, Any]], identifier: str) -> int:
    for row in rows:
        if row.get("id") == identifier:
            return int(row.get("final_direct_evidence_rank") or 2**31)
    raise ValueError(f"Required calibration row missing: {identifier}")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read readiness input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Readiness input is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    from app.phase7 import write_json_atomic

    write_json_atomic(path, value)


if __name__ == "__main__":
    raise SystemExit(main())
