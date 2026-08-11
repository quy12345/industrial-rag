"""Aggregate exactly three independent sanitized Phase 7 calibration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from app.phase7 import write_json_atomic

REQUIRED_REGRESSION_IDS = (
    "phase7_calibration_003",
    "phase7_calibration_007",
    "phase7_calibration_010",
)


def main() -> int:
    args = _parse_args()
    payload = aggregate_stability_runs(args.run)
    write_json_atomic(args.output, payload)
    status = "PASS" if payload["quality_gates"]["overall_pass"] else "FAIL"
    print(f"Phase 7 calibration stability {status}: {args.output}")
    return 0 if status == "PASS" else 2


def aggregate_stability_runs(paths: list[Path]) -> dict[str, Any]:
    """Apply worst-run gates without selecting a lucky provider result."""

    if len(paths) != 3 or len({path.resolve() for path in paths}) != 3:
        raise ValueError("Stability aggregation requires three distinct run paths.")
    payloads = [_read_run(path) for path in paths]
    file_hashes = [_file_sha256(path) for path in paths]
    if len(set(file_hashes)) != 3:
        raise ValueError("Duplicate calibration run artifacts are not independent runs.")
    identities = [payload["run_identity"] for payload in payloads]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Calibration stability run identities do not match.")
    rows_by_run = [{row["id"]: row for row in payload["per_query"]} for payload in payloads]
    ids = set(rows_by_run[0])
    if any(set(rows) != ids for rows in rows_by_run[1:]):
        raise ValueError("Calibration runs do not contain the same query IDs.")
    if len(ids) != 20 or any(
        sum(bool(row["answerable"]) for row in rows.values()) != 12
        for rows in rows_by_run
    ):
        raise ValueError("Calibration stability requires 20 rows with exactly 12 answerable.")
    retrieval_signatures = [
        {
            identifier: (
                row.get("candidate_direct_evidence_rank"),
                row.get("ranked_direct_evidence_rank"),
                row.get("direct_evidence_rank"),
                tuple(row.get("evidence_candidate_ids", [])),
            )
            for identifier, row in rows.items()
        }
        for rows in rows_by_run
    ]
    retrieval_stable = all(
        signature == retrieval_signatures[0] for signature in retrieval_signatures[1:]
    )
    per_run = [
        _run_metrics(rows, payload)
        for rows, payload in zip(rows_by_run, payloads, strict=True)
    ]
    fact_counts = [metrics["deterministic_fact_match_count"] for metrics in per_run]
    per_item = {
        identifier: {
            "deterministic_fact_pass_count": sum(
                bool(rows[identifier].get("deterministic_fact_match")) for rows in rows_by_run
            ),
            "citation_document_pass_count": sum(
                rows[identifier].get("citation_document_correct") is not False
                for rows in rows_by_run
            ),
        }
        for identifier in sorted(ids)
    }
    regression_pass = all(
        per_item[identifier]["deterministic_fact_pass_count"] == 3
        and all(
            (rows[identifier].get("direct_evidence_rank") or 2**31) <= 5
            for rows in rows_by_run
        )
        for identifier in REQUIRED_REGRESSION_IDS
    )
    gates = {
        "worst_run_fact_accuracy_at_least_11_of_12": min(fact_counts) >= 11,
        "valid_citation_ids_all_runs": all(
            metrics["referential_valid_rate"] == 1.0 for metrics in per_run
        ),
        "unsupported_citations_zero_all_runs": all(
            metrics["unsupported_citation_count"] == 0 for metrics in per_run
        ),
        "wrong_document_citations_zero_all_runs": all(
            metrics["wrong_document_citation_count"] == 0 for metrics in per_run
        ),
        "abstention_precision_all_runs": all(
            metrics["abstention_precision"] >= 0.90 for metrics in per_run
        ),
        "abstention_recall_all_runs": all(
            metrics["abstention_recall"] >= 0.80 for metrics in per_run
        ),
        "retrieval_ranks_and_evidence_stable": retrieval_stable,
        "required_items_no_regression": regression_pass,
    }
    return {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "three independent frozen calibration runs; worst-run headline",
        "run_identity": identities[0],
        "source_runs": [
            {"path": str(path).replace("\\", "/"), "sha256": digest}
            for path, digest in zip(paths, file_hashes, strict=True)
        ],
        "fact_accuracy": {
            "counts": fact_counts,
            "minimum": min(fact_counts),
            "maximum": max(fact_counts),
            "mean": mean(fact_counts),
            "denominator": 12,
        },
        "per_run": per_run,
        "per_item": per_item,
        "quality_gates": {"overall_pass": all(gates.values()), "gates": gates},
        "sanitization": {
            "question": "excluded",
            "answer": "excluded",
            "evidence": "excluded",
            "provider_response": "excluded",
        },
    }


def _run_metrics(rows: dict[str, dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    answerable = [row for row in rows.values() if row["answerable"]]
    citations = payload["overall"]["citations"]
    abstention = payload["overall"]["abstention"]
    return {
        "deterministic_fact_match_count": sum(
            bool(row.get("deterministic_fact_match")) for row in answerable
        ),
        "referential_valid_rate": float(citations["referential_valid_rate_when_answered"]),
        "unsupported_citation_count": int(citations["unsupported_citation_count"]),
        "wrong_document_citation_count": int(citations["wrong_document_citation_count"]),
        "abstention_precision": float(abstention["precision"]),
        "abstention_recall": float(abstention["recall"]),
    }


def _read_run(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read sanitized calibration run: {path}") from exc
    if payload.get("schema_version") != 5:
        raise ValueError("Stability aggregation requires schema-v5 calibration runs.")
    identity = payload.get("run_identity")
    if not isinstance(identity, dict) or identity.get("dataset") != "calibration":
        raise ValueError("Stability input is not a sealed calibration run.")
    if payload.get("overall") is None or not isinstance(payload.get("per_query"), list):
        raise ValueError("Stability input is incomplete or partial.")
    if _contains_forbidden_content(payload):
        raise ValueError("Stability input contains forbidden raw or secret fields.")
    return payload


def _contains_forbidden_content(value: Any) -> bool:
    forbidden = {
        "question",
        "raw_question",
        "answer",
        "raw_answer",
        "evidence",
        "evidence_text",
        "provider_response",
        "provider_response_body",
        "api_key",
        "openai_api_key",
        "gemini_api_key",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "sanitization":
                continue
            if str(key).casefold() in forbidden:
                return True
            if _contains_forbidden_content(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_content(item) for item in value)
    return False


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-stability-v1.json"),
    )
    args = parser.parse_args()
    if len(args.run) != 3:
        parser.error("--run must be supplied exactly three times")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
