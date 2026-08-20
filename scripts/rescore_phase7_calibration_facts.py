"""Rescore sanitized calibration fact diagnostics with the deterministic text matcher.

The source artifact contains no answer text, so this tool is intentionally limited
to legacy text facts whose stored maximum alias token recall is sufficient to
reconstruct the deterministic token-set decision. It never calls a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation_e2e import FACT_EVALUATOR_ID
from app.phase7 import read_phase7_dataset, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-e2e-v2-diagnostics.json"),
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-fact-rescore-v1.json"),
    )
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    calibration = read_phase7_dataset(args.calibration)
    answerable = {item.id: item for item in calibration if item.answerable}
    _validate_reconstructable_facts(answerable)
    rows = rescore_records(source.get("per_query", []), answerable)
    strict_count = sum(row["strict_phrase_match"] for row in rows)
    deterministic_count = sum(row["deterministic_fact_match"] for row in rows)
    query_count = len(rows)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_artifact": str(args.source),
        "source_artifact_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "derivation": (
            "Reconstructed from sanitized per-fact strict match and max alias token recall; "
            "no answer text or provider call was available."
        ),
        "fact_evaluator_id": FACT_EVALUATOR_ID,
        "query_count": query_count,
        "strict_phrase_accuracy": strict_count / query_count,
        "deterministic_fact_accuracy": deterministic_count / query_count,
        "strict_phrase_match_count": strict_count,
        "deterministic_fact_match_count": deterministic_count,
        "required_match_count": 11,
        "quality_gate_passed": deterministic_count >= 11,
        "per_query": rows,
        "provider_calls": 0,
        "held_out_queries_executed": 0,
    }
    write_json_atomic(args.output, payload)
    print(f"Phase 7 calibration fact rescore PASS: {args.output}")
    return 0


def rescore_records(
    records: list[dict[str, Any]], answerable: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {record.get("id"): record for record in records if record.get("answerable")}
    if set(by_id) != set(answerable):
        raise ValueError("Source artifact answerable IDs differ from calibration.")
    rows: list[dict[str, Any]] = []
    for item_id in sorted(answerable):
        record = by_id[item_id]
        results = record.get("answer_fact_results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"Source artifact has no fact diagnostics for {item_id}.")
        strict = all(bool(result.get("matched")) for result in results)
        deterministic = all(
            float(result.get("max_alias_token_recall", 0.0)) == 1.0
            for result in results
        )
        rows.append(
            {
                "id": item_id,
                "strict_phrase_match": strict,
                "deterministic_fact_match": deterministic,
                "strict_missing_fact_ids": [
                    result.get("id") for result in results if not result.get("matched")
                ],
                "deterministic_missing_fact_ids": [
                    result.get("id")
                    for result in results
                    if float(result.get("max_alias_token_recall", 0.0)) != 1.0
                ],
            }
        )
    return rows


def _validate_reconstructable_facts(answerable: dict[str, Any]) -> None:
    unsupported = [
        item.id
        for item in answerable.values()
        if any(
            fact.type != "text" or fact.required_token_groups
            for fact in item.expected_answer_facts
        )
    ]
    if unsupported:
        raise ValueError(
            "Sanitized reconstruction supports only legacy text facts: "
            + ", ".join(sorted(unsupported))
        )


if __name__ == "__main__":
    raise SystemExit(main())
