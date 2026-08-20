"""Bounded Phase 7.5 CPU reranker benchmark without provider or held-out data.

``micro`` measures the documented finite grid on three calibration IDs after a
warmup.  ``full`` remeasures only the two micro Pareto configurations across
all approved answerable calibration rows and three warm repetitions each.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from app.config import Settings
from app.evaluation import direct_evidence_rank, load_frozen_chunks, percentile_nearest_rank
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, build_query_retriever
from scripts.evaluate_phase7_retrieval_closure import aggregate_closure_rows

Stage = Literal["micro", "full"]
MICRO_QUERY_IDS = {
    "phase7_calibration_001",
    "phase7_calibration_004",
    "phase7_calibration_010",
}
MICRO_BATCH_SIZES = (8, 16, 30)
MICRO_THREADS: tuple[int | None, ...] = (None, 1, 2, 4)
MICRO_BUDGETS = (26, 28, 30)
FULL_REPETITIONS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("micro", "full"), default="micro")
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--micro-input",
        type=Path,
        default=Path("artifacts/metrics/phase-7-cpu-reranker-micro-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-cpu-reranker-ablation-v1.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    held_out = read_phase7_dataset(args.test)
    validation = validate_phase7_datasets(calibration, held_out, load_frozen_chunks(args.chunks))
    answerable = [item for item in calibration if item.answerable]
    if args.stage == "micro":
        selected = [item for item in answerable if item.id in MICRO_QUERY_IDS]
        payload = _micro_payload(selected, calibration, held_out, validation)
    else:
        configs = _read_micro_pareto(args.micro_input, calibration_sha=dataset_sha256(calibration))
        payload = _full_payload(answerable, configs, calibration, held_out, validation)
    write_json_atomic(args.output, payload)
    status = "PASS" if payload["quality"]["overall_pass"] else "PARTIAL"
    print(f"Phase 7 CPU reranker {args.stage} benchmark {status}: {args.output}")
    return 0 if payload["quality"]["overall_pass"] else 2


def _micro_payload(
    items: list[Any], calibration: list[Any], held_out: list[Any], validation: dict[str, Any]
) -> dict[str, Any]:
    configs = [
        {"candidate_budget": budget, "batch_size": batch_size, "threads": threads}
        for budget in MICRO_BUDGETS
        for batch_size in MICRO_BATCH_SIZES
        for threads in MICRO_THREADS
    ]
    results = [_benchmark_config(config, items, repetitions=1) for config in configs]
    pareto = _select_pareto(results, maximum=2)
    return _payload(
        stage="micro",
        calibration=calibration,
        held_out=held_out,
        validation=validation,
        results=results,
        pareto_configs=pareto,
    )


def _full_payload(
    items: list[Any],
    configs: list[dict[str, Any]],
    calibration: list[Any],
    held_out: list[Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    results = [_benchmark_config(config, items, repetitions=FULL_REPETITIONS) for config in configs]
    selected = _select_pareto(results, maximum=1)
    return _payload(
        stage="full",
        calibration=calibration,
        held_out=held_out,
        validation=validation,
        results=results,
        pareto_configs=selected,
    )


def _benchmark_config(
    config: dict[str, Any], items: list[Any], *, repetitions: int
) -> dict[str, Any]:
    contract = replace(
        PHASE7_RETRIEVAL_CONTRACT,
        phase7_fusion_profile=replace(
            PHASE7_RETRIEVAL_CONTRACT.phase7_fusion_profile,
            max_candidates=int(config["candidate_budget"]),
        ),
    )
    settings = _settings_for_config(config, contract)
    started = perf_counter()
    try:
        retriever = build_query_retriever(settings, contract=contract)
        # The first query initializes the lazy cross-encoder.  It is intentionally excluded.
        retriever.retrieve(items[0].question, document_id=None)
        cold_initialization_ms = (perf_counter() - started) * 1000
    except Exception as exc:
        return {
            "config": config,
            "valid": False,
            "failure": type(exc).__name__,
            "provider_calls": 0,
            "held_out_queries_executed": 0,
        }

    quality_rows: list[dict[str, Any]] = []
    samples: list[dict[str, float]] = []
    for repetition in range(repetitions):
        for item in items:
            try:
                execution = retriever.retrieve(item.question, document_id=None)
            except Exception as exc:
                return {
                    "config": config,
                    "valid": False,
                    "failure": type(exc).__name__,
                    "provider_calls": 0,
                    "held_out_queries_executed": 0,
                }
            if repetition == 0:
                relevant = set(item.relevant_chunk_ids)
                quality_rows.append(
                    {
                        "id": item.id,
                        "language": item.language,
                        "candidate_count": len(execution.candidate_pool or execution.candidates),
                        "final_candidate_count": len(execution.candidates),
                        "candidate_direct_evidence_rank": direct_evidence_rank(
                            execution.candidate_pool or execution.candidates, relevant
                        ),
                        "final_direct_evidence_rank": direct_evidence_rank(
                            execution.candidates, relevant
                        ),
                        "failure_class": "hit",
                        "wrong_document_top1": bool(execution.candidates)
                        and execution.candidates[0].document_id not in item.expected_document_ids,
                        "wrong_document_candidate_count_at_5": sum(
                            candidate.document_id not in item.expected_document_ids
                            for candidate in execution.candidates[:5]
                        ),
                        "document_context_complete": all(
                            candidate.metadata.get("document_role")
                            in {"installation", "programming"}
                            for candidate in execution.candidates
                        ),
                        "retrieval_ms": execution.retrieval_ms,
                        "rerank_ms": execution.rerank_ms,
                    }
                )
            samples.append(
                {
                    "retrieval_ms": execution.retrieval_ms,
                    "rerank_ms": execution.rerank_ms,
                    "total_ms": execution.retrieval_ms + execution.rerank_ms,
                }
            )
    return {
        "config": config,
        "valid": True,
        "cold_initialization_ms": cold_initialization_ms,
        "quality": aggregate_closure_rows(quality_rows),
        "per_query": quality_rows,
        "warm_latency_ms": _latency(samples),
        "provider_calls": 0,
        "held_out_queries_executed": 0,
    }


def _settings_for_config(config: dict[str, Any], contract: Any) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "qdrant_collection": contract.dense_collection,
            "qdrant_hybrid_collection": contract.hybrid_collection,
            "bm25_avg_len": contract.bm25_avg_len,
            "dense_candidate_limit": contract.dense_candidate_limit,
            "sparse_candidate_limit": contract.sparse_candidate_limit,
            "rrf_k": contract.rrf_k,
            "retrieval_strategy": "union",
            "rerank_enabled": True,
            "rerank_deduplicate_content": True,
            "rerank_batch_size": int(config["batch_size"]),
            "rerank_threads": config["threads"],
        }
    )


def _select_pareto(results: list[dict[str, Any]], *, maximum: int) -> list[dict[str, Any]]:
    valid = [result for result in results if result.get("valid") and _quality_passes(result)]
    ordered = sorted(
        valid,
        key=lambda result: (
            float(result["warm_latency_ms"]["total_ms"]["p95"]),
            int(result["config"]["candidate_budget"]),
            int(result["config"]["batch_size"]),
            0 if result["config"]["threads"] is None else int(result["config"]["threads"]),
        ),
    )
    return [result["config"] for result in ordered[:maximum]]


def _quality_passes(result: dict[str, Any]) -> bool:
    metrics = result["quality"]
    rows = result["per_query"]
    if len(rows) != 12:
        # Microbenchmark has one English, one Vietnamese, and the known rank-6
        # programming case. Full-quality gates are intentionally applied only
        # after the two fastest safe configurations are remeasured on all 12 rows.
        return (
            metrics["candidate_recall"] == 1.0
            and metrics["wrong_document_top1_rate"] == 0.0
            and metrics["candidate_count_maximum"] <= 30
            and _direct_rank(rows, "phase7_calibration_010") <= 6
        )
    if not (
        metrics["candidate_recall"] == 1.0
        and metrics["hit_rate_at_5"] >= 11 / 12
        and metrics["mrr_at_5"] >= 0.875
        and metrics["wrong_document_top1_rate"] == 0.0
        and metrics["wrong_document_candidate_rate_at_5"] <= 0.15
        and metrics["candidate_count_maximum"] <= 30
    ):
        return False
    english = [row for row in rows if row["language"] == "en"]
    vietnamese = [row for row in rows if row["language"] == "vi"]
    return (
        len(english) == len(vietnamese) == 6
        and sum((row["final_direct_evidence_rank"] or 2**31) <= 5 for row in english) == 6
        and sum((row["final_direct_evidence_rank"] or 2**31) <= 5 for row in vietnamese) >= 5
        and _direct_rank(rows, "phase7_calibration_010") <= 6
    )


def _payload(
    *,
    stage: Stage,
    calibration: list[Any],
    held_out: list[Any],
    validation: dict[str, Any],
    results: list[dict[str, Any]],
    pareto_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = pareto_configs[0] if pareto_configs else None
    selected_result = next(
        (result for result in results if result.get("config") == selected and result.get("valid")),
        None,
    )
    target_ms = min(10_000.0, 13_399.0 * 0.75)
    latency_target = (
        bool(selected_result)
        and selected_result["warm_latency_ms"]["rerank_ms"]["p95"] <= target_ms
    )
    return {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "scope": "approved answerable calibration rows only",
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "held_out_dataset_sha256": dataset_sha256(held_out),
        "corpus": validation["corpus"],
        "hardware": {"cpu_count": os.cpu_count(), "platform": platform.platform()},
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("fastembed", "onnxruntime", "qdrant-client")
        },
        "methodology": {
            "warmup": "one unmeasured query per configuration",
            "full_repetitions": FULL_REPETITIONS if stage == "full" else 1,
            "candidate_budgets": list(MICRO_BUDGETS),
            "batch_sizes": list(MICRO_BATCH_SIZES),
            "threads": ["default" if value is None else value for value in MICRO_THREADS],
            "target_rerank_p95_ms": target_ms,
        },
        "results": results,
        "pareto_configs": pareto_configs,
        "quality": {
            "overall_pass": bool(selected_result) and _quality_passes(selected_result),
            "latency_target_met": latency_target,
            "selected_config": selected,
        },
        "sanitization": {
            "question": "excluded",
            "evidence": "excluded",
            "answer": "not applicable",
        },
    }


def _latency(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "average": sum(sample[name] for sample in samples) / len(samples),
            "p50": percentile_nearest_rank([sample[name] for sample in samples], 50),
            "p95": percentile_nearest_rank([sample[name] for sample in samples], 95),
            "maximum": max(sample[name] for sample in samples),
        }
        for name in ("retrieval_ms", "rerank_ms", "total_ms")
    }


def _direct_rank(rows: list[dict[str, Any]], identifier: str) -> int:
    for row in rows:
        if row["id"] == identifier:
            return int(row["final_direct_evidence_rank"] or 2**31)
    raise ValueError(f"Required calibration row missing: {identifier}")


def _read_micro_pareto(path: Path, *, calibration_sha: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to read Phase 7 CPU microbenchmark artifact.") from exc
    if payload.get("stage") != "micro" or payload.get("provider_calls") != 0:
        raise ValueError("CPU microbenchmark must be calibration-only.")
    if payload.get("calibration_dataset_sha256") != calibration_sha:
        raise ValueError("CPU microbenchmark dataset hash mismatch.")
    configs = payload.get("pareto_configs")
    if not isinstance(configs, list) or not 1 <= len(configs) <= 2:
        raise ValueError("CPU microbenchmark has no bounded Pareto configurations.")
    return configs


if __name__ == "__main__":
    raise SystemExit(main())
