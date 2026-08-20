"""Evaluate all Phase 5 candidate pools with one lazy cross-encoder lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import get_settings
from app.evaluation import (
    EvaluationError,
    chunk_set_metadata,
    load_evaluation_cases,
    load_frozen_chunks,
    validate_cases_against_chunks,
)
from app.reranking import (
    CANDIDATE_TEXT_FORMAT,
    RerankingError,
    evaluate_reranked_cases,
    rerank_candidates,
)
from app.retrieval import RetrievalError
from scripts.rerank_runtime import build_rerank_runtime

METRICS_DIR = Path("artifacts/metrics")
DEFAULT_DATASET = Path("data/eval/dense_smoke.jsonl")
DEFAULT_CHUNKS = Path("artifacts/manual-batched.jsonl")
OUTPUTS = {
    "sparse": METRICS_DIR / "rerank-sparse.json",
    "hybrid": METRICS_DIR / "rerank-hybrid.json",
    "union": METRICS_DIR / "rerank-union.json",
}
CANDIDATE_AUDIT = METRICS_DIR / "candidate-pool-audit.json"
PHASE5_AUDIT = METRICS_DIR / "phase-5-candidate-audit.json"
COMPARISON = METRICS_DIR / "phase-5-comparison.json"


def main(argv: list[str] | None = None) -> int:
    """Run selected real reranker evaluations and write additive artifacts."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    if args.comparison_only:
        try:
            _rebuild_comparison_from_artifacts()
        except (EvaluationError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Comparison: {COMPARISON}")
        return 0
    settings = get_settings()
    strategies = ("sparse", "hybrid", "union") if args.strategy == "all" else (args.strategy,)
    try:
        cases = load_evaluation_cases(args.dataset)
        chunks = load_frozen_chunks(args.chunks)
        validate_cases_against_chunks(cases, chunks)
        frozen = chunk_set_metadata(chunks)
        pipeline, runtime_contract = build_rerank_runtime(settings, frozen, chunks)
        candidate_audit = _load_json(CANDIDATE_AUDIT)
        _validate_candidate_audit(candidate_audit, frozen)
        _write_candidate_audit_reference(candidate_audit, frozen)

        first_case = cases[0]
        first_pool = pipeline.prepare_pool(
            first_case.question,
            strategy=strategies[0],
            document_id=first_case.document_id,
        )
        cold_started = perf_counter()
        rerank_candidates(
            first_case.question,
            first_pool.candidates,
            pipeline.cross_encoder,
            strategy=strategies[0],
            batch_size=settings.rerank_batch_size,
        )
        cold_reranker_ms = (perf_counter() - cold_started) * 1000

        reports: dict[str, dict[str, Any]] = {}
        for strategy in strategies:
            pipeline.search(
                first_case.question,
                strategy=strategy,
                document_id=first_case.document_id,
            )
            evaluation = evaluate_reranked_cases(
                cases,
                lambda question, document_id, selected=strategy: pipeline.search(
                    question, strategy=selected, document_id=document_id
                ),
                cutoff=20,
            )
            report = _build_report(
                strategy=strategy,
                evaluation=evaluation,
                dataset=args.dataset,
                chunks_path=args.chunks,
                frozen=frozen,
                runtime_contract=runtime_contract,
                cold_reranker_ms=cold_reranker_ms,
                batch_size=settings.rerank_batch_size,
            )
            reports[strategy] = report
            _write_json_atomic(OUTPUTS[strategy], report)
            _print_summary(report, OUTPUTS[strategy])

        if args.strategy == "all":
            comparison = _build_comparison(reports, candidate_audit, frozen)
            _write_json_atomic(COMPARISON, comparison)
            print(f"Comparison: {COMPARISON}")
    except (EvaluationError, OSError, RetrievalError, RerankingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_report(
    *,
    strategy: str,
    evaluation: dict[str, Any],
    dataset: Path,
    chunks_path: Path,
    frozen: dict[str, Any],
    runtime_contract: dict[str, Any],
    cold_reranker_ms: float,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "strategy": f"{strategy}_rerank",
        "document_id": frozen["document_ids"][0],
        "evaluation_dataset": {"path": str(dataset), "sha256": _sha256_file(dataset)},
        "frozen_chunk_set": {"path": str(chunks_path), **frozen},
        "candidate_strategy": strategy,
        "candidate_limits": {"dense": 20, "sparse": 20, "hybrid": 20, "union": "unbounded"},
        "candidate_text_format": CANDIDATE_TEXT_FORMAT,
        "rerank_batch_size": batch_size,
        "runtime_contract": runtime_contract,
        "runtime": {
            "python": platform.python_version(),
            "fastembed": _package_version("fastembed"),
            "qdrant_client": _package_version("qdrant-client"),
        },
        "latency_methodology": {
            "unit": "milliseconds",
            "cold_reranker_initialization_and_first_inference": cold_reranker_ms,
            "warm_queries_exclude": ["model download", "model initialization", "one warmup query"],
            "warm_stages": [
                "dense_retrieval when applicable",
                "sparse_retrieval",
                "fusion or union preparation when applicable",
                "rerank",
                "total",
            ],
            "percentile_method": "nearest-rank",
        },
        **evaluation,
    }


def _build_comparison(
    reports: dict[str, dict[str, Any]],
    candidate_audit: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    summaries = {strategy: _strategy_summary(report) for strategy, report in reports.items()}
    best = max(summaries, key=lambda strategy: _strategy_sort_key(summaries[strategy]))
    union_recall = candidate_audit["aggregate_metrics"]["pools"]["dense_sparse_union"][
        "candidate_recall"
    ]
    complete_passes = [
        strategy
        for strategy, summary in summaries.items()
        if summary["quality_gate"]["pass"] and union_recall >= 0.933
    ]
    recommended = (
        max(complete_passes, key=lambda strategy: _strategy_sort_key(summaries[strategy]))
        if complete_passes
        else None
    )
    return {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "document_id": frozen["document_ids"][0],
        "frozen_chunk_set": frozen,
        "primary_baseline": {
            "strategy": "sparse",
            "hit_rate_at_5": 0.633,
            "mrr_at_5": 0.441,
            "mrr_at_20": 0.469,
            "candidate_recall": 0.867,
        },
        "union_candidate_recall": union_recall,
        "strategies": summaries,
        "best_observed_strategy": best,
        "recommended_default_strategy": recommended,
        "quality_gate": (
            "PASS"
            if recommended is not None
            else "PARTIAL"
            if any(_ranking_gates_pass(summary) for summary in summaries.values())
            else "FAIL"
        ),
        "selection_order": [
            "critical intent pairs top 5",
            "Hit@5",
            "MRR@5",
            "Vietnamese metrics",
            "English cross-lingual metrics",
            "candidate recall",
            "p95 latency",
            "candidate pool size",
            "license limitation",
        ],
        "license_limitation": (
            "CC-BY-NC-4.0 is non-commercial; a production/commercial deployment needs a "
            "different license or model."
        ),
    }


def _strategy_summary(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    critical_pairs = _critical_pair_count(report["critical_questions"])
    p95 = overall["stage_latency_ms"]["total"]["p95"]
    gates = {
        "critical_pairs_top5": {"actual": critical_pairs, "target": 3},
        "hit_rate_at_5": {"actual": overall["hit_rate_at_5"], "target": 0.633},
        "mrr_at_5": {"actual": overall["mrr_at_5"], "target": 0.485},
        "warm_total_p95_ms": {"actual": p95, "target_less_than": 1500.0},
    }
    gates["pass"] = (
        critical_pairs == 3
        and overall["hit_rate_at_5"] >= 0.633
        and overall["mrr_at_5"] >= 0.485
        and p95 < 1500
    )
    return {
        "overall": overall,
        "vietnamese": report["per_retrieval_scenario"]["monolingual"],
        "english_cross_lingual": report["per_retrieval_scenario"]["cross_lingual"],
        "critical_questions": [
            {
                "id": row["id"],
                "candidate_evidence_rank": row["candidate_evidence_rank"],
                "direct_evidence_rank": row["direct_evidence_rank"],
                "failure_class": row["failure_class"],
            }
            for row in report["critical_questions"]
        ],
        "quality_gate": gates,
    }


def _ranking_gates_pass(summary: dict[str, Any]) -> bool:
    gates = summary["quality_gate"]
    return (
        gates["critical_pairs_top5"]["actual"] == gates["critical_pairs_top5"]["target"]
        and gates["hit_rate_at_5"]["actual"] >= gates["hit_rate_at_5"]["target"]
        and gates["mrr_at_5"]["actual"] >= gates["mrr_at_5"]["target"]
    )


def _rebuild_comparison_from_artifacts() -> None:
    reports = {strategy: _load_json(path) for strategy, path in OUTPUTS.items()}
    frozen_sets = [
        {key: value for key, value in report["frozen_chunk_set"].items() if key != "path"}
        for report in reports.values()
    ]
    if not frozen_sets or any(frozen != frozen_sets[0] for frozen in frozen_sets[1:]):
        raise EvaluationError("Reranking reports do not use one frozen chunk set.")
    candidate_audit = _load_json(CANDIDATE_AUDIT)
    _validate_candidate_audit(candidate_audit, frozen_sets[0])
    comparison = _build_comparison(reports, candidate_audit, frozen_sets[0])
    _write_json_atomic(COMPARISON, comparison)


def _critical_pair_count(rows: list[dict[str, Any]]) -> int:
    by_id = {row["id"]: row["direct_evidence_rank"] for row in rows}
    return sum(
        by_id.get(first) is not None
        and by_id[first] <= 5
        and by_id.get(second) is not None
        and by_id[second] <= 5
        for first, second in (
            ("dense_001", "dense_002"),
            ("dense_003", "dense_004"),
            ("dense_005", "dense_006"),
        )
    )


def _strategy_sort_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    overall = summary["overall"]
    return (
        summary["quality_gate"]["critical_pairs_top5"]["actual"],
        overall["hit_rate_at_5"],
        overall["mrr_at_5"],
        summary["vietnamese"]["mrr_at_5"],
        summary["english_cross_lingual"]["mrr_at_5"],
        overall["candidate_recall"],
        -overall["stage_latency_ms"]["total"]["p95"],
        -overall["candidate_count"]["average"],
    )


def _write_candidate_audit_reference(audit: dict[str, Any], frozen: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_artifact": str(CANDIDATE_AUDIT),
        "source_sha256": _sha256_file(CANDIDATE_AUDIT),
        "frozen_chunk_set": frozen,
        "aggregate_metrics": audit["aggregate_metrics"],
        "critical_query_coverage": audit["critical_query_coverage"],
    }
    _write_json_atomic(PHASE5_AUDIT, payload)


def _validate_candidate_audit(audit: dict[str, Any], frozen: dict[str, Any]) -> None:
    if audit.get("frozen_chunk_set") != frozen:
        raise EvaluationError("Phase 4.1 candidate audit does not match frozen chunks.")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Unable to read candidate audit {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"Candidate audit must contain a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary.replace(path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _print_summary(report: dict[str, Any], output: Path) -> None:
    overall = report["overall"]
    print(f"{report['strategy']} evaluation")
    print(f"Hit@5={overall['hit_rate_at_5']:.3f} MRR@5={overall['mrr_at_5']:.3f}")
    print(f"Hit@20={overall['hit_rate_at_20']:.3f} MRR@20={overall['mrr_at_20']:.3f}")
    print(f"Candidate recall={overall['candidate_recall']:.3f}")
    print(f"Warm total p95={overall['stage_latency_ms']['total']['p95']:.2f} ms")
    print(f"Report: {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 cross-encoder reranking.")
    parser.add_argument("--strategy", choices=("sparse", "hybrid", "union", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Rebuild comparison from existing strategy artifacts without loading models.",
    )
    return parser


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
