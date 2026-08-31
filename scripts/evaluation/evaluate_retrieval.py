"""Run provider-free ATV320 retrieval experiments against frozen collections.

Artifacts contain only stable IDs, ranks, timings, and configuration identity.
They never contain questions, chunk text, expected phrases, or model output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.application.reranking_service import CANDIDATE_TEXT_FORMAT
from app.config import Settings
from app.domain.retrieval_contracts import (
    ATV320_RETRIEVAL_CONTRACT,
    FrozenRetrievalContract,
)
from app.infrastructure.corpus_artifacts import (
    load_frozen_chunks,
    write_json_atomic,
    write_jsonl_atomic,
)
from evaluation.dataset import (
    DatasetValidationError,
    EvaluationItem,
    dataset_sha256,
    read_dataset,
    validate_dataset,
    validate_dataset_splits,
    validate_evaluation_manifest,
)
from evaluation.retrieval import (
    DEVELOPMENT_RETRIEVAL_VARIANTS,
    EVIDENCE_SELECTION_POLICY_ID,
    EVIDENCE_SELECTION_TOP_K,
    DevelopmentRetrievalVariant,
    aggregate_retrieval_rows,
    aggregate_retrieval_rows_by_language,
    build_corpus_exact_content_equivalence,
    development_retrieval_variant,
    evaluate_retrieval_quality_gates,
    score_retrieval_execution,
    select_evaluation_evidence,
)

LEGACY_CALIBRATION_PATH = Path("data/eval/phase7/calibration.jsonl")
LEGACY_TEST_PATH = Path("data/eval/phase7/test.jsonl")
DEVELOPMENT_PATH = Path("data/eval/atv320/development-v1.jsonl")
DEVELOPMENT_MANIFEST_PATH = Path("data/eval/atv320/development-v1-manifest.json")
FROZEN_CHUNKS_PATH = Path("artifacts/phase7/frozen-chunks.jsonl")
ARTIFACT_SCHEMA_VERSION = 4
DEFAULT_OUTPUT = Path("artifacts/metrics/atv320-retrieval-evaluation-v4.json")
DEFAULT_CHECKPOINT = Path(
    "artifacts/metrics/atv320-retrieval-evaluation-v4-checkpoint.jsonl"
)
SOURCE_IDENTITY_PATHS = {
    "content_identity": Path("app/domain/content_identity.py"),
    "dataset_contract": Path("evaluation/dataset.py"),
    "dense_search_adapter": Path("app/infrastructure/qdrant/dense.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_retrieval.py"),
    "evidence_policy": Path("app/domain/evidence.py"),
    "fusion_policy": Path("app/domain/policies/fusion.py"),
    "list_completeness_policy": Path("app/domain/policies/list_completeness.py"),
    "query_analysis_policy": Path("app/domain/policies/query_analysis.py"),
    "query_role_policy": Path("app/domain/policies/query_roles.py"),
    "reranker_adapter": Path("app/infrastructure/models/reranker.py"),
    "reranking_service": Path("app/application/reranking_service.py"),
    "retrieval_composition": Path("app/composition/retrieval.py"),
    "retrieval_contract": Path("app/domain/retrieval_contracts.py"),
    "retrieval_metrics": Path("evaluation/retrieval.py"),
    "retrieval_types": Path("app/domain/retrieval.py"),
    "search_adapters": Path("app/infrastructure/qdrant/search.py"),
    "sparse_search_adapter": Path("app/infrastructure/qdrant/hybrid.py"),
}


def main() -> int:
    args = _parse_args()
    variant = _active_retrieval_variant(args.dataset, args.retrieval_variant)
    chunks = load_frozen_chunks(args.chunks)
    try:
        dataset, selected, validation, sealed_hash = _load_selected_dataset(
            args,
            chunks,
        )
    except DatasetValidationError as exc:
        raise SystemExit(f"ATV320 retrieval evaluation input is invalid: {exc}") from None

    contract = variant.apply(ATV320_RETRIEVAL_CONTRACT)
    settings = _runtime_settings(Settings(), contract)
    corpus_equivalence = build_corpus_exact_content_equivalence(chunks)
    runtime_identity = _runtime_identity(contract, variant)
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_sha256": dataset_sha256(dataset),
        "sealed_comparison_dataset_sha256": sealed_hash,
        "selected_item_ids": [item.id for item in selected],
        "corpus": validation["corpus"],
        "contract_chunk_ids_sha256": contract.chunk_ids_sha256,
        "retrieval_variant": variant.identity(),
        "runtime": runtime_identity,
        "source_identity": _source_identity(),
        "libraries": _library_versions(),
        "trace_policy": (
            "sanitized_ids_counts_ranks_role_signals_and_evidence_provenance"
        ),
        "evidence_selection_policy": {
            "id": EVIDENCE_SELECTION_POLICY_ID,
            "top_k": EVIDENCE_SELECTION_TOP_K,
        },
    }
    rows = _load_checkpoint(args.checkpoint, identity)
    completed_ids = {str(row["id"]) for row in rows}

    from app.composition.retrieval import build_query_retriever

    retriever = build_query_retriever(
        settings,
        contract=contract,
        capture_trace=True,
    )
    for item in selected:
        if item.id in completed_ids:
            continue
        execution = retriever.retrieve(item.question, document_id=None)
        evidence_selection = select_evaluation_evidence(
            item.question,
            execution.candidates,
        )
        rows.append(
            score_retrieval_execution(
                item,
                execution,
                corpus_equivalence=corpus_equivalence,
                evidence_selection=evidence_selection,
            )
        )
        _write_checkpoint(args.checkpoint, identity, rows)

    if len(rows) != len(selected):
        raise RuntimeError("Retrieval checkpoint count does not match the selected dataset.")
    overall = aggregate_retrieval_rows(rows)
    per_language = aggregate_retrieval_rows_by_language(rows)
    quality_gates = evaluate_retrieval_quality_gates(
        args.dataset,
        rows=rows,
        overall=overall,
        per_language=per_language,
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": (
            "approved answerable calibration rows only"
            if args.dataset == "legacy"
            else "approved answerable development rows only"
        ),
        "run_identity": identity,
        "dataset_validation": validation,
        "overall": overall,
        "per_language": per_language,
        "quality_gates": quality_gates,
        "quality_gate_scope": "candidate_and_ranked_retrieval_only",
        "evidence_diagnostics_status": "measured_not_gated",
        "per_query": rows,
        "provider_calls": 0,
        "confirmation_queries_executed": 0,
        "held_out_queries_executed": 0,
        "sanitization": {
            "question": "used_in_memory_for_selection_then_excluded",
            "answer": "not_applicable",
            "evidence_text": "excluded",
            "expected_fact_text": "excluded",
            "evidence_provenance": (
                "stable_ids_document_ids_pages_and_heading_counts_only"
            ),
            "unlisted_candidate_metadata": "excluded",
        },
    }
    write_json_atomic(args.output, payload)
    passed = all(quality_gates.values())
    status = "PASS" if passed else "FAIL"
    print(
        f"ATV320 {args.dataset} provider-free retrieval gates {status}: "
        f"{args.output}"
    )
    return 0 if passed else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("legacy", "development"),
        default="legacy",
        help="Use the frozen legacy calibration pair or the inspectable development split.",
    )
    parser.add_argument("--calibration", type=Path, default=LEGACY_CALIBRATION_PATH)
    parser.add_argument("--test", type=Path, default=LEGACY_TEST_PATH)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEVELOPMENT_MANIFEST_PATH)
    parser.add_argument("--chunks", type=Path, default=FROZEN_CHUNKS_PATH)
    parser.add_argument(
        "--retrieval-variant",
        choices=tuple(DEVELOPMENT_RETRIEVAL_VARIANTS),
        default="baseline",
        help="Preregistered retrieval policy; non-baseline choices are development-only.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dataset != "development" and args.retrieval_variant != "baseline":
        parser.error("Non-baseline retrieval variants are permitted only for development")
    if args.output is None:
        args.output = _default_output(args.dataset, args.retrieval_variant)
    if args.checkpoint is None:
        args.checkpoint = _default_checkpoint(args.dataset, args.retrieval_variant)
    return args


def _default_output(dataset: str, variant: str) -> Path:
    if dataset == "legacy":
        return DEFAULT_OUTPUT
    return Path(
        "artifacts/metrics/"
        f"atv320-development-retrieval-{variant}-v{ARTIFACT_SCHEMA_VERSION}.json"
    )


def _default_checkpoint(dataset: str, variant: str) -> Path:
    if dataset == "legacy":
        return DEFAULT_CHECKPOINT
    return Path(
        "artifacts/metrics/"
        f"atv320-development-retrieval-{variant}-v{ARTIFACT_SCHEMA_VERSION}"
        "-checkpoint.jsonl"
    )


def _active_retrieval_variant(
    dataset: str,
    identifier: str,
) -> DevelopmentRetrievalVariant:
    variant = development_retrieval_variant(identifier)
    if dataset == "development":
        return variant
    if dataset != "legacy" or identifier != "baseline":
        raise ValueError("Retrieval variants other than baseline are development-only.")
    return replace(variant, diagnostic_component_limit=None)


def _load_selected_dataset(
    args: argparse.Namespace,
    chunks: list[Any],
) -> tuple[list[EvaluationItem], list[EvaluationItem], dict[str, Any], str]:
    if args.dataset == "legacy":
        calibration = read_dataset(args.calibration)
        test = read_dataset(args.test)
        validation = validate_dataset_splits(calibration, test, chunks)
        return (
            calibration,
            [item for item in calibration if item.answerable],
            validation,
            str(validation["test_dataset_sha256"]),
        )

    development = read_dataset(args.development)
    validation = validate_dataset(development, chunks, kind="development")
    if any(item.review_status != "approved" for item in development):
        raise DatasetValidationError(
            "Development retrieval evaluation requires every item to be approved."
        )
    manifest = validate_evaluation_manifest(
        args.manifest,
        chunks,
        development,
        dataset_key="development_dataset_sha256",
    )
    return (
        development,
        [item for item in development if item.answerable],
        validation,
        str(manifest["confirmation_dataset_sha256"]),
    )


def _runtime_identity(
    contract: FrozenRetrievalContract,
    variant: DevelopmentRetrievalVariant,
) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "dense_collection": contract.dense_collection,
        "hybrid_collection": contract.hybrid_collection,
        "dense_candidate_limit": contract.dense_candidate_limit,
        "sparse_candidate_limit": contract.sparse_candidate_limit,
        "rrf_k": contract.rrf_k,
        "candidate_budget": contract.union_rrf_prune_limit,
        "diagnostic_component_limit": contract.diagnostic_component_limit,
        "dense_model": contract.dense_model,
        "dense_dimension": contract.dense_dimension,
        "sparse_model": contract.sparse_model,
        "bm25_k": contract.bm25_k,
        "bm25_b": contract.bm25_b,
        "bm25_avg_len": contract.bm25_avg_len,
        "bm25_disable_stemmer": contract.bm25_disable_stemmer,
        "reranker": contract.rerank_model,
        "rerank_batch_size": contract.frozen_rerank_batch_size,
        "freeze_rerank_threads": contract.freeze_rerank_threads,
        "rerank_threads": contract.frozen_rerank_threads,
        "deduplicate_exact_content": True,
        "candidate_text_format": CANDIDATE_TEXT_FORMAT,
        "retrieval_variant": variant.identity(),
    }


def _json_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    """Hash every canonical behavior owner used by the v4 evaluator."""

    files = {name: _file_sha256(path) for name, path in SOURCE_IDENTITY_PATHS.items()}
    return {
        "files": files,
        "files_sha256": _json_sha256(files),
    }


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("fastembed", "onnxruntime", "qdrant-client"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    try:
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed ATV320 retrieval checkpoint: {path}") from exc
    if header != {"run_identity": identity}:
        raise RuntimeError(
            "Existing ATV320 retrieval checkpoint belongs to a different frozen run."
        )
    row_ids = [row.get("id") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Existing ATV320 retrieval checkpoint has duplicate item IDs.")
    selected_ids = identity.get("selected_item_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(item_id, str) for item_id in selected_ids
    ):
        raise RuntimeError("ATV320 retrieval run identity has invalid selected item IDs.")
    allowed_ids = set(selected_ids)
    if any(row_id not in allowed_ids for row_id in row_ids):
        raise RuntimeError("Existing ATV320 retrieval checkpoint has an unexpected item ID.")
    return rows


def _write_checkpoint(
    path: Path,
    identity: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    write_jsonl_atomic(path, [{"run_identity": identity}, *rows])


def _runtime_settings(
    settings: Settings,
    contract: FrozenRetrievalContract = ATV320_RETRIEVAL_CONTRACT,
) -> Settings:
    return settings.model_copy(
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
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
