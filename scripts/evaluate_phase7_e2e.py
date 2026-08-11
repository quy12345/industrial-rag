"""Run the approved Phase 7 end-to-end evaluation against separate collections.

The output is intentionally sanitized: it records IDs, ranks, aggregate quality,
latency, and token counts, but never questions, manual text, prompts, answers, or
provider responses.  It is an explicit integration CLI and is not part of pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import Settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.evaluation_e2e import (
    FACT_EVALUATOR_ID,
    aggregate_phase7_records,
    evaluate_phase7_quality_gates,
    score_phase7_execution,
)
from app.generation import SYSTEM_PROMPT, GeneratedAnswer, LangChainOpenAIGenerator
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_dataset,
    write_json_atomic,
)
from app.query_service import EvidenceGate, QueryService
from app.reranking import PHASE7_CANDIDATE_TEXT_FORMAT
from app.retrieval import create_qdrant_client
from app.retrieval_runtime import (
    PHASE7_RETRIEVAL_CONTRACT,
    LazyQueryRetriever,
    build_query_retriever,
    validate_frozen_runtime,
)

CALIBRATION_PROVIDER_APPROVAL_TOKEN = "APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS"
HELDOUT_PROVIDER_APPROVAL_TOKEN = "APPROVE PHASE 7 HELDOUT PROVIDER EGRESS"
HELDOUT_GOVERNANCE_BLOCK = (
    "Phase 7 held-out execution is BLOCKED_GOVERNANCE because historical tracked documentation "
    "and calibration tooling exposed the current split."
)
ACTIVE_CALIBRATION_PATH = Path("data/eval/phase7/calibration-v3.jsonl")
ACTIVE_MANIFEST_PATH = Path("artifacts/metrics/phase-7-evaluation-manifest-v3.json")


def main() -> int:
    args = _parse_args()
    _validate_execution_approval(args)
    chunks = load_frozen_chunks(args.chunks)
    dataset, validation, manifest = _load_selected_dataset(args, chunks)
    selected = list(dataset)
    if args.item_id is not None:
        selected = [item for item in selected if item.id == args.item_id]
        if not selected:
            raise SystemExit(f"Calibration item ID was not found: {args.item_id}")
    if args.max_queries is not None:
        selected = selected[: args.max_queries]

    settings = _phase7_settings(Settings())
    client = create_qdrant_client(settings)
    validate_frozen_runtime(
        client,
        collection_names=(settings.qdrant_collection, settings.qdrant_hybrid_collection),
        contract=PHASE7_RETRIEVAL_CONTRACT,
    )
    service = QueryService(
        retriever=LazyQueryRetriever(
            lambda: build_query_retriever(settings, contract=PHASE7_RETRIEVAL_CONTRACT)
        ),
        evidence_gate=EvidenceGate(score_threshold=settings.evidence_score_threshold),
        generator=LangChainOpenAIGenerator(settings),
        settings=settings,
    )
    full_dataset_hash = dataset_sha256(dataset)
    identity = {
        "dataset": args.dataset,
        "dataset_sha256": full_dataset_hash,
        "selected_item_ids": [item.id for item in selected],
        "corpus": chunk_set_metadata(chunks),
        "contract_chunk_ids_sha256": PHASE7_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "strategy": settings.retrieval_strategy,
        "rerank_enabled": settings.rerank_enabled,
        "dense_candidate_limit": settings.dense_candidate_limit,
        "sparse_candidate_limit": settings.sparse_candidate_limit,
        "union_rrf_prune_limit": PHASE7_RETRIEVAL_CONTRACT.union_rrf_prune_limit,
        "query_expansion_profile": PHASE7_RETRIEVAL_CONTRACT.query_expansion_profile,
        "candidate_text_format": PHASE7_CANDIDATE_TEXT_FORMAT,
        "generation_provider": settings.generation_provider,
        "generation_model": settings.generation_model,
        "generation_configuration": _generation_configuration(settings),
        "source_identity": _source_identity(),
        "runtime_profile": _runtime_profile(),
        "libraries": _library_versions(),
        "held_out_dataset_sha256": manifest.get("test_dataset_sha256"),
        "deduplicate_exact_content": settings.rerank_deduplicate_content,
    }
    records = _load_checkpoint(args.checkpoint, identity)
    completed_ids = {record["id"] for record in records}
    for item in selected:
        if item.id in completed_ids:
            continue
        try:
            execution = service.execute(
                question=item.question,
                document_id=None,
                top_k=args.top_k,
            )
            record = score_phase7_execution(item, execution)
        except Exception as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(f"Phase 7 execution failed for dataset item {item.id}.") from exc
        records.append(record)
        _write_checkpoint(args.checkpoint, identity, records)

    if len(records) != len(selected):
        raise RuntimeError("Checkpoint result count does not match the selected dataset.")
    complete_evaluation = len(selected) == len(dataset)
    overall = aggregate_phase7_records(records) if complete_evaluation else None
    quality_gates = (
        evaluate_phase7_quality_gates(overall)
        if overall is not None
        else {
            "overall_pass": False,
            "status": "NOT_EVALUATED_PARTIAL_SELECTION",
            "gates": {},
        }
    )
    output = {
        "schema_version": 5,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "dataset_validation": validation,
        "overall": overall,
        "quality_gates": quality_gates,
        "evaluation_methodology": {
            "headline_answer_metric": "deterministic_fact_accuracy_when_answered",
            "fact_evaluator_id": FACT_EVALUATOR_ID,
            "strict_phrase_accuracy": "diagnostic_only",
            "token_coverage": "diagnostic_only",
            "retrieval_relevance": "stable relevant_chunk_ids only",
        },
        "per_query": records,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "evidence_text": "excluded",
            "provider_response": "excluded",
        },
    }
    write_json_atomic(args.output, output)
    quality_passed = bool(output["quality_gates"]["overall_pass"])
    if not complete_evaluation:
        print(f"Phase 7 {args.dataset} partial diagnostic completed: {args.output}")
        return 0
    quality_status = "PASS" if quality_passed else "FAIL"
    print(
        f"Phase 7 {args.dataset} E2E evaluation completed; "
        f"quality gates {quality_status}: {args.output}"
    )
    return 0 if quality_passed else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("calibration", "test"), required=True)
    parser.add_argument("--calibration", type=Path, default=ACTIVE_CALIBRATION_PATH)
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--manifest", type=Path, default=ACTIVE_MANIFEST_PATH
    )
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument(
        "--provider-approval-token",
        required=True,
        help="Required exact approval before a question/evidence is sent to a provider.",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--item-id",
        default=None,
        help="Calibration-only exact item ID for a sanitized targeted diagnostic.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    if args.item_id is not None and args.dataset != "calibration":
        parser.error("--item-id is permitted only for calibration diagnostics")
    if args.item_id is not None and args.max_queries is not None:
        parser.error("--item-id and --max-queries are mutually exclusive")
    if args.output is None:
        args.output = Path(
            f"artifacts/metrics/phase-7-{args.dataset}-e2e-v5.json"
        )
    if args.checkpoint is None:
        args.checkpoint = Path(
            f"artifacts/metrics/phase-7-{args.dataset}-e2e-v5-checkpoint.jsonl"
        )
    return args


def _validate_execution_approval(args: argparse.Namespace) -> None:
    if args.dataset == "calibration":
        expected = CALIBRATION_PROVIDER_APPROVAL_TOKEN
    else:
        raise SystemExit(HELDOUT_GOVERNANCE_BLOCK)
    if args.provider_approval_token != expected:
        raise SystemExit("Phase 7 provider approval token is missing or invalid for this dataset.")


def _load_selected_dataset(
    args: argparse.Namespace, chunks: list[Any]
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Load exactly one approved split; calibration never touches the held-out path."""

    dataset_path = args.calibration if args.dataset == "calibration" else args.test
    dataset = read_phase7_dataset(dataset_path)
    validation = validate_phase7_dataset(dataset, chunks, kind=args.dataset)
    if any(item.review_status != "approved" for item in dataset):
        raise SystemExit("Phase 7 evaluation requires an explicitly approved dataset.")
    manifest = _validate_evaluation_manifest(
        args.manifest,
        chunks,
        dataset,
        kind=args.dataset,
    )
    return dataset, validation, manifest


def _phase7_settings(settings: Settings) -> Settings:
    """Make the fixed Phase 7 runtime selection explicit and artifact-independent."""

    return settings.model_copy(
        update={
            "qdrant_collection": PHASE7_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": PHASE7_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
            "dense_candidate_limit": PHASE7_RETRIEVAL_CONTRACT.dense_candidate_limit,
            "sparse_candidate_limit": PHASE7_RETRIEVAL_CONTRACT.sparse_candidate_limit,
            "rrf_k": PHASE7_RETRIEVAL_CONTRACT.rrf_k,
            "retrieval_strategy": "union",
            "rerank_enabled": True,
            "rerank_deduplicate_content": True,
        }
    )


def _validate_evaluation_manifest(
    path: Path,
    chunks: list[Any],
    dataset: list[Any],
    *,
    kind: str,
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load Phase 7 evaluation manifest: {path}") from exc
    dataset_key = (
        "calibration_dataset_sha256" if kind == "calibration" else "test_dataset_sha256"
    )
    expected = {"corpus": chunk_set_metadata(chunks), dataset_key: dataset_sha256(dataset)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Phase 7 evaluation manifest mismatch: {key}")
    sealed_hash = manifest.get("test_dataset_sha256")
    if not isinstance(sealed_hash, str) or len(sealed_hash) != 64:
        raise RuntimeError("Phase 7 manifest has no valid sealed test dataset hash.")
    return manifest


def _generation_configuration(settings: Settings) -> dict[str, Any]:
    base_url_host = (
        urlparse(settings.gemini_base_url).netloc
        if settings.generation_provider == "gemini"
        else "api.openai.com"
    )
    return {
        "base_url_host": base_url_host,
        "reasoning_effort": (
            settings.gemini_reasoning_effort
            if settings.generation_provider == "gemini"
            else settings.openai_reasoning_effort
        ),
        "temperature": (
            settings.gemini_temperature
            if settings.generation_provider == "gemini"
            else None
        ),
        "maximum_output_tokens": settings.openai_max_output_tokens,
        "timeout_seconds": settings.openai_timeout_seconds,
        "provider_max_retries": settings.openai_max_retries,
        "store": settings.openai_store,
        "structured_output": {
            "schema": GeneratedAnswer.__name__,
            "method": "json_schema",
            "strict": True,
            "include_raw": True,
        },
        "maximum_correction_retries": 1,
    }


def _source_identity() -> dict[str, str]:
    paths = {
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
    identity = {name: _file_sha256(path) for name, path in paths.items()}
    identity["system_prompt_sha256"] = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    return identity


def _runtime_profile() -> dict[str, Any]:
    contract = PHASE7_RETRIEVAL_CONTRACT
    profile = contract.phase7_fusion_profile
    return {
        "dense_candidate_limit": contract.dense_candidate_limit,
        "sparse_candidate_limit": contract.sparse_candidate_limit,
        "candidate_budget": contract.union_rrf_prune_limit,
        "rrf_k": contract.rrf_k,
        "rerank_batch_size": contract.frozen_rerank_batch_size,
        "rerank_threads": contract.frozen_rerank_threads,
        "fusion_profile": None if profile is None else profile.__dict__,
    }


def _library_versions() -> dict[str, str]:
    names = (
        "fastembed",
        "langchain-core",
        "langchain-openai",
        "onnxruntime",
        "qdrant-client",
    )
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    try:
        header = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed Phase 7 checkpoint: {path}") from exc
    if header != {"run_identity": identity}:
        raise RuntimeError("Existing Phase 7 checkpoint belongs to a different frozen run.")
    if len({record.get("id") for record in records}) != len(records):
        raise RuntimeError("Existing Phase 7 checkpoint has duplicate item IDs.")
    return records


def _write_checkpoint(path: Path, identity: dict[str, Any], records: list[dict[str, Any]]) -> None:
    payload = [{"run_identity": identity}, *records]
    from app.phase7 import write_jsonl_atomic

    write_jsonl_atomic(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
