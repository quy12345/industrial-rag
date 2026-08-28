"""Run the approved ATV320 end-to-end evaluation against separate collections.

The output is intentionally sanitized: it records IDs, ranks, aggregate quality,
latency, and token counts, but never questions, manual text, prompts, answers, or
provider responses.  It is an explicit integration CLI and is not part of pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.application.generation_prompt import (
    HUMAN_PROMPT,
    SYSTEM_PROMPT,
    build_correction_text,
    format_evidence,
)
from app.application.query_service import QueryExecution, QueryService
from app.application.reranking_service import CANDIDATE_TEXT_FORMAT
from app.composition.retrieval import (
    LazyQueryRetriever,
    build_query_retriever,
    validate_frozen_runtime,
)
from app.config import Settings
from app.domain.evidence import EvidenceGate
from app.domain.generation import GeneratedAnswer
from app.domain.retrieval_contracts import ATV320_RETRIEVAL_CONTRACT
from app.infrastructure.corpus_artifacts import (
    chunk_set_metadata,
    load_frozen_chunks,
    write_json_atomic,
    write_jsonl_atomic,
)
from app.infrastructure.generation.langchain_structured import LangChainStructuredGenerator
from app.infrastructure.qdrant.client import create_qdrant_client
from evaluation.dataset import (
    EvaluationItem,
    dataset_sha256,
    read_dataset,
    validate_dataset,
)
from evaluation.e2e import (
    FACT_EVALUATOR_ID,
    aggregate_query_records,
    evaluate_quality_gates,
    score_query_execution,
)
from evaluation.semantic import (
    RagasSemanticJudge,
    SemanticConfigurationError,
    SemanticEvaluationInput,
    SemanticJudge,
    SemanticJudgeError,
    SemanticScores,
    aggregate_semantic_records,
    semantic_run_identity,
)

CALIBRATION_PROVIDER_APPROVAL_TOKEN = "APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS"
HELDOUT_PROVIDER_APPROVAL_TOKEN = "APPROVE PHASE 7 HELDOUT PROVIDER EGRESS"
CALIBRATION_SEMANTIC_APPROVAL_TOKEN = (
    "APPROVE ATV320 CALIBRATION OPENROUTER RAGAS EGRESS"
)
HELDOUT_SEMANTIC_APPROVAL_TOKEN = "APPROVE ATV320 HELDOUT OPENROUTER RAGAS EGRESS"
ACTIVE_CALIBRATION_PATH = Path("data/eval/phase7/calibration-v3.jsonl")
ACTIVE_MANIFEST_PATH = Path("artifacts/metrics/phase-7-evaluation-manifest-v3.json")
PRIVATE_HELDOUT_ROOT = Path("data/eval/phase7/private-heldout-v2")
PRIVATE_HELDOUT_DATASET_PATH = PRIVATE_HELDOUT_ROOT / "heldout-v2.jsonl"
PRIVATE_HELDOUT_MANIFEST_PATH = PRIVATE_HELDOUT_ROOT / "heldout-v2-manifest.json"
ARTIFACT_SCHEMA_VERSION = 8
SOURCE_IDENTITY_VERSION = 4
SOURCE_IDENTITY_PATHS = {
    "citation_policy": Path("app/domain/citations.py"),
    "dataset_contract": Path("evaluation/dataset.py"),
    "dense_search_adapter": Path("app/infrastructure/qdrant/dense.py"),
    "evaluator": Path("evaluation/e2e.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_e2e.py"),
    "evidence_policy": Path("app/domain/evidence.py"),
    "fusion_policy": Path("app/domain/policies/fusion.py"),
    "generation_adapter": Path("app/infrastructure/generation/langchain_structured.py"),
    "generation_prompt_policy": Path("app/application/generation_prompt.py"),
    "list_completeness_policy": Path("app/domain/policies/list_completeness.py"),
    "query_analysis_policy": Path("app/domain/policies/query_analysis.py"),
    "query_role_policy": Path("app/domain/policies/query_roles.py"),
    "query_service": Path("app/application/query_service.py"),
    "reranker_adapter": Path("app/infrastructure/models/reranker.py"),
    "reranking_service": Path("app/application/reranking_service.py"),
    "retrieval_metrics": Path("evaluation/retrieval.py"),
    "semantic_evaluator": Path("evaluation/semantic.py"),
    "retrieval_composition": Path("app/composition/retrieval.py"),
    "retrieval_contract": Path("app/domain/retrieval_contracts.py"),
    "search_adapters": Path("app/infrastructure/qdrant/search.py"),
    "sparse_search_adapter": Path("app/infrastructure/qdrant/hybrid.py"),
}


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

    settings = _runtime_settings(Settings())
    _validate_semantic_approval(args)
    semantic_judge = _build_semantic_judge(args)
    client = create_qdrant_client(settings)
    validate_frozen_runtime(
        client,
        collection_names=(settings.qdrant_collection, settings.qdrant_hybrid_collection),
        contract=ATV320_RETRIEVAL_CONTRACT,
    )
    service = QueryService(
        retriever=LazyQueryRetriever(
            lambda: build_query_retriever(settings, contract=ATV320_RETRIEVAL_CONTRACT)
        ),
        evidence_gate=EvidenceGate(score_threshold=settings.evidence_score_threshold),
        generator=LangChainStructuredGenerator(
            settings,
            system_prompt=SYSTEM_PROMPT,
            human_prompt=HUMAN_PROMPT,
            correction_text_builder=build_correction_text,
        ),
        settings=settings,
    )
    full_dataset_hash = dataset_sha256(dataset)
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_sha256": full_dataset_hash,
        "selected_item_ids": [item.id for item in selected],
        "corpus": chunk_set_metadata(chunks),
        "contract_chunk_ids_sha256": ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "strategy": settings.retrieval_strategy,
        "rerank_enabled": settings.rerank_enabled,
        "dense_candidate_limit": settings.dense_candidate_limit,
        "sparse_candidate_limit": settings.sparse_candidate_limit,
        "union_rrf_prune_limit": ATV320_RETRIEVAL_CONTRACT.union_rrf_prune_limit,
        "query_expansion_profile": ATV320_RETRIEVAL_CONTRACT.query_expansion_profile,
        "candidate_text_format": CANDIDATE_TEXT_FORMAT,
        "generation_provider": settings.generation_provider,
        "generation_model": settings.generation_model,
        "generation_configuration": _generation_configuration(settings),
        "semantic_evaluation": semantic_run_identity(args.semantic_judge),
        "source_identity": _source_identity(),
        "runtime_profile": _runtime_profile(),
        "libraries": _library_versions(semantic_mode=args.semantic_judge),
        "held_out_dataset_sha256": manifest.get(
            "heldout_dataset_sha256", manifest.get("test_dataset_sha256")
        ),
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
            record = score_query_execution(item, execution)
            if semantic_judge is not None:
                record["semantic_evaluation"] = _score_semantic_execution(
                    item,
                    execution,
                    judge=semantic_judge,
                    max_context_chars=settings.generation_max_context_chars,
                ).to_payload()
        except SemanticJudgeError as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(str(exc)) from None
        except Exception as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(f"ATV320 execution failed for dataset item {item.id}.") from exc
        records.append(record)
        _write_checkpoint(args.checkpoint, identity, records)

    if len(records) != len(selected):
        raise RuntimeError("Checkpoint result count does not match the selected dataset.")
    complete_evaluation = len(selected) == len(dataset)
    overall = aggregate_query_records(records) if complete_evaluation else None
    semantic_evaluation = (
        {"status": "disabled", "run_identity": semantic_run_identity("none")}
        if semantic_judge is None
        else {
            "status": "complete",
            "run_identity": semantic_run_identity("ragas"),
            "aggregate": aggregate_semantic_records(records),
        }
    )
    quality_gates = (
        evaluate_quality_gates(overall)
        if overall is not None
        else {
            "overall_pass": False,
            "status": "NOT_EVALUATED_PARTIAL_SELECTION",
            "gates": {},
        }
    )
    output = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "dataset_validation": validation,
        "overall": overall,
        "semantic_evaluation": semantic_evaluation,
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
            "judge_reasoning": "excluded",
        },
    }
    write_json_atomic(args.output, output)
    quality_passed = bool(output["quality_gates"]["overall_pass"])
    if not complete_evaluation:
        print(f"ATV320 {args.dataset} partial diagnostic completed: {args.output}")
        return 0
    quality_status = "PASS" if quality_passed else "FAIL"
    print(
        f"ATV320 {args.dataset} E2E evaluation completed; "
        f"quality gates {quality_status}: {args.output}"
    )
    return 0 if quality_passed else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("calibration", "test", "heldout-v2"), required=True
    )
    parser.add_argument("--calibration", type=Path, default=ACTIVE_CALIBRATION_PATH)
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument(
        "--heldout-v2", type=Path, default=PRIVATE_HELDOUT_DATASET_PATH
    )
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument(
        "--provider-approval-token",
        required=True,
        help="Required exact approval before a question/evidence is sent to a provider.",
    )
    parser.add_argument(
        "--semantic-judge",
        choices=("none", "ragas"),
        default="none",
        help="Optional Ragas semantic scoring through the pinned OpenRouter gateway.",
    )
    parser.add_argument(
        "--judge-provider-approval-token",
        default=None,
        help="Separate exact approval for OpenRouter semantic-judge egress.",
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
    if args.manifest is None:
        args.manifest = (
            PRIVATE_HELDOUT_MANIFEST_PATH
            if args.dataset == "heldout-v2"
            else ACTIVE_MANIFEST_PATH
        )
    if args.output is None:
        args.output = Path(f"artifacts/metrics/atv320-{args.dataset}-e2e-v8.json")
    if args.checkpoint is None:
        args.checkpoint = Path(
            f"artifacts/metrics/atv320-{args.dataset}-e2e-v8-checkpoint.jsonl"
        )
    return args


def _validate_execution_approval(args: argparse.Namespace) -> None:
    if args.dataset == "calibration":
        expected = CALIBRATION_PROVIDER_APPROVAL_TOKEN
    else:
        expected = HELDOUT_PROVIDER_APPROVAL_TOKEN
    if args.provider_approval_token != expected:
        raise SystemExit("ATV320 provider approval token is missing or invalid for this dataset.")


def _validate_semantic_approval(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    if args.semantic_judge == "none":
        if args.judge_provider_approval_token is not None:
            raise SystemExit(
                "Semantic approval token was supplied while semantic evaluation is disabled."
            )
        return
    expected = (
        CALIBRATION_SEMANTIC_APPROVAL_TOKEN
        if args.dataset == "calibration"
        else HELDOUT_SEMANTIC_APPROVAL_TOKEN
    )
    if args.judge_provider_approval_token != expected:
        raise SystemExit(
            "OpenRouter semantic-judge approval token is missing or invalid for this dataset."
        )
    if _openrouter_api_key(environ) is None:
        raise SystemExit("OPENROUTER_API_KEY is required for Ragas semantic evaluation.")


def _build_semantic_judge(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> SemanticJudge | None:
    if args.semantic_judge == "none":
        return None
    api_key = _openrouter_api_key(environ)
    if api_key is None:
        raise SemanticConfigurationError(
            "OPENROUTER_API_KEY is required for Ragas semantic evaluation."
        )
    return RagasSemanticJudge(api_key=api_key)


def _openrouter_api_key(environ: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if environ is None else environ
    value = source.get("OPENROUTER_API_KEY", "").strip()
    return value or None


def _score_semantic_execution(
    item: EvaluationItem,
    execution: QueryExecution,
    *,
    judge: SemanticJudge,
    max_context_chars: int,
) -> SemanticScores:
    if execution.response.abstained:
        return SemanticScores.not_applicable_abstained()
    candidates = execution.evidence_candidates
    if not candidates:
        raise SemanticJudgeError(
            item_id=item.id,
            metric="input_boundary",
            category="missing_final_evidence",
        )
    evidence = format_evidence(candidates, max_chars=max_context_chars)
    return judge.score(
        SemanticEvaluationInput(
            item_id=item.id,
            question=item.question,
            answer=execution.response.answer,
            rendered_generation_evidence=evidence.text,
            ranked_final_contexts=tuple(candidate.text for candidate in candidates),
        )
    )


def _load_selected_dataset(
    args: argparse.Namespace, chunks: list[Any]
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Load exactly one approved split; calibration never touches the held-out path."""

    if args.dataset == "calibration":
        dataset_path = args.calibration
        validation_kind = "calibration"
        dataset_key = "calibration_dataset_sha256"
    elif args.dataset == "test":
        dataset_path = args.test
        validation_kind = "test"
        dataset_key = "test_dataset_sha256"
    else:
        dataset_path = args.heldout_v2
        validation_kind = "test"
        dataset_key = "heldout_dataset_sha256"
        _require_private_heldout_path(dataset_path)
        _require_private_heldout_path(args.manifest)
    dataset = read_dataset(dataset_path)
    validation = validate_dataset(dataset, chunks, kind=validation_kind)
    if any(item.review_status != "approved" for item in dataset):
        raise SystemExit("ATV320 evaluation requires an explicitly approved dataset.")
    manifest = _validate_evaluation_manifest(
        args.manifest,
        chunks,
        dataset,
        dataset_key=dataset_key,
    )
    return dataset, validation, manifest


def _require_private_heldout_path(path: Path) -> None:
    try:
        path.resolve().relative_to(PRIVATE_HELDOUT_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"Held-out v2 input and manifest must remain under {PRIVATE_HELDOUT_ROOT}."
        ) from exc


def _runtime_settings(settings: Settings) -> Settings:
    """Make the fixed ATV320 runtime selection explicit and artifact-independent."""

    return settings.model_copy(
        update={
            "qdrant_collection": ATV320_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": ATV320_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": ATV320_RETRIEVAL_CONTRACT.bm25_avg_len,
            "dense_candidate_limit": ATV320_RETRIEVAL_CONTRACT.dense_candidate_limit,
            "sparse_candidate_limit": ATV320_RETRIEVAL_CONTRACT.sparse_candidate_limit,
            "rrf_k": ATV320_RETRIEVAL_CONTRACT.rrf_k,
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
    dataset_key: str,
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load ATV320 evaluation manifest: {path}") from exc
    expected = {"corpus": chunk_set_metadata(chunks), dataset_key: dataset_sha256(dataset)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"ATV320 evaluation manifest mismatch: {key}")
    sealed_hash = manifest.get(
        "heldout_dataset_sha256"
        if dataset_key == "heldout_dataset_sha256"
        else "test_dataset_sha256"
    )
    if not isinstance(sealed_hash, str) or len(sealed_hash) != 64:
        raise RuntimeError("ATV320 manifest has no valid sealed test dataset hash.")
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


def _source_identity() -> dict[str, Any]:
    """Hash canonical behavior owners under the explicit v4 provenance schema."""

    return {
        "version": SOURCE_IDENTITY_VERSION,
        "files": {
            name: _file_sha256(path) for name, path in SOURCE_IDENTITY_PATHS.items()
        },
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }


def _runtime_profile() -> dict[str, Any]:
    contract = ATV320_RETRIEVAL_CONTRACT
    profile = contract.fusion_profile
    return {
        "dense_candidate_limit": contract.dense_candidate_limit,
        "sparse_candidate_limit": contract.sparse_candidate_limit,
        "candidate_budget": contract.union_rrf_prune_limit,
        "rrf_k": contract.rrf_k,
        "rerank_batch_size": contract.frozen_rerank_batch_size,
        "rerank_threads": contract.frozen_rerank_threads,
        "fusion_profile": None if profile is None else profile.__dict__,
    }


def _library_versions(*, semantic_mode: str = "none") -> dict[str, str]:
    names = [
        "fastembed",
        "langchain-core",
        "langchain-openai",
        "onnxruntime",
        "qdrant-client",
    ]
    if semantic_mode == "ragas":
        names.extend(("langchain-community", "openai", "ragas"))
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
        raise RuntimeError(f"Malformed ATV320 checkpoint: {path}") from exc
    if header != {"run_identity": identity}:
        raise RuntimeError("Existing ATV320 checkpoint belongs to a different frozen run.")
    if len({record.get("id") for record in records}) != len(records):
        raise RuntimeError("Existing ATV320 checkpoint has duplicate item IDs.")
    return records


def _write_checkpoint(path: Path, identity: dict[str, Any], records: list[dict[str, Any]]) -> None:
    payload = [{"run_identity": identity}, *records]
    write_jsonl_atomic(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
