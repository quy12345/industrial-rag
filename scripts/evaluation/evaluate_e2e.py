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

from app.application.generation_prompt import build_correction_text, format_evidence
from app.application.query_service import QueryExecution, QueryService
from app.application.reranking_service import CANDIDATE_TEXT_FORMAT
from app.config import Settings
from app.domain.evidence import EvidenceGate
from app.domain.retrieval_contracts import ATV320_RETRIEVAL_CONTRACT
from app.infrastructure.corpus_artifacts import (
    chunk_set_metadata,
    load_frozen_chunks,
    write_json_atomic,
    write_jsonl_atomic,
)
from app.infrastructure.generation.endpoints import generation_endpoint_identity
from app.infrastructure.generation.langchain_structured import LangChainStructuredGenerator
from evaluation.dataset import (
    EvaluationItem,
    dataset_sha256,
    read_dataset,
    validate_dataset,
    validate_evaluation_manifest,
)
from evaluation.e2e import (
    CLAIM_COMPLETENESS_EVALUATOR_ID,
    FACT_EVALUATOR_ID,
    aggregate_repeated_query_records,
    score_query_execution,
)
from evaluation.generation_experiments import (
    BASELINE_GENERATION_VARIANT,
    GENERATION_EXPERIMENTS,
    GenerationExperiment,
    generation_experiment_for,
    generation_experiment_identity,
)
from evaluation.retrieval import build_corpus_exact_content_equivalence
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
DEVELOPMENT_PROVIDER_APPROVAL_TOKEN = "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS"
CONFIRMATION_PROVIDER_APPROVAL_TOKEN = "APPROVE ATV320 CONFIRMATION GENERATOR EGRESS"
CALIBRATION_SEMANTIC_APPROVAL_TOKEN = (
    "APPROVE ATV320 CALIBRATION OPENROUTER RAGAS EGRESS"
)
HELDOUT_SEMANTIC_APPROVAL_TOKEN = "APPROVE ATV320 HELDOUT OPENROUTER RAGAS EGRESS"
DEVELOPMENT_SEMANTIC_APPROVAL_TOKEN = "APPROVE ATV320 DEVELOPMENT OPENROUTER RAGAS EGRESS"
CONFIRMATION_SEMANTIC_APPROVAL_TOKEN = "APPROVE ATV320 CONFIRMATION OPENROUTER RAGAS EGRESS"
ACTIVE_CALIBRATION_PATH = Path("data/eval/phase7/calibration-v3.jsonl")
ACTIVE_MANIFEST_PATH = Path("artifacts/metrics/phase-7-evaluation-manifest-v3.json")
ACTIVE_DEVELOPMENT_PATH = Path("data/eval/atv320/development-v1.jsonl")
ACTIVE_DEVELOPMENT_MANIFEST_PATH = Path(
    "data/eval/atv320/development-v1-manifest.json"
)
PRIVATE_CONFIRMATION_ROOT = Path("data/eval/atv320/private-confirmation-v1")
PRIVATE_CONFIRMATION_DATASET_PATH = PRIVATE_CONFIRMATION_ROOT / "confirmation-v1.jsonl"
PRIVATE_CONFIRMATION_MANIFEST_PATH = PRIVATE_CONFIRMATION_ROOT / "confirmation-v1-manifest.json"
PRIVATE_HELDOUT_ROOT = Path("data/eval/phase7/private-heldout-v2")
PRIVATE_HELDOUT_DATASET_PATH = PRIVATE_HELDOUT_ROOT / "heldout-v2.jsonl"
PRIVATE_HELDOUT_MANIFEST_PATH = PRIVATE_HELDOUT_ROOT / "heldout-v2-manifest.json"
ARTIFACT_SCHEMA_VERSION = 11
SOURCE_IDENTITY_VERSION = 7
SOURCE_IDENTITY_PATHS = {
    "citation_policy": Path("app/domain/citations.py"),
    "content_identity": Path("app/domain/content_identity.py"),
    "dataset_contract": Path("evaluation/dataset.py"),
    "dense_search_adapter": Path("app/infrastructure/qdrant/dense.py"),
    "evaluator": Path("evaluation/e2e.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_e2e.py"),
    "evidence_policy": Path("app/domain/evidence.py"),
    "fusion_policy": Path("app/domain/policies/fusion.py"),
    "generation_adapter": Path("app/infrastructure/generation/langchain_structured.py"),
    "generation_endpoints": Path("app/infrastructure/generation/endpoints.py"),
    "generation_contract": Path("app/domain/generation.py"),
    "generation_experiments": Path("evaluation/generation_experiments.py"),
    "generation_prompt_policy": Path("app/application/generation_prompt.py"),
    "list_completeness_policy": Path("app/domain/policies/list_completeness.py"),
    "query_analysis_policy": Path("app/domain/policies/query_analysis.py"),
    "query_role_policy": Path("app/domain/policies/query_roles.py"),
    "query_service": Path("app/application/query_service.py"),
    "retrieval_types": Path("app/domain/retrieval.py"),
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
    experiment = generation_experiment_for(args.generation_variant)
    chunks = load_frozen_chunks(args.chunks)
    dataset, validation, manifest = _load_selected_dataset(args, chunks)
    selected = list(dataset)
    if args.item_id is not None:
        selected = [item for item in selected if item.id == args.item_id]
        if not selected:
            raise SystemExit(f"Diagnostic item ID was not found: {args.item_id}")
    if args.max_queries is not None:
        selected = selected[: args.max_queries]

    settings = _runtime_settings(Settings())
    _validate_semantic_approval(args)
    semantic_judge = _build_semantic_judge(args)

    from app.composition.retrieval import (
        LazyQueryRetriever,
        build_query_retriever,
        validate_frozen_runtime,
    )
    from app.infrastructure.qdrant.client import create_qdrant_client

    client = create_qdrant_client(settings)
    validate_frozen_runtime(
        client,
        collection_names=(settings.qdrant_collection, settings.qdrant_hybrid_collection),
        contract=ATV320_RETRIEVAL_CONTRACT,
    )
    service = QueryService(
        retriever=LazyQueryRetriever(
            lambda: build_query_retriever(
                settings,
                contract=ATV320_RETRIEVAL_CONTRACT,
                capture_trace=True,
            )
        ),
        evidence_gate=EvidenceGate(score_threshold=settings.evidence_score_threshold),
        generator=_build_generation_adapter(settings, experiment),
        settings=settings,
    )
    full_dataset_hash = dataset_sha256(dataset)
    exact_content_equivalence = build_corpus_exact_content_equivalence(chunks)
    generation_configuration = _generation_configuration(settings, experiment)
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
        "generation_provider": generation_configuration["provider"],
        "generation_endpoint_profile": settings.generation_endpoint_profile,
        "generation_model": settings.generation_model,
        "generation_configuration": generation_configuration,
        "generation_experiment": generation_experiment_identity(experiment),
        "answer_boundary_configuration": _answer_boundary_configuration(
            settings,
            top_k=args.top_k,
        ),
        "repeat_schedule": {
            "count": args.repeat_count,
            "ordering": "repeat-major-dataset-order",
        },
        "semantic_evaluation": semantic_run_identity(args.semantic_judge),
        "source_identity": _source_identity(experiment),
        "runtime_profile": _runtime_profile(),
        "libraries": _library_versions(semantic_mode=args.semantic_judge),
        "sealed_comparison_dataset_sha256": _sealed_comparison_dataset_sha256(
            manifest, args.dataset
        ),
        "deduplicate_exact_content": settings.rerank_deduplicate_content,
    }
    records = _load_checkpoint(args.checkpoint, identity)
    schedule = _execution_schedule(selected, repeat_count=args.repeat_count)
    _validate_checkpoint_prefix(records, schedule)
    observed_keys = [_checkpoint_key(record) for record in records]
    completed_keys = set(observed_keys)
    for repeat_index, item in schedule:
        if (item.id, repeat_index) in completed_keys:
            continue
        try:
            execution = service.execute(
                question=item.question,
                document_id=None,
                top_k=args.top_k,
            )
            record = score_query_execution(
                item,
                execution,
                exact_content_equivalence=exact_content_equivalence,
            )
            record["repeat_index"] = repeat_index
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

    if len(records) != len(schedule):
        raise RuntimeError("Checkpoint result count does not match the selected dataset.")
    complete_evaluation = len(selected) == len(dataset)
    run_aggregate = _aggregate_run_records(
        records,
        repeat_count=args.repeat_count,
        complete_evaluation=complete_evaluation,
    )
    overall = run_aggregate["overall"]
    repeated_runs = run_aggregate["repeated_runs"]
    semantic_evaluation = (
        {"status": "disabled", "run_identity": semantic_run_identity("none")}
        if semantic_judge is None
        else {
            "status": "complete",
            "run_identity": semantic_run_identity("ragas"),
            "aggregate": aggregate_semantic_records(records),
        }
    )
    quality_gates = run_aggregate["quality_gates"]
    output = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "dataset_validation": validation,
        "overall": overall,
        "repeated_runs": repeated_runs,
        "semantic_evaluation": semantic_evaluation,
        "quality_gates": quality_gates,
        "evaluation_methodology": {
            "headline_answer_metric": "deterministic_fact_accuracy_when_answered",
            "fact_evaluator_id": FACT_EVALUATOR_ID,
            "claim_completeness_evaluator_id": CLAIM_COMPLETENESS_EVALUATOR_ID,
            "generation_variant": experiment.identifier,
            "repeat_schedule": "repeat-major-dataset-order",
            "repeat_aggregation": "per_run_only_not_pooled",
            "strict_phrase_accuracy": "diagnostic_only",
            "token_coverage": "diagnostic_only",
            "retrieval_relevance": "stable relevant_chunk_ids only",
            "exact_content_equivalence": (
                "diagnostic_only_corpus_wide_exact_normalized_content"
            ),
            "retrieval_trace": "sanitized_ids_counts_ranks_and_allowlisted_role_signals",
        },
        "per_query": records,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "evidence_text": "excluded",
            "provider_response": "excluded",
            "judge_reasoning": "excluded",
            "expected_fact_text": "excluded",
            "candidate_text_and_unlisted_metadata": "excluded",
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
        f"variant {experiment.identifier} r{args.repeat_count}; "
        f"quality gates {quality_status}: {args.output}"
    )
    return 0 if quality_passed else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("calibration", "development", "confirmation", "test", "heldout-v2"),
        required=True,
    )
    parser.add_argument("--calibration", type=Path, default=ACTIVE_CALIBRATION_PATH)
    parser.add_argument("--development", type=Path, default=ACTIVE_DEVELOPMENT_PATH)
    parser.add_argument(
        "--confirmation", type=Path, default=PRIVATE_CONFIRMATION_DATASET_PATH
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument(
        "--heldout-v2", type=Path, default=PRIVATE_HELDOUT_DATASET_PATH
    )
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument(
        "--generation-variant",
        choices=tuple(GENERATION_EXPERIMENTS),
        default=BASELINE_GENERATION_VARIANT,
        help="Preregistered generation contract used only by this evaluation run.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        choices=(1, 3),
        default=1,
        help="Evaluate once, or repeat the complete development split three times.",
    )
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
        help=(
            "Calibration/development item ID for a sanitized targeted diagnostic; "
            "development requires explicit output and checkpoint paths."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    if args.item_id is not None and args.dataset not in {"calibration", "development"}:
        parser.error("--item-id is permitted only for calibration or development diagnostics")
    if args.item_id is not None and args.max_queries is not None:
        parser.error("--item-id and --max-queries are mutually exclusive")
    if (
        args.generation_variant != BASELINE_GENERATION_VARIANT
        and args.dataset != "development"
    ):
        parser.error("Experimental generation variants are development-only")
    if args.repeat_count == 3 and args.dataset != "development":
        parser.error("--repeat-count 3 is permitted only for the development dataset")
    if args.repeat_count == 3 and (
        args.item_id is not None or args.max_queries is not None
    ):
        parser.error("--repeat-count 3 requires the complete development dataset")
    if args.repeat_count == 3 and args.semantic_judge != "none":
        parser.error("Semantic evaluation cannot be combined with repeated generation runs")
    if args.dataset in {"confirmation", "heldout-v2"} and args.max_queries is not None:
        parser.error("--max-queries is not permitted for sealed evaluation datasets")
    if (
        args.dataset == "development"
        and (args.item_id is not None or args.max_queries is not None)
        and (args.output is None or args.checkpoint is None)
    ):
        parser.error(
            "Partial development diagnostics require explicit --output and --checkpoint paths"
        )
    if args.manifest is None:
        args.manifest = {
            "development": ACTIVE_DEVELOPMENT_MANIFEST_PATH,
            "confirmation": PRIVATE_CONFIRMATION_MANIFEST_PATH,
            "heldout-v2": PRIVATE_HELDOUT_MANIFEST_PATH,
        }.get(args.dataset, ACTIVE_MANIFEST_PATH)
    if args.output is None:
        args.output = Path(
            f"artifacts/metrics/atv320-{args.dataset}-{args.generation_variant}"
            f"-r{args.repeat_count}-e2e-v{ARTIFACT_SCHEMA_VERSION}.json"
        )
    if args.checkpoint is None:
        args.checkpoint = Path(
            f"artifacts/metrics/atv320-{args.dataset}-{args.generation_variant}"
            f"-r{args.repeat_count}-e2e-v{ARTIFACT_SCHEMA_VERSION}"
            "-checkpoint.jsonl"
        )
    return args


def _validate_execution_approval(args: argparse.Namespace) -> None:
    expected = {
        "calibration": CALIBRATION_PROVIDER_APPROVAL_TOKEN,
        "development": DEVELOPMENT_PROVIDER_APPROVAL_TOKEN,
        "confirmation": CONFIRMATION_PROVIDER_APPROVAL_TOKEN,
        "test": HELDOUT_PROVIDER_APPROVAL_TOKEN,
        "heldout-v2": HELDOUT_PROVIDER_APPROVAL_TOKEN,
    }[args.dataset]
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
    expected = {
        "calibration": CALIBRATION_SEMANTIC_APPROVAL_TOKEN,
        "development": DEVELOPMENT_SEMANTIC_APPROVAL_TOKEN,
        "confirmation": CONFIRMATION_SEMANTIC_APPROVAL_TOKEN,
        "test": HELDOUT_SEMANTIC_APPROVAL_TOKEN,
        "heldout-v2": HELDOUT_SEMANTIC_APPROVAL_TOKEN,
    }[args.dataset]
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
    """Load exactly one approved split without opening its sealed counterpart."""

    if args.dataset == "calibration":
        dataset_path = args.calibration
        validation_kind = "calibration"
        dataset_key = "calibration_dataset_sha256"
    elif args.dataset == "development":
        dataset_path = args.development
        validation_kind = "development"
        dataset_key = "development_dataset_sha256"
    elif args.dataset == "confirmation":
        dataset_path = args.confirmation
        validation_kind = "confirmation"
        dataset_key = "confirmation_dataset_sha256"
        _require_private_confirmation_path(dataset_path)
        _require_private_confirmation_path(args.manifest)
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
    manifest = validate_evaluation_manifest(
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


def _require_private_confirmation_path(path: Path) -> None:
    try:
        path.resolve().relative_to(PRIVATE_CONFIRMATION_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            "Confirmation input and manifest must remain under "
            f"{PRIVATE_CONFIRMATION_ROOT}."
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


def _build_generation_adapter(
    settings: Settings,
    experiment: GenerationExperiment,
) -> LangChainStructuredGenerator:
    """Compose the selected evaluation-only generation contract without model I/O."""

    return LangChainStructuredGenerator(
        settings,
        system_prompt=experiment.system_prompt,
        human_prompt=experiment.human_prompt,
        correction_text_builder=build_correction_text,
        output_schema=experiment.output_schema,
        output_mapper=experiment.output_mapper,
    )


def _sealed_comparison_dataset_sha256(
    manifest: Mapping[str, Any], dataset: str
) -> str:
    key = {
        "calibration": "test_dataset_sha256",
        "development": "confirmation_dataset_sha256",
        "confirmation": "development_dataset_sha256",
        "test": "test_dataset_sha256",
        "heldout-v2": "heldout_dataset_sha256",
    }[dataset]
    value = manifest.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"ATV320 manifest has no valid sealed comparison hash: {key}")
    return value


def _generation_configuration(
    settings: Settings,
    experiment: GenerationExperiment,
) -> dict[str, Any]:
    return {
        **generation_endpoint_identity(settings),
        "structured_output": {
            "schema": experiment.output_schema.__name__,
            "method": "json_schema",
            "strict": True,
            "include_raw": True,
        },
        "maximum_correction_retries": 1,
    }


def _answer_boundary_configuration(
    settings: Settings,
    *,
    top_k: int,
) -> dict[str, int | float | None]:
    """Seal every resolved setting that changes answer evidence or citations."""

    return {
        "evidence_top_k": top_k,
        "evidence_score_threshold": settings.evidence_score_threshold,
        "generation_max_context_chars": settings.generation_max_context_chars,
        "citation_excerpt_max_chars": settings.citation_excerpt_max_chars,
    }


def _source_identity(
    experiment: GenerationExperiment | None = None,
) -> dict[str, Any]:
    """Hash canonical behavior owners under the explicit v7 provenance schema."""

    selected = experiment or generation_experiment_for(BASELINE_GENERATION_VARIANT)

    return {
        "version": SOURCE_IDENTITY_VERSION,
        "files": {
            name: _file_sha256(path) for name, path in SOURCE_IDENTITY_PATHS.items()
        },
        "system_prompt_sha256": hashlib.sha256(
            selected.system_prompt.encode("utf-8")
        ).hexdigest(),
    }


def _aggregate_run_records(
    records: list[dict[str, Any]],
    *,
    repeat_count: int,
    complete_evaluation: bool,
) -> dict[str, Any]:
    if not complete_evaluation:
        return {
            "overall": None,
            "repeated_runs": None,
            "quality_gates": {
                "overall_pass": False,
                "status": "NOT_EVALUATED_PARTIAL_SELECTION",
                "gates": {},
            },
        }
    repeated_runs = aggregate_repeated_query_records(
        records,
        repeat_count=repeat_count,
    )
    if repeat_count == 1:
        first_run = repeated_runs["per_repeat"][0]
        return {
            "overall": first_run["overall"],
            "repeated_runs": repeated_runs,
            "quality_gates": first_run["quality_gates"],
        }
    per_repeat = [
        {
            "repeat_index": row["repeat_index"],
            "quality_gates": row["quality_gates"],
        }
        for row in repeated_runs["per_repeat"]
    ]
    return {
        "overall": None,
        "repeated_runs": repeated_runs,
        "quality_gates": {
            "overall_pass": all(
                row["quality_gates"]["overall_pass"] for row in per_repeat
            ),
            "status": "ALL_REPEATS_EVALUATED",
            "selection_policy": "all_repeats_must_pass",
            "per_repeat": per_repeat,
        },
    }


def _runtime_profile() -> dict[str, Any]:
    contract = ATV320_RETRIEVAL_CONTRACT
    profile = contract.fusion_profile
    return {
        "dense_candidate_limit": contract.dense_candidate_limit,
        "sparse_candidate_limit": contract.sparse_candidate_limit,
        "candidate_budget": contract.union_rrf_prune_limit,
        "refill_after_content_dedup": contract.refill_after_content_dedup,
        "diagnostic_component_limit": contract.diagnostic_component_limit,
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
        "pydantic",
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
    keys = [_checkpoint_key(record) for record in records]
    if len(set(keys)) != len(records):
        raise RuntimeError("Existing ATV320 checkpoint has duplicate item/repeat keys.")
    return records


def _write_checkpoint(path: Path, identity: dict[str, Any], records: list[dict[str, Any]]) -> None:
    payload = [{"run_identity": identity}, *records]
    write_jsonl_atomic(path, payload)


def _execution_schedule(
    items: list[EvaluationItem],
    *,
    repeat_count: int,
) -> list[tuple[int, EvaluationItem]]:
    """Return a deterministic repeat-major schedule for resumable provider calls."""

    return [
        (repeat_index, item)
        for repeat_index in range(1, repeat_count + 1)
        for item in items
    ]


def _checkpoint_key(record: Mapping[str, Any]) -> tuple[str, int]:
    if not isinstance(record, Mapping):
        raise RuntimeError("Existing ATV320 checkpoint contains a malformed record.")
    item_id = record.get("id")
    repeat_index = record.get("repeat_index")
    if not isinstance(item_id, str) or not item_id:
        raise RuntimeError("Existing ATV320 checkpoint record has no stable item ID.")
    if type(repeat_index) is not int or repeat_index not in {1, 2, 3}:
        raise RuntimeError(
            "Existing ATV320 checkpoint record has an invalid repeat index."
        )
    return item_id, repeat_index


def _validate_checkpoint_prefix(
    records: list[dict[str, Any]],
    schedule: list[tuple[int, EvaluationItem]],
) -> None:
    expected_keys = [(item.id, repeat_index) for repeat_index, item in schedule]
    observed_keys = [_checkpoint_key(record) for record in records]
    if observed_keys != expected_keys[: len(observed_keys)]:
        raise RuntimeError(
            "Existing ATV320 checkpoint does not match repeat-major dataset order."
        )


if __name__ == "__main__":
    raise SystemExit(main())
