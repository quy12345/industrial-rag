"""Run the approval-gated ATV320 E2E Ragas scorecard.

The command writes only compact sanitized schema-v15 artifacts. Raw review data is
allowed only in the ignored ``artifacts/local-review`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.application.generation_prompt import (
    HUMAN_PROMPT,
    SYSTEM_PROMPT,
    build_correction_text,
    format_evidence,
)
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
from evaluation.e2e import build_case_record, build_scorecard
from evaluation.semantic import (
    MetricScore,
    RagasEvaluationInput,
    RagasJudge,
    RagasScorecardJudge,
    RagasScores,
    SemanticConfigurationError,
    SemanticJudgeError,
    render_reviewed_reference,
    semantic_run_identity,
)

DEVELOPMENT_PROVIDER_APPROVAL_TOKEN = "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS"
CONFIRMATION_PROVIDER_APPROVAL_TOKEN = "APPROVE ATV320 CONFIRMATION GENERATOR EGRESS"
HELDOUT_PROVIDER_APPROVAL_TOKEN = "APPROVE PHASE 7 HELDOUT PROVIDER EGRESS"
RAGAS_APPROVAL_TOKENS = {
    "development": "APPROVE ATV320 DEVELOPMENT OPENROUTER RAGAS SCORECARD EGRESS",
    "confirmation": "APPROVE ATV320 CONFIRMATION OPENROUTER RAGAS SCORECARD EGRESS",
    "heldout-v2": "APPROVE ATV320 HELDOUT OPENROUTER RAGAS SCORECARD EGRESS",
}
ACTIVE_DEVELOPMENT_PATH = Path("data/eval/atv320/development-v1.jsonl")
ACTIVE_DEVELOPMENT_MANIFEST_PATH = Path("data/eval/atv320/development-v1-manifest.json")
PRIVATE_CONFIRMATION_ROOT = Path("data/eval/atv320/private-confirmation-v1")
PRIVATE_CONFIRMATION_DATASET_PATH = PRIVATE_CONFIRMATION_ROOT / "confirmation-v1.jsonl"
PRIVATE_CONFIRMATION_MANIFEST_PATH = PRIVATE_CONFIRMATION_ROOT / "confirmation-v1-manifest.json"
PRIVATE_HELDOUT_ROOT = Path("data/eval/phase7/private-heldout-v2")
PRIVATE_HELDOUT_DATASET_PATH = PRIVATE_HELDOUT_ROOT / "heldout-v2.jsonl"
PRIVATE_HELDOUT_MANIFEST_PATH = PRIVATE_HELDOUT_ROOT / "heldout-v2-manifest.json"
FROZEN_CHUNKS_PATH = Path("artifacts/phase7/frozen-chunks.jsonl")
ARTIFACT_SCHEMA_VERSION = 15
SOURCE_IDENTITY_VERSION = 11
SOURCE_IDENTITY_PATHS = {
    "dataset_contract": Path("evaluation/dataset.py"),
    "evaluator": Path("evaluation/e2e.py"),
    "semantic_evaluator": Path("evaluation/semantic.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_e2e.py"),
    "generation_prompt": Path("app/application/generation_prompt.py"),
    "query_service": Path("app/application/query_service.py"),
    "retrieval_composition": Path("app/composition/retrieval.py"),
    "retrieval_contract": Path("app/domain/retrieval_contracts.py"),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_approvals(args)
    chunks = load_frozen_chunks(FROZEN_CHUNKS_PATH)
    dataset, validation, manifest = _load_dataset(args.dataset, chunks)
    selected = _select_items(dataset, args)
    settings = _runtime_settings(Settings())
    judge = _build_judge(args)

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
                capture_trace=False,
            )
        ),
        evidence_gate=EvidenceGate(score_threshold=settings.evidence_score_threshold),
        generator=_build_generation_adapter(settings),
        settings=settings,
    )
    identity = _run_identity(
        args=args,
        dataset=dataset,
        selected=selected,
        chunks=chunks,
        settings=settings,
        manifest=manifest,
    )
    records = _load_checkpoint(args.checkpoint, identity)
    _validate_checkpoint_prefix(records, selected)
    completed_ids = {str(record["id"]) for record in records}
    for item in selected:
        if item.id in completed_ids:
            continue
        try:
            execution = service.execute(question=item.question, document_id=None, top_k=5)
            scores = _score_execution(
                item,
                execution,
                judge=judge,
                include_diagnostics=args.ragas_diagnostics,
                max_context_chars=settings.generation_max_context_chars,
            )
            records.append(build_case_record(item, execution, scores))
            _write_checkpoint(args.checkpoint, identity, records)
        except SemanticJudgeError as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(str(exc)) from None
        except Exception as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(f"ATV320 execution failed for dataset item {item.id}.") from exc

    if len(records) != len(selected):
        raise RuntimeError("Checkpoint result count does not match selected items.")
    output = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "scorecard": build_scorecard(records),
        "cases": records,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "reference_facts": "excluded",
            "evidence_text": "excluded",
            "provider_response": "excluded",
            "judge_reasoning": "excluded",
            "retrieval_trace": "excluded_use_evaluate_retrieval",
        },
    }
    write_json_atomic(args.output, output)
    print(f"ATV320 {args.dataset} Ragas scorecard completed: {args.output}")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("development", "confirmation", "heldout-v2"), required=True
    )
    parser.add_argument("--provider-approval-token", required=True)
    parser.add_argument("--judge-provider-approval-token", required=True)
    parser.add_argument("--ragas-diagnostics", action="store_true")
    parser.add_argument("--review-cohort", choices=("judge-validation-v1",))
    parser.add_argument("--review-item-id", action="append", default=[])
    parser.add_argument("--local-review-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.review_cohort and args.review_item_id:
        parser.error("--review-cohort and --review-item-id are mutually exclusive")
    if (args.review_cohort or args.review_item_id) and args.dataset != "development":
        parser.error("Review selection is permitted only for development")
    if len(args.review_item_id) > 40 or len(set(args.review_item_id)) != len(args.review_item_id):
        parser.error("Targeted review requires at most 40 unique item IDs")
    partial = bool(args.review_cohort or args.review_item_id)
    if partial and args.local_review_output is None:
        parser.error("Review selection requires --local-review-output")
    if args.local_review_output is not None and not partial:
        parser.error("--local-review-output requires a review selection")
    if partial and (args.output is None or args.checkpoint is None):
        parser.error("Partial review requires explicit --output and --checkpoint")
    if args.local_review_output is not None:
        _require_private_review_path(args.local_review_output)
    if args.output is None:
        args.output = Path(f"artifacts/metrics/atv320-{args.dataset}-e2e-v15.json")
    if args.checkpoint is None:
        args.checkpoint = Path(f"artifacts/metrics/atv320-{args.dataset}-e2e-v15-checkpoint.jsonl")
    return args


def _validate_approvals(
    args: argparse.Namespace, *, environ: Mapping[str, str] | None = None
) -> None:
    expected_generator = {
        "development": DEVELOPMENT_PROVIDER_APPROVAL_TOKEN,
        "confirmation": CONFIRMATION_PROVIDER_APPROVAL_TOKEN,
        "heldout-v2": HELDOUT_PROVIDER_APPROVAL_TOKEN,
    }[args.dataset]
    if args.provider_approval_token != expected_generator:
        raise SystemExit("Generator approval token is missing or invalid for this dataset.")
    if args.judge_provider_approval_token != RAGAS_APPROVAL_TOKENS[args.dataset]:
        raise SystemExit("Ragas scorecard approval token is missing or invalid for this dataset.")
    if _openrouter_api_key(environ) is None:
        raise SystemExit("OPENROUTER_API_KEY is required for the Ragas scorecard.")


def _build_judge(
    args: argparse.Namespace, *, environ: Mapping[str, str] | None = None
) -> RagasJudge:
    api_key = _openrouter_api_key(environ)
    if api_key is None:
        raise SemanticConfigurationError("OPENROUTER_API_KEY is required for the Ragas scorecard.")
    return RagasScorecardJudge(api_key=api_key, include_diagnostics=args.ragas_diagnostics)


def _score_execution(
    item: EvaluationItem,
    execution: QueryExecution,
    *,
    judge: RagasJudge,
    include_diagnostics: bool,
    max_context_chars: int,
) -> RagasScores:
    not_applicable = MetricScore.not_applicable()
    if not item.answerable:
        return RagasScores(*(not_applicable for _ in range(5)))

    generator_saw_evidence = execution.evidence_context_chars > 0
    generation_context = ""
    if generator_saw_evidence and execution.evidence_candidates:
        generation_context = format_evidence(
            execution.evidence_candidates, max_chars=max_context_chars
        ).text
    answered = not execution.response.abstained
    citation_ids = [citation.chunk_id for citation in execution.response.citations]
    candidates_by_id = {row.chunk_id: row for row in execution.evidence_candidates}
    citations_valid = bool(citation_ids) and set(citation_ids) <= set(candidates_by_id)
    cited_contexts = tuple(
        candidates_by_id[chunk_id].text
        for chunk_id in dict.fromkeys(citation_ids)
        if chunk_id in candidates_by_id
    )
    if not generation_context:
        return RagasScores(
            not_applicable,
            MetricScore.deterministic_zero("invalid_citations") if answered else not_applicable,
            MetricScore.deterministic_zero("empty_context"),
            not_applicable,
            not_applicable,
        )

    scores = judge.score(
        RagasEvaluationInput(
            item_id=item.id,
            question=item.question,
            answer=execution.response.answer if answered else "",
            reference=render_reviewed_reference(item),
            cited_contexts=cited_contexts,
            generation_context=generation_context,
            final_contexts=tuple(row.text for row in execution.evidence_candidates),
        ),
        score_factual_correctness=answered,
        score_faithfulness=answered and citations_valid,
        score_context_recall=True,
        include_diagnostics=include_diagnostics,
    )
    if answered and not citations_valid:
        scores = replace(
            scores,
            faithfulness=MetricScore.deterministic_zero("invalid_citations"),
        )
    return scores


def _select_items(items: list[EvaluationItem], args: argparse.Namespace) -> list[EvaluationItem]:
    if args.review_cohort or args.review_item_id:
        raise SystemExit("Private judge-validation review is implemented in E04 Batch 2.")
    return items


def _load_dataset(
    dataset_name: str, chunks: list[Any]
) -> tuple[list[EvaluationItem], dict[str, Any], dict[str, Any]]:
    paths = {
        "development": (
            ACTIVE_DEVELOPMENT_PATH,
            ACTIVE_DEVELOPMENT_MANIFEST_PATH,
            "development",
            "development_dataset_sha256",
        ),
        "confirmation": (
            PRIVATE_CONFIRMATION_DATASET_PATH,
            PRIVATE_CONFIRMATION_MANIFEST_PATH,
            "confirmation",
            "confirmation_dataset_sha256",
        ),
        "heldout-v2": (
            PRIVATE_HELDOUT_DATASET_PATH,
            PRIVATE_HELDOUT_MANIFEST_PATH,
            "test",
            "heldout_dataset_sha256",
        ),
    }
    dataset_path, manifest_path, validation_kind, hash_key = paths[dataset_name]
    if dataset_name == "confirmation":
        _require_under(dataset_path, PRIVATE_CONFIRMATION_ROOT, "Confirmation")
        _require_under(manifest_path, PRIVATE_CONFIRMATION_ROOT, "Confirmation")
    if dataset_name == "heldout-v2":
        _require_under(dataset_path, PRIVATE_HELDOUT_ROOT, "Held-out v2")
        _require_under(manifest_path, PRIVATE_HELDOUT_ROOT, "Held-out v2")
    items = read_dataset(dataset_path)
    validation = validate_dataset(items, chunks, kind=validation_kind)
    if any(item.review_status != "approved" for item in items):
        raise SystemExit("ATV320 evaluation requires an approved dataset.")
    manifest = validate_evaluation_manifest(manifest_path, chunks, items, dataset_key=hash_key)
    return items, validation, manifest


def _runtime_settings(settings: Settings) -> Settings:
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


def _build_generation_adapter(settings: Settings) -> LangChainStructuredGenerator:
    return LangChainStructuredGenerator(
        settings,
        system_prompt=SYSTEM_PROMPT,
        human_prompt=HUMAN_PROMPT,
        correction_text_builder=build_correction_text,
    )


def _run_identity(
    *,
    args: argparse.Namespace,
    dataset: list[EvaluationItem],
    selected: list[EvaluationItem],
    chunks: list[Any],
    settings: Settings,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    generation = generation_endpoint_identity(settings)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_sha256": dataset_sha256(dataset),
        "selected_item_ids": [item.id for item in selected],
        "corpus": chunk_set_metadata(chunks),
        "contract_chunk_ids_sha256": ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "candidate_text_format": CANDIDATE_TEXT_FORMAT,
        "evidence_top_k": 5,
        "generation": generation,
        "generation_max_context_chars": settings.generation_max_context_chars,
        "ragas": semantic_run_identity(include_diagnostics=args.ragas_diagnostics),
        "source_identity": _source_identity(),
        "runtime": _runtime_profile(),
        "libraries": _library_versions(),
        "sealed_comparison_dataset_sha256": _sealed_comparison_hash(manifest, args.dataset),
        "review": {
            "cohort": args.review_cohort,
            "selected_ids_sha256": _selected_ids_hash(selected),
        },
    }


def _source_identity() -> dict[str, Any]:
    return {
        "version": SOURCE_IDENTITY_VERSION,
        "files": {name: _file_sha256(path) for name, path in SOURCE_IDENTITY_PATHS.items()},
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }


def _runtime_profile() -> dict[str, Any]:
    contract = ATV320_RETRIEVAL_CONTRACT
    return {
        "contract_id": contract.contract_id,
        "dense_candidate_limit": contract.dense_candidate_limit,
        "sparse_candidate_limit": contract.sparse_candidate_limit,
        "candidate_budget": contract.union_rrf_prune_limit,
        "capture_trace": False,
    }


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in (
        "fastembed",
        "langchain-core",
        "langchain-openai",
        "onnxruntime",
        "pydantic",
        "qdrant-client",
        "langchain-community",
        "openai",
        "ragas",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0]) if lines else {}
        records = [json.loads(line) for line in lines[1:]]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Malformed ATV320 checkpoint: {path}") from exc
    if header != {"run_identity": identity}:
        raise RuntimeError("Existing ATV320 checkpoint belongs to a different frozen run.")
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Existing ATV320 checkpoint contains duplicate item IDs.")
    return records


def _write_checkpoint(path: Path, identity: dict[str, Any], records: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(path, [{"run_identity": identity}, *records])


def _validate_checkpoint_prefix(
    records: list[dict[str, Any]], selected: list[EvaluationItem]
) -> None:
    if [record.get("id") for record in records] != [item.id for item in selected[: len(records)]]:
        raise RuntimeError("Existing ATV320 checkpoint does not match dataset order.")


def _sealed_comparison_hash(manifest: Mapping[str, Any], dataset: str) -> str:
    key = {
        "development": "confirmation_dataset_sha256",
        "confirmation": "development_dataset_sha256",
        "heldout-v2": "heldout_dataset_sha256",
    }[dataset]
    value = manifest.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"ATV320 manifest has no valid sealed comparison hash: {key}")
    return value


def _selected_ids_hash(items: Sequence[EvaluationItem]) -> str:
    return hashlib.sha256("\n".join(item.id for item in items).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _openrouter_api_key(environ: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if environ is None else environ
    return source.get("OPENROUTER_API_KEY", "").strip() or None


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"{label} inputs must remain under {root}.") from None


def _require_private_review_path(path: Path) -> None:
    root = (Path.cwd() / "artifacts/local-review").resolve()
    target = (path if path.is_absolute() else Path.cwd() / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise SystemExit("Local review output must remain under artifacts/local-review.") from None


if __name__ == "__main__":
    raise SystemExit(main())
