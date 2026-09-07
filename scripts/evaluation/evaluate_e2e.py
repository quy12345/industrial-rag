"""Run the approval-gated ATV320 E2E Ragas scorecard.

The command writes only compact sanitized schema-v19 artifacts. Raw review data is
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
from app.domain.policies.evidence_scope import EVIDENCE_SCOPE_PROFILE, evaluate_evidence_scope
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
from evaluation.e2e import build_case_record, build_scorecard, retrieval_evidence_fingerprint
from evaluation.human_review import (
    PrivateReviewPack,
    import_legacy_review,
    select_judge_validation_cohort,
    selected_ids_sha256,
)
from evaluation.review_inputs import (
    INPUT_POLICY_ID,
    PrivateInputJournal,
    ReviewInput,
    ReviewInputError,
    load_replay_source,
    load_scope_replay_source,
    payload_sha256,
    scores_from_payload,
    verify_evidence,
)
from evaluation.semantic import (
    MetricScore,
    RagasEvaluationInput,
    RagasJudge,
    RagasScorecardJudge,
    RagasScores,
    SemanticConfigurationError,
    SemanticJudgeError,
    headed_cited_contexts,
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
ARTIFACT_SCHEMA_VERSION = 19
SOURCE_IDENTITY_VERSION = 15
SOURCE_IDENTITY_PATHS = {
    "dataset_contract": Path("evaluation/dataset.py"),
    "evaluator": Path("evaluation/e2e.py"),
    "human_review": Path("evaluation/human_review.py"),
    "review_inputs": Path("evaluation/review_inputs.py"),
    "semantic_evaluator": Path("evaluation/semantic.py"),
    "evaluation_command": Path("scripts/evaluation/evaluate_e2e.py"),
    "generation_prompt": Path("app/application/generation_prompt.py"),
    "query_service": Path("app/application/query_service.py"),
    "evidence_scope_policy": Path("app/domain/policies/evidence_scope.py"),
    "retrieval_composition": Path("app/composition/retrieval.py"),
    "retrieval_contract": Path("app/domain/retrieval_contracts.py"),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_approvals(args)
    chunks = load_frozen_chunks(FROZEN_CHUNKS_PATH)
    dataset, validation, manifest = _load_dataset(args.dataset, chunks)
    if args.scope_replay_artifact is not None:
        return _run_scope_replay(args, chunks=chunks, dataset=dataset)
    if args.replay_artifact is not None:
        return _run_replay(args, chunks=chunks, dataset=dataset)
    selected = _select_items(dataset, args)
    _require_reviewed_references(selected)
    settings = _runtime_settings(Settings())
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
    _protect_output(args.output, identity)

    corpus = {chunk.chunk_id: chunk for chunk in chunks}
    journal = (
        PrivateInputJournal.open(
            args.local_review_output.with_name("inputs.jsonl"), identity, selected, corpus
        )
        if args.local_review_output is not None
        else None
    )
    review_pack = (
        PrivateReviewPack.open(
            args.local_review_output,
            selected_ids=[item.id for item in selected],
            checkpoint_ids=[str(record["id"]) for record in records],
        )
        if args.local_review_output is not None
        else None
    )
    if review_pack is not None and journal is not None:
        for record in records:
            saved = journal.rows.get(str(record["id"]))
            if saved is None or saved.to_payload()["input_sha256"] != record["input_sha256"]:
                raise ReviewInputError("Completed checkpoint is missing its bound private input.")
            if saved.item.id not in review_pack._completed_ids:
                review_pack.append(
                    item=saved.item,
                    execution=saved.execution,
                    scores=scores_from_payload(record["ragas"]),
                    max_context_chars=saved.max_context_chars,
                )
    if len(records) == len(selected):
        return _finish_run(args, identity, records)
    judge = _build_judge(args)
    completed_ids = {str(record["id"]) for record in records}
    service = None
    for item in selected:
        if item.id in completed_ids:
            continue
        try:
            saved = journal.rows.get(item.id) if journal is not None else None
            if saved is None:
                if service is None:
                    service = _build_query_service(settings)
                execution = service.execute(question=item.question, document_id=None, top_k=5)
                verify_evidence(execution, corpus)
                saved = ReviewInput(item, execution, settings.generation_max_context_chars)
                if journal is not None:
                    journal.append(saved)
            execution = saved.execution
            scores = _score_execution(
                item,
                execution,
                judge=judge,
                include_diagnostics=args.ragas_diagnostics,
                max_context_chars=settings.generation_max_context_chars,
                corpus=corpus,
            )
            record = build_case_record(
                item, execution, scores, input_sha256=saved.to_payload()["input_sha256"]
            )
            records.append(record)
            _write_checkpoint(args.checkpoint, identity, records)
            if review_pack is not None:
                review_pack.append(
                    item=item,
                    execution=execution,
                    scores=scores,
                    max_context_chars=settings.generation_max_context_chars,
                )
        except SemanticJudgeError as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(str(exc)) from None
        except Exception:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(f"ATV320 execution failed for dataset item {item.id}.") from None

    if len(records) != len(selected):
        raise RuntimeError("Checkpoint result count does not match selected items.")
    return _finish_run(args, identity, records)


def _build_query_service(settings: Settings) -> QueryService:
    """Construct external retrieval dependencies only when a new answer is needed."""
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
    return QueryService(
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


def _finish_run(
    args: argparse.Namespace, identity: dict[str, Any], records: list[dict[str, Any]]
) -> int:
    scorecard = build_scorecard(records)
    scorecard["operational"]["measurement_provenance"] = {
        "retrieval_generation": "historical_source_run"
        if identity["execution_mode"] == "judge_replay"
        else "live",
        "judge": "current_run",
        "generation_usage": "historical_not_rebilled"
        if identity["execution_mode"] == "judge_replay"
        else "live",
        "judge_token_usage": "unavailable_from_current_adapter",
    }
    output = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "scorecard": scorecard,
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


def _run_replay(
    args: argparse.Namespace, *, chunks: list[Any], dataset: list[EvaluationItem]
) -> int:
    """Judge fixed inputs only. This branch never constructs the runtime service."""
    source = load_replay_source(
        args.replay_artifact,
        items=dataset,
        chunks=chunks,
        source_files=_source_identity()["files"],
    )
    origin = source["run_identity"]
    if source["schema_version"] >= 18 and (
        origin.get("execution_mode") != "live"
        or origin.get("frozen_chunks_sha256") != _file_sha256(FROZEN_CHUNKS_PATH)
    ):
        raise ReviewInputError("V18 replay requires an unchanged corpus and a live source run.")
    if origin["generation"].get("model") != "qwen/qwen3-235b-a22b-2507":
        raise ReviewInputError("This replay approval is scoped to the existing Qwen generator.")
    by_id = {item.id: item for item in dataset}
    selected = [by_id[item_id] for item_id in origin["selected_item_ids"]]
    _require_reviewed_references(selected)
    corpus = {chunk.chunk_id: chunk for chunk in chunks}
    imported = import_legacy_review(
        args.replay_review,
        items=selected,
        cases=source["cases"],
        corpus=corpus,
        max_context_chars=origin["generation_max_context_chars"],
    )
    import_policy = "legacy_review_import"
    if source["schema_version"] >= 18:
        journal_path = args.replay_review.with_name("inputs.jsonl")
        if not journal_path.is_file():
            raise ReviewInputError("V18 replay requires its original private input journal.")
        journal = PrivateInputJournal.open(journal_path, origin, selected, corpus)
        captured = []
        for value, record in zip(imported, source["cases"], strict=True):
            saved = journal.rows.get(value.item.id)
            if saved is None or saved.to_payload()["input_sha256"] != record["input_sha256"]:
                raise ReviewInputError("Artifact and captured input hash differ.")
            if (
                retrieval_evidence_fingerprint(saved.execution)
                != record["retrieval_evidence_fingerprint"]
            ):
                raise ReviewInputError("Captured retrieval fingerprint differs from its artifact.")
            if (
                saved.execution.response.answer != value.execution.response.answer
                or saved.generation_context() != value.generation_context()
                or [row.chunk_id for row in saved.execution.response.citations]
                != record["citation_ids"]
                or [row.chunk_id for row in saved.execution.evidence_candidates]
                != record["evidence_candidate_ids"]
            ):
                raise ReviewInputError("Markdown and captured input differ.")
            captured.append(saved)
        imported = captured
        import_policy = INPUT_POLICY_ID
    inputs_hash = payload_sha256([value.to_payload()["input_sha256"] for value in imported])
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "execution_mode": "judge_replay",
        "dataset": "development",
        "dataset_sha256": dataset_sha256(dataset),
        "selected_item_ids": [item.id for item in selected],
        "corpus": chunk_set_metadata(chunks),
        "contract_chunk_ids_sha256": ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "frozen_chunks_sha256": _file_sha256(FROZEN_CHUNKS_PATH),
        "evidence_top_k": 5,
        "generation_max_context_chars": origin["generation_max_context_chars"],
        "generation": {"model": "qwen/qwen3-235b-a22b-2507", "execution": "not_called"},
        "ragas": semantic_run_identity(include_diagnostics=False),
        "source_identity": _source_identity(),
        "libraries": _library_versions(),
        "runtime": _runtime_profile(),
        "review": {"cohort": None, "selected_ids_sha256": selected_ids_sha256(selected)},
        "replay": {
            "source_artifact_sha256": _file_sha256(args.replay_artifact),
            "source_run_identity_sha256": payload_sha256(origin),
            "input_policy": import_policy,
            "inputs_sha256": inputs_hash,
            "original_answer_hash_verified": import_policy != "legacy_review_import",
        },
    }
    records = _load_checkpoint(args.checkpoint, identity)
    _validate_checkpoint_prefix(records, selected)
    _protect_output(args.output, identity)
    if len(records) == len(selected):
        return _finish_run(args, identity, records)
    # Every imported item and source boundary is validated before judge construction.
    judge = _build_judge(args)
    for value, original_record in zip(
        imported[len(records) :], source["cases"][len(records) :], strict=True
    ):
        try:
            scores = _score_execution(
                value.item,
                value.execution,
                judge=judge,
                include_diagnostics=False,
                max_context_chars=value.max_context_chars,
                corpus=corpus,
            )
            record = build_case_record(
                value.item,
                value.execution,
                scores,
                input_sha256=value.to_payload()["input_sha256"],
            )
            record["retrieval_evidence_fingerprint"] = original_record[
                "retrieval_evidence_fingerprint"
            ]
            records.append(record)
            _write_checkpoint(args.checkpoint, identity, records)
        except SemanticJudgeError as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(str(exc)) from None
        except Exception:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(f"Judge replay failed for item {value.item.id}.") from None
    return _finish_run(args, identity, records)


def _run_scope_replay(
    args: argparse.Namespace, *, chunks: list[Any], dataset: list[EvaluationItem]
) -> int:
    """Replay the deterministic scope guard without constructing external clients."""

    source = load_scope_replay_source(
        args.scope_replay_artifact,
        items=dataset,
        chunks=chunks,
    )
    origin = source["run_identity"]
    if (
        origin.get("execution_mode") != "live"
        or origin.get("frozen_chunks_sha256") != _file_sha256(FROZEN_CHUNKS_PATH)
    ):
        raise ReviewInputError("Scope replay requires an unchanged corpus and a live source run.")
    by_id = {item.id: item for item in dataset}
    selected = [by_id[item_id] for item_id in origin["selected_item_ids"]]
    corpus = {chunk.chunk_id: chunk for chunk in chunks}
    journal = PrivateInputJournal.open(args.scope_replay_inputs, origin, selected, corpus)
    source_cases = {str(case["id"]): case for case in source["cases"]}
    saved_inputs: list[ReviewInput] = []
    for item in selected:
        saved = journal.rows.get(item.id)
        record = source_cases.get(item.id)
        if saved is None or record is None:
            raise ReviewInputError("Scope replay source is missing an identity-bound input.")
        if saved.to_payload()["input_sha256"] != record["input_sha256"]:
            raise ReviewInputError("Scope replay artifact and captured input hash differ.")
        if (
            retrieval_evidence_fingerprint(saved.execution)
            != record["retrieval_evidence_fingerprint"]
        ):
            raise ReviewInputError("Scope replay retrieval fingerprint differs from its artifact.")
        saved_inputs.append(saved)

    identity = {
        "scope_replay_schema_version": 1,
        "execution_mode": "scope_replay",
        "dataset": "development",
        "dataset_sha256": dataset_sha256(dataset),
        "selected_item_ids": [item.id for item in selected],
        "corpus": chunk_set_metadata(chunks),
        "contract_chunk_ids_sha256": ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "candidate_policy_sha256": _file_sha256(Path("app/domain/retrieval_contracts.py")),
        "target_policy": {
            "profile": EVIDENCE_SCOPE_PROFILE,
            "evidence_scope_policy_sha256": _file_sha256(
                Path("app/domain/policies/evidence_scope.py")
            ),
            "generation_prompt_sha256": _file_sha256(Path("app/application/generation_prompt.py")),
            "query_service_sha256": _file_sha256(Path("app/application/query_service.py")),
        },
        "source_identity": _source_identity(),
        "replay": {
            "source_artifact_sha256": _file_sha256(args.scope_replay_artifact),
            "source_run_identity_sha256": payload_sha256(origin),
            "input_policy": INPUT_POLICY_ID,
            "inputs_sha256": payload_sha256(
                [value.to_payload()["input_sha256"] for value in saved_inputs]
            ),
            "origin_source_identity": origin["source_identity"],
        },
    }
    records = _load_scope_checkpoint(args.checkpoint, identity)
    _validate_scope_checkpoint_prefix(records, selected)
    _protect_output(args.output, identity)
    if len(records) == len(selected):
        return _finish_scope_replay(args, identity, records)

    for value in saved_inputs[len(records) :]:
        bundle = format_evidence(
            value.execution.evidence_candidates, max_chars=value.max_context_chars
        )
        pre_generation = evaluate_evidence_scope(
            value.item.question,
            _rendered_scope_texts(bundle),
        )
        cited_chunk_ids = {citation.chunk_id for citation in value.execution.response.citations}
        cited_source_ids = tuple(
            source_id
            for source_id, candidate in bundle.source_map.items()
            if candidate.chunk_id in cited_chunk_ids
        )
        cited_scope = (
            evaluate_evidence_scope(
                value.item.question,
                _rendered_scope_texts(bundle, source_ids=cited_source_ids),
            )
            if cited_source_ids
            else None
        )
        proposed_reason = (
            "evidence_scope_missing"
            if pre_generation.status == "missing_scope"
            else (
                "cited_evidence_scope_missing"
                if cited_scope is not None and cited_scope.status == "missing_scope"
                else None
            )
        )
        record = {
            "id": value.item.id,
            "answerable": value.item.answerable,
            "original_abstained": value.execution.response.abstained,
            "original_abstention_reason": value.execution.response.abstention_reason,
            "evidence_candidate_ids": [
                candidate.chunk_id for candidate in value.execution.evidence_candidates
            ],
            "citation_ids": [citation.chunk_id for citation in value.execution.response.citations],
            "retrieval_evidence_fingerprint": retrieval_evidence_fingerprint(value.execution),
            "pre_generation_scope": _scope_decision_payload(pre_generation),
            "cited_scope": _scope_decision_payload(cited_scope) if cited_scope else None,
            "proposed_abstention_reason": proposed_reason,
        }
        records.append(record)
        _write_scope_checkpoint(args.checkpoint, identity, records)
    return _finish_scope_replay(args, identity, records)


def _rendered_scope_texts(
    evidence, *, source_ids: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Read only title, headings, and content actually rendered for the provider."""

    ids = source_ids if source_ids is not None else tuple(evidence.rendered_sources)
    return tuple(evidence.rendered_sources[source_id].scope_text for source_id in ids)


def _scope_decision_payload(decision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "query_concept_ids": list(decision.query_concept_ids),
        "missing_concept_ids": list(decision.missing_concept_ids),
    }


def _finish_scope_replay(
    args: argparse.Namespace, identity: dict[str, Any], records: list[dict[str, Any]]
) -> int:
    output = {
        "schema_version": 1,
        "kind": "evidence_scope_replay",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "cases": records,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "evidence_text": "excluded",
            "provider_response": "not_loaded",
        },
    }
    write_json_atomic(args.output, output)
    print(f"ATV320 development evidence-scope replay completed: {args.output}")
    return 0


def _protect_output(path: Path, identity: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["run_identity"] != identity:
                raise ValueError
        except Exception:
            raise ReviewInputError(
                "Refusing to overwrite an unrelated evaluation artifact."
            ) from None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("development", "confirmation", "heldout-v2"), required=True
    )
    parser.add_argument("--provider-approval-token")
    parser.add_argument("--judge-provider-approval-token")
    parser.add_argument("--ragas-diagnostics", action="store_true")
    parser.add_argument("--review-cohort", choices=("judge-validation-v1",))
    parser.add_argument("--review-item-id", action="append", default=[])
    parser.add_argument("--local-review-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-artifact", type=Path)
    parser.add_argument("--replay-review", type=Path)
    parser.add_argument("--scope-replay-artifact", type=Path)
    parser.add_argument("--scope-replay-inputs", type=Path)
    args = parser.parse_args(argv)
    replay = args.replay_artifact is not None
    scope_replay = args.scope_replay_artifact is not None
    if replay != (args.replay_review is not None):
        parser.error("--replay-artifact and --replay-review must be provided together")
    if scope_replay != (args.scope_replay_inputs is not None):
        parser.error("--scope-replay-artifact and --scope-replay-inputs must be provided together")
    if replay and scope_replay:
        parser.error("Judge replay and scope replay are mutually exclusive")
    if not scope_replay and args.judge_provider_approval_token is None:
        parser.error("the following arguments are required: --judge-provider-approval-token")
    if replay:
        if args.dataset != "development":
            parser.error("Judge replay is permitted only for development")
        if (
            args.review_cohort
            or args.review_item_id
            or args.local_review_output
            or args.ragas_diagnostics
        ):
            parser.error(
                "Replay uses the source selection and core metrics only; "
                "review/diagnostic flags are not allowed"
            )
        if args.output is None or args.checkpoint is None:
            parser.error("Replay requires explicit --output and --checkpoint")
        _require_private_review_path(args.replay_review)
        sources = {
            args.replay_artifact.resolve(),
            args.replay_review.resolve(),
            args.replay_review.with_name("inputs.jsonl").resolve(),
            args.replay_review.with_name("labels.jsonl").resolve(),
        }
        if args.output.resolve() in sources or args.checkpoint.resolve() in sources:
            parser.error("Replay must not overwrite source files")
    elif scope_replay:
        if args.dataset != "development":
            parser.error("Scope replay is permitted only for development")
        if (
            args.provider_approval_token
            or args.judge_provider_approval_token
            or args.review_cohort
            or args.review_item_id
            or args.local_review_output
            or args.ragas_diagnostics
        ):
            parser.error("Scope replay accepts only source, input, output, and checkpoint paths")
        if args.output is None or args.checkpoint is None:
            parser.error("Scope replay requires explicit --output and --checkpoint")
        _require_private_review_path(args.scope_replay_inputs)
        sources = {
            args.scope_replay_artifact.resolve(),
            args.scope_replay_inputs.resolve(),
        }
        if args.output.resolve() in sources or args.checkpoint.resolve() in sources:
            parser.error("Scope replay must not overwrite source files")
    elif args.provider_approval_token is None:
        parser.error("the following arguments are required: --provider-approval-token")
    if args.output and args.checkpoint and args.output.resolve() == args.checkpoint.resolve():
        parser.error("Output and checkpoint must be different files")
    if args.review_cohort and args.review_item_id:
        parser.error("--review-cohort and --review-item-id are mutually exclusive")
    if (args.review_cohort or args.review_item_id) and args.dataset != "development":
        parser.error("Review selection is permitted only for development")
    if len(args.review_item_id) > 40 or len(set(args.review_item_id)) != len(args.review_item_id):
        parser.error("Targeted review requires at most 40 unique item IDs")
    partial = bool(args.review_cohort or args.review_item_id)
    if partial and args.local_review_output is None:
        parser.error("Review selection requires --local-review-output")
    if args.local_review_output is not None and args.dataset != "development":
        parser.error("Private input capture is permitted only for development")
    if partial and (args.output is None or args.checkpoint is None):
        parser.error("Partial review requires explicit --output and --checkpoint")
    if args.local_review_output is not None:
        _require_private_review_path(args.local_review_output)
    if args.output is None:
        args.output = Path(f"artifacts/metrics/atv320-{args.dataset}-e2e-v19.json")
    if args.checkpoint is None:
        args.checkpoint = Path(f"artifacts/metrics/atv320-{args.dataset}-e2e-v19-checkpoint.jsonl")
    return args


def _validate_approvals(
    args: argparse.Namespace, *, environ: Mapping[str, str] | None = None
) -> None:
    if getattr(args, "scope_replay_artifact", None):
        return
    expected_generator = {
        "development": DEVELOPMENT_PROVIDER_APPROVAL_TOKEN,
        "confirmation": CONFIRMATION_PROVIDER_APPROVAL_TOKEN,
        "heldout-v2": HELDOUT_PROVIDER_APPROVAL_TOKEN,
    }[args.dataset]
    if (
        not getattr(args, "replay_artifact", None)
        and args.provider_approval_token != expected_generator
    ):
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
    corpus: Mapping[str, Any],
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
    cited_contexts = (
        headed_cited_contexts(execution.evidence_candidates, citation_ids, corpus)
        if citations_valid
        else ()
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
    required = {
        "factual_correctness": answered,
        "faithfulness": answered and citations_valid,
        "context_recall": True,
        "answer_relevancy": include_diagnostics and answered,
        "context_precision_without_reference": include_diagnostics and answered,
    }
    for name, enabled in required.items():
        metric = getattr(scores, name, None)
        if (
            not isinstance(metric, MetricScore)
            or (enabled and (metric.status != "complete" or metric.score is None))
            or (not enabled and metric.status != "not_applicable")
        ):
            raise SemanticJudgeError(item_id=item.id, metric=name, category="invalid_score")
    if answered and not citations_valid:
        scores = replace(
            scores,
            faithfulness=MetricScore.deterministic_zero("invalid_citations"),
        )
    return scores


def _select_items(items: list[EvaluationItem], args: argparse.Namespace) -> list[EvaluationItem]:
    if args.review_cohort:
        return list(select_judge_validation_cohort(items))
    if not args.review_item_id:
        return items
    by_id = {item.id: item for item in items}
    missing = [item_id for item_id in args.review_item_id if item_id not in by_id]
    if missing:
        raise SystemExit("Review item IDs were not found: " + ", ".join(missing))
    return [by_id[item_id] for item_id in args.review_item_id]


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


def _require_reviewed_references(items: Sequence[EvaluationItem]) -> None:
    """Fail before provider/model construction when selected references are not approved."""

    pending = [
        item.id
        for item in items
        if item.answerable
        and (item.reference_answer is None or item.reference_answer_review_status != "approved")
    ]
    if pending:
        raise SystemExit(
            "ATV320 E2E evaluation requires approved natural reference answers; "
            f"pending record count: {len(pending)}."
        )


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
        "execution_mode": "live",
        "private_input_policy_id": INPUT_POLICY_ID,
        "frozen_chunks_sha256": _file_sha256(FROZEN_CHUNKS_PATH),
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
            "selected_ids_sha256": selected_ids_sha256(selected),
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
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Malformed ATV320 checkpoint.") from None
    if header != {"run_identity": identity}:
        raise RuntimeError("Existing ATV320 checkpoint belongs to a different frozen run.")
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Existing ATV320 checkpoint contains duplicate item IDs.")
    return records


def _write_checkpoint(path: Path, identity: dict[str, Any], records: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(path, [{"run_identity": identity}, *records])


def _load_scope_checkpoint(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0]) if lines else {}
        records = [json.loads(line) for line in lines[1:]]
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Malformed evidence-scope replay checkpoint.") from None
    if header != {"run_identity": identity}:
        raise RuntimeError("Existing evidence-scope replay checkpoint belongs to another run.")
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Evidence-scope replay checkpoint contains duplicate item IDs.")
    return records


def _write_scope_checkpoint(
    path: Path, identity: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    write_jsonl_atomic(path, [{"run_identity": identity}, *records])


def _validate_checkpoint_prefix(
    records: list[dict[str, Any]], selected: list[EvaluationItem]
) -> None:
    if [record.get("id") for record in records] != [item.id for item in selected[: len(records)]]:
        raise RuntimeError("Existing ATV320 checkpoint does not match dataset order.")


def _validate_scope_checkpoint_prefix(
    records: list[dict[str, Any]], selected: list[EvaluationItem]
) -> None:
    if [record.get("id") for record in records] != [item.id for item in selected[: len(records)]]:
        raise RuntimeError("Evidence-scope replay checkpoint does not match dataset order.")


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
