"""Run the approved three-attempt Phase 7 calibration-005 diagnostic.

Retrieval and reranking execute exactly once. Three provider generations then
reuse the same frozen evidence bundle. Raw answers are permitted only beneath
the ignored ``artifacts/private-debug`` directory; the metrics artifact remains
sanitized and contains no question, answer, prompt, evidence, or provider body.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.citations import validate_generated_answer
from app.config import Settings
from app.errors import (
    CitationValidationError,
    GenerationValidationError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.evaluation import load_frozen_chunks
from app.evaluation_e2e import FACT_EVALUATOR_ID, score_expected_answer_fact
from app.evidence_selection import select_evidence_candidates
from app.generation import EvidenceBundle, LangChainOpenAIGenerator, format_evidence
from app.phase7 import dataset_sha256, write_json_atomic
from app.query_service import EvidenceGate
from app.retrieval import create_qdrant_client
from app.retrieval_runtime import (
    PHASE7_RETRIEVAL_CONTRACT,
    build_query_retriever,
    validate_frozen_runtime,
)
from scripts.evaluate_phase7_e2e import (
    ACTIVE_CALIBRATION_PATH,
    ACTIVE_MANIFEST_PATH,
    _generation_configuration,
    _load_selected_dataset,
    _phase7_settings,
    _source_identity,
)

APPROVAL_TOKEN = "APPROVE PHASE 7 CALIBRATION 005 DIAGNOSTIC EGRESS"
ITEM_ID = "phase7_calibration_005"


def main() -> int:
    args = _parse_args()
    if args.approval_token != APPROVAL_TOKEN:
        raise SystemExit("Calibration-005 diagnostic provider approval is missing or invalid.")
    private_output = _validated_private_debug_path(args.private_output)
    chunks = load_frozen_chunks(args.chunks)
    dataset_args = argparse.Namespace(
        dataset="calibration",
        calibration=args.calibration,
        test=None,
        manifest=args.manifest,
    )
    dataset, validation, manifest = _load_selected_dataset(dataset_args, chunks)
    item = next((value for value in dataset if value.id == ITEM_ID), None)
    if item is None or not item.answerable:
        raise RuntimeError("Approved calibration item 005 is unavailable or not answerable.")

    settings = _phase7_settings(Settings())
    client = create_qdrant_client(settings)
    validate_frozen_runtime(
        client,
        collection_names=(settings.qdrant_collection, settings.qdrant_hybrid_collection),
        contract=PHASE7_RETRIEVAL_CONTRACT,
    )
    retriever = build_query_retriever(settings, contract=PHASE7_RETRIEVAL_CONTRACT)
    retrieved = retriever.retrieve(item.question, document_id=None)
    selection = select_evidence_candidates(
        item.question,
        retrieved.candidates,
        top_k=args.top_k,
    )
    gate = EvidenceGate(score_threshold=settings.evidence_score_threshold).evaluate(
        selection.candidates,
        requested_document_id=None,
    )
    if not gate.passed:
        raise RuntimeError(f"Calibration-005 fixed evidence failed the gate: {gate.reason}")
    evidence = format_evidence(
        selection.candidates,
        max_chars=settings.generation_max_context_chars,
    )
    generator = LangChainOpenAIGenerator(settings)
    generator.ensure_configured()
    sanitized_attempts, private_attempts = _run_fixed_evidence_attempts(
        generator,
        item=item,
        evidence=evidence,
        attempts=3,
    )
    identity = {
        "item_id": ITEM_ID,
        "calibration_dataset_sha256": dataset_sha256(dataset),
        "held_out_dataset_sha256": manifest["test_dataset_sha256"],
        "corpus": validation["corpus"],
        "contract_chunk_ids_sha256": PHASE7_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "evaluator_id": FACT_EVALUATOR_ID,
        "generation_configuration": _generation_configuration(settings),
        "source_identity": _source_identity(),
        "top_k": args.top_k,
        "retrieval_executions": 1,
        "reranker_executions": 1,
        "provider_attempts": 3,
        "evidence_manifest_sha256": _evidence_manifest_sha256(evidence),
    }
    sanitized = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "fixed_evidence": {
            "source_labels": list(evidence.source_map),
            "chunk_ids": [candidate.chunk_id for candidate in selection.candidates],
            "duplicate_group_count": len(selection.duplicate_groups),
        },
        "attempts": sanitized_attempts,
        "sanitization": {
            "question": "excluded",
            "answer": "excluded",
            "evidence": "excluded",
            "provider_response": "excluded",
        },
    }
    private = {
        "warning": "LOCAL PRIVATE DEBUG: do not commit or share",
        "run_identity": identity,
        "attempts": private_attempts,
    }
    write_json_atomic(args.output, sanitized)
    write_json_atomic(private_output, private)
    print(f"Phase 7 calibration-005 diagnostic completed: {args.output}")
    return 0


def _run_fixed_evidence_attempts(
    generator: Any,
    *,
    item: Any,
    evidence: EvidenceBundle,
    attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if attempts != 3:
        raise ValueError("Calibration-005 diagnostic requires exactly three attempts.")
    sanitized: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    evidence_identity = id(evidence)
    for attempt in range(1, attempts + 1):
        try:
            generated = generator.generate(question=item.question, evidence=evidence)
            validated = validate_generated_answer(
                generated.output,
                source_map=evidence.source_map,
            )
            fact_results = [
                score_expected_answer_fact(fact, validated.answer)
                for fact in item.expected_answer_facts
            ]
            status = "model_abstention" if validated.insufficient_evidence else "completed"
            sanitized.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "deterministic_fact_match": bool(fact_results)
                    and all(result["deterministic_matched"] for result in fact_results),
                    "source_ids": list(validated.source_ids),
                    "fact_results": fact_results,
                    "usage": (
                        None
                        if generated.usage is None
                        else {
                            "input_tokens": generated.usage.input_tokens,
                            "output_tokens": generated.usage.output_tokens,
                            "cached_input_tokens": generated.usage.cached_input_tokens,
                        }
                    ),
                }
            )
            private.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "answer": validated.answer,
                    "source_ids": list(validated.source_ids),
                    "insufficient_evidence": validated.insufficient_evidence,
                    "fact_results": fact_results,
                }
            )
        except LLMTimeoutError:
            sanitized.append({"attempt": attempt, "status": "provider_timeout"})
            private.append({"attempt": attempt, "status": "provider_timeout"})
        except LLMRefusalError:
            sanitized.append({"attempt": attempt, "status": "provider_refusal"})
            private.append({"attempt": attempt, "status": "provider_refusal"})
        except LLMUnavailableError:
            sanitized.append({"attempt": attempt, "status": "provider_unavailable"})
            private.append({"attempt": attempt, "status": "provider_unavailable"})
        except (GenerationValidationError, CitationValidationError):
            sanitized.append({"attempt": attempt, "status": "structured_output_invalid"})
            private.append({"attempt": attempt, "status": "structured_output_invalid"})
        if id(evidence) != evidence_identity:
            raise RuntimeError("Diagnostic evidence changed between provider attempts.")
    return sanitized, private


def _validated_private_debug_path(path: Path) -> Path:
    root = (Path.cwd() / "artifacts" / "private-debug").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Raw diagnostic output must remain under artifacts/private-debug.")
    return resolved


def _evidence_manifest_sha256(evidence: EvidenceBundle) -> str:
    manifest = "\n".join(
        f"{source_id}:{candidate.chunk_id}"
        for source_id, candidate in evidence.source_map.items()
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-token", required=True)
    parser.add_argument("--calibration", type=Path, default=ACTIVE_CALIBRATION_PATH)
    parser.add_argument("--manifest", type=Path, default=ACTIVE_MANIFEST_PATH)
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-005-diagnostic-v1.json"),
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=Path("artifacts/private-debug/phase-7-calibration-005-raw-v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
