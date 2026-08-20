"""Run the explicitly approved private Phase 7 replacement held-out v2 once.

This integration CLI is intentionally separate from the historic held-out
runner. It reads only the Git-ignored v2 dataset and manifest, validates their
hashes against the frozen corpus, and writes sanitized metrics. It never opens
the historic Phase 7 test JSONL or calibration dataset.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import Settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.evaluation_e2e import FACT_EVALUATOR_ID, aggregate_phase7_records, score_phase7_execution
from app.generation import LangChainOpenAIGenerator
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_dataset,
    write_json_atomic,
    write_jsonl_atomic,
)
from app.query_service import EvidenceGate, QueryService
from app.retrieval import create_qdrant_client
from app.retrieval_runtime import (
    PHASE7_RETRIEVAL_CONTRACT,
    LazyQueryRetriever,
    build_query_retriever,
    validate_frozen_runtime,
)
from scripts.evaluate_phase7_e2e import (
    _generation_configuration,
    _library_versions,
    _phase7_settings,
    _runtime_profile,
    _source_identity,
)

PROVIDER_APPROVAL_TOKEN = (
    "I approve sending Phase 7 heldout-v2 questions and retrieved excerpts to Gemini "
    "at generativelanguage.googleapis.com for one final heldout-v2 evaluation."
)
PRIVATE_ROOT = Path("data/eval/phase7/private-heldout-v2")


def main() -> int:
    args = _parse_args()
    _require_private_path(args.dataset)
    _require_private_path(args.manifest)
    _require_private_path(args.checkpoint)
    _validate_egress_approval(args.provider_approval_token)
    chunks = load_frozen_chunks(args.chunks)
    dataset = read_phase7_dataset(args.dataset)
    validation = validate_phase7_dataset(dataset, chunks, kind="test")
    if any(item.review_status != "approved" for item in dataset):
        raise SystemExit("Held-out v2 evaluation requires an explicitly approved frozen dataset.")
    manifest = _validate_manifest(args.manifest, chunks, dataset)

    settings = _phase7_settings(Settings())
    _validate_gemini_contract(settings)
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
    identity = {
        "dataset": "heldout-v2",
        "dataset_sha256": dataset_sha256(dataset),
        "corpus": chunk_set_metadata(chunks),
        "manifest_sha256": _file_sha256(args.manifest),
        "strategy": settings.retrieval_strategy,
        "rerank_enabled": settings.rerank_enabled,
        "generation_provider": settings.generation_provider,
        "generation_model": settings.generation_model,
        "generation_configuration": _generation_configuration(settings),
        "source_identity": _source_identity(),
        "runtime_profile": _runtime_profile(),
        "libraries": _library_versions(),
    }
    records = _load_checkpoint(args.checkpoint, identity)
    completed_ids = {record["id"] for record in records}
    for item in dataset:
        if item.id in completed_ids:
            continue
        try:
            execution = service.execute(question=item.question, document_id=None, top_k=args.top_k)
            record = score_phase7_execution(item, execution)
        except Exception as exc:
            _write_checkpoint(args.checkpoint, identity, records)
            raise RuntimeError(f"Held-out v2 execution failed for dataset item {item.id}.") from exc
        records.append(record)
        _write_checkpoint(args.checkpoint, identity, records)

    if len(records) != len(dataset):
        raise RuntimeError("Held-out v2 checkpoint result count does not match frozen dataset.")
    overall = aggregate_phase7_records(records)
    output = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "one approved final replacement held-out-v2 execution; not a tuning input",
        "run_identity": identity,
        "dataset_validation": validation,
        "manifest": {
            "heldout_dataset_sha256": manifest["heldout_dataset_sha256"],
            "corpus": manifest["corpus"],
            "runtime_configuration": manifest["runtime_configuration"],
        },
        "overall": overall,
        "evaluation_methodology": {
            "headline_answer_metric": "deterministic_fact_accuracy_when_answered",
            "fact_evaluator_id": FACT_EVALUATOR_ID,
            "retrieval_relevance": "stable relevant_chunk_ids only",
            "quality_gate_note": (
                "Calibration thresholds are not re-tuned or used to select runtime on held-out-v2."
            ),
        },
        "per_query": records,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "evidence_text": "excluded",
            "provider_response": "excluded",
            "api_key": "excluded",
        },
    }
    write_json_atomic(args.output, output)
    print(f"Phase 7 replacement held-out v2 E2E evaluation completed: {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PRIVATE_ROOT / "heldout-v2.jsonl")
    parser.add_argument("--manifest", type=Path, default=PRIVATE_ROOT / "heldout-v2-manifest.json")
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--provider-approval-token", required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=PRIVATE_ROOT / "heldout-v2-e2e-checkpoint.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/phase-7-heldout-v2-e2e.json")
    )
    return parser.parse_args()


def _require_private_path(path: Path) -> None:
    try:
        path.resolve().relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"Held-out v2 input/checkpoint must remain under {PRIVATE_ROOT}.") from exc


def _validate_egress_approval(value: str) -> None:
    if value != PROVIDER_APPROVAL_TOKEN:
        raise SystemExit("Held-out v2 provider approval is missing or invalid.")


def _validate_gemini_contract(settings: Settings) -> None:
    host = urlparse(settings.gemini_base_url).netloc
    if settings.generation_provider != "gemini" or host != "generativelanguage.googleapis.com":
        raise SystemExit(
            "Held-out v2 is approved only for Gemini at generativelanguage.googleapis.com."
        )
    if settings.gemini_temperature != 0:
        raise SystemExit("Held-out v2 requires explicit Gemini temperature=0 for reproducibility.")
    if settings.generation_api_key is None:
        raise SystemExit("Gemini API key is unavailable; no provider request was made.")


def _validate_manifest(path: Path, chunks: list[Any], dataset: list[Any]) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to load held-out v2 manifest: {path}") from exc
    expected = {
        "heldout_dataset_sha256": dataset_sha256(dataset),
        "corpus": chunk_set_metadata(chunks),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"Held-out v2 manifest mismatch: {key}")
    return manifest


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Malformed held-out v2 checkpoint: {path}") from exc
    if header != {"run_identity": identity}:
        raise RuntimeError("Existing held-out v2 checkpoint belongs to a different frozen run.")
    if len({record.get("id") for record in records}) != len(records):
        raise RuntimeError("Existing held-out v2 checkpoint has duplicate item IDs.")
    return records


def _write_checkpoint(path: Path, identity: dict[str, Any], records: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(path, [{"run_identity": identity}, *records])


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
