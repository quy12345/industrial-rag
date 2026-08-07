"""Run the approved Phase 7 end-to-end evaluation against separate collections.

The output is intentionally sanitized: it records IDs, ranks, aggregate quality,
latency, and token counts, but never questions, manual text, prompts, answers, or
provider responses.  It is an explicit integration CLI and is not part of pytest.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.evaluation_e2e import aggregate_phase7_records, score_phase7_execution
from app.generation import LangChainOpenAIGenerator
from app.phase7 import (
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.query_service import EvidenceGate, QueryService
from app.retrieval import create_qdrant_client
from app.retrieval_runtime import (
    PHASE7_RETRIEVAL_CONTRACT,
    LazyQueryRetriever,
    build_query_retriever,
    validate_frozen_runtime,
)


def main() -> int:
    args = _parse_args()
    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    chunks = load_frozen_chunks(args.chunks)
    validation = validate_phase7_datasets(calibration, test, chunks)
    if any(item.review_status != "approved" for item in [*calibration, *test]):
        raise SystemExit("Phase 7 evaluation requires explicitly approved datasets.")
    selected = calibration if args.dataset == "calibration" else test
    if args.max_queries is not None:
        selected = selected[: args.max_queries]

    _validate_evaluation_manifest(args.manifest, chunks, calibration, test)
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
    identity = {
        "dataset": args.dataset,
        "dataset_sha256": dataset_sha256(selected),
        "corpus": chunk_set_metadata(chunks),
        "contract_chunk_ids_sha256": PHASE7_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "strategy": settings.retrieval_strategy,
        "rerank_enabled": settings.rerank_enabled,
        "generation_provider": settings.generation_provider,
        "generation_model": settings.generation_model,
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
    output = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_identity": identity,
        "dataset_validation": validation,
        "overall": aggregate_phase7_records(records),
        "per_query": records,
        "sanitization": {
            "raw_question": "excluded",
            "raw_answer": "excluded",
            "evidence_text": "excluded",
            "provider_response": "excluded",
        },
    }
    write_json_atomic(args.output, output)
    print(f"Phase 7 {args.dataset} E2E evaluation PASS: {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("calibration", "test"), required=True)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/metrics/phase-7-evaluation-manifest.json")
    )
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/metrics/phase-7-e2e-checkpoint.jsonl")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    if args.output is None:
        args.output = Path(f"artifacts/metrics/phase-7-{args.dataset}-e2e.json")
    return args


def _phase7_settings(settings: Settings) -> Settings:
    """Make the fixed Phase 7 runtime selection explicit and artifact-independent."""

    return settings.model_copy(
        update={
            "qdrant_collection": PHASE7_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": PHASE7_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
            "retrieval_strategy": "union",
            "rerank_enabled": True,
        }
    )


def _validate_evaluation_manifest(
    path: Path, chunks: list[Any], calibration: list[Any], test: list[Any]
) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load Phase 7 evaluation manifest: {path}") from exc
    expected = {
        "corpus": chunk_set_metadata(chunks),
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "test_dataset_sha256": dataset_sha256(test),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Phase 7 evaluation manifest mismatch: {key}")


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
