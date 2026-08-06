"""Write a machine-readable Phase 5 handoff from frozen Phase 4 artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from app.config import get_settings
from app.evaluation import chunk_set_metadata, load_frozen_chunks
from app.retrieval import RetrievalError, create_qdrant_client

METRICS_DIR = Path("artifacts/metrics")
DEFAULT_OUTPUT = METRICS_DIR / "phase-5-readiness.json"


def main(argv: list[str] | None = None) -> int:
    """Collect current collection state plus immutable Phase 4 benchmark artifacts."""

    _configure_output_encoding()
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    try:
        chunks = load_frozen_chunks(args.chunks)
        frozen_metadata = chunk_set_metadata(chunks)
        dense_metrics = _load_json(args.dense_metrics)
        sparse_metrics = _load_json(args.sparse_metrics)
        hybrid_metrics = _load_json(args.hybrid_metrics)
        audit = _load_json(args.candidate_audit)
        _validate_frozen_contract(
            frozen_metadata,
            document_id=chunks[0].document_id,
            artifacts=(dense_metrics, sparse_metrics, hybrid_metrics, audit),
        )
        client = create_qdrant_client(settings)
        payload = {
            "readiness_schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": args.status,
            "phase_4_commit": args.phase_4_commit,
            "git": {
                "branch": args.git_branch or _git_value("branch", "--show-current"),
                "working_tree_short": args.working_tree_short or _git_lines("status", "--short"),
            },
            "document_id": chunks[0].document_id,
            "frozen_chunk_set": frozen_metadata,
            "evaluation_dataset": dense_metrics["evaluation_dataset"],
            "dense": dense_metrics["index"],
            "sparse": {
                **hybrid_metrics["index"],
                "hybrid_manifest": _load_json(args.hybrid_manifest),
            },
            "rrf": {
                "dense_candidate_limit": audit["candidate_limits"]["dense"],
                "sparse_candidate_limit": audit["candidate_limits"]["sparse"],
                "rrf_k": audit["rrf_k"],
            },
            "runtime": {
                "python": sys.version.split()[0],
                "qdrant_client": _package_version("qdrant-client"),
                "fastembed": _package_version("fastembed"),
                "qdrant_server": _qdrant_server_version(settings.qdrant_url, settings.qdrant_port),
            },
            "collections": {
                "v1": _collection_summary(client, settings.qdrant_collection),
                "v2": _collection_summary(client, settings.qdrant_hybrid_collection),
            },
            "metrics": {
                "dense": dense_metrics["overall"],
                "sparse": sparse_metrics["overall"],
                "hybrid": hybrid_metrics["overall"],
            },
            "candidate_pool_audit": {
                "artifact": str(args.candidate_audit),
                "aggregate_metrics": audit["aggregate_metrics"],
                "per_retrieval_scenario": audit["per_retrieval_scenario"],
                "critical_query_coverage": audit["critical_query_coverage"],
                "rrf_diagnosis": audit["rrf_diagnosis"],
            },
            "critical_query_ranks": _critical_ranks(hybrid_metrics),
            "docker_validation": {
                "status": args.docker_status,
                "detail": args.docker_detail,
            },
            "recommended_phase_5_candidate_strategies": [
                "sparse_top20",
                "hybrid_top20",
                "dense20_union_sparse20",
            ],
            "known_limitations": [
                "Sparse BM25 currently outranks hybrid RRF on the retrieval development set.",
                "Two of three bilingual critical intents remain outside hybrid top 5.",
                "English-to-Vietnamese evidence has lower candidate coverage than monolingual "
                "retrieval.",
                "The 30-query set is a development set, not the Phase 6 held-out evaluation set.",
                "OCR is disabled; page-batch boundaries can affect chunks and multi-page tables.",
            ],
            "known_blockers": args.blocker,
        }
        _write_json_atomic(args.output, payload)
    except (OSError, ValueError, KeyError, RetrievalError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 5 readiness artifact written: {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Phase 5 readiness handoff artifact.")
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/manual-batched.jsonl"))
    parser.add_argument(
        "--dense-metrics", type=Path, default=METRICS_DIR / "dense-baseline-closure.json"
    )
    parser.add_argument("--sparse-metrics", type=Path, default=METRICS_DIR / "sparse-baseline.json")
    parser.add_argument("--hybrid-metrics", type=Path, default=METRICS_DIR / "hybrid-baseline.json")
    parser.add_argument(
        "--candidate-audit", type=Path, default=METRICS_DIR / "candidate-pool-audit.json"
    )
    parser.add_argument(
        "--hybrid-manifest", type=Path, default=METRICS_DIR / "hybrid-index-manifest.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-4-commit", default="51ead18")
    parser.add_argument(
        "--status",
        choices=("ready", "ready_with_documented_deviation", "blocked"),
        default="ready_with_documented_deviation",
    )
    parser.add_argument(
        "--docker-status",
        choices=("pass", "pending", "blocked"),
        default="pending",
    )
    parser.add_argument(
        "--docker-detail",
        default=(
            "Baked Phase 4 modules are present in existing images; "
            "reproducible closure build pending."
        ),
    )
    parser.add_argument("--blocker", action="append", default=[])
    parser.add_argument(
        "--git-branch",
        help="Branch recorded when generation runs in an image without the .git directory.",
    )
    parser.add_argument(
        "--working-tree-short",
        action="append",
        default=[],
        help="One git status --short line; repeat to record a host working tree from a container.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _collection_summary(client: Any, collection_name: str) -> dict[str, Any]:
    collection = client.get_collection(collection_name)
    return {
        "name": collection_name,
        "point_count": client.count(collection_name, exact=True).count,
        "vectors": _model_value(collection.config.params.vectors),
        "sparse_vectors": _model_value(collection.config.params.sparse_vectors),
        "payload_schema": {
            key: value.model_dump(mode="json")
            for key, value in (getattr(collection, "payload_schema", None) or {}).items()
        },
    }


def _model_value(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {key: _model_value(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if dump is not None else value


def _validate_frozen_contract(
    frozen_metadata: dict[str, Any],
    *,
    document_id: str,
    artifacts: tuple[dict[str, Any], ...],
) -> None:
    expected_hash = frozen_metadata["chunk_ids_sha256"]
    expected_count = frozen_metadata["chunk_count"]
    for artifact in artifacts:
        artifact_frozen = artifact.get("frozen_chunk_set")
        if artifact_frozen is None:
            raise ValueError("Artifact is missing frozen_chunk_set metadata")
        artifact_document_ids = artifact_frozen.get("document_ids", [])
        artifact_document_id = artifact.get("document_id")
        if (
            (artifact_document_id is not None and artifact_document_id != document_id)
            or (artifact_document_id is None and artifact_document_ids != [document_id])
            or artifact_frozen.get("chunk_count") != expected_count
            or artifact_frozen.get("chunk_ids_sha256") != expected_hash
        ):
            raise ValueError("Artifact does not match the current frozen chunk-set contract")


def _critical_ranks(hybrid_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "language": row["language"],
            "retrieval_scenario": row["retrieval_scenario"],
            "direct_evidence_rank": row["direct_evidence_rank"],
        }
        for row in hybrid_metrics["critical_questions"]
    ]


def _qdrant_server_version(url: str, port: int) -> str | None:
    try:
        with urlopen(f"{url}:{port}", timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload.get("version") if isinstance(payload, dict) else None


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() or None


def _git_lines(*args: str) -> list[str]:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except OSError:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary_path.replace(output_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
