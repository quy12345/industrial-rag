"""Run a bounded real Phase 7 query smoke and write only sanitized diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.bootstrap import get_query_service
from app.config import get_settings
from app.errors import QueryPipelineError
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT

DEFAULT_OUTPUT = Path("artifacts/metrics/phase-7-query-smoke.json")

INSTALLATION_DOCUMENT_ID, PROGRAMMING_DOCUMENT_ID = PHASE7_RETRIEVAL_CONTRACT.document_ids

SCENARIOS = (
    (
        "vi_installation_answerable",
        "Tài liệu Installation Manual hỗ trợ những công việc nào?",
        INSTALLATION_DOCUMENT_ID,
        False,
    ),
    (
        "en_programming_answerable",
        "How do I navigate and configure the ATV320 display menus?",
        PROGRAMMING_DOCUMENT_ID,
        False,
    ),
    (
        "clearly_unanswerable",
        "What is the maximum hydraulic pressure of the excavator described in this document?",
        INSTALLATION_DOCUMENT_ID,
        True,
    ),
)


def main() -> int:
    args = _build_parser().parse_args()
    settings = get_settings()
    if settings.retrieval_profile != "phase7":
        print("Phase 7 query smoke requires RETRIEVAL_PROFILE=phase7; no request was made.")
        return 2
    artifact: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_commit": os.getenv("PHASE7_BASE_COMMIT") or _git_commit(),
        "retrieval_profile": settings.retrieval_profile,
        "corpus": {
            "document_ids": list(PHASE7_RETRIEVAL_CONTRACT.document_ids),
            "chunk_count": PHASE7_RETRIEVAL_CONTRACT.chunk_count,
            "chunk_ids_sha256": PHASE7_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        },
        "runtime_versions": _runtime_versions(),
        "generation_provider": settings.generation_provider,
        "model": settings.generation_model,
        "retrieval_strategy": settings.retrieval_strategy,
        "reranker_enabled": settings.rerank_enabled,
        "store": settings.openai_store,
        "test_scenarios": [],
    }
    if settings.generation_api_key is None:
        artifact.update(
            {
                "integration_run_status": "not_run",
                "reason": "api_key_unavailable",
            }
        )
        _atomic_write(args.output, artifact)
        print(f"Phase 7 real smoke NOT RUN: API key unavailable. Artifact: {args.output}")
        return 0

    service = get_query_service()
    all_passed = True
    for scenario_id, question, document_id, expected_abstained in SCENARIOS:
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "expected_abstained": expected_abstained,
        }
        try:
            execution = service.execute(
                question=question,
                document_id=document_id,
                top_k=5,
            )
            response = execution.response
            scenario_passed = (
                response.abstained == expected_abstained
                and (response.abstained or bool(response.citations))
            )
            row.update(
                {
                    "result_status": "pass" if scenario_passed else "fail",
                    "abstained": response.abstained,
                    "abstention_reason": response.abstention_reason,
                    "citation_count": len(response.citations),
                    "citation_ids_valid": all(
                        bool(citation.chunk_id) for citation in response.citations
                    ),
                    "stage_latency_ms": asdict(execution.timings),
                    "token_usage": asdict(execution.usage) if execution.usage else None,
                }
            )
            all_passed = all_passed and scenario_passed
        except QueryPipelineError as exc:
            all_passed = False
            row.update(
                {
                    "result_status": "dependency_error",
                    "error_class": type(exc).__name__,
                }
            )
        artifact["test_scenarios"].append(row)
    artifact["integration_run_status"] = "pass" if all_passed else "fail"
    _atomic_write(args.output, artifact)
    print(f"Phase 7 real smoke {artifact['integration_run_status'].upper()}: {args.output}")
    return 0 if all_passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_name = handle.name
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _runtime_versions() -> dict[str, str | None]:
    packages = ("industrial-rag", "langchain-core", "langchain-openai", "openai")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
