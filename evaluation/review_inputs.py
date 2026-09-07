"""Private, identity-bound generation snapshots; no provider or retrieval clients."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.application.generation_prompt import format_evidence
from app.application.query_service import QueryExecution
from app.domain.documents import DocumentChunk
from app.domain.retrieval import RetrievalCandidate
from app.domain.retrieval_contracts import ATV320_RETRIEVAL_CONTRACT
from app.infrastructure.corpus_artifacts import chunk_set_metadata, write_jsonl_atomic
from evaluation.dataset import EvaluationItem, dataset_sha256
from evaluation.semantic import RagasScores, render_reviewed_reference

INPUT_POLICY_ID = "private_generation_snapshot_v1"
_EXECUTION = TypeAdapter(QueryExecution)


class ReviewInputError(ValueError):
    """Sanitized error: never include a raw validation exception or payload."""


def load_replay_source(
    path: Path,
    *,
    items: Sequence[EvaluationItem],
    chunks: Sequence[DocumentChunk],
    source_files: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the sanitized source before reading private answers or creating clients."""
    return _load_e2e_source(
        path,
        items=items,
        chunks=chunks,
        source_files=source_files,
        allowed_versions={17, 18, 19},
    )


def load_scope_replay_source(
    path: Path,
    *,
    items: Sequence[EvaluationItem],
    chunks: Sequence[DocumentChunk],
) -> dict[str, Any]:
    """Validate an identity-bound live source without pinning it to current code.

    Scope replay compares a historical source boundary with a new policy. It
    therefore validates the origin identity shape but deliberately does not
    require its production source hashes to equal the target policy's hashes.
    """
    return _load_e2e_source(
        path,
        items=items,
        chunks=chunks,
        source_files=None,
        allowed_versions={18, 19},
    )


def _load_e2e_source(
    path: Path,
    *,
    items: Sequence[EvaluationItem],
    chunks: Sequence[DocumentChunk],
    source_files: Mapping[str, str] | None,
    allowed_versions: set[int],
) -> dict[str, Any]:
    """Validate sanitized E2E records before loading a private input journal."""
    from evaluation.e2e import CASE_KEYS, RAGAS_METRICS

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        version = artifact["schema_version"]
        identity = artifact["run_identity"]
        if version not in allowed_versions or identity["artifact_schema_version"] != version:
            raise ValueError
        source_identity = identity["source_identity"]
        if source_identity["version"] != {17: 13, 18: 14, 19: 15}[version]:
            raise ValueError
        if identity["dataset"] != "development" or identity["dataset_sha256"] != dataset_sha256(
            items
        ):
            raise ValueError
        if identity["corpus"] != chunk_set_metadata(chunks):
            raise ValueError
        if identity["contract_chunk_ids_sha256"] != ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256:
            raise ValueError
        if (
            identity["evidence_top_k"] != 5
            or type(identity["generation_max_context_chars"]) is not int
        ):
            raise ValueError
        if identity["generation_max_context_chars"] <= 0:
            raise ValueError
        required_source_keys = (
            "generation_prompt",
            "query_service",
            "retrieval_composition",
            "retrieval_contract",
        )
        if source_files is not None:
            for key in required_source_keys:
                if source_identity["files"][key] != source_files[key]:
                    raise ValueError
        elif any(
            not isinstance(source_identity["files"].get(key), str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_identity["files"][key])
            for key in required_source_keys
        ):
            raise ValueError
        ids = identity["selected_item_ids"]
        if not ids or len(ids) > 120 or len(set(ids)) != len(ids):
            raise ValueError
        by_id = {item.id: item for item in items}
        if [case["id"] for case in artifact["cases"]] != ids:
            raise ValueError
        corpus_ids = {chunk.chunk_id for chunk in chunks}
        for case in artifact["cases"]:
            expected_keys = CASE_KEYS - {"input_sha256"} if version == 17 else CASE_KEYS
            if set(case) != expected_keys:
                raise ValueError
            item = by_id[case["id"]]
            for key in ("id", "fact_family_id", "language", "question_type", "answerable"):
                if case[key] != getattr(item, key):
                    raise ValueError
            if (
                type(case["abstained"]) is not bool
                or type(case["citation_reference_valid"]) is not bool
            ):
                raise ValueError
            if case["outcome"] != ("abstained" if case["abstained"] else "answered"):
                raise ValueError
            evidence = case["evidence_candidate_ids"]
            citations = case["citation_ids"]
            if not isinstance(evidence, list) or not isinstance(citations, list):
                raise ValueError
            if len(evidence) > 5 or len(set(evidence)) != len(evidence):
                raise ValueError
            if not set(evidence + citations) <= corpus_ids:
                raise ValueError
            valid = (
                not citations
                if case["abstained"]
                else bool(citations) and set(citations) <= set(evidence)
            )
            if case["citation_reference_valid"] != valid:
                raise ValueError
            if not re.fullmatch(r"[0-9a-f]{64}", case["retrieval_evidence_fingerprint"]):
                raise ValueError
            if version >= 18 and not re.fullmatch(r"[0-9a-f]{64}", case["input_sha256"]):
                raise ValueError
            if set(case["timings_ms"]) != {
                "retrieval",
                "rerank",
                "evidence_gate",
                "generation",
                "citation_validation",
                "judge",
                "total",
            }:
                raise ValueError
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in case["timings_ms"].values()
            ):
                raise ValueError
            usage = case["usage"]
            if usage is not None and set(usage) != {
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
            }:
                raise ValueError
            if any(
                value is not None and (type(value) is not int or value < 0)
                for value in (usage or {}).values()
            ):
                raise ValueError
            if set(case["ragas"]) != set(RAGAS_METRICS):
                raise ValueError
            scores_from_payload(case["ragas"])
        return artifact
    except Exception:
        raise ReviewInputError(
            "Replay artifact schema, dataset, corpus, or runtime identity mismatch."
        ) from None


def scores_from_payload(payload: Mapping[str, Any]) -> RagasScores:
    """Decode only the five sanitized metric payloads."""
    from evaluation.semantic import CORE_METRICS, DIAGNOSTIC_METRICS, MetricScore

    try:
        return RagasScores(
            *(MetricScore(**payload[name]) for name in CORE_METRICS + DIAGNOSTIC_METRICS)
        )
    except Exception:
        raise ReviewInputError("Invalid saved metric payload.") from None


def payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def private_path(path: Path) -> Path:
    target = path.resolve()
    if not target.is_relative_to(Path("artifacts/local-review").resolve()):
        raise ReviewInputError("Review inputs must remain under artifacts/local-review.")
    return target


def corpus_candidate(chunk: DocumentChunk) -> RetrievalCandidate:
    """Reconstruct source metadata, never ranking, from the frozen corpus."""
    metadata = dict(chunk.metadata)
    metadata.update(ATV320_RETRIEVAL_CONTRACT.document_context_by_id.get(chunk.document_id, {}))
    return RetrievalCandidate(**(chunk.model_dump() | {"metadata": metadata}), score=0.0)


def verify_evidence(execution: QueryExecution, corpus: Mapping[str, DocumentChunk]) -> None:
    fields = ("document_id", "filename", "text", "page_numbers", "headings", "content_type")
    ids = [row.chunk_id for row in execution.evidence_candidates]
    if len(ids) != len(set(ids)) or len(ids) > 5:
        raise ReviewInputError("Private input has an invalid final-evidence boundary.")
    for row in execution.evidence_candidates:
        chunk = corpus.get(row.chunk_id)
        if chunk is None or any(getattr(row, name) != getattr(chunk, name) for name in fields):
            raise ReviewInputError("Private input evidence differs from the frozen corpus.")
        canonical = corpus_candidate(chunk)
        for name in ("document_title", "document_role"):
            if row.metadata.get(name, "n/a") != canonical.metadata.get(name, "n/a"):
                raise ReviewInputError("Private input document context differs from the corpus.")


@dataclass(frozen=True)
class ReviewInput:
    """One raw answer and its exact generation boundary, retained only privately."""

    item: EvaluationItem
    execution: QueryExecution
    max_context_chars: int

    def to_payload(self) -> dict[str, Any]:
        execution = _EXECUTION.dump_python(self.execution, mode="json", exclude={"retrieval_trace"})
        payload = {
            "id": self.item.id,
            "question": self.item.question,
            "reference": render_reviewed_reference(self.item),
            "execution": execution,
            "max_context_chars": self.max_context_chars,
            "generation_context": self.generation_context(),
        }
        return payload | {"input_sha256": payload_sha256(payload)}

    def generation_context(self) -> str:
        if self.execution.evidence_context_chars <= 0 or not self.execution.evidence_candidates:
            return ""
        return format_evidence(
            self.execution.evidence_candidates, max_chars=self.max_context_chars
        ).text

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        item: EvaluationItem,
        corpus: Mapping[str, DocumentChunk],
    ) -> ReviewInput:
        try:
            expected = {
                "id",
                "question",
                "reference",
                "execution",
                "max_context_chars",
                "generation_context",
                "input_sha256",
            }
            if set(payload) != expected:
                raise ValueError
            raw = {key: value for key, value in payload.items() if key != "input_sha256"}
            if payload_sha256(raw) != payload["input_sha256"]:
                raise ValueError
            if (payload["id"], payload["question"], payload["reference"]) != (
                item.id,
                item.question,
                render_reviewed_reference(item),
            ):
                raise ValueError
            execution = _EXECUTION.validate_python(payload["execution"])
            maximum = payload["max_context_chars"]
            if type(maximum) is not int or maximum <= 0:
                raise ValueError
            result = cls(item, execution, maximum)
            verify_evidence(execution, corpus)
            if result.generation_context() != payload["generation_context"]:
                raise ValueError
            return result
        except Exception:
            raise ReviewInputError(
                "Private input hash, schema, or source boundary mismatch."
            ) from None


@dataclass
class PrivateInputJournal:
    """Write generation before judging; never regenerate an already stored answer."""

    path: Path
    identity: dict[str, Any]
    rows: dict[str, ReviewInput]

    @classmethod
    def open(
        cls,
        path: Path,
        identity: dict[str, Any],
        items: Sequence[EvaluationItem],
        corpus: Mapping[str, DocumentChunk],
    ) -> PrivateInputJournal:
        target = private_path(path)
        rows: dict[str, ReviewInput] = {}
        if target.exists():
            try:
                content = [
                    json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
                ]
                if content[0] != {"policy": INPUT_POLICY_ID, "run_identity": identity}:
                    raise ValueError
                by_id = {item.id: item for item in items}
                for payload in content[1:]:
                    item_id = payload["id"]
                    if item_id in rows or item_id not in by_id:
                        raise ValueError
                    rows[item_id] = ReviewInput.from_payload(payload, by_id[item_id], corpus)
                if list(rows) != [item.id for item in items[: len(rows)]]:
                    raise ValueError
            except Exception:
                raise ReviewInputError(
                    "Private snapshot journal is invalid or belongs to another run."
                ) from None
        return cls(target, identity, rows)

    def append(self, value: ReviewInput) -> None:
        if value.item.id in self.rows:
            raise ReviewInputError("Refusing to overwrite a captured generation input.")
        updated = self.rows | {value.item.id: value}
        write_jsonl_atomic(
            self.path,
            [
                {"policy": INPUT_POLICY_ID, "run_identity": self.identity},
                *(row.to_payload() for row in updated.values()),
            ],
        )
        self.rows = updated
