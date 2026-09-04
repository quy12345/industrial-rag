"""Local, provider-free human review of sanitized development evaluation records.

The E2E artifact deliberately excludes questions, expected fact text, and manual
content.  This module joins those local-only inputs for terminal inspection without
changing or enriching the persisted artifact.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from app.application.query_service import QueryExecution
from app.domain.documents import DocumentChunk
from evaluation.dataset import EvaluationItem, dataset_sha256


class DevelopmentReviewError(ValueError):
    """Raised when local review inputs do not describe the same development run."""


def build_live_development_review(
    *,
    item: EvaluationItem,
    execution: QueryExecution,
    record: Mapping[str, Any],
    chunks: Sequence[DocumentChunk],
    destination: Literal[
        "terminal_stdout_only", "private_local_markdown"
    ] = "terminal_stdout_only",
) -> dict[str, Any]:
    """Build one ephemeral review payload after an approved development rerun."""

    chunk_by_id = _unique_chunks(chunks)
    _validate_record_contract(item, record)
    case = _review_case(item, record, chunk_by_id)
    case["observed"].update(
        {
            "raw_answer_available": True,
            "answer": execution.response.answer,
            "citations": [
                citation.model_dump(mode="json")
                for citation in execution.response.citations
            ],
        }
    )
    case["generation_evidence"] = [
        {
            "source_id": f"S{index}",
            "chunk_id": candidate.chunk_id,
            "document_id": candidate.document_id,
            "filename": candidate.filename,
            "page_numbers": candidate.page_numbers,
            "headings": candidate.headings,
            "content_type": candidate.content_type,
            "text": candidate.text,
        }
        for index, candidate in enumerate(execution.evidence_candidates, start=1)
    ]
    return {
        "review_policy": {
            "dataset": "development",
            "destination": destination,
            "provider_calls": "approved_targeted_rerun",
            "persistence": "raw_content_forbidden_in_sanitized_artifacts",
            "interpretation": "new_provider_sample_not_original_artifact_output",
        },
        "case": case,
    }


def render_live_development_reviews_markdown(
    reviews: Sequence[Mapping[str, Any]],
) -> str:
    """Render approved raw development reviews for an ignored local file."""

    lines = [
        "# Private local development review",
        "",
        "> **LOCAL PRIVATE REPORT — DO NOT COMMIT OR PASTE INTO PUBLIC LOGS.**",
        "> This file may contain evaluation questions, generated answers, and manual text.",
        "",
    ]
    for review in reviews:
        case = _required_mapping(review, "case")
        item = _required_mapping(case, "item")
        expected = _required_mapping(case, "expected")
        observed = _required_mapping(case, "observed")
        item_id = str(item.get("id", "unknown-item"))
        lines.extend(
            [
                f"## {_markdown_inline(item_id)}",
                "",
                (
                    f"- Language: `{_markdown_inline(item.get('language'))}`\n"
                    f"- Question type: `{_markdown_inline(item.get('question_type'))}`\n"
                    f"- Answerable: `{_markdown_inline(item.get('answerable'))}`\n"
                    f"- Abstained: `{_markdown_inline(observed.get('abstained'))}`"
                ),
                "",
                "### Question",
                "",
                _markdown_pre(item.get("question")),
                "",
                "### Generated answer",
                "",
                _markdown_pre(observed.get("answer")),
                "",
                "### Expected facts and evaluator verdicts",
                "",
            ]
        )
        fact_results = {
            str(result.get("id")): result
            for result in _mapping_list(observed.get("answer_fact_results"))
        }
        facts = _mapping_list(expected.get("answer_facts"))
        if not facts:
            lines.extend(["No expected facts.", ""])
        for fact in facts:
            fact_id = str(fact.get("id", "unknown-fact"))
            result = fact_results.get(fact_id, {})
            lines.extend(
                [
                    f"#### Fact `{_markdown_inline(fact_id)}`",
                    "",
                    (
                        f"- Deterministic match: `"
                        f"{_markdown_inline(result.get('deterministic_matched'))}`\n"
                        f"- Final match: `{_markdown_inline(result.get('final_matched'))}`\n"
                        f"- Resolution: `{_markdown_inline(result.get('resolution'))}`\n"
                        f"- Judge verdict: `{_markdown_inline(result.get('judge_verdict'))}`"
                    ),
                    "",
                    "Reviewed fact contract:",
                    "",
                    _markdown_pre(json.dumps(fact, ensure_ascii=False, indent=2)),
                    "",
                ]
            )
        lines.extend(["### Authoritative evidence requirements", ""])
        requirements = _mapping_list(expected.get("evidence_requirements"))
        if not requirements:
            lines.extend(["No answerable evidence requirements.", ""])
        for requirement in requirements:
            requirement_id = str(requirement.get("id", "unknown-requirement"))
            coverage_value = observed.get("citation_authoritative_requirement_coverage")
            coverage = coverage_value if isinstance(coverage_value, Mapping) else {}
            requirement_result = next(
                (
                    candidate
                    for candidate in _mapping_list(coverage.get("requirements"))
                    if candidate.get("id") == requirement_id
                ),
                {},
            )
            lines.extend(
                [
                    f"#### Requirement `{_markdown_inline(requirement_id)}`",
                    "",
                    (
                        "- Citation coverage: `"
                        f"{_markdown_inline(requirement_result.get('rank') is not None)}`\n"
                        "- Matched chunk: `"
                        f"{_markdown_inline(requirement_result.get('matched_chunk_id'))}`\n"
                        "- Provenance: `"
                        f"{_markdown_inline(requirement_result.get('provenance'))}`"
                    ),
                    "",
                    "Canonical sources:",
                    "",
                    _markdown_pre(
                        json.dumps(
                            requirement.get("canonical_sources", []),
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                    "",
                    "Reviewed alternatives:",
                    "",
                    _markdown_pre(
                        json.dumps(
                            requirement.get("reviewed_alternative_sources", []),
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                    "",
                ]
            )
        lines.extend(["### Citations and cited evidence", ""])
        citations = _mapping_list(observed.get("citations"))
        evidence_by_chunk_id = {
            str(evidence.get("chunk_id")): evidence
            for evidence in _mapping_list(case.get("generation_evidence"))
        }
        if not citations:
            lines.extend(["No citations.", ""])
        for index, citation in enumerate(citations, start=1):
            chunk_id = str(citation.get("chunk_id", "unknown-chunk"))
            pages = ", ".join(str(page) for page in citation.get("page_numbers", []))
            filename = str(citation.get("filename", "unknown-file"))
            lines.extend(
                [
                    (
                        f"#### Citation {index}: {_markdown_inline(filename)} — "
                        f"page {pages or 'unknown'}"
                    ),
                    "",
                    f"- Chunk ID: `{_markdown_inline(chunk_id)}`",
                    "",
                    "Citation excerpt:",
                    "",
                    _markdown_pre(citation.get("excerpt")),
                    "",
                ]
            )
            evidence = evidence_by_chunk_id.get(chunk_id)
            if evidence is not None:
                lines.extend(
                    [
                        "<details>",
                        "<summary>Full cited generation evidence</summary>",
                        "",
                        _markdown_pre(evidence.get("text")),
                        "",
                        "</details>",
                        "",
                    ]
                )
        lines.extend(
            [
                "### Human review",
                "",
                "- [ ] The answer addresses the question.",
                "- [ ] Every required fact is present and correctly related.",
                "- [ ] Citation document and page are correct.",
                "- [ ] Cited evidence supports the answer.",
                "- [ ] The evaluator verdict is correct.",
                "",
                "Reviewer notes:",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def build_development_review(
    *,
    dataset: Sequence[EvaluationItem],
    chunks: Sequence[DocumentChunk],
    artifact: Mapping[str, Any],
    item_ids: Sequence[str],
    repeat_index: int = 1,
) -> dict[str, Any]:
    """Join local raw development inputs with selected sanitized result records."""

    requested_ids = _validated_item_ids(item_ids)
    if repeat_index not in {1, 2, 3}:
        raise DevelopmentReviewError("Repeat index must be 1, 2, or 3.")

    run_identity = _required_mapping(artifact, "run_identity")
    if run_identity.get("dataset") != "development":
        raise DevelopmentReviewError("Only development artifacts may be reviewed.")
    if run_identity.get("dataset_sha256") != dataset_sha256(dataset):
        raise DevelopmentReviewError(
            "The artifact dataset identity does not match the local development dataset."
        )

    chunk_by_id = _unique_chunks(chunks)
    corpus = _required_mapping(run_identity, "corpus")
    expected_chunk_hash = _ordered_chunk_id_hash(chunk_by_id)
    if corpus.get("chunk_ids_sha256") != expected_chunk_hash:
        raise DevelopmentReviewError(
            "The artifact corpus identity does not match the local frozen chunks."
        )

    item_by_id = {item.id: item for item in dataset}
    missing_items = [item_id for item_id in requested_ids if item_id not in item_by_id]
    if missing_items:
        raise DevelopmentReviewError(
            f"Development item IDs were not found: {', '.join(missing_items)}"
        )

    records = artifact.get("per_query")
    if not isinstance(records, list):
        raise DevelopmentReviewError("The E2E artifact has no per_query record list.")
    record_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise DevelopmentReviewError("The E2E artifact contains a malformed record.")
        record_id = record.get("id")
        record_repeat = record.get("repeat_index")
        if not isinstance(record_id, str) or type(record_repeat) is not int:
            raise DevelopmentReviewError("The E2E artifact contains an invalid record key.")
        key = record_id, record_repeat
        if key in record_by_key:
            raise DevelopmentReviewError("The E2E artifact contains duplicate record keys.")
        record_by_key[key] = record

    cases = []
    for item_id in requested_ids:
        key = item_id, repeat_index
        record = record_by_key.get(key)
        if record is None:
            raise DevelopmentReviewError(
                f"Artifact record was not found for {item_id} repeat {repeat_index}."
            )
        item = item_by_id[item_id]
        _validate_record_contract(item, record)
        cases.append(_review_case(item, record, chunk_by_id))

    return {
        "review_policy": {
            "dataset": "development",
            "destination": "terminal_stdout_only",
            "provider_calls": "none",
            "raw_answer": "unavailable_in_sanitized_artifact",
            "interpretation": "diagnostic_hints_require_human_confirmation",
        },
        "artifact": {
            "schema_version": artifact.get("schema_version"),
            "generation_variant": _generation_variant(run_identity),
            "generation_model": run_identity.get("generation_model"),
            "repeat_index": repeat_index,
        },
        "cases": cases,
    }


def _review_case(
    item: EvaluationItem,
    record: Mapping[str, Any],
    chunk_by_id: Mapping[str, DocumentChunk],
) -> dict[str, Any]:
    evidence_ids = _string_list(record, "evidence_candidate_ids")
    return {
        "item": {
            "id": item.id,
            "fact_family_id": item.fact_family_id,
            "language": item.language,
            "question_type": item.question_type,
            "answerable": item.answerable,
            "question": item.question,
        },
        "expected": {
            "document_ids": item.expected_document_ids,
            "pages": item.expected_pages,
            "phrases": item.expected_phrases,
            "answer_facts": [
                fact.model_dump(mode="json", exclude_none=True)
                for fact in item.expected_answer_facts
            ],
            "qrel_chunks": [
                _chunk_payload(_required_chunk(chunk_by_id, chunk_id), source_id=None)
                for chunk_id in item.relevant_chunk_ids
            ],
            "evidence_requirements": _review_evidence_requirements(item, chunk_by_id),
            "unanswerable_reason": item.unanswerable_reason,
        },
        "observed": {
            "raw_answer_available": False,
            "abstained": record.get("abstained"),
            "abstention_reason": record.get("abstention_reason"),
            "failure_class": record.get("failure_class"),
            "citation_ids": record.get("citation_ids"),
            "citation_document_ids": record.get("citation_document_ids"),
            "citation_authoritative_requirement_coverage": record.get(
                "citation_authoritative_requirement_coverage"
            ),
            "citation_authoritative_provenance": record.get(
                "citation_authoritative_provenance"
            ),
            "missing_answer_fact_ids": record.get("missing_answer_fact_ids"),
            "answer_fact_results": record.get("answer_fact_results"),
            "candidate_direct_evidence_rank": record.get(
                "candidate_direct_evidence_rank"
            ),
            "ranked_direct_evidence_rank": record.get("ranked_direct_evidence_rank"),
            "direct_evidence_rank": record.get("direct_evidence_rank"),
            "unexpected_citation_document_ids": record.get(
                "unexpected_citation_document_ids"
            ),
            "evidence_rendering": record.get("evidence_rendering"),
        },
        "retrieval_trace": record.get("retrieval_trace"),
        "generation_evidence": [
            _chunk_payload(
                _required_chunk(chunk_by_id, chunk_id),
                source_id=f"S{index}",
            )
            for index, chunk_id in enumerate(evidence_ids, start=1)
        ],
        "diagnostic_hints": _diagnostic_hints(item, record),
    }


def _diagnostic_hints(
    item: EvaluationItem,
    record: Mapping[str, Any],
) -> list[str]:
    hints: list[str] = []
    if not item.answerable:
        if record.get("abstained") is False:
            hints.append("false_answer_requires_generation_safety_review")
        return hints or ["no_known_failure_signal"]

    relevant = set(item.relevant_chunk_ids)
    trace = record.get("retrieval_trace")
    if record.get("candidate_direct_evidence_rank") is None:
        if isinstance(trace, Mapping) and relevant.intersection(
            _string_list(trace, "removed_by_candidate_selection_ids")
        ):
            hints.append("qrel_removed_by_candidate_selection")
        elif isinstance(trace, Mapping) and relevant.intersection(
            _string_list(trace, "removed_by_same_document_dedup_ids")
        ):
            hints.append("qrel_removed_by_same_document_dedup")
        else:
            hints.append("retrieval_ceiling_or_qrel_gap")
    elif record.get("direct_evidence_rank") is None:
        hints.append("qrel_did_not_reach_generation_evidence")

    if record.get("abstained") is True and record.get("direct_evidence_rank") is not None:
        hints.append("false_abstention_with_direct_evidence")
    if record.get("missing_answer_fact_ids") and record.get("direct_evidence_rank") is not None:
        hints.append("generator_omission_or_evaluator_false_negative")
    if record.get("unexpected_citation_document_ids"):
        hints.append("citation_provenance_mismatch")
    return hints or ["no_known_failure_signal"]


def _review_evidence_requirements(
    item: EvaluationItem,
    chunk_by_id: Mapping[str, DocumentChunk],
) -> list[dict[str, Any]]:
    requirements = item.evidence_requirements
    if not requirements and item.relevant_chunk_ids:
        return [
            {
                "id": "legacy-qrel-group",
                "canonical_sources": [
                    _chunk_payload(_required_chunk(chunk_by_id, chunk_id), source_id=None)
                    for chunk_id in item.relevant_chunk_ids
                ],
                "reviewed_alternative_sources": [],
            }
        ]
    return [
        {
            "id": requirement.id,
            "canonical_sources": [
                _chunk_payload(_required_chunk(chunk_by_id, chunk_id), source_id=None)
                for chunk_id in requirement.canonical_chunk_ids
            ],
            "reviewed_alternative_sources": [
                _chunk_payload(_required_chunk(chunk_by_id, chunk_id), source_id=None)
                for chunk_id in requirement.reviewed_alternative_chunk_ids
            ],
        }
        for requirement in requirements
    ]


def _chunk_payload(chunk: DocumentChunk, *, source_id: str | None) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "page_numbers": chunk.page_numbers,
        "headings": chunk.headings,
        "content_type": chunk.content_type,
        "text": chunk.text,
    }


def _validate_record_contract(
    item: EvaluationItem,
    record: Mapping[str, Any],
) -> None:
    expected = {
        "answerable": item.answerable,
        "fact_family_id": item.fact_family_id,
        "language": item.language,
        "question_type": item.question_type,
        "scenario": item.scenario,
    }
    mismatched = [key for key, value in expected.items() if record.get(key) != value]
    if mismatched:
        raise DevelopmentReviewError(
            f"Artifact record {item.id} does not match dataset fields: {', '.join(mismatched)}"
        )


def _validated_item_ids(item_ids: Sequence[str]) -> list[str]:
    normalized = [item_id.strip() for item_id in item_ids]
    if not normalized or any(not item_id for item_id in normalized):
        raise DevelopmentReviewError("At least one non-blank development item ID is required.")
    if len(set(normalized)) != len(normalized):
        raise DevelopmentReviewError("Development review item IDs must be unique.")
    return normalized


def _unique_chunks(chunks: Sequence[DocumentChunk]) -> dict[str, DocumentChunk]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if not chunk_by_id:
        raise DevelopmentReviewError("The local frozen chunk set is empty.")
    if len(chunk_by_id) != len(chunks):
        raise DevelopmentReviewError("The local frozen chunk set contains duplicate IDs.")
    return chunk_by_id


def _required_chunk(
    chunk_by_id: Mapping[str, DocumentChunk], chunk_id: str
) -> DocumentChunk:
    chunk = chunk_by_id.get(chunk_id)
    if chunk is None:
        raise DevelopmentReviewError(f"Local frozen chunk was not found: {chunk_id}")
    return chunk


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DevelopmentReviewError(f"The E2E artifact has no valid {key} mapping.")
    return value


def _mapping_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _markdown_pre(value: object) -> str:
    rendered = "" if value is None else str(value)
    return f"<pre>{html.escape(rendered)}</pre>"


def _markdown_inline(value: object) -> str:
    rendered = "" if value is None else str(value)
    return html.escape(rendered.replace("`", "'"))


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DevelopmentReviewError(f"Artifact field {key} must be a string list.")
    return value


def _generation_variant(run_identity: Mapping[str, Any]) -> str | None:
    experiment = run_identity.get("generation_experiment")
    if not isinstance(experiment, Mapping):
        return None
    identifier = experiment.get("id")
    return identifier if isinstance(identifier, str) else None


def _ordered_chunk_id_hash(chunk_by_id: Mapping[str, DocumentChunk]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(chunk_by_id)).encode("utf-8")
    ).hexdigest()
