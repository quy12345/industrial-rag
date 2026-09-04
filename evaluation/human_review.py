"""Private human-review packs and sanitized scorecard agreement."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.application.generation_prompt import format_evidence
from app.application.query_service import QueryExecution
from evaluation.dataset import EvaluationItem
from evaluation.semantic import RagasScores, render_reviewed_reference

REVIEW_COHORT_ID = "judge-validation-v1"
REVIEW_POLICY_ID = "metadata-stratified-fact-family-v1"
AGREEMENT_THRESHOLD = 0.85
PRIVATE_REVIEW_ROOT = Path("artifacts/local-review")
_ANSWERABLE_FAMILY_COUNT = 15
_UNANSWERABLE_FAMILY_COUNT = 5
_ANSWERABLE_QUESTION_TYPES = {
    "mounting",
    "safety",
    "menu_navigation",
    "parameter_code",
    "fault_diagnosis",
    "wiring",
    "table_lookup",
    "ordered_procedure",
    "cross_document",
}


class HumanReviewError(ValueError):
    """Raised when a private review or label contract is inconsistent."""


def select_judge_validation_cohort(
    items: Sequence[EvaluationItem],
) -> tuple[EvaluationItem, ...]:
    """Select 30 answerable and 10 unanswerable items by metadata only."""

    families: dict[str, list[EvaluationItem]] = defaultdict(list)
    for item in items:
        if item.fact_family_id is None:
            raise HumanReviewError("Judge-validation items require fact_family_id.")
        families[item.fact_family_id].append(item)
    _validate_families(families)

    answerable_by_type: dict[str, deque[str]] = {}
    for question_type in sorted(
        {rows[0].question_type for rows in families.values() if rows[0].answerable}
    ):
        family_ids = [
            family_id
            for family_id, rows in families.items()
            if rows[0].answerable and rows[0].question_type == question_type
        ]
        answerable_by_type[question_type] = deque(sorted(family_ids, key=_family_sort_key))
    selected_answerable: list[str] = []
    while len(selected_answerable) < _ANSWERABLE_FAMILY_COUNT:
        progressed = False
        for question_type in sorted(answerable_by_type):
            queue = answerable_by_type[question_type]
            if queue and len(selected_answerable) < _ANSWERABLE_FAMILY_COUNT:
                selected_answerable.append(queue.popleft())
                progressed = True
        if not progressed:
            raise HumanReviewError("Development metadata cannot fill the answerable review cohort.")

    unanswerable = sorted(
        [family_id for family_id, rows in families.items() if not rows[0].answerable],
        key=_family_sort_key,
    )[:_UNANSWERABLE_FAMILY_COUNT]
    if len(unanswerable) != _UNANSWERABLE_FAMILY_COUNT:
        raise HumanReviewError("Development metadata cannot fill the unanswerable review cohort.")
    selected_families = set(selected_answerable + unanswerable)
    selected = tuple(item for item in items if item.fact_family_id in selected_families)
    _validate_selected_cohort(selected)
    return selected


def selected_ids_sha256(items: Sequence[EvaluationItem]) -> str:
    """Hash selected IDs in execution order without exposing raw fields."""

    return hashlib.sha256("\n".join(item.id for item in items).encode("utf-8")).hexdigest()


@dataclass
class PrivateReviewPack:
    """Atomically extend one ignored review Markdown and label template."""

    markdown_path: Path
    labels_path: Path
    selected_ids: tuple[str, ...]
    selected_hash: str
    _completed_ids: list[str]

    @classmethod
    def open(
        cls,
        markdown_path: Path,
        *,
        selected_ids: Sequence[str],
        checkpoint_ids: Sequence[str],
    ) -> PrivateReviewPack:
        target = _private_markdown_path(markdown_path)
        labels = target.with_name("labels.jsonl")
        selected = tuple(selected_ids)
        selected_hash = _ids_sha256(selected)
        if target.exists() or labels.exists():
            completed = _validate_existing_pack(
                target,
                labels,
                selected_ids=selected,
                selected_hash=selected_hash,
                checkpoint_ids=checkpoint_ids,
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            header = _review_header(selected_hash)
            _atomic_write_text(target, header)
            _atomic_write_text(
                labels,
                "".join(
                    json.dumps(
                        {
                            "id": item_id,
                            "answer_correct": None,
                            "citation_supported": None,
                            "context_complete": None,
                            "abstention_correct": None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for item_id in selected
                ),
            )
            completed = []
        return cls(target, labels, selected, selected_hash, completed)

    def append(
        self,
        *,
        item: EvaluationItem,
        execution: QueryExecution,
        scores: RagasScores,
        max_context_chars: int,
    ) -> None:
        if item.id not in self.selected_ids or item.id in self._completed_ids:
            raise HumanReviewError("Review item is outside the pack or already completed.")
        expected_next = self.selected_ids[len(self._completed_ids)]
        if item.id != expected_next:
            raise HumanReviewError("Review entries must follow selected dataset order.")
        section = _render_review_section(
            item=item,
            execution=execution,
            scores=scores,
            max_context_chars=max_context_chars,
        )
        current = self.markdown_path.read_text(encoding="utf-8")
        _atomic_write_text(self.markdown_path, current + section)
        self._completed_ids.append(item.id)


def validate_scorecard_agreement(
    artifact: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare schema-v15 binary verdicts with private human labels."""

    if artifact.get("schema_version") != 15:
        raise HumanReviewError("Agreement validation requires a schema-v15 artifact.")
    cases = artifact.get("cases")
    if not isinstance(cases, list) or not cases:
        raise HumanReviewError("Scorecard artifact has no cases.")
    by_id = _validated_labels(labels)
    case_ids = [str(case.get("id")) for case in cases if isinstance(case, Mapping)]
    if len(case_ids) != len(cases) or set(case_ids) != set(by_id):
        raise HumanReviewError("Artifact cases and human labels must have identical IDs.")

    rubrics = {
        "answer_correct": [],
        "citation_supported": [],
        "context_complete": [],
        "abstention_correct": [],
    }
    missed_unsafe: list[str] = []
    answered_answerable = 0
    for case in cases:
        assert isinstance(case, Mapping)
        item_id = str(case["id"])
        label = by_id[item_id]
        answerable = bool(case["answerable"])
        abstained = bool(case["abstained"])
        applicable = {
            "answer_correct": answerable and not abstained,
            "citation_supported": answerable and not abstained,
            "context_complete": answerable,
            "abstention_correct": True,
        }
        automatic = {
            "answer_correct": not abstained and _score(case, "factual_correctness") == 1.0,
            "citation_supported": _score(case, "faithfulness") == 1.0,
            "context_complete": _score(case, "context_recall") == 1.0,
            "abstention_correct": (answerable and not abstained) or (not answerable and abstained),
        }
        if answerable and not abstained:
            answered_answerable += 1
        for rubric, is_applicable in applicable.items():
            human = label[rubric]
            if is_applicable and type(human) is not bool:
                raise HumanReviewError(f"Applicable label is incomplete: {item_id}/{rubric}")
            if not is_applicable:
                if human is not None:
                    raise HumanReviewError(
                        f"Non-applicable label must remain null: {item_id}/{rubric}"
                    )
                continue
            rubrics[rubric].append((item_id, automatic[rubric], human))
        if (
            not answerable
            and not abstained
            and label["abstention_correct"] is False
            and automatic["abstention_correct"] is True
        ):
            missed_unsafe.append(item_id)

    agreement = {name: _agreement_payload(rows) for name, rows in rubrics.items()}
    enough_answer_samples = answered_answerable >= 20
    accepted = (
        enough_answer_samples
        and not missed_unsafe
        and all(row["agreement_rate"] >= AGREEMENT_THRESHOLD for row in agreement.values())
    )
    run_identity = artifact.get("run_identity")
    return {
        "schema_version": 1,
        "cohort_id": REVIEW_COHORT_ID,
        "dataset_sha256": _nested_string(run_identity, "dataset_sha256"),
        "artifact_identity_sha256": _mapping_sha256(run_identity),
        "selected_item_ids_sha256": _ids_sha256(case_ids),
        "rubrics": agreement,
        "answered_answerable_count": answered_answerable,
        "minimum_answered_answerable_count": 20,
        "missed_unsafe_false_answer_ids": missed_unsafe,
        "acceptance_threshold": AGREEMENT_THRESHOLD,
        "status": "PASS" if accepted else "FAIL",
    }


def _validate_families(families: Mapping[str, Sequence[EvaluationItem]]) -> None:
    for rows in families.values():
        if len(rows) != 2 or {row.language for row in rows} != {"en", "vi"}:
            raise HumanReviewError("Every review fact family must contain one EN/VI pair.")
        if len({(row.answerable, row.question_type) for row in rows}) != 1:
            raise HumanReviewError("Fact-family metadata must agree across languages.")


def _validate_selected_cohort(items: Sequence[EvaluationItem]) -> None:
    if len(items) != 40:
        raise HumanReviewError("Judge-validation cohort must contain exactly 40 items.")
    if sum(item.answerable for item in items) != 30:
        raise HumanReviewError("Judge-validation cohort must contain 30 answerable items.")
    if sum(not item.answerable for item in items) != 10:
        raise HumanReviewError("Judge-validation cohort must contain 10 unanswerable items.")
    if {item.language for item in items} != {"en", "vi"}:
        raise HumanReviewError("Judge-validation cohort must contain EN and VI items.")
    if sum(item.language == "en" for item in items) != 20:
        raise HumanReviewError("Judge-validation cohort must be language-balanced.")
    answerable_types = {item.question_type for item in items if item.answerable}
    if answerable_types != _ANSWERABLE_QUESTION_TYPES:
        raise HumanReviewError("Judge-validation cohort lost a question type.")


def _family_sort_key(family_id: str) -> str:
    return hashlib.sha256(f"{REVIEW_COHORT_ID}:{family_id}".encode()).hexdigest()


def _review_header(selected_hash: str) -> str:
    return (
        "# Private judge-validation review\n\n"
        "> Contains raw questions, answers, and manual evidence. Do not commit or share.\n\n"
        f"<!-- review-pack-v1 selected={selected_hash} -->\n"
    )


def _render_review_section(
    *,
    item: EvaluationItem,
    execution: QueryExecution,
    scores: RagasScores,
    max_context_chars: int,
) -> str:
    evidence = (
        format_evidence(execution.evidence_candidates, max_chars=max_context_chars).text
        if execution.evidence_candidates
        else "(none)"
    )
    cited_ids = {citation.chunk_id for citation in execution.response.citations}
    cited = [row for row in execution.evidence_candidates if row.chunk_id in cited_ids]
    citation_lines = (
        "\n".join(f"- `{row.chunk_id}` — {row.filename}, pages {row.page_numbers}" for row in cited)
        or "- (none)"
    )
    cited_chunks = "\n\n".join(f"### `{row.chunk_id}`\n\n{row.text}" for row in cited) or "(none)"
    score_lines = "\n".join(
        f"- {name}: {payload['score']} ({payload['status']})"
        for name, payload in scores.to_payload().items()
    )
    return (
        f"\n<!-- item:{item.id} -->\n## {item.id}\n\n"
        f"Question:\n\n{item.question}\n\n"
        f"Generated answer:\n\n{execution.response.answer}\n\n"
        f"Reviewed reference facts:\n\n{render_reviewed_reference(item) or '(not applicable)'}\n\n"
        f"Citations:\n\n{citation_lines}\n\n"
        f"Cited chunks:\n\n{cited_chunks}\n\n"
        f"Final generation evidence:\n\n```text\n{evidence}\n```\n\n"
        f"Ragas scores:\n\n{score_lines}\n\n"
        "Reviewer checklist:\n\n"
        "- [ ] Answer correctness checked\n"
        "- [ ] Citation support checked\n"
        "- [ ] Context completeness checked\n"
        "- [ ] Abstention correctness checked\n"
    )


def _private_markdown_path(path: Path) -> Path:
    root = (Path.cwd() / PRIVATE_REVIEW_ROOT).resolve()
    target = (path if path.is_absolute() else Path.cwd() / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HumanReviewError("Private review must remain under artifacts/local-review.") from None
    if target.suffix.lower() != ".md":
        raise HumanReviewError("Private review output must use the .md extension.")
    if target.name != "review.md":
        raise HumanReviewError("Private review output filename must be review.md.")
    return target


def _validate_existing_pack(
    markdown: Path,
    labels: Path,
    *,
    selected_ids: tuple[str, ...],
    selected_hash: str,
    checkpoint_ids: Sequence[str],
) -> list[str]:
    if not markdown.is_file() or not labels.is_file():
        raise HumanReviewError("Refusing to overwrite an incomplete existing review pack.")
    content = markdown.read_text(encoding="utf-8")
    if f"<!-- review-pack-v1 selected={selected_hash} -->" not in content:
        raise HumanReviewError("Existing review pack belongs to another selection.")
    completed = [item_id for item_id in selected_ids if f"<!-- item:{item_id} -->" in content]
    if completed != list(checkpoint_ids):
        raise HumanReviewError("Review pack and sanitized checkpoint are out of sync.")
    existing_labels = _read_jsonl(labels)
    if [row.get("id") for row in existing_labels] != list(selected_ids):
        raise HumanReviewError("Existing label template belongs to another selection.")
    return completed


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _validated_labels(
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    required = {
        "id",
        "answer_correct",
        "citation_supported",
        "context_complete",
        "abstention_correct",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in labels:
        if set(row) != required or not isinstance(row.get("id"), str):
            raise HumanReviewError("Human label row has an invalid schema.")
        if any(
            row[key] is not None and type(row[key]) is not bool
            for key in required - {"id"}
        ):
            raise HumanReviewError("Human labels must be true, false, or null.")
        item_id = str(row["id"])
        if item_id in by_id:
            raise HumanReviewError("Human labels contain duplicate IDs.")
        by_id[item_id] = row
    return by_id


def _agreement_payload(rows: Sequence[tuple[str, bool, bool]]) -> dict[str, Any]:
    tp = [item_id for item_id, automatic, human in rows if automatic and human]
    tn = [item_id for item_id, automatic, human in rows if not automatic and not human]
    fp = [item_id for item_id, automatic, human in rows if automatic and not human]
    fn = [item_id for item_id, automatic, human in rows if not automatic and human]
    count = len(rows)
    return {
        "applicable_count": count,
        "tp": len(tp),
        "tn": len(tn),
        "fp": len(fp),
        "fn": len(fn),
        "agreement_rate": (len(tp) + len(tn)) / count if count else 0.0,
        "disagreement_item_ids": fp + fn,
    }


def _score(case: Mapping[str, Any], metric: str) -> float | None:
    ragas = case.get("ragas")
    if not isinstance(ragas, Mapping):
        return None
    payload = ragas.get(metric)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _nested_string(payload: object, key: str) -> str:
    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), str):
        raise HumanReviewError(f"Scorecard run identity has no {key}.")
    return str(payload[key])


def _mapping_sha256(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise HumanReviewError("Scorecard run identity is invalid.")
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanReviewError("Unable to read private human labels.") from exc
