"""Offline regression tests for sanitized E2E and scope checkpoint boundaries."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.policies.evidence_scope import evaluate_evidence_scope
from app.domain.retrieval import RetrievalCandidate
from evaluation.dataset import EvaluationItem, ExpectedAnswerFact
from evaluation.e2e import build_case_record
from evaluation.semantic import DIAGNOSTIC_METRICS, MetricScore, RagasScores
from scripts.evaluation import evaluate_e2e as cli

CORPUS_IDS = frozenset({"c0", "c1", "c2"})


def _case() -> dict:
    scored = {"status": "complete", "score": 0.5, "latency_ms": 1.0}
    na = {"status": "not_applicable", "score": None, "latency_ms": 0.0}
    return {
        "id": "a1",
        "fact_family_id": "family-a1",
        "language": "en",
        "question_type": "safety",
        "answerable": True,
        "abstained": False,
        "outcome": "answered",
        "ragas": {
            "factual_correctness": dict(scored),
            "faithfulness": dict(scored),
            "context_recall": dict(scored),
            "answer_relevancy": dict(na),
            "context_precision_without_reference": dict(na),
        },
        "citation_ids": ["c1"],
        "citation_reference_valid": True,
        "evidence_candidate_ids": ["c1"],
        "retrieval_evidence_fingerprint": "a" * 64,
        "timings_ms": dict.fromkeys(
            (
                "retrieval", "rerank", "evidence_gate", "generation",
                "citation_validation", "judge", "total",
            ),
            1.0,
        ),
        "usage": {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": None},
        "input_sha256": "b" * 64,
    }


def _scope_case() -> dict:
    scope = {
        "status": "covered",
        "query_concept_ids": ["bluetooth", "credential"],
        "missing_concept_ids": [],
    }
    return {
        "id": "a1",
        "answerable": True,
        "original_abstained": False,
        "original_abstention_reason": None,
        "evidence_candidate_ids": ["c1"],
        "citation_ids": ["c1"],
        "retrieval_evidence_fingerprint": "a" * 64,
        "pre_generation_scope": dict(scope),
        "cited_scope": dict(scope),
        "proposed_abstention_reason": None,
    }


def _checkpoint(path: Path, identity: dict, records: list) -> None:
    path.write_text(
        "\n".join(json.dumps(value) for value in [{"run_identity": identity}, *records]),
        encoding="utf-8",
    )


def _identity(*, diagnostics: bool = False) -> dict:
    return {
        "execution_mode": "live",
        "selected_item_ids": ["a1"],
        "ragas": {"diagnostic_metrics": list(DIAGNOSTIC_METRICS) if diagnostics else []},
    }


def _assert_rejected_without_replacement(
    tmp_path: Path, record: dict, identity: dict, *, scope: bool = False
) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    output = tmp_path / "result.json"
    previous = json.dumps({"run_identity": identity, "cases": []}).encode() + b"\r\n"
    output.write_bytes(previous)
    _checkpoint(checkpoint, identity, [record])
    original_checkpoint = checkpoint.read_bytes()
    load = cli._load_scope_checkpoint if scope else cli._load_checkpoint
    write = cli._write_scope_checkpoint if scope else cli._write_checkpoint
    finish = cli._finish_scope_replay if scope else cli._finish_run

    with pytest.raises(RuntimeError, match="sanitized"):
        load(checkpoint, identity, CORPUS_IDS)
    assert checkpoint.read_bytes() == original_checkpoint
    _checkpoint(checkpoint, identity, [])
    saved_checkpoint = checkpoint.read_bytes()
    with pytest.raises(RuntimeError, match="sanitized"):
        write(checkpoint, identity, [record], CORPUS_IDS)
    assert checkpoint.read_bytes() == saved_checkpoint
    with pytest.raises(RuntimeError, match="sanitized"):
        finish(Namespace(output=output, dataset="development"), identity, [record], CORPUS_IDS)
    assert output.read_bytes() == previous


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
def test_checkpoint_rejects_evidence_outside_frozen_corpus(tmp_path: Path, scope: bool) -> None:
    record = _scope_case() if scope else _case()
    record.update(evidence_candidate_ids=["unknown"], citation_ids=["unknown"])
    _assert_rejected_without_replacement(tmp_path, record, _identity(), scope=scope)


def test_checkpoint_rejects_citation_outside_frozen_corpus(tmp_path: Path) -> None:
    record = _case()
    record.update(citation_ids=["unknown"], citation_reference_valid=False)
    invalid = MetricScore.deterministic_zero("invalid_citations")
    record["ragas"]["faithfulness"] = invalid.to_payload()
    _assert_rejected_without_replacement(tmp_path, record, _identity())


@pytest.mark.parametrize(
    "name",
    [
        "factual_correctness", "faithfulness", "context_recall",
        "context_precision_without_reference",
    ],
)
def test_checkpoint_rejects_negative_unsigned_metric(tmp_path: Path, name: str) -> None:
    record = _case()
    for diagnostic in DIAGNOSTIC_METRICS:
        record["ragas"][diagnostic] = dict(record["ragas"]["faithfulness"])
    record["ragas"][name]["score"] = -0.25
    _assert_rejected_without_replacement(tmp_path, record, _identity(diagnostics=True))


@pytest.mark.parametrize("enabled", [True, False])
def test_checkpoint_rejects_diagnostics_inconsistent_with_run(
    tmp_path: Path, enabled: bool
) -> None:
    record = _case()
    if not enabled:
        for diagnostic in DIAGNOSTIC_METRICS:
            record["ragas"][diagnostic] = dict(record["ragas"]["faithfulness"])
    _assert_rejected_without_replacement(tmp_path, record, _identity(diagnostics=enabled))


@pytest.mark.parametrize("missing", [[], ["bluetooth"]])
def test_empty_scope_requires_all_protected_concepts_missing(
    tmp_path: Path, missing: list[str]
) -> None:
    record = _scope_case()
    record.update(
        original_abstained=True, original_abstention_reason="no_candidates",
        evidence_candidate_ids=[], citation_ids=[], cited_scope=None,
    )
    if missing:
        record["pre_generation_scope"].update(status="missing_scope", missing_concept_ids=missing)
        record["proposed_abstention_reason"] = "evidence_scope_missing"
    _assert_rejected_without_replacement(tmp_path, record, _identity(), scope=True)


@pytest.mark.parametrize("question", ["General question", "Bluetooth", "Bluetooth PIN"])
def test_empty_evidence_accepts_decision_from_scope_policy(tmp_path: Path, question: str) -> None:
    pre = evaluate_evidence_scope(question, [])
    record = _scope_case()
    record.update(
        original_abstained=True, original_abstention_reason="no_candidates",
        evidence_candidate_ids=[], citation_ids=[], cited_scope=None,
        pre_generation_scope=cli._scope_decision_payload(pre),
        proposed_abstention_reason=(
            "evidence_scope_missing" if pre.status == "missing_scope" else None
        ),
    )
    path = tmp_path / "scope.jsonl"
    cli._write_scope_checkpoint(path, {}, [record], CORPUS_IDS)
    assert cli._load_scope_checkpoint(path, {}, CORPUS_IDS) == [record]


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
def test_checkpoint_rejects_raw_fields_before_resume(tmp_path: Path, scope: bool) -> None:
    identity = _identity()
    record = _scope_case() if scope else _case()
    record["raw_answer"] = "SYNTHETIC_RAW_SENTINEL"
    path = tmp_path / "checkpoint.jsonl"
    _checkpoint(path, identity, [record])
    load = cli._load_scope_checkpoint if scope else cli._load_checkpoint

    with pytest.raises(RuntimeError, match="sanitized") as caught:
        load(path, identity, CORPUS_IDS)

    assert "SYNTHETIC_RAW_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
@pytest.mark.parametrize("boundary", ["write", "finish"])
def test_publication_rejects_raw_fields_without_touching_output(
    tmp_path: Path, scope: bool, boundary: str
) -> None:
    path = tmp_path / "existing.json"
    path.write_text("UNCHANGED", encoding="utf-8")
    identity = _identity()
    record = _scope_case() if scope else _case()
    record["raw_answer"] = "SYNTHETIC_RAW_SENTINEL"

    with pytest.raises(RuntimeError, match="sanitized"):
        if boundary == "write":
            write = cli._write_scope_checkpoint if scope else cli._write_checkpoint
            write(path, identity, [record], CORPUS_IDS)
        else:
            finish = cli._finish_scope_replay if scope else cli._finish_run
            finish(Namespace(output=path, dataset="development"), identity, [record], CORPUS_IDS)

    assert path.read_text(encoding="utf-8") == "UNCHANGED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("usage", "raw_answer"), "SYNTHETIC_RAW_SENTINEL"),
        (("ragas", "raw_answer"), "SYNTHETIC_RAW_SENTINEL"),
        (("ragas", "faithfulness", "reasoning"), "SYNTHETIC_RAW_SENTINEL"),
        (("timings_ms", "raw_answer"), "SYNTHETIC_RAW_SENTINEL"),
        (("id",), ["a1"]),
        (("id",), "raw question with spaces"),
        (("language",), "raw-language"),
        (("question_type",), "raw-type"),
        (("answerable",), "false"),
        (("abstained",), 0),
        (("outcome",), "abstained"),
        (("citation_reference_valid",), False),
        (("citation_ids",), [{"raw_answer": "SYNTHETIC_RAW_SENTINEL"}]),
        (("evidence_candidate_ids",), ["c1", "c1"]),
        (("evidence_candidate_ids",), [f"c{i}" for i in range(6)]),
        (("retrieval_evidence_fingerprint",), "not-a-hash"),
        (("input_sha256",), None),
        (("timings_ms", "total"), float("nan")),
        (("timings_ms", "total"), True),
        (("usage", "input_tokens"), -1),
        (("usage", "input_tokens"), True),
        (("ragas", "faithfulness", "score"), float("inf")),
        (("ragas", "faithfulness", "score"), True),
        (("ragas", "faithfulness", "score"), 2),
        (("ragas", "faithfulness", "status"), "SYNTHETIC_RAW_SENTINEL"),
        (("ragas", "faithfulness", "latency_ms"), -1),
        (("ragas", "faithfulness"), {"status": "not_applicable", "score": None, "latency_ms": 0}),
        (("ragas", "context_recall"), {"status": "invalid_citations", "score": 0, "latency_ms": 0}),
    ],
)
def test_e2e_checkpoint_rejects_nested_or_inconsistent_records(
    tmp_path: Path, field: tuple[str, ...], value: object
) -> None:
    record = _case()
    target = record
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    identity = _identity()
    path = tmp_path / "checkpoint.jsonl"
    _checkpoint(path, identity, [record])

    with pytest.raises(RuntimeError, match="sanitized") as caught:
        cli._load_checkpoint(path, identity, CORPUS_IDS)

    assert "SYNTHETIC_RAW_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("pre_generation_scope", "raw_answer"), "SYNTHETIC_RAW_SENTINEL"),
        (("cited_scope", "raw_answer"), "SYNTHETIC_RAW_SENTINEL"),
        (("id",), ["a1"]),
        (("answerable",), "false"),
        (("original_abstained",), 0),
        (("original_abstention_reason",), "SYNTHETIC_RAW_SENTINEL"),
        (("citation_ids",), ["not-in-evidence"]),
        (("evidence_candidate_ids",), ["c1", "c1"]),
        (("retrieval_evidence_fingerprint",), "not-a-hash"),
        (("pre_generation_scope", "query_concept_ids"), ["unknown-concept"]),
        (("pre_generation_scope", "missing_concept_ids"), ["bluetooth"]),
        (("pre_generation_scope", "status"), "missing_scope"),
        (("cited_scope",), None),
        (("proposed_abstention_reason",), "evidence_scope_missing"),
    ],
)
def test_scope_checkpoint_rejects_nested_or_inconsistent_records(
    tmp_path: Path, field: tuple[str, ...], value: object
) -> None:
    record = _scope_case()
    target = record
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    identity = _identity()
    path = tmp_path / "checkpoint.jsonl"
    _checkpoint(path, identity, [record])

    with pytest.raises(RuntimeError, match="sanitized") as caught:
        cli._load_scope_checkpoint(path, identity, CORPUS_IDS)

    assert "SYNTHETIC_RAW_SENTINEL" not in str(caught.value)


def _item() -> EvaluationItem:
    return EvaluationItem(
        id="a1", fact_family_id="family-a1", question="Synthetic question?", language="en",
        answerable=True, scenario="en_to_en", question_type="safety",
        expected_document_ids=["manual"], relevant_chunk_ids=["c1"], expected_pages=[1],
        expected_phrases=["synthetic"],
        expected_answer_facts=[ExpectedAnswerFact(id="f1", aliases=["synthetic"])],
        reference_answer="Synthetic reference.", reference_answer_review_status="approved",
        citation_required=True, review_status="approved",
    )


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
@pytest.mark.parametrize("boundary", ["write", "finish"])
def test_publication_rejects_duplicate_records(tmp_path: Path, scope: bool, boundary: str) -> None:
    path = tmp_path / "existing.json"
    path.write_text("UNCHANGED", encoding="utf-8")
    record = _scope_case() if scope else _case()
    identity = _identity()

    with pytest.raises(RuntimeError, match="duplicate"):
        if boundary == "write":
            write = cli._write_scope_checkpoint if scope else cli._write_checkpoint
            write(path, identity, [record, record], CORPUS_IDS)
        else:
            finish = cli._finish_scope_replay if scope else cli._finish_run
            args = Namespace(output=path, dataset="development")
            finish(args, identity, [record, record], CORPUS_IDS)

    assert path.read_text(encoding="utf-8") == "UNCHANGED"


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
def test_checkpoint_prefix_rejects_changed_dataset_metadata(scope: bool) -> None:
    record = _scope_case() if scope else _case()
    record["answerable"] = False
    validate = cli._validate_scope_checkpoint_prefix if scope else cli._validate_checkpoint_prefix

    with pytest.raises(RuntimeError, match="metadata"):
        validate([record], [_item()])


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
def test_valid_checkpoints_round_trip_without_rewriting_records(
    tmp_path: Path, scope: bool
) -> None:
    identity = _identity()
    record = _scope_case() if scope else _case()
    path = tmp_path / "checkpoint.jsonl"
    load = cli._load_scope_checkpoint if scope else cli._load_checkpoint
    write = cli._write_scope_checkpoint if scope else cli._write_checkpoint
    validate = cli._validate_scope_checkpoint_prefix if scope else cli._validate_checkpoint_prefix
    finish = cli._finish_scope_replay if scope else cli._finish_run
    assert load(path, identity, CORPUS_IDS) == []
    write(path, identity, [], CORPUS_IDS)
    assert load(path, identity, CORPUS_IDS) == []
    write(path, identity, [record], CORPUS_IDS)
    restored = load(path, identity, CORPUS_IDS)
    assert restored == [record]
    validate(restored, [_item()])
    output = tmp_path / "result.json"
    args = Namespace(output=output, dataset="development")
    assert finish(args, identity, restored, CORPUS_IDS) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["cases"] == [record]


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
@pytest.mark.parametrize("mutation", ["identity", "duplicate", "unknown", "order", "extra"])
def test_checkpoint_identity_and_dataset_prefix_still_fail_closed(
    tmp_path: Path, scope: bool, mutation: str
) -> None:
    record = _scope_case() if scope else _case()
    identity = _identity()
    records = [record]
    if mutation == "duplicate":
        records.append(dict(record))
    elif mutation == "unknown":
        record["id"] = "unknown"
    elif mutation in {"order", "extra"}:
        other = dict(record, id="a2")
        records = [other, record] if mutation == "order" else [record, other]
    path = tmp_path / "checkpoint.jsonl"
    _checkpoint(path, identity, records)
    load = cli._load_scope_checkpoint if scope else cli._load_checkpoint
    validate = cli._validate_scope_checkpoint_prefix if scope else cli._validate_checkpoint_prefix

    with pytest.raises(RuntimeError):
        restored = load(path, {} if mutation == "identity" else identity, CORPUS_IDS)
        validate(restored, [_item()])


@pytest.mark.parametrize("scope", [False, True], ids=["e2e", "scope"])
@pytest.mark.parametrize("record", [None, [], "SYNTHETIC_RAW_SENTINEL", 1])
def test_non_mapping_checkpoint_record_has_sanitized_error(
    tmp_path: Path, scope: bool, record: object
) -> None:
    path = tmp_path / "checkpoint.jsonl"
    _checkpoint(path, _identity(), [record])
    load = cli._load_scope_checkpoint if scope else cli._load_checkpoint

    with pytest.raises(RuntimeError, match="sanitized") as caught:
        load(path, _identity(), CORPUS_IDS)

    assert "SYNTHETIC_RAW_SENTINEL" not in str(caught.value)


def test_real_e2e_projection_is_accepted_by_checkpoint_validator(tmp_path: Path) -> None:
    candidate = RetrievalCandidate(
        chunk_id="c1", document_id="manual", filename="manual.pdf", text="Synthetic evidence.",
        page_numbers=[1], headings=["Safety"], content_type="text", score=1.0,
    )
    execution = QueryExecution(
        response=QueryResponse(
            answer="Synthetic answer.", abstained=False,
            citations=[Citation(
                chunk_id="c1", document_id="manual", filename="manual.pdf",
                page_numbers=[1], headings=["Safety"], excerpt="Synthetic evidence.",
            )],
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15), usage=None,
        evidence_candidates=(candidate,),
    )
    score = MetricScore("complete", 0.5, 1.0)
    na = MetricScore.not_applicable()
    record = build_case_record(
        _item(), execution, RagasScores(score, score, score, na, na), input_sha256="b" * 64
    )
    path = tmp_path / "checkpoint.jsonl"
    identity = _identity()
    cli._write_checkpoint(path, identity, [record], CORPUS_IDS)
    assert cli._load_checkpoint(path, identity, CORPUS_IDS) == [record]


@pytest.mark.parametrize(
    "kind",
    [
        "abstained", "unanswerable", "invalid_citations",
        "empty_context", "unrendered", "diagnostics",
    ],
)
def test_valid_metric_applicability_is_preserved(tmp_path: Path, kind: str) -> None:
    record = _case()
    na = {"status": "not_applicable", "score": None, "latency_ms": 0.0}
    if kind in {"abstained", "unanswerable", "unrendered"}:
        record.update(abstained=True, outcome="abstained", citation_ids=[])
        record["ragas"]["factual_correctness"] = dict(na)
        record["ragas"]["faithfulness"] = dict(na)
        if kind == "unanswerable":
            record["answerable"] = False
            record["question_type"] = "unanswerable"
            record["ragas"]["context_recall"] = dict(na)
        elif kind == "unrendered":
            record["ragas"]["context_recall"] = {
                "status": "empty_context", "score": 0.0, "latency_ms": 0.0,
            }
    elif kind in {"invalid_citations", "empty_context"}:
        record.update(citation_ids=[], citation_reference_valid=False)
        record["ragas"]["faithfulness"] = {
            "status": "invalid_citations", "score": 0.0, "latency_ms": 0.0,
        }
        if kind == "empty_context":
            record["evidence_candidate_ids"] = []
            record["ragas"]["factual_correctness"] = dict(na)
            record["ragas"]["context_recall"] = {
                "status": "empty_context", "score": 0.0, "latency_ms": 0.0,
            }
    else:
        for name in ("answer_relevancy", "context_precision_without_reference"):
            record["ragas"][name] = {"status": "complete", "score": 0.5, "latency_ms": 1.0}
        record["ragas"]["answer_relevancy"]["score"] = -0.25
    path = tmp_path / "checkpoint.jsonl"
    identity = _identity(diagnostics=kind in {"diagnostics", "unrendered", "unanswerable"})
    cli._write_checkpoint(path, identity, [record], CORPUS_IDS)
    assert cli._load_checkpoint(path, identity, CORPUS_IDS) == [record]


@pytest.mark.parametrize("mutation", [None, "raw_field", "unknown_evidence"])
def test_completed_main_resume_validates_before_external_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    identity = _identity()
    checkpoint, output = tmp_path / "checkpoint.jsonl", tmp_path / "result.json"
    record = _case()
    previous = json.dumps({"run_identity": identity, "cases": []}).encode()
    output.write_bytes(previous)
    if mutation == "raw_field":
        record["raw_answer"] = "SYNTHETIC_RAW_SENTINEL"
    elif mutation == "unknown_evidence":
        record.update(evidence_candidate_ids=["unknown"], citation_ids=["unknown"])
    _checkpoint(checkpoint, identity, [record])
    monkeypatch.setattr(cli, "_validate_approvals", lambda _: None)
    monkeypatch.setattr(cli, "load_frozen_chunks", lambda _: [SimpleNamespace(chunk_id="c1")])
    monkeypatch.setattr(cli, "_load_dataset", lambda *args: ([_item()], {}, {}))
    monkeypatch.setattr(cli, "Settings", lambda: object())
    monkeypatch.setattr(cli, "_runtime_settings", lambda value: value)
    monkeypatch.setattr(cli, "_run_identity", lambda **kwargs: identity)
    monkeypatch.setattr(cli, "_build_judge", lambda _: pytest.fail("judge must not be built"))
    monkeypatch.setattr(
        cli, "_build_query_service", lambda _: pytest.fail("runtime must not be built")
    )
    argv = [
        "--dataset", "development", "--checkpoint", str(checkpoint), "--output", str(output),
        "--provider-approval-token", "synthetic", "--judge-provider-approval-token", "synthetic",
    ]

    if mutation:
        with pytest.raises(RuntimeError, match="sanitized"):
            cli.main(argv)
        assert output.read_bytes() == previous
    else:
        assert cli.main(argv) == 0
        assert json.loads(output.read_text(encoding="utf-8"))["cases"] == [record]


@pytest.mark.parametrize("boundary", ["load", "write", "finish"])
def test_empty_evidence_cannot_claim_scored_context_recall(
    tmp_path: Path, boundary: str
) -> None:
    record = _case()
    record.update(abstained=True, outcome="abstained", evidence_candidate_ids=[], citation_ids=[])
    for name in ("factual_correctness", "faithfulness"):
        record["ragas"][name] = {"status": "not_applicable", "score": None, "latency_ms": 0}
    record["ragas"]["context_recall"]["score"] = 0.99
    path = tmp_path / "checkpoint.jsonl"
    output = tmp_path / "existing.json"
    output.write_text("UNCHANGED", encoding="utf-8")
    _checkpoint(path, _identity(), [record])

    with pytest.raises(RuntimeError, match="sanitized"):
        if boundary == "load":
            cli._load_checkpoint(path, _identity(), CORPUS_IDS)
        elif boundary == "write":
            cli._write_checkpoint(output, _identity(), [record], CORPUS_IDS)
        else:
            cli._finish_run(
                Namespace(output=output, dataset="development"), _identity(),
                [record], CORPUS_IDS,
            )

    assert output.read_text(encoding="utf-8") == "UNCHANGED"


@pytest.mark.parametrize("boundary", ["load", "write", "finish"])
def test_identical_scope_sources_cannot_disagree_on_missing_concepts(
    tmp_path: Path, boundary: str
) -> None:
    record = _scope_case()
    record["pre_generation_scope"].update(
        status="missing_scope", missing_concept_ids=["credential"]
    )
    record["proposed_abstention_reason"] = "evidence_scope_missing"
    path = tmp_path / "checkpoint.jsonl"
    output = tmp_path / "existing.json"
    output.write_text("UNCHANGED", encoding="utf-8")
    _checkpoint(path, _identity(), [record])

    with pytest.raises(RuntimeError, match="sanitized"):
        if boundary == "load":
            cli._load_scope_checkpoint(path, _identity(), CORPUS_IDS)
        elif boundary == "write":
            cli._write_scope_checkpoint(output, {}, [record], CORPUS_IDS)
        else:
            cli._finish_scope_replay(Namespace(output=output), {}, [record], CORPUS_IDS)

    assert output.read_text(encoding="utf-8") == "UNCHANGED"


def test_identical_scope_sources_cannot_introduce_a_new_missing_concept(tmp_path: Path) -> None:
    record = _scope_case()
    record.update(evidence_candidate_ids=["c1", "c2"], citation_ids=["c2", "c1"])
    record["cited_scope"].update(status="missing_scope", missing_concept_ids=["credential"])
    record["proposed_abstention_reason"] = "cited_evidence_scope_missing"
    path = tmp_path / "checkpoint.jsonl"
    _checkpoint(path, _identity(), [record])

    with pytest.raises(RuntimeError, match="sanitized"):
        cli._load_scope_checkpoint(path, _identity(), CORPUS_IDS)


@pytest.mark.parametrize(
    ("question", "sources", "cited_indices"),
    [
        ("Bluetooth PIN", ["Bluetooth", "PIN"], [0]),
        ("Bluetooth mobile app", ["Bluetooth mobile", "unrelated source", "app"], [0, 2]),
    ],
)
def test_different_scope_source_sets_can_have_different_valid_decisions(
    tmp_path: Path, question: str, sources: list[str], cited_indices: list[int]
) -> None:
    record = _scope_case()
    pre = evaluate_evidence_scope(question, sources)
    cited = evaluate_evidence_scope(question, [sources[index] for index in cited_indices])
    record.update(
        evidence_candidate_ids=[f"c{index}" for index in range(len(sources))],
        citation_ids=[f"c{index}" for index in cited_indices],
        pre_generation_scope=cli._scope_decision_payload(pre),
        cited_scope=cli._scope_decision_payload(cited),
        proposed_abstention_reason=(
            "evidence_scope_missing" if pre.status == "missing_scope"
            else "cited_evidence_scope_missing"
        ),
    )
    path = tmp_path / "checkpoint.jsonl"
    cli._write_scope_checkpoint(path, _identity(), [record], CORPUS_IDS)
    assert cli._load_scope_checkpoint(path, _identity(), CORPUS_IDS) == [record]
