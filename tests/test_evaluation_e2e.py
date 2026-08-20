"""Offline scoring tests for the Phase 7 end-to-end evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation_e2e import (
    aggregate_phase7_records,
    evaluate_phase7_quality_gates,
    score_expected_answer_fact,
    score_phase7_execution,
)
from app.generation import TokenUsage
from app.models import Citation, QueryResponse, RetrievalCandidate
from app.phase7 import ExpectedAnswerFact, Phase7DatasetItem
from app.query_service import QueryExecution, QueryTimings
from scripts import evaluate_phase7_e2e


def _candidate(chunk_id: str, *, document_id: str = "installation") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="manual.pdf",
        text="The documented range is -599 to +599 Hz.",
        page_numbers=[7],
        headings=["Parameters"],
        content_type="text",
        score=1.0,
    )


def _item(*, answerable: bool = True, phrase_mode: str = "all") -> Phase7DatasetItem:
    if not answerable:
        return Phase7DatasetItem(
            id="unsupported",
            question="Unknown?",
            language="en",
            answerable=False,
            scenario="en_to_en",
            question_type="unanswerable",
            expected_document_ids=[],
            relevant_chunk_ids=[],
            expected_pages=[],
            expected_phrases=[],
            citation_required=False,
            unanswerable_reason="Verified absent.",
            review_status="approved",
        )
    return Phase7DatasetItem(
        id="answerable",
        question="Range?",
        language="en",
        answerable=True,
        scenario="en_to_en",
        question_type="parameter_code",
        expected_document_ids=["installation"],
        relevant_chunk_ids=["qrel"],
        expected_pages=[7],
        expected_phrases=["-599 to +599 Hz"],
        phrase_match_mode=phrase_mode,
        expected_answer_facts=[
            {
                "id": "frequency-range",
                "aliases": ["-599 to +599 Hz", "từ -599 đến +599 Hz"],
            }
        ],
        citation_required=True,
        review_status="approved",
    )


def _execution(
    *,
    final: list[RetrievalCandidate],
    pool: list[RetrievalCandidate],
    abstained: bool = False,
    evidence: list[RetrievalCandidate] | None = None,
) -> QueryExecution:
    actual_evidence = evidence if evidence is not None else final
    citations = []
    if not abstained:
        candidate = actual_evidence[0]
        citations = [
            Citation(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                filename=candidate.filename,
                page_numbers=candidate.page_numbers,
                headings=candidate.headings,
                excerpt="excerpt",
            )
        ]
    return QueryExecution(
        response=QueryResponse(
            answer="The documented range is -599 to +599 Hz." if not abstained else "insufficient",
            abstained=abstained,
            abstention_reason="llm_insufficient_evidence" if abstained else None,
            citations=citations,
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=TokenUsage(10, 2, 1),
        candidates=tuple(final),
        candidate_pool=tuple(pool),
        evidence_candidates=tuple(actual_evidence),
    )


def test_scores_direct_evidence_and_candidate_miss_without_page_fallback() -> None:
    item = _item()
    pool = [_candidate("same-page"), _candidate("qrel")]
    record = score_phase7_execution(item, _execution(final=[_candidate("same-page")], pool=pool))
    assert record["candidate_direct_evidence_rank"] == 2
    assert record["direct_evidence_rank"] is None
    assert record["failure_class"] == "reranker_miss_top20"
    assert record["citation_direct_evidence"] is False
    assert record["citation_ids_in_final_candidates"] is True
    assert record["qrel_candidate_diagnostics"] == [
        {
            "chunk_id": "qrel",
            "matched_relevant_chunk_ids": ["qrel"],
            "ordinal_rank": 2,
            "dense_rank": None,
            "sparse_rank": None,
            "rrf_rank": None,
            "rerank_rank": None,
            "document_id": "installation",
            "page_numbers": [7],
        }
    ]
    assert record["qrel_final_diagnostics"] == []


def test_scoring_distinguishes_full_ranking_from_actual_generation_evidence() -> None:
    qrel = _candidate("qrel")
    record = score_phase7_execution(
        _item(),
        _execution(
            final=[_candidate("other"), qrel],
            pool=[_candidate("other"), qrel],
            evidence=[qrel],
        ),
    )
    assert record["ranked_direct_evidence_rank"] == 2
    assert record["direct_evidence_rank"] == 1
    assert record["evidence_candidate_ids"] == ["qrel"]
    assert record["citation_ids_in_final_candidates"] is True


def test_full_rank_six_excluded_from_actual_top_five_is_a_top5_miss() -> None:
    leading = [_candidate(f"other-{index}") for index in range(1, 6)]
    qrel = _candidate("qrel")
    record = score_phase7_execution(
        _item(),
        _execution(
            final=[*leading, qrel],
            pool=[*leading, qrel],
            evidence=leading,
        ),
    )
    assert record["ranked_direct_evidence_rank"] == 6
    assert record["direct_evidence_rank"] is None
    assert record["failure_class"] == "reranker_miss_top5"


def test_aggregate_reports_retrieval_citations_abstention_and_latency() -> None:
    answerable = score_phase7_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    unsupported = score_phase7_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
    )
    metrics = aggregate_phase7_records([answerable, unsupported])
    assert metrics["retrieval"]["hit_rate_at_1"] == 1.0
    assert metrics["answer_quality"]["answer_fact_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["deterministic_fact_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["strict_phrase_accuracy_when_answered"] == 1.0
    assert metrics["answer_quality"]["fact_count"] == 1
    assert metrics["answer_quality"]["matched_fact_count"] == 1
    assert (
        metrics["answer_quality"]["all_alias_tokens_covered_accuracy_when_answered"]
        == 1.0
    )
    assert metrics["citations"]["direct_evidence_rate_when_answered"] == 1.0
    assert metrics["abstention"]["true_positive"] == 1
    assert metrics["abstention"]["precision"] == 1.0
    assert metrics["abstention"]["recall"] == 1.0
    assert metrics["latency_ms"]["total"]["p95"] == 15
    assert metrics["per_language"]["en"]["query_count"] == 2

    gates = evaluate_phase7_quality_gates(metrics)
    assert gates["overall_pass"] is True
    assert gates["gates"]["unsupported_citation_ids"]["actual"] == 0
    assert gates["gates"]["wrong_document_citations"]["actual"] == 0


def test_answerable_abstention_does_not_claim_answer_or_citation_success() -> None:
    record = score_phase7_execution(
        _item(), _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")], abstained=True)
    )
    assert record["answer_fact_match"] is None
    assert record["answer_fact_results"] == []
    assert record["missing_answer_fact_ids"] == []
    assert record["citation_direct_evidence"] is False


def test_answer_fact_uses_language_aliases_not_evidence_phrase() -> None:
    item = _item().model_copy(
        update={
            "language": "vi",
            "scenario": "vi_to_en",
            "expected_phrases": ["English evidence wording"],
        }
    )
    execution = _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    execution = execution.__class__(
        response=execution.response.model_copy(
            update={"answer": "Dải tần số là từ -599 đến +599 Hz."}
        ),
        timings=execution.timings,
        usage=execution.usage,
        candidates=execution.candidates,
        candidate_pool=execution.candidate_pool,
    )
    record = score_phase7_execution(item, execution)
    assert record["answer_fact_match"] is True
    assert record["answer_fact_results"][0]["id"] == "frequency-range"
    assert record["answer_fact_results"][0]["matched"] is True
    assert record["answer_fact_results"][0]["max_alias_token_recall"] == 1.0


def test_answer_fact_diagnostics_report_ids_without_alias_or_answer_content() -> None:
    item = Phase7DatasetItem.model_validate(
        _item().model_dump()
        | {
            "expected_answer_facts": [
                {"id": "range", "aliases": ["-599 to +599 Hz"]},
                {"id": "unit", "aliases": ["rpm"]},
            ]
        }
    )
    record = score_phase7_execution(
        item, _execution(final=[_candidate("qrel")], pool=[_candidate("qrel")])
    )
    assert record["answer_fact_match"] is False
    assert [result["id"] for result in record["answer_fact_results"]] == [
        "range",
        "unit",
    ]
    assert [result["matched"] for result in record["answer_fact_results"]] == [
        True,
        False,
    ]
    assert record["answer_fact_results"][0]["max_alias_token_recall"] == 1.0
    assert record["answer_fact_results"][1]["max_alias_token_recall"] == 0.0
    assert record["missing_answer_fact_ids"] == ["unit"]
    assert "aliases" not in record["answer_fact_results"][0]


def test_text_fact_is_order_insensitive_but_strict_phrase_remains_diagnostic() -> None:
    fact = ExpectedAnswerFact(
        id="power",
        aliases=["disconnect all power sources"],
    )
    result = score_expected_answer_fact(
        fact, "Before work, all power sources must disconnect safely."
    )
    assert result["deterministic_matched"] is True
    assert result["strict_phrase_matched"] is False
    assert result["matcher"] == "text_alias_token_set_v2"


def test_typed_numeric_identifier_and_required_token_group_matchers() -> None:
    numeric = ExpectedAnswerFact(
        id="voltage",
        aliases=["24 VDC"],
        type="numeric_unit",
        value="24",
        unit="VDC",
    )
    assert score_expected_answer_fact(numeric, "Supply: 24VDC.")[
        "deterministic_matched"
    ]
    decimal = ExpectedAnswerFact(
        id="decimal-voltage",
        aliases=["24.0 VDC"],
        type="numeric_unit",
        value="24.0",
        unit="VDC",
    )
    assert score_expected_answer_fact(decimal, "Supply: 24,0VDC.")["deterministic_matched"]
    assert not score_expected_answer_fact(numeric, "Supply: 124 VDC.")[
        "deterministic_matched"
    ]

    identifier = ExpectedAnswerFact(
        id="rating",
        aliases=["IP65"],
        type="identifier",
        acceptable_values=["IP65"],
    )
    assert score_expected_answer_fact(identifier, "The enclosure is rated IP65.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(identifier, "The enclosure is rated IP650.")[
        "deterministic_matched"
    ]

    text = ExpectedAnswerFact(
        id="mounting",
        aliases=["vertical mounting"],
        required_token_groups=[["vertical", "upright"], ["mount", "install"]],
    )
    result = score_expected_answer_fact(text, "Install the drive in an upright position.")
    assert result["deterministic_matched"] is True
    assert result["strict_phrase_matched"] is False


@pytest.mark.parametrize(
    ("expected", "answer"),
    [
        ("block", "The guard is blocked."),
        ("contact", "Inspect all contacts."),
        ("install", "The unit is installed."),
        ("secure", "The cover is secured."),
    ],
)
def test_text_fact_accepts_only_bounded_regular_inflections(expected, answer) -> None:
    fact = ExpectedAnswerFact(id="lexical", aliases=[expected])
    result = score_expected_answer_fact(fact, answer)
    assert result["deterministic_matched"] is True
    assert result["match_mode"] == "inflection"


def test_text_inflection_does_not_use_arbitrary_prefix_matching() -> None:
    fact = ExpectedAnswerFact(id="mode", aliases=["MODE"])
    assert score_expected_answer_fact(fact, "Select the model.")[
        "deterministic_matched"
    ] is False


def test_typed_boundaries_reject_identifier_substrings_wrong_value_unit_and_sign() -> None:
    identifier = ExpectedAnswerFact(
        id="menu",
        aliases=["rEF"],
        type="identifier",
        acceptable_values=["rEF"],
    )
    assert score_expected_answer_fact(identifier, "Open rEF.")["deterministic_matched"]
    assert not score_expected_answer_fact(identifier, "Open prEFix.")[
        "deterministic_matched"
    ]

    voltage = ExpectedAnswerFact(
        id="voltage",
        aliases=["24 VDC"],
        type="numeric_unit",
        value="24",
        unit="VDC",
    )
    for invalid in ("124 VDC", "24 VAC", "-24 VDC"):
        assert not score_expected_answer_fact(voltage, invalid)["deterministic_matched"]
    negative_voltage = ExpectedAnswerFact(
        id="negative-voltage",
        aliases=["-24 VDC"],
        type="numeric_unit",
        value="-24",
        unit="VDC",
    )
    assert score_expected_answer_fact(negative_voltage, "-24 VDC")["deterministic_matched"]
    assert not score_expected_answer_fact(negative_voltage, "--24 VDC")[
        "deterministic_matched"
    ]


def test_multiword_text_alternative_preserves_internal_order() -> None:
    fact = ExpectedAnswerFact(
        id="shaft",
        aliases=["motor shaft"],
        required_token_groups=[["motor shaft"]],
    )
    assert score_expected_answer_fact(fact, "Inspect the motor shaft.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(fact, "Inspect the shaft motor.")[
        "deterministic_matched"
    ]


@pytest.mark.parametrize(
    ("answer", "matched", "polarity"),
    [
        ("No hazards remain; protective equipment is installed and closed.", True, "positive"),
        ("Protective equipment is not installed.", False, "negative"),
        ("Do not verify that protective equipment is installed.", False, "negative"),
        ("Not only installed but also closed protective equipment.", True, "positive"),
        ("It was not installed before; protective equipment is now installed.", True, "positive"),
        ("Protective equipment is not never installed.", False, "ambiguous"),
        ("The motor is not energized; protective equipment is installed.", True, "positive"),
    ],
)
def test_span_aware_negation_policy(answer, matched, polarity) -> None:
    fact = ExpectedAnswerFact(
        id="guard",
        aliases=["protective equipment installed"],
        required_token_groups=[["protective equipment"], ["installed"]],
    )
    result = score_expected_answer_fact(fact, answer)
    assert result["deterministic_matched"] is matched
    assert result["polarity"] == polarity


def test_vietnamese_negation_is_boundary_aware() -> None:
    fact = ExpectedAnswerFact(
        id="installation",
        aliases=["thiết bị lắp đặt"],
        required_token_groups=[["thiết bị"], ["lắp đặt"]],
    )
    result = score_expected_answer_fact(fact, "Thiết bị chưa được lắp đặt.")
    assert result["deterministic_matched"] is False
    assert result["polarity"] == "negative"


def test_fact_matcher_rejects_plain_negation_even_when_alias_tokens_are_present() -> None:
    fact = ExpectedAnswerFact(
        id="disconnect-power",
        aliases=["disconnect all power"],
    )
    result = score_expected_answer_fact(fact, "Do not disconnect all power before electrical work.")
    assert result["deterministic_matched"] is False
    assert result["matcher"] == "text_alias_token_set_v2_negation_guard_v2"


def test_document_contamination_metrics_and_gate_are_explicit() -> None:
    wrong = score_phase7_execution(
        _item(),
        _execution(
            final=[_candidate("qrel", document_id="programming")],
            pool=[_candidate("qrel", document_id="programming")],
        ),
    )
    unsupported = score_phase7_execution(
        _item(answerable=False),
        _execution(final=[_candidate("other")], pool=[_candidate("other")], abstained=True),
    )
    metrics = aggregate_phase7_records([wrong, unsupported])
    assert metrics["document_contamination"]["wrong_document_retrieval_at_1_rate"] == 1.0
    assert metrics["citations"]["wrong_document_citation_rate_when_answered"] == 1.0
    gates = evaluate_phase7_quality_gates(metrics)
    assert gates["overall_pass"] is False
    assert gates["gates"]["wrong_document_citations"]["passed"] is False


def test_phase7_v5_cli_uses_new_artifact_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_phase7_e2e",
            "--dataset",
            "calibration",
            "--provider-approval-token",
            evaluate_phase7_e2e.CALIBRATION_PROVIDER_APPROVAL_TOKEN,
        ],
    )
    args = evaluate_phase7_e2e._parse_args()
    assert args.output.name == "phase-7-calibration-e2e-v5.json"
    assert args.checkpoint.name == "phase-7-calibration-e2e-v5-checkpoint.jsonl"
    settings = evaluate_phase7_e2e._phase7_settings(evaluate_phase7_e2e.Settings())
    assert settings.rerank_deduplicate_content is True
    assert settings.dense_candidate_limit == 60
    assert settings.sparse_candidate_limit == 40


def test_provider_execution_requires_dataset_specific_approval() -> None:
    with pytest.raises(SystemExit, match="missing or invalid"):
        evaluate_phase7_e2e._validate_execution_approval(
            SimpleNamespace(dataset="calibration", provider_approval_token="wrong")
        )
    evaluate_phase7_e2e._validate_execution_approval(
        SimpleNamespace(
            dataset="calibration",
            provider_approval_token=evaluate_phase7_e2e.CALIBRATION_PROVIDER_APPROVAL_TOKEN,
        )
    )
    with pytest.raises(SystemExit, match="BLOCKED_GOVERNANCE"):
        evaluate_phase7_e2e._validate_execution_approval(
            SimpleNamespace(
                dataset="test",
                provider_approval_token=evaluate_phase7_e2e.HELDOUT_PROVIDER_APPROVAL_TOKEN,
            )
        )


def test_calibration_loader_never_opens_held_out_path(monkeypatch) -> None:
    opened = []
    approved = SimpleNamespace(review_status="approved")

    def fake_read(path):
        opened.append(path)
        if path == "poison-held-out":
            raise AssertionError("held-out path was opened")
        return [approved]

    monkeypatch.setattr(evaluate_phase7_e2e, "read_phase7_dataset", fake_read)
    monkeypatch.setattr(
        evaluate_phase7_e2e,
        "validate_phase7_dataset",
        lambda dataset, chunks, *, kind: {"kind": kind},
    )
    monkeypatch.setattr(
        evaluate_phase7_e2e,
        "_validate_evaluation_manifest",
        lambda path, chunks, dataset, *, kind: {"test_dataset_sha256": "a" * 64},
    )
    args = SimpleNamespace(
        dataset="calibration",
        calibration="active-calibration",
        test="poison-held-out",
        manifest="manifest",
    )
    dataset, validation, _ = evaluate_phase7_e2e._load_selected_dataset(args, [])
    assert dataset == [approved]
    assert validation == {"kind": "calibration"}
    assert opened == ["active-calibration"]


def test_item_id_is_calibration_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_phase7_e2e",
            "--dataset",
            "test",
            "--item-id",
            "phase7_test_001",
            "--provider-approval-token",
            evaluate_phase7_e2e.HELDOUT_PROVIDER_APPROVAL_TOKEN,
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_phase7_e2e._parse_args()


def test_checkpoint_fails_closed_when_provider_identity_changes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    identity = {"generation_configuration": {"temperature": 0.0}}
    evaluate_phase7_e2e._write_checkpoint(checkpoint, identity, [])
    changed = {"generation_configuration": {"temperature": 0.1}}
    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_phase7_e2e._load_checkpoint(checkpoint, changed)
