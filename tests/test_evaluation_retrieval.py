"""Offline metric tests for the provider-free retrieval evaluator."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.domain.retrieval import QueryRetrievalResult, RetrievalCandidate, RetrievalTrace
from evaluation.retrieval import (
    DEVELOPMENT_RETRIEVAL_VARIANTS,
    aggregate_retrieval_rows,
    aggregate_retrieval_rows_by_language,
    authoritative_requirement_coverage,
    build_corpus_exact_content_equivalence,
    development_retrieval_variant,
    direct_evidence_rank,
    evidence_failure_class,
    evidence_rank,
    exact_equivalent_evidence_rank,
    percentile_nearest_rank,
    phrase_matches,
    score_retrieval_execution,
    select_evaluation_evidence,
)
from scripts.evaluation import evaluate_retrieval


def _row(identifier: str, rank: int | None, *, wrong_document: bool = False) -> dict:
    return {
        "id": identifier,
        "language": "en",
        "candidate_count": 30,
        "final_candidate_count": 30,
        "evidence_candidate_count": 5,
        "candidate_direct_evidence_rank": rank,
        "final_direct_evidence_rank": rank,
        "evidence_direct_evidence_rank": rank,
        "evidence_exact_equivalent_evidence_rank": rank,
        "failure_class": "candidate_miss" if rank is None else "hit",
        "evidence_failure_class": "candidate_miss" if rank is None else "hit",
        "wrong_document_top1": wrong_document,
        "wrong_document_candidate_count_at_5": int(wrong_document),
        "wrong_document_evidence_top1": wrong_document,
        "wrong_document_evidence_candidate_count_at_5": int(wrong_document),
        "wrong_document_evidence_top1_excluding_exact_equivalents": wrong_document,
        "wrong_document_evidence_candidate_count_at_5_excluding_exact_equivalents": int(
            wrong_document
        ),
        "evidence_selection": {"duplicate_groups": []},
        "candidate_authoritative_requirement_coverage": {
            "requirement_count": 1,
            "covered_requirement_count": int(rank is not None),
            "complete": rank is not None,
        },
        "final_authoritative_requirement_coverage": {
            "requirement_count": 1,
            "covered_requirement_count": int(rank is not None),
            "complete": rank is not None,
        },
        "evidence_authoritative_requirement_coverage": {
            "requirement_count": 1,
            "covered_requirement_count": int(rank is not None),
            "complete": rank is not None,
        },
        "document_context_complete": True,
        "retrieval_ms": 10.0,
        "rerank_ms": 100.0,
    }


def test_retrieval_aggregation_reports_recall_ranks_contamination_and_latency() -> None:
    metrics = aggregate_retrieval_rows([_row("a", 1), _row("b", None, wrong_document=True)])
    assert metrics["candidate_recall"] == 0.5
    assert metrics["authoritative_candidate_complete_rate"] == 0.5
    assert metrics["authoritative_evidence_complete_rate"] == 0.5
    assert metrics["raw_component_recall"] == 0.5
    assert metrics["candidate_exact_equivalent_recall"] == 0.5
    assert metrics["hit_rate_at_5"] == 0.5
    assert metrics["evidence_hit_rate_at_5"] == 0.5
    assert metrics["evidence_exact_equivalent_hit_rate_at_5"] == 0.5
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["candidate_count_maximum"] == 30
    assert metrics["candidate_count_minimum"] == 30
    assert metrics["candidate_count_average"] == 30
    assert metrics["wrong_document_top1_rate"] == 0.5
    assert metrics["wrong_document_candidate_rate_at_5"] == 0.1
    assert metrics["wrong_document_evidence_top1_rate"] == 0.5
    assert metrics["wrong_document_evidence_candidate_rate_at_5"] == 0.1
    assert (
        metrics[
            "wrong_document_evidence_candidate_rate_at_5_excluding_exact_equivalents"
        ]
        == 0.1
    )
    assert metrics["evidence_failure_classes"] == {"candidate_miss": 1, "hit": 1}
    assert metrics["document_context_complete_rate"] == 1.0
    assert metrics["rerank_latency_ms"]["p95"] == 100.0


def test_retrieval_aggregation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_retrieval_rows([])


def test_evidence_aggregation_handles_real_empty_retrieval_and_language_groups() -> None:
    empty_execution = QueryRetrievalResult(
        candidates=[],
        candidate_pool=[],
        retrieval_ms=1.0,
        rerank_ms=2.0,
    )
    empty_selection = select_evaluation_evidence(
        "Which wiring step is required?",
        empty_execution.candidates,
    )
    english = score_retrieval_execution(
        SimpleNamespace(
            id="en-empty",
            fact_family_id="family-empty",
            language="en",
            question_type="wiring",
            relevant_chunk_ids=["missing"],
            expected_document_ids=["installation"],
        ),
        empty_execution,
        corpus_equivalence={},
        evidence_selection=empty_selection,
    )
    vietnamese = _row("vi-hit", 1)
    vietnamese["language"] = "vi"

    overall = aggregate_retrieval_rows([english])
    per_language = aggregate_retrieval_rows_by_language([english, vietnamese])

    assert overall["wrong_document_evidence_candidate_rate_at_5"] == 0.0
    assert per_language["en"]["evidence_hit_rate_at_5"] == 0.0
    assert per_language["vi"]["evidence_hit_rate_at_5"] == 1.0


def test_evidence_failure_class_treats_selected_qrel_as_a_hit() -> None:
    assert evidence_failure_class(6, 6, 1) == "hit"


def test_retrieval_metric_primitives_are_deterministic() -> None:
    results = [SimpleNamespace(chunk_id="a"), SimpleNamespace(chunk_id="b")]

    assert direct_evidence_rank(results, {"b"}) == 2
    assert direct_evidence_rank(results, {"missing"}) is None
    assert phrase_matches("Cafe\u0301\n menu", "CAF\u00c9 menu")
    assert percentile_nearest_rank([30.0, 10.0, 20.0], 95) == 30.0


def test_evidence_rank_keeps_strict_and_exact_equivalent_policies_distinct() -> None:
    results = [
        SimpleNamespace(
            chunk_id="representative",
            metadata={"equivalent_chunk_ids": ["qrel", "representative", 42]},
        )
    ]

    assert direct_evidence_rank(results, {"qrel"}) is None
    assert (
        evidence_rank(results, {"qrel"}, include_exact_equivalents=False) is None
    )
    assert exact_equivalent_evidence_rank(results, {"qrel"}) == 1
    assert evidence_rank(results, {"qrel"}, include_exact_equivalents=True) == 1


def test_corpus_equivalence_detects_cross_document_exact_content_without_text_output() -> None:
    chunks = [
        SimpleNamespace(chunk_id="installation-copy", text="Disconnect all power."),
        SimpleNamespace(chunk_id="programming-copy", text="  DISCONNECT  ALL POWER.  "),
        SimpleNamespace(chunk_id="different", text="Block the motor shaft."),
    ]

    equivalence = build_corpus_exact_content_equivalence(chunks)

    assert equivalence == {
        "installation-copy": ("installation-copy", "programming-copy"),
        "programming-copy": ("installation-copy", "programming-copy"),
    }
    assert (
        exact_equivalent_evidence_rank(
            [SimpleNamespace(chunk_id="programming-copy")],
            {"installation-copy"},
            equivalence=equivalence,
        )
        == 1
    )
    assert direct_evidence_rank(
        [SimpleNamespace(chunk_id="programming-copy")], {"installation-copy"}
    ) is None


def test_authoritative_requirements_need_every_group_and_ignore_candidate_metadata() -> None:
    item = SimpleNamespace(
        relevant_chunk_ids=["installation-safety", "installation-wiring"],
        evidence_requirements=[
            SimpleNamespace(
                id="safety",
                canonical_chunk_ids=["installation-safety"],
                reviewed_alternative_chunk_ids=["programming-safety"],
            ),
            SimpleNamespace(
                id="wiring",
                canonical_chunk_ids=["installation-wiring"],
                reviewed_alternative_chunk_ids=[],
            ),
        ],
    )
    reviewed_alternative = SimpleNamespace(
        chunk_id="programming-safety",
        metadata={"equivalent_chunk_ids": ["installation-wiring"]},
    )

    partial = authoritative_requirement_coverage(
        [reviewed_alternative], item, equivalence={}
    )

    assert partial["contract_mode"] == "explicit"
    assert partial["covered_requirement_count"] == 1
    assert partial["complete"] is False
    assert partial["requirements"][0]["provenance"] == "reviewed_authoritative_alternative"
    assert partial["requirements"][1]["provenance"] == "unreviewed"

    unrelated_same_document = authoritative_requirement_coverage(
        [SimpleNamespace(chunk_id="installation-unrelated")], item, equivalence={}
    )
    assert unrelated_same_document["covered_requirement_count"] == 0
    assert all(
        requirement["provenance"] == "unreviewed"
        for requirement in unrelated_same_document["requirements"]
    )

    exact_copy = authoritative_requirement_coverage(
        [SimpleNamespace(chunk_id="programming-wiring")],
        item,
        equivalence={
            "installation-wiring": ("installation-wiring", "programming-wiring"),
            "programming-wiring": ("installation-wiring", "programming-wiring"),
        },
    )
    assert exact_copy["covered_requirement_count"] == 1
    assert exact_copy["requirements"][1]["provenance"] == "exact_content_equivalent"


def test_retrieval_evaluator_hashes_existing_canonical_behavior_owners() -> None:
    expected = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in evaluate_retrieval.SOURCE_IDENTITY_PATHS.items()
    }

    identity = evaluate_retrieval._source_identity()
    assert identity["files"] == expected
    assert {
        "list_completeness_policy",
        "query_role_policy",
    } <= identity["files"].keys()
    assert identity["files_sha256"] == evaluate_retrieval._json_sha256(expected)
    assert identity == evaluate_retrieval._source_identity()
    assert "evidence_policy" in identity["files"]
    assert evaluate_retrieval.ARTIFACT_SCHEMA_VERSION == 5
    assert evaluate_retrieval.DEFAULT_OUTPUT.name == "atv320-retrieval-evaluation-v5.json"


def test_six_development_variants_are_preregistered_and_behaviorally_distinct() -> None:
    assert tuple(DEVELOPMENT_RETRIEVAL_VARIANTS) == (
        "baseline",
        "refill",
        "query-expansion-v2",
        "refill-query-expansion-v2",
        "dense-reserve-d20-s5",
        "sparse-reserve-d5-s20",
    )
    identities = [variant.identity() for variant in DEVELOPMENT_RETRIEVAL_VARIANTS.values()]
    assert len({json.dumps(identity, sort_keys=True) for identity in identities}) == 6

    baseline = development_retrieval_variant("baseline")
    refill = development_retrieval_variant("refill")
    dense = development_retrieval_variant("dense-reserve-d20-s5")
    sparse = development_retrieval_variant("sparse-reserve-d5-s20")
    assert baseline.fusion_profile.dense_reserve == 5
    assert baseline.fusion_profile.sparse_reserve == 24
    assert refill.refill_after_content_dedup is True
    assert all(identity["diagnostic_component_limit"] == 100 for identity in identities)
    applied = refill.apply(evaluate_retrieval.ATV320_RETRIEVAL_CONTRACT)
    assert applied.refill_after_content_dedup is True
    assert applied.diagnostic_component_limit == 100
    assert evaluate_retrieval.ATV320_RETRIEVAL_CONTRACT.refill_after_content_dedup is False
    assert evaluate_retrieval.ATV320_RETRIEVAL_CONTRACT.diagnostic_component_limit is None
    assert (dense.fusion_profile.dense_reserve, dense.fusion_profile.sparse_reserve) == (
        20,
        5,
    )
    assert (sparse.fusion_profile.dense_reserve, sparse.fusion_profile.sparse_reserve) == (
        5,
        20,
    )
    with pytest.raises(ValueError, match="Unknown development retrieval variant"):
        development_retrieval_variant("arbitrary")


def test_retrieval_variant_cli_is_development_only_and_uses_distinct_paths() -> None:
    legacy = evaluate_retrieval._parse_args([])
    assert legacy.dataset == "legacy"
    assert legacy.retrieval_variant == "baseline"
    assert legacy.output == evaluate_retrieval.DEFAULT_OUTPUT
    assert legacy.checkpoint == evaluate_retrieval.DEFAULT_CHECKPOINT
    legacy_variant = evaluate_retrieval._active_retrieval_variant(
        legacy.dataset,
        legacy.retrieval_variant,
    )
    assert legacy_variant.diagnostic_component_limit is None
    assert (
        legacy_variant.apply(evaluate_retrieval.ATV320_RETRIEVAL_CONTRACT)
        .diagnostic_component_limit
        is None
    )

    development = evaluate_retrieval._parse_args(
        [
            "--dataset",
            "development",
            "--retrieval-variant",
            "refill-query-expansion-v2",
        ]
    )
    assert development.output.name == (
        "atv320-development-retrieval-refill-query-expansion-v2-v5.json"
    )
    assert development.checkpoint.name == (
        "atv320-development-retrieval-refill-query-expansion-v2-v5-checkpoint.jsonl"
    )
    assert (
        evaluate_retrieval._active_retrieval_variant(
            development.dataset,
            development.retrieval_variant,
        ).diagnostic_component_limit
        == 100
    )

    with pytest.raises(SystemExit):
        evaluate_retrieval._parse_args(
            ["--dataset", "legacy", "--retrieval-variant", "refill"]
        )
    with pytest.raises(ValueError, match="development-only"):
        evaluate_retrieval._active_retrieval_variant("legacy", "refill")


def test_retrieval_checkpoint_fails_closed_across_variants_and_legacy_schema(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "retrieval-checkpoint.jsonl"
    baseline = {
        "artifact_schema_version": 5,
        "variant": "baseline",
        "selected_item_ids": ["dev-001"],
    }
    refill = {
        "artifact_schema_version": 5,
        "variant": "refill",
        "selected_item_ids": ["dev-001"],
    }
    evaluate_retrieval._write_checkpoint(checkpoint, baseline, [{"id": "dev-001"}])

    assert evaluate_retrieval._load_checkpoint(checkpoint, baseline) == [
        {"id": "dev-001"}
    ]
    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_retrieval._load_checkpoint(checkpoint, refill)
    with pytest.raises(RuntimeError, match="different frozen run"):
        evaluate_retrieval._load_checkpoint(
            checkpoint,
            {
                "artifact_schema_version": 3,
                "variant": "baseline",
                "selected_item_ids": ["dev-001"],
            },
        )


def test_development_loader_never_opens_the_sealed_confirmation_payload(
    monkeypatch,
) -> None:
    item = SimpleNamespace(
        id="dev-001",
        answerable=True,
        review_status="approved",
    )
    opened = []
    args = SimpleNamespace(
        dataset="development",
        development="development-only",
        manifest="hash-manifest-only",
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "read_dataset",
        lambda path: opened.append(path) or [item],
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "validate_dataset",
        lambda dataset, chunks, *, kind: {"corpus": {}, "kind": kind},
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "validate_evaluation_manifest",
        lambda path, chunks, dataset, *, dataset_key: {
            "confirmation_dataset_sha256": "c" * 64
        },
    )

    dataset, selected, validation, sealed_hash = (
        evaluate_retrieval._load_selected_dataset(args, [])
    )

    assert dataset == selected == [item]
    assert validation["kind"] == "development"
    assert sealed_hash == "c" * 64
    assert opened == ["development-only"]


def test_retrieval_scoring_trace_excludes_question_text_and_candidate_content() -> None:
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        document_id="installation",
        filename="TRACE-SECRET-FILENAME.pdf",
        text="TRACE-SECRET-EVIDENCE",
        page_numbers=[1],
        headings=["TRACE-SECRET-HEADING"],
        content_type="text",
        metadata={
            "document_role": "installation",
            "document_title": "TRACE-SECRET-TITLE",
            "private": "TRACE-SECRET-METADATA",
        },
        score=1.0,
        dense_rank=1,
        rrf_rank=1,
        rerank_rank=1,
    )
    trace = RetrievalTrace(
        dense_candidates=(candidate,),
        fused_candidates=(candidate,),
        selected_candidates_pre_dedup=(candidate,),
        candidate_pool_post_dedup=(candidate,),
        candidates_before_rerank=(candidate,),
        raw_cross_encoder_candidates=(candidate,),
        final_candidates=(candidate,),
        candidate_count_pre_dedup=1,
        candidate_count_post_dedup=1,
        candidate_count_before_rerank=1,
    )
    record = score_retrieval_execution(
        SimpleNamespace(
            id="dev-001",
            fact_family_id="family-001",
            language="en",
            question_type="safety",
            question="TRACE-SECRET-QUESTION: which wiring step is required?",
            relevant_chunk_ids=["chunk-1"],
            expected_document_ids=["installation"],
        ),
        QueryRetrievalResult(
            candidates=[candidate],
            candidate_pool=[candidate],
            retrieval_ms=1.0,
            rerank_ms=2.0,
            trace=trace,
        ),
        corpus_equivalence={},
        evidence_selection=select_evaluation_evidence(
            "TRACE-SECRET-QUESTION: which wiring step is required?",
            [candidate],
        ),
    )

    serialized = json.dumps(record, sort_keys=True)
    assert record["raw_dense_direct_evidence_rank"] == 1
    assert record["evidence_direct_evidence_rank"] == 1
    assert record["evidence_exact_equivalent_evidence_rank"] == 1
    assert record["evidence_failure_class"] == "hit"
    assert record["evidence_selection"] == {
        "policy_id": "canonical-evidence-top5-v1",
        "top_k": 5,
        "candidate_count": 1,
        "candidates": [
            {
                "chunk_id": "chunk-1",
                "document_id": "installation",
                "ordinal_rank": 1,
                "page_numbers": [1],
                "exact_content_provenance": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "installation",
                        "page_numbers": [1],
                        "heading_count": 1,
                    }
                ],
            }
        ],
        "duplicate_groups": [],
    }
    assert "TRACE-SECRET" not in serialized


def test_evaluation_evidence_boundary_has_a_fixed_top_five_limit() -> None:
    candidates = [
        RetrievalCandidate(
            chunk_id=f"chunk-{index}",
            document_id="installation",
            filename="manual.pdf",
            text=f"Unique evidence {index}.",
            page_numbers=[index],
            headings=["Safety"],
            content_type="text",
            metadata={"document_role": "installation"},
            score=1.0,
            rerank_rank=index,
        )
        for index in range(1, 7)
    ]

    selection = select_evaluation_evidence("Which wiring step is required?", candidates)

    assert [candidate.chunk_id for candidate in selection.candidates] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
        "chunk-5",
    ]


def test_retrieval_scoring_distinguishes_strict_and_exact_duplicate_provenance() -> None:
    programming_copy = RetrievalCandidate(
        chunk_id="programming-copy",
        document_id="programming",
        filename="programming.pdf",
        text="Disconnect all power.",
        page_numbers=[9],
        headings=["Safety"],
        content_type="text",
        metadata={
            "document_role": "programming",
            "document_title": "Programming manual",
        },
        score=1.0,
        rerank_rank=1,
    )
    execution = QueryRetrievalResult(
        candidates=[programming_copy],
        candidate_pool=[programming_copy],
        retrieval_ms=1.0,
        rerank_ms=2.0,
    )
    selection = select_evaluation_evidence(
        "Give the safety requirement.",
        execution.candidates,
    )

    record = score_retrieval_execution(
        SimpleNamespace(
            id="dev-duplicate",
            fact_family_id="family-duplicate",
            language="en",
            question_type="safety",
            relevant_chunk_ids=["installation-copy"],
            expected_document_ids=["installation"],
        ),
        execution,
        corpus_equivalence={
            "installation-copy": ("installation-copy", "programming-copy"),
            "programming-copy": ("installation-copy", "programming-copy"),
        },
        evidence_selection=selection,
    )

    assert record["evidence_direct_evidence_rank"] is None
    assert record["evidence_exact_equivalent_evidence_rank"] == 1
    assert record["wrong_document_evidence_top1"] is True
    assert (
        record["wrong_document_evidence_top1_excluding_exact_equivalents"]
        is False
    )
