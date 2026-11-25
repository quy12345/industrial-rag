"""Offline fixed-input replay and private snapshot failure boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.generation_prompt import format_evidence
from app.application.query_service import QueryExecution, QueryTimings
from app.contracts.query import Citation, QueryResponse
from app.domain.documents import DocumentChunk
from app.domain.retrieval_contracts import ATV320_RETRIEVAL_CONTRACT
from app.infrastructure.corpus_artifacts import chunk_set_metadata, write_json_atomic
from evaluation.dataset import EvaluationItem, dataset_sha256
from evaluation.e2e import build_case_record
from evaluation.human_review import PrivateReviewPack, import_legacy_review
from evaluation.review_inputs import (
    PrivateInputJournal,
    ReviewInput,
    ReviewInputError,
    corpus_candidate,
    payload_sha256,
)
from evaluation.semantic import MetricScore, RagasScores, SemanticJudgeError
from scripts.evaluation import evaluate_e2e as cli


def _item(item_id: str = "a1") -> EvaluationItem:
    return EvaluationItem(
        id=item_id,
        question="PRIVATE_QUESTION",
        language="en",
        answerable=True,
        scenario="en_to_en",
        question_type="safety",
        expected_document_ids=["manual"],
        relevant_chunk_ids=["c1"],
        expected_pages=[1],
        expected_phrases=["cool"],
        expected_answer_facts=[{"id": "f1", "aliases": ["PRIVATE_ALIAS"]}],
        reference_answer="PRIVATE_REFERENCE",
        reference_answer_review_status="approved",
        citation_required=True,
        review_status="approved",
    )


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c1",
        document_id="manual",
        filename="manual.pdf",
        text="PRIVATE_EVIDENCE",
        page_numbers=[1],
        headings=["Handling"],
    )


def _execution() -> QueryExecution:
    candidate = corpus_candidate(_chunk())
    return QueryExecution(
        response=QueryResponse(
            answer="PRIVATE_ANSWER",
            abstained=False,
            citations=[
                Citation(
                    **{
                        key: value
                        for key, value in _chunk().model_dump().items()
                        if key
                        in {"chunk_id", "document_id", "filename", "page_numbers", "headings"}
                    },
                    excerpt="short",
                )
            ],
        ),
        timings=QueryTimings(1, 2, 3, 4, 5, 15),
        usage=None,
        candidates=(candidate,),
        candidate_pool=(candidate,),
        evidence_candidates=(candidate,),
        evidence_context_chars=len(format_evidence([candidate], max_chars=4000).text),
    )


def _scores() -> RagasScores:
    scored = MetricScore("complete", 0.5, 1)
    na = MetricScore.not_applicable()
    return RagasScores(scored, scored, scored, na, na)


class FakeJudge:
    def __init__(self, fail_id: str | None = None) -> None:
        self.inputs = []
        self.fail_id = fail_id

    def score(self, value, **flags):
        self.inputs.append(value)
        if value.item_id == self.fail_id:
            raise SemanticJudgeError(
                item_id=value.item_id, metric="faithfulness", category="provider"
            )
        return _scores()


def test_snapshot_binds_raw_inputs_and_rejects_tampering_without_raw_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    item = _item()
    corpus = {"c1": _chunk()}
    identity = {"source_identity": "one", "dataset": "development"}
    path = Path("artifacts/local-review/fixed/inputs.jsonl")
    value = ReviewInput(item, _execution(), 4000)
    journal = PrivateInputJournal.open(path, identity, [item], corpus)
    journal.append(value)
    restored = PrivateInputJournal.open(path, identity, [item], corpus).rows[item.id]
    assert restored.to_payload() == value.to_payload()
    assert restored.execution.response.answer == "PRIVATE_ANSWER"
    with pytest.raises(ReviewInputError, match="another run"):
        PrivateInputJournal.open(path, {"source_identity": "two"}, [item], corpus)
    with pytest.raises(ReviewInputError, match="overwrite"):
        journal.append(value)
    changed = value.to_payload()
    changed["question"] = "RAW_MALICIOUS_SENTINEL"
    with pytest.raises(ReviewInputError) as caught:
        ReviewInput.from_payload(changed, item, corpus)
    assert "RAW_MALICIOUS_SENTINEL" not in str(caught.value)
    assert caught.value.__suppress_context__
    # Rehashing altered evidence must still fail independent corpus validation.
    changed = value.to_payload()
    changed["execution"]["evidence_candidates"][0]["text"] = "PRIVATE_FORGED"
    changed["input_sha256"] = payload_sha256(
        {k: v for k, v in changed.items() if k != "input_sha256"}
    )
    with pytest.raises(ReviewInputError):
        ReviewInput.from_payload(changed, item, corpus)


@pytest.fixture
def replay_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    items = [_item(), _item("a2")]
    chunks = [_chunk()]
    corpus = {"c1": chunks[0]}
    source_identity = {
        "version": 13,
        "files": {
            name: "a" * 64
            for name in (
                "generation_prompt",
                "query_service",
                "retrieval_composition",
                "retrieval_contract",
            )
        },
    }
    current_identity = json.loads(json.dumps(source_identity)) | {"version": 14}
    monkeypatch.setattr(cli, "_source_identity", lambda: current_identity)
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text("synthetic frozen corpus", encoding="utf-8")
    monkeypatch.setattr(cli, "FROZEN_CHUNKS_PATH", frozen)
    identity = {
        "artifact_schema_version": 17,
        "source_identity": source_identity,
        "dataset": "development",
        "dataset_sha256": dataset_sha256(items),
        "corpus": chunk_set_metadata(chunks),
        "selected_item_ids": [item.id for item in items],
        "contract_chunk_ids_sha256": ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "generation_max_context_chars": 4000,
        "evidence_top_k": 5,
        "generation": {"model": "qwen/qwen3-235b-a22b-2507"},
    }
    records = []
    review_path = Path("artifacts/local-review/source/review.md")
    pack = PrivateReviewPack.open(
        review_path, selected_ids=[i.id for i in items], checkpoint_ids=[]
    )
    for item in items:
        record = build_case_record(item, _execution(), _scores())
        record.pop("input_sha256")
        records.append(record)
        pack.append(item=item, execution=_execution(), scores=_scores(), max_context_chars=4000)
    artifact = {"schema_version": 17, "run_identity": identity, "cases": records}
    source_path = Path("source.json")
    write_json_atomic(source_path, artifact)
    args = cli._parse_args(
        [
            "--dataset",
            "development",
            "--replay-artifact",
            str(source_path),
            "--replay-review",
            str(review_path),
            "--output",
            "result.json",
            "--checkpoint",
            "checkpoint.jsonl",
            "--judge-provider-approval-token",
            cli.RAGAS_APPROVAL_TOKENS["development"],
        ]
    )
    return args, items, chunks, corpus, artifact


def test_replay_never_builds_runtime_and_resume_preserves_answer_and_historical_costs(
    replay_setup,
    monkeypatch,
    capsys,
):
    args, items, chunks, _, source = replay_setup

    def prohibited(*a, **kw):
        raise AssertionError("Runtime construction is prohibited in replay")

    monkeypatch.setattr(cli, "Settings", prohibited)
    monkeypatch.setattr(cli, "QueryService", prohibited)
    monkeypatch.setattr(cli, "_build_generation_adapter", prohibited)
    judge = FakeJudge(fail_id="a2")
    monkeypatch.setattr(cli, "_build_judge", lambda _: judge)
    with pytest.raises(RuntimeError, match="a2"):
        cli._run_replay(args, chunks=chunks, dataset=items)
    checkpoint = [json.loads(line) for line in args.checkpoint.read_text().splitlines()]
    assert len(checkpoint) == 2  # header plus only the completed first item
    judge = FakeJudge()
    monkeypatch.setattr(cli, "_build_judge", lambda _: judge)
    assert cli._run_replay(args, chunks=chunks, dataset=items) == 0
    assert [value.item_id for value in judge.inputs] == ["a2"]
    assert judge.inputs[0].answer == "PRIVATE_ANSWER"
    assert "Heading: Handling" in judge.inputs[0].cited_contexts[0]
    output = json.loads(args.output.read_text())
    assert output["schema_version"] == 19
    assert output["run_identity"]["execution_mode"] == "judge_replay"
    assert output["run_identity"]["replay"]["input_policy"] == "legacy_review_import"
    assert output["run_identity"]["replay"]["original_answer_hash_verified"] is False
    assert (
        output["cases"][0]["retrieval_evidence_fingerprint"]
        == source["cases"][0]["retrieval_evidence_fingerprint"]
    )
    assert (
        output["scorecard"]["operational"]["measurement_provenance"]["generation_usage"]
        == "historical_not_rebilled"
    )
    assert output["cases"][0]["timings_ms"]["judge"] == 3
    encoded = args.output.read_text() + args.checkpoint.read_text() + capsys.readouterr().out
    for sentinel in (
        "PRIVATE_ANSWER",
        "PRIVATE_QUESTION",
        "PRIVATE_EVIDENCE",
        "PRIVATE_REFERENCE",
        "PRIVATE_ALIAS",
    ):
        assert sentinel not in encoded


@pytest.mark.parametrize(
    "field", ["question", "reference", "evidence", "heading", "case_id", "citation"]
)
def test_legacy_mismatch_is_rejected_before_judge_construction(replay_setup, monkeypatch, field):
    args, items, chunks, _, _ = replay_setup
    text = args.replay_review.read_text(encoding="utf-8")
    replacements = {
        "question": ("PRIVATE_QUESTION", "different_question"),
        "reference": ("PRIVATE_REFERENCE", "different_reference"),
        "evidence": ("PRIVATE_EVIDENCE", "different_evidence"),
        "heading": ("heading: Handling", "heading: different"),
        "case_id": ("<!-- item:a2 -->", "<!-- item:other -->"),
        "citation": ("- `c1`", "- `unknown`"),
    }
    old, new = replacements[field]
    assert old in text
    args.replay_review.write_text(text.replace(old, new, 1), encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "_build_judge", lambda _: calls.append(True))
    with pytest.raises(ReviewInputError):
        cli._run_replay(args, chunks=chunks, dataset=items)
    assert calls == []


def test_private_pack_recovers_checkpoint_ahead_without_overwriting_labels(replay_setup):
    args, items, _, corpus, source = replay_setup
    values = import_legacy_review(
        args.replay_review,
        items=items,
        cases=source["cases"],
        corpus=corpus,
        max_context_chars=4000,
    )
    target = Path("artifacts/local-review/new/review.md")
    pack = PrivateReviewPack.open(target, selected_ids=[i.id for i in items], checkpoint_ids=[])
    labels = target.with_name("labels.jsonl")
    edited = labels.read_text().replace('"answer_correct": null', '"answer_correct": true')
    labels.write_text(edited, encoding="utf-8")
    # Simulate checkpoint written, then process termination before Markdown append.
    pack = PrivateReviewPack.open(target, selected_ids=[i.id for i in items], checkpoint_ids=["a1"])
    pack.append(
        item=items[0], execution=values[0].execution, scores=_scores(), max_context_chars=4000
    )
    assert labels.read_text() == edited
    assert "PRIVATE_ANSWER" in target.read_text()


def test_headed_context_preserves_order_and_rejects_forged_chunk():
    from evaluation.semantic import SemanticConfigurationError, headed_cited_contexts

    first = corpus_candidate(_chunk())
    second = first.model_copy(update={"chunk_id": "c2", "text": "SECOND"})
    corpus = {
        "c1": _chunk(),
        "c2": _chunk().model_copy(update={"chunk_id": "c2", "text": "SECOND"}),
    }
    result = headed_cited_contexts((first, second, first), ["c2", "c1", "c1"], corpus)
    assert result == (
        "Document title: manual.pdf\nHeading: Handling\n\nPRIVATE_EVIDENCE",
        "Document title: manual.pdf\nHeading: Handling\n\nSECOND",
    )
    with pytest.raises(SemanticConfigurationError):
        headed_cited_contexts((first.model_copy(update={"headings": ["FORGED"]}),), ["c1"], corpus)


def test_replay_parser_and_approval_are_judge_only(replay_setup):
    args, *_ = replay_setup
    cli._validate_approvals(args, environ={"OPENROUTER_API_KEY": "offline-sentinel"})
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        cli._validate_approvals(args, environ={})
    with pytest.raises(SystemExit, match="Ragas"):
        cli._validate_approvals(
            replace_namespace(args, judge_provider_approval_token="invalid"), environ={}
        )


def replace_namespace(args, **values):
    from argparse import Namespace

    return Namespace(**(vars(args) | values))


def test_live_judge_failure_resumes_saved_answer_without_regeneration(replay_setup, monkeypatch):
    _, items, chunks, _, _ = replay_setup
    generated = []

    class Service:
        def __init__(self, **kwargs):
            pass

        def execute(self, **kwargs):
            generated.append(kwargs)
            return _execution()

    settings = SimpleNamespace(
        generation_max_context_chars=4000,
        evidence_score_threshold=0.1,
        qdrant_collection="fake",
        qdrant_hybrid_collection="fake",
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "_runtime_settings", lambda value: value)
    monkeypatch.setattr(cli, "load_frozen_chunks", lambda _: chunks)
    monkeypatch.setattr(cli, "_load_dataset", lambda *_: (items, {}, {}))
    monkeypatch.setattr(
        cli,
        "_run_identity",
        lambda **_: {
            "artifact_schema_version": 18,
            "source_identity": {"version": 14},
            "execution_mode": "live",
            "dataset": "development",
            "ragas": {"diagnostic_metrics": []},
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "app.composition.retrieval",
        SimpleNamespace(
            LazyQueryRetriever=lambda _: object(),
            build_query_retriever=lambda *a, **kw: None,
            validate_frozen_runtime=lambda *a, **kw: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.infrastructure.qdrant.client",
        SimpleNamespace(
            create_qdrant_client=lambda _: object(),
        ),
    )
    monkeypatch.setattr(cli, "QueryService", Service)
    monkeypatch.setattr(cli, "_build_generation_adapter", lambda _: object())
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    judge = FakeJudge(fail_id="a2")
    monkeypatch.setattr(cli, "_build_judge", lambda _: judge)
    args = [
        "--dataset",
        "development",
        "--local-review-output",
        "artifacts/local-review/live/review.md",
        "--output",
        "live.json",
        "--checkpoint",
        "live-checkpoint.jsonl",
        "--provider-approval-token",
        cli.DEVELOPMENT_PROVIDER_APPROVAL_TOKEN,
        "--judge-provider-approval-token",
        cli.RAGAS_APPROVAL_TOKENS["development"],
    ]
    with pytest.raises(RuntimeError, match="a2"):
        cli.main(args)
    assert len(generated) == 2
    journal = Path("artifacts/local-review/live/inputs.jsonl")
    assert len(journal.read_text().splitlines()) == 3
    labels = journal.with_name("labels.jsonl")
    edited = labels.read_text().replace('"answer_correct": null', '"answer_correct": true')
    labels.write_text(edited, encoding="utf-8")
    judge = FakeJudge()
    monkeypatch.setattr(cli, "_build_judge", lambda _: judge)
    assert cli.main(args) == 0
    assert len(generated) == 2
    assert [row.item_id for row in judge.inputs] == ["a2"]
    assert labels.read_text() == edited


def test_v18_replay_requires_captured_answer_hash(replay_setup, monkeypatch):
    args, items, chunks, corpus, source = replay_setup
    origin = source["run_identity"]
    source["schema_version"] = origin["artifact_schema_version"] = 18
    origin["source_identity"]["version"] = 14
    origin["execution_mode"] = "live"
    origin["frozen_chunks_sha256"] = cli._file_sha256(cli.FROZEN_CHUNKS_PATH)
    path = args.replay_review.with_name("inputs.jsonl")
    journal = PrivateInputJournal.open(path, origin, items, corpus)
    for item, case in zip(items, source["cases"], strict=True):
        saved = ReviewInput(item, _execution(), 4000)
        journal.append(saved)
        case["input_sha256"] = saved.to_payload()["input_sha256"]
    write_json_atomic(args.replay_artifact, source)
    monkeypatch.setattr(cli, "_build_judge", lambda _: FakeJudge())
    assert cli._run_replay(args, chunks=chunks, dataset=items) == 0
    output = json.loads(args.output.read_text())
    assert output["run_identity"]["replay"]["original_answer_hash_verified"] is True
    text = args.replay_review.read_text(encoding="utf-8")
    args.replay_review.write_text(
        text.replace("PRIVATE_ANSWER", "CHANGED_ANSWER"), encoding="utf-8"
    )
    with pytest.raises(ReviewInputError, match="captured input differ"):
        cli._run_replay(args, chunks=chunks, dataset=items)


def test_scope_replay_resumes_captured_inputs_without_external_clients(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    item = _item().model_copy(update={"question": "What Bluetooth pairing PIN is required?"})
    items = [item, item.model_copy(update={"id": "a2"})]
    chunks = [_chunk()]
    corpus = {"c1": chunks[0]}
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text("synthetic frozen corpus", encoding="utf-8")
    monkeypatch.setattr(cli, "FROZEN_CHUNKS_PATH", frozen)
    monkeypatch.setattr(cli, "_file_sha256", lambda _: "b" * 64)
    origin = {
        "artifact_schema_version": 18,
        "execution_mode": "live",
        "frozen_chunks_sha256": "b" * 64,
        "source_identity": {
            "version": 14,
            "files": {
                key: "a" * 64
                for key in (
                    "generation_prompt",
                    "query_service",
                    "retrieval_composition",
                    "retrieval_contract",
                )
            },
        },
        "dataset": "development",
        "dataset_sha256": dataset_sha256(items),
        "corpus": chunk_set_metadata(chunks),
        "selected_item_ids": [row.id for row in items],
        "contract_chunk_ids_sha256": ATV320_RETRIEVAL_CONTRACT.chunk_ids_sha256,
        "generation_max_context_chars": 4000,
        "evidence_top_k": 5,
        "generation": {"model": "qwen/qwen3-235b-a22b-2507"},
    }
    input_path = Path("artifacts/local-review/scope/inputs.jsonl")
    journal = PrivateInputJournal.open(input_path, origin, items, corpus)
    source_records = []
    for row in items:
        saved = ReviewInput(row, _execution(), 4000)
        journal.append(saved)
        source_records.append(build_case_record(
            row, saved.execution, _scores(), input_sha256=saved.to_payload()["input_sha256"]
        ))
    source_path = Path("source-v18.json")
    write_json_atomic(
        source_path,
        {"schema_version": 18, "run_identity": origin, "cases": source_records},
    )
    args = cli._parse_args(
        [
            "--dataset",
            "development",
            "--scope-replay-artifact",
            str(source_path),
            "--scope-replay-inputs",
            str(input_path),
            "--output",
            "scope.json",
            "--checkpoint",
            "scope-checkpoint.jsonl",
        ]
    )
    cli._validate_approvals(args, environ={})
    monkeypatch.setattr(cli, "Settings", lambda: pytest.fail("scope replay built Settings"))
    monkeypatch.setattr(cli, "QueryService", lambda *args, **kwargs: pytest.fail("runtime built"))
    monkeypatch.setattr(cli, "_build_judge", lambda _: pytest.fail("judge built"))

    write = cli._write_scope_checkpoint
    writes = []

    def interrupt_second_write(path, identity, records, corpus_ids):
        writes.append(records[-1]["id"])
        if writes == ["a1", "a2"]:
            raise InterruptedError
        write(path, identity, records, corpus_ids)

    monkeypatch.setattr(cli, "_write_scope_checkpoint", interrupt_second_write)
    with pytest.raises(InterruptedError):
        cli._run_scope_replay(args, chunks=chunks, dataset=items)
    first = [json.loads(line) for line in args.checkpoint.read_text().splitlines()[1:]]
    assert [record["id"] for record in first] == ["a1"]
    assert not args.output.exists()
    assert cli._run_scope_replay(args, chunks=chunks, dataset=items) == 0
    assert writes == ["a1", "a2", "a2"]
    output = json.loads(args.output.read_text(encoding="utf-8"))
    assert output["cases"][:1] == first
    assert [record["id"] for record in output["cases"]] == ["a1", "a2"]
    assert output["kind"] == "evidence_scope_replay"
    assert output["cases"][0]["proposed_abstention_reason"] == "evidence_scope_missing"
    assert output["cases"][0]["pre_generation_scope"]["missing_concept_ids"] == [
        "bluetooth",
        "credential",
        "pairing",
    ]
    assert output["run_identity"]["replay"]["source_artifact_sha256"] == "b" * 64
    encoded = args.output.read_text() + args.checkpoint.read_text()
    for sentinel in ("PRIVATE_QUESTION", "PRIVATE_ANSWER", "PRIVATE_EVIDENCE"):
        assert sentinel not in encoded


@pytest.mark.parametrize(
    "mutation", ["extra_raw", "dataset", "source", "selected", "v17_checkpoint"]
)
def test_replay_rejects_identity_or_schema_drift_before_egress(replay_setup, monkeypatch, mutation):
    args, items, chunks, _, source = replay_setup
    if mutation == "extra_raw":
        source["cases"][0]["answer"] = "RAW_SENTINEL"
    elif mutation == "dataset":
        source["run_identity"]["dataset_sha256"] = "0" * 64
    elif mutation == "source":
        source["run_identity"]["source_identity"]["files"]["generation_prompt"] = "0" * 64
    elif mutation == "selected":
        source["run_identity"]["selected_item_ids"] = ["a1"]
    else:
        cli._write_checkpoint(args.checkpoint, source["run_identity"], [], set())
    write_json_atomic(args.replay_artifact, source)
    called = []
    monkeypatch.setattr(cli, "_build_judge", lambda _: called.append(True))
    with pytest.raises((ReviewInputError, RuntimeError)):
        cli._run_replay(args, chunks=chunks, dataset=items)
    assert called == []
