"""Offline contracts for the active Phase 7 operational smoke commands."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.config import Settings
from app.domain.retrieval_contracts import PHASE7_RETRIEVAL_CONTRACT
from scripts.operations import query_smoke, validate_query_runtime


def test_retrieval_smoke_defaults_and_builds_the_resolved_phase7_contract(
    monkeypatch, capsys
) -> None:
    args = validate_query_runtime._build_parser().parse_args([])
    assert args.document_id == PHASE7_RETRIEVAL_CONTRACT.document_ids[0]
    assert "ATV320" in args.question

    settings = Settings()
    calls: list[tuple[object, object]] = []

    class Retriever:
        def retrieve(self, question, *, document_id):
            calls.append((question, document_id))
            return SimpleNamespace(candidates=[], retrieval_ms=1.0, rerank_ms=2.0)

    def build(resolved_settings, *, contract):
        calls.append((resolved_settings, contract))
        return Retriever()

    monkeypatch.setattr(validate_query_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(
        validate_query_runtime,
        "resolve_retrieval_runtime",
        lambda value: (value, PHASE7_RETRIEVAL_CONTRACT),
    )
    monkeypatch.setattr(validate_query_runtime, "build_query_retriever", build)
    monkeypatch.setattr(
        "sys.argv",
        ["validate_query_runtime", "menu navigation"],
    )

    assert validate_query_runtime.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval_profile"] == "phase7"
    assert payload["contract_chunk_count"] == 2753
    assert calls[0] == (settings, PHASE7_RETRIEVAL_CONTRACT)
    assert calls[1] == ("menu navigation", PHASE7_RETRIEVAL_CONTRACT.document_ids[0])


def test_retrieval_smoke_rejects_phase6_before_building_retriever(monkeypatch) -> None:
    settings = SimpleNamespace(retrieval_profile="phase6")
    monkeypatch.setattr(validate_query_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(
        validate_query_runtime,
        "resolve_retrieval_runtime",
        lambda value: (value, object()),
    )
    monkeypatch.setattr(
        validate_query_runtime,
        "build_query_retriever",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("retriever must not be constructed")
        ),
    )
    monkeypatch.setattr("sys.argv", ["validate_query_runtime"])

    assert validate_query_runtime.main() == 2


def test_query_smoke_scenarios_use_only_phase7_documents() -> None:
    scenario_document_ids = {scenario[2] for scenario in query_smoke.SCENARIOS}

    assert query_smoke.DEFAULT_OUTPUT.name == "phase-7-query-smoke.json"
    assert scenario_document_ids == set(PHASE7_RETRIEVAL_CONTRACT.document_ids)


def test_query_smoke_without_key_writes_sanitized_phase7_artifact(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "smoke.json"
    settings = Settings(openai_api_key=None, gemini_api_key=None)
    monkeypatch.setattr(query_smoke, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_smoke,
        "get_query_service",
        lambda: (_ for _ in ()).throw(AssertionError("service must not be constructed")),
    )
    monkeypatch.setattr(query_smoke, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(query_smoke, "_runtime_versions", lambda: {})
    monkeypatch.setattr("sys.argv", ["query_smoke", "--output", str(output)])

    assert query_smoke.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["retrieval_profile"] == "phase7"
    assert payload["corpus"]["document_ids"] == list(PHASE7_RETRIEVAL_CONTRACT.document_ids)
    assert payload["integration_run_status"] == "not_run"
    assert payload["reason"] == "api_key_unavailable"
    assert payload["test_scenarios"] == []


def test_query_smoke_rejects_phase6_before_constructing_service(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "smoke.json"
    monkeypatch.setattr(
        query_smoke,
        "get_settings",
        lambda: SimpleNamespace(retrieval_profile="phase6"),
    )
    monkeypatch.setattr(
        query_smoke,
        "get_query_service",
        lambda: (_ for _ in ()).throw(AssertionError("service must not be constructed")),
    )
    monkeypatch.setattr("sys.argv", ["query_smoke", "--output", str(output)])

    assert query_smoke.main() == 2
    assert not output.exists()
