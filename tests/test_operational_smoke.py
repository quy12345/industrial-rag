"""Offline contracts for the active ATV320 operational smoke commands."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.config import Settings
from app.domain.retrieval_contracts import ATV320_RETRIEVAL_CONTRACT
from scripts.operations import query_smoke, validate_query_runtime


def test_retrieval_smoke_defaults_and_builds_the_resolved_atv320_contract(
    monkeypatch, capsys
) -> None:
    args = validate_query_runtime._build_parser().parse_args([])
    assert args.document_id == ATV320_RETRIEVAL_CONTRACT.document_ids[0]
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
        lambda value: (value, ATV320_RETRIEVAL_CONTRACT),
    )
    monkeypatch.setattr(validate_query_runtime, "_build_query_retriever", build)
    monkeypatch.setattr(
        "sys.argv",
        ["validate_query_runtime", "menu navigation"],
    )

    assert validate_query_runtime.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract_id"] == "atv320-2025-04-v1"
    assert payload["contract_chunk_count"] == 2753
    assert calls[0] == (settings, ATV320_RETRIEVAL_CONTRACT)
    assert calls[1] == ("menu navigation", ATV320_RETRIEVAL_CONTRACT.document_ids[0])


def test_query_smoke_scenarios_use_only_atv320_documents() -> None:
    scenario_document_ids = {scenario[2] for scenario in query_smoke.SCENARIOS}

    assert query_smoke.DEFAULT_OUTPUT.name == "atv320-query-smoke-v3.json"
    assert scenario_document_ids == set(ATV320_RETRIEVAL_CONTRACT.document_ids)


def test_query_smoke_without_key_writes_sanitized_atv320_artifact(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "smoke.json"
    settings = Settings(openai_api_key=None, gemini_api_key=None)
    monkeypatch.setattr(query_smoke, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_smoke,
        "_get_query_service",
        lambda: (_ for _ in ()).throw(AssertionError("service must not be constructed")),
    )
    monkeypatch.setattr(query_smoke, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(query_smoke, "_runtime_versions", lambda: {})
    monkeypatch.setattr("sys.argv", ["query_smoke", "--output", str(output)])

    assert query_smoke.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["contract_id"] == "atv320-2025-04-v1"
    assert payload["corpus"]["document_ids"] == list(ATV320_RETRIEVAL_CONTRACT.document_ids)
    assert payload["integration_run_status"] == "not_run"
    assert payload["reason"] == "api_key_unavailable"
    assert payload["test_scenarios"] == []
