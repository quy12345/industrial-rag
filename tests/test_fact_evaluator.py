"""Offline characterization tests for deterministic and hybrid fact evaluation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from evaluation.dataset import ExpectedAnswerFact
from evaluation.facts import (
    DETERMINISTIC_FACT_EVALUATOR,
    FACT_EVALUATOR_ID,
    HYBRID_FACT_EVALUATOR,
    FactJudgeDecision,
    FactJudgeError,
    FactJudgeResult,
    OpenRouterLunaFactJudge,
    fact_evaluator_run_identity,
    score_answer_facts,
    score_expected_answer_fact,
)


def _text_fact(
    *,
    fact_id: str = "f1",
    aliases: list[str] | None = None,
    groups: list[list[str]] | None = None,
) -> ExpectedAnswerFact:
    return ExpectedAnswerFact(
        id=fact_id,
        aliases=aliases or ["disconnect all power sources"],
        required_token_groups=groups or [],
    )


def _item(*facts: ExpectedAnswerFact):
    return SimpleNamespace(
        id="synthetic-item",
        question="What must be done?",
        expected_answer_facts=list(facts),
    )


def test_v3_identity_and_default_result_contract() -> None:
    result = score_expected_answer_fact(
        _text_fact(), "Before work, all power sources must disconnect safely."
    )

    assert FACT_EVALUATOR_ID == "deterministic_typed_facts_v3"
    assert result["deterministic_matched"] is True
    assert result["final_matched"] is True
    assert result["matched"] is True
    assert result["resolution"] == "deterministic"
    assert result["judge_verdict"] is None
    assert result["judge_latency_ms"] is None
    assert "aliases" not in result


def test_alias_only_match_ignores_fixed_english_function_words() -> None:
    fact = _text_fact(aliases=["return to the previous menu"])

    result = score_expected_answer_fact(fact, "Return previous menu.")

    assert result["deterministic_matched"] is True
    assert result["matcher"] == "text_alias_content_tokens_v3"


def test_alias_only_match_ignores_fixed_vietnamese_function_words() -> None:
    fact = _text_fact(aliases=["quay về các menu trước"])

    assert score_expected_answer_fact(fact, "Quay về menu trước.")[
        "deterministic_matched"
    ]


def test_decimal_does_not_split_required_groups_across_clauses() -> None:
    fact = _text_fact(
        aliases=["default baud rate is 19.2 Kbps"],
        groups=[["default"], ["19.2 Kbps"]],
    )

    result = score_expected_answer_fact(fact, "The default baud rate is 19.2 Kbps.")

    assert result["deterministic_matched"] is True
    assert result["matched_clause_index"] == 0


def test_numeric_and_identifier_matching_remain_exact() -> None:
    numeric = ExpectedAnswerFact(
        id="voltage",
        aliases=["24 VDC"],
        type="numeric_unit",
        value="24",
        unit="VDC",
    )
    identifier = ExpectedAnswerFact(
        id="rating",
        aliases=["IP65"],
        type="identifier",
        acceptable_values=["IP65"],
    )

    assert score_expected_answer_fact(numeric, "Supply: 24VDC.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(numeric, "Supply: 124 VDC.")[
        "deterministic_matched"
    ]
    assert score_expected_answer_fact(identifier, "Rated IP65.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(identifier, "Rated IP650.")[
        "deterministic_matched"
    ]

    decimal = ExpectedAnswerFact(
        id="decimal-voltage",
        aliases=["24.0 VDC"],
        type="numeric_unit",
        value="24.0",
        unit="VDC",
    )
    negative = ExpectedAnswerFact(
        id="negative-voltage",
        aliases=["-24 VDC"],
        type="numeric_unit",
        value="-24",
        unit="VDC",
    )
    assert score_expected_answer_fact(decimal, "Supply: 24,0VDC.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(numeric, "Supply: -24 VDC.")[
        "deterministic_matched"
    ]
    assert score_expected_answer_fact(negative, "Supply: -24 VDC.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(negative, "Supply: --24 VDC.")[
        "deterministic_matched"
    ]


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
    result = score_expected_answer_fact(_text_fact(aliases=[expected]), answer)
    assert result["deterministic_matched"] is True
    assert result["match_mode"] == "inflection"


def test_text_inflection_does_not_use_arbitrary_prefix_matching() -> None:
    assert not score_expected_answer_fact(
        _text_fact(aliases=["MODE"]), "Select the model."
    )["deterministic_matched"]


def test_multiword_alternative_preserves_reviewed_order() -> None:
    fact = _text_fact(aliases=["motor shaft"], groups=[["motor shaft"]])

    assert score_expected_answer_fact(fact, "Inspect the motor shaft.")[
        "deterministic_matched"
    ]
    assert not score_expected_answer_fact(fact, "Inspect the shaft motor.")[
        "deterministic_matched"
    ]


def test_vietnamese_reordered_noun_phrase_requires_reviewed_alternative() -> None:
    without_review = _text_fact(
        aliases=["nắp bảo vệ bằng nhựa"],
        groups=[["nắp bảo vệ bằng nhựa"]],
    )
    with_review = _text_fact(
        aliases=["nắp bảo vệ bằng nhựa"],
        groups=[["nắp bảo vệ bằng nhựa", "nắp nhựa bảo vệ"]],
    )
    answer = "Tháo nắp nhựa bảo vệ."

    assert not score_expected_answer_fact(without_review, answer)[
        "deterministic_matched"
    ]
    assert score_expected_answer_fact(with_review, answer)["deterministic_matched"]


@pytest.mark.parametrize(
    ("answer", "matched", "polarity"),
    [
        ("Protective equipment is installed.", True, "positive"),
        ("Protective equipment is not installed.", False, "negative"),
        ("Do not verify that protective equipment is installed.", False, "negative"),
        ("Not only installed but also closed protective equipment.", True, "positive"),
        ("Protective equipment is not never installed.", False, "ambiguous"),
    ],
)
def test_negation_guard_remains_span_aware(answer, matched, polarity) -> None:
    fact = _text_fact(
        aliases=["protective equipment installed"],
        groups=[["protective equipment"], ["installed"]],
    )

    result = score_expected_answer_fact(fact, answer)

    assert result["deterministic_matched"] is matched
    assert result["polarity"] == polarity


def test_default_relation_is_not_inferred_from_parameter_and_value() -> None:
    fact = _text_fact(
        aliases=["default baud rate is 19.2 Kbps"],
        groups=[["default"], ["baud rate"], ["19.2 Kbps"]],
    )

    assert not score_expected_answer_fact(
        fact, "Set the baud rate; 19.2 Kbps is one available value."
    )["deterministic_matched"]


@dataclass
class _FakeJudge:
    decisions: tuple[FactJudgeDecision, ...]
    calls: int = 0
    received_fact_ids: tuple[str, ...] = ()

    def score(self, *, item_id, question, answer, unresolved_facts):
        self.calls += 1
        self.received_fact_ids = tuple(fact.id for fact in unresolved_facts)
        return FactJudgeResult(decisions=self.decisions, latency_ms=12.5)


def test_hybrid_batches_only_unresolved_text_facts_once() -> None:
    matched = _text_fact(fact_id="matched", aliases=["disconnect power"])
    unresolved = _text_fact(fact_id="unresolved", aliases=["return previous menu"])
    numeric = ExpectedAnswerFact(
        id="numeric",
        aliases=["24 VDC"],
        type="numeric_unit",
        value="24",
        unit="VDC",
    )
    judge = _FakeJudge((FactJudgeDecision("unresolved", "present"),))

    result = score_answer_facts(
        _item(matched, unresolved, numeric),
        "Disconnect power. The voltage is wrong.",
        judge=judge,
    )

    assert judge.calls == 1
    assert judge.received_fact_ids == ("unresolved",)
    assert result.invocation_count == 1
    assert result.fallback_fact_count == 1
    by_id = {fact["id"]: fact for fact in result.facts}
    assert by_id["matched"]["resolution"] == "deterministic"
    assert by_id["numeric"]["resolution"] == "deterministic"
    assert by_id["numeric"]["final_matched"] is False
    assert by_id["unresolved"]["resolution"] == "luna_fallback"
    assert by_id["unresolved"]["final_matched"] is True


def test_deterministic_match_does_not_call_judge() -> None:
    judge = _FakeJudge(())

    result = score_answer_facts(
        _item(_text_fact(aliases=["disconnect power"])),
        "Disconnect power.",
        judge=judge,
    )

    assert judge.calls == 0
    assert result.invocation_count == 0


@pytest.mark.parametrize(
    "decisions",
    [
        (),
        (FactJudgeDecision("unexpected", "present"),),
        (
            FactJudgeDecision("f1", "present"),
            FactJudgeDecision("f1", "absent"),
        ),
    ],
)
def test_hybrid_fails_closed_on_missing_extra_or_duplicate_ids(decisions) -> None:
    judge = _FakeJudge(decisions)

    with pytest.raises(FactJudgeError, match="invalid_verdict_contract"):
        score_answer_facts(_item(_text_fact()), "Unrelated answer.", judge=judge)


def test_run_identity_is_offline_and_versions_hybrid_configuration() -> None:
    deterministic = fact_evaluator_run_identity(DETERMINISTIC_FACT_EVALUATOR)
    hybrid = fact_evaluator_run_identity(HYBRID_FACT_EVALUATOR)

    assert "judge" not in deterministic
    assert hybrid["judge"]["model"] == "openai/gpt-5.6-luna"
    assert hybrid["judge"]["reasoning_effort"] == "low"
    assert hybrid["judge"]["store"] is False
    assert len(hybrid["judge"]["prompt_sha256"]) == 64


def test_luna_adapter_uses_pinned_openrouter_route_and_structured_output(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"verdicts":[{"fact_id":"f1","verdict":"present"}]}'
                        )
                    )
                ]
            )

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    module = ModuleType("openai")
    module.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    judge = OpenRouterLunaFactJudge(api_key="test-only")

    result = judge.score(
        item_id="synthetic-item",
        question="SENTINEL-QUESTION",
        answer="SENTINEL-ANSWER",
        unresolved_facts=[_text_fact()],
    )

    assert result.decisions == (FactJudgeDecision("f1", "present"),)
    assert captured["model"] == "openai/gpt-5.6-luna"
    assert captured["store"] is False
    assert captured["extra_body"] == {
        "provider": {
            "order": ["OpenAI"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "reasoning": {"effort": "low", "exclude": True},
    }
    response_format = captured["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["json_schema"]["strict"] is True


def test_luna_adapter_sanitizes_provider_failures(monkeypatch) -> None:
    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        RuntimeError("SENTINEL-PROVIDER-BODY")
                    )
                )
            )

    module = ModuleType("openai")
    module.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)

    with pytest.raises(FactJudgeError) as caught:
        OpenRouterLunaFactJudge(api_key="test-only").score(
            item_id="synthetic-item",
            question="SENTINEL-QUESTION",
            answer="SENTINEL-ANSWER",
            unresolved_facts=[_text_fact()],
        )

    message = str(caught.value)
    assert "synthetic-item" in message
    assert "SENTINEL" not in message
