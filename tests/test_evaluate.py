"""Unit tests for the dependency-free dense smoke metrics."""

from types import SimpleNamespace

import pytest

from scripts.evaluate import evaluate_records


def test_evaluate_records_matches_page_or_phrase_and_calculates_metrics() -> None:
    records = [
        {"id": "one", "question": "first", "expected_pages": [2]},
        {"id": "two", "question": "second", "expected_phrases": ["Mahalanobis"]},
    ]
    results = {
        "first": [
            SimpleNamespace(page_numbers=[1], text="unrelated"),
            SimpleNamespace(page_numbers=[2], text="evidence"),
        ],
        "second": [SimpleNamespace(page_numbers=[3], text="no match")],
    }

    metrics = evaluate_records(records, lambda question, limit, document_id: results[question])

    assert [row["first_relevant_rank"] for row in metrics["rows"]] == [2, None]
    assert metrics["hit_rate_at_1"] == pytest.approx(0.0)
    assert metrics["hit_rate_at_3"] == pytest.approx(0.5)
    assert metrics["hit_rate_at_5"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.25)
