"""Compatibility facade for offline retrieval evaluation utilities.

New evaluation code should import from :mod:`evaluation.retrieval`. Existing
scripts keep this module path until their R07 classification slice.
"""

from evaluation.retrieval import DocumentLanguage as DocumentLanguage
from evaluation.retrieval import EvaluationCase as EvaluationCase
from evaluation.retrieval import EvaluationCategory as EvaluationCategory
from evaluation.retrieval import EvaluationError as EvaluationError
from evaluation.retrieval import EvaluationLanguage as EvaluationLanguage
from evaluation.retrieval import RetrievalScenario as RetrievalScenario
from evaluation.retrieval import RetrievedLike as RetrievedLike
from evaluation.retrieval import aggregate_rows as aggregate_rows
from evaluation.retrieval import chunk_set_metadata as chunk_set_metadata
from evaluation.retrieval import diagnostic_page_rank as diagnostic_page_rank
from evaluation.retrieval import diagnostic_phrase_rank as diagnostic_phrase_rank
from evaluation.retrieval import direct_evidence_rank as direct_evidence_rank
from evaluation.retrieval import evaluate_cases as evaluate_cases
from evaluation.retrieval import load_evaluation_cases as load_evaluation_cases
from evaluation.retrieval import load_frozen_chunks as load_frozen_chunks
from evaluation.retrieval import percentile_nearest_rank as percentile_nearest_rank
from evaluation.retrieval import phrase_matches as phrase_matches
from evaluation.retrieval import validate_cases_against_chunks as validate_cases_against_chunks

__all__ = [
    "DocumentLanguage",
    "EvaluationCase",
    "EvaluationCategory",
    "EvaluationError",
    "EvaluationLanguage",
    "RetrievedLike",
    "RetrievalScenario",
    "aggregate_rows",
    "chunk_set_metadata",
    "diagnostic_page_rank",
    "diagnostic_phrase_rank",
    "direct_evidence_rank",
    "evaluate_cases",
    "load_evaluation_cases",
    "load_frozen_chunks",
    "percentile_nearest_rank",
    "phrase_matches",
    "validate_cases_against_chunks",
]
