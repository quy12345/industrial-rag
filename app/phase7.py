"""Compatibility facade for Phase 7 corpus artifacts and offline datasets."""

from app.infrastructure.corpus_artifacts import PHASE7_CORPUS_VERSION as PHASE7_CORPUS_VERSION
from app.infrastructure.corpus_artifacts import PHASE7_DENSE_COLLECTION as PHASE7_DENSE_COLLECTION
from app.infrastructure.corpus_artifacts import PHASE7_HYBRID_COLLECTION as PHASE7_HYBRID_COLLECTION
from app.infrastructure.corpus_artifacts import PROTECTED_COLLECTIONS as PROTECTED_COLLECTIONS
from app.infrastructure.corpus_artifacts import file_sha256 as file_sha256
from app.infrastructure.corpus_artifacts import write_json_atomic as write_json_atomic
from app.infrastructure.corpus_artifacts import write_jsonl_atomic as write_jsonl_atomic
from evaluation.phase7_dataset import AnswerFactType as AnswerFactType
from evaluation.phase7_dataset import DatasetKind as DatasetKind
from evaluation.phase7_dataset import ExpectedAnswerFact as ExpectedAnswerFact
from evaluation.phase7_dataset import Phase7DatasetItem as Phase7DatasetItem
from evaluation.phase7_dataset import Phase7Error as Phase7Error
from evaluation.phase7_dataset import Phase7Source as Phase7Source
from evaluation.phase7_dataset import PhraseMatchMode as PhraseMatchMode
from evaluation.phase7_dataset import QuestionLanguage as QuestionLanguage
from evaluation.phase7_dataset import QuestionType as QuestionType
from evaluation.phase7_dataset import ReviewStatus as ReviewStatus
from evaluation.phase7_dataset import Scenario as Scenario
from evaluation.phase7_dataset import (
    build_exact_content_equivalence as build_exact_content_equivalence,
)
from evaluation.phase7_dataset import chunk_ids_sha256 as chunk_ids_sha256
from evaluation.phase7_dataset import dataset_sha256 as dataset_sha256
from evaluation.phase7_dataset import expand_exact_equivalent_qrels as expand_exact_equivalent_qrels
from evaluation.phase7_dataset import read_phase7_dataset as read_phase7_dataset
from evaluation.phase7_dataset import validate_phase7_dataset as validate_phase7_dataset
from evaluation.phase7_dataset import validate_phase7_datasets as validate_phase7_datasets
from evaluation.phase7_dataset import validate_source_records as validate_source_records

__all__ = [
    "AnswerFactType",
    "DatasetKind",
    "ExpectedAnswerFact",
    "PHASE7_CORPUS_VERSION",
    "PHASE7_DENSE_COLLECTION",
    "PHASE7_HYBRID_COLLECTION",
    "PROTECTED_COLLECTIONS",
    "Phase7DatasetItem",
    "Phase7Error",
    "Phase7Source",
    "PhraseMatchMode",
    "QuestionLanguage",
    "QuestionType",
    "ReviewStatus",
    "Scenario",
    "build_exact_content_equivalence",
    "chunk_ids_sha256",
    "dataset_sha256",
    "expand_exact_equivalent_qrels",
    "file_sha256",
    "read_phase7_dataset",
    "validate_phase7_dataset",
    "validate_phase7_datasets",
    "validate_source_records",
    "write_json_atomic",
    "write_jsonl_atomic",
]
