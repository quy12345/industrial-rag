"""Immutable corpus and retrieval identity for the active Phase 7 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.phase7_optimization import PHASE7_CALIBRATION_FUSION_PROFILE, Phase7FusionProfile
from app.query_expansion import QUERY_EXPANSION_PROFILE

RetrievalProfile = Literal["phase7"]


@dataclass(frozen=True)
class FrozenDocumentContext:
    """Trusted document metadata included in reranking and generation context."""

    document_id: str
    document_title: str
    document_role: str


@dataclass(frozen=True)
class FrozenRetrievalContract:
    """Immutable index identity required by a frozen retrieval runtime."""

    document_id: str
    chunk_count: int
    chunk_ids_sha256: str
    dense_collection: str
    hybrid_collection: str
    bm25_avg_len: float
    dense_candidate_limit: int
    sparse_candidate_limit: int
    rrf_k: int
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    dense_dimension: int = 384
    bm25_k: float = 1.2
    bm25_b: float = 0.75
    bm25_disable_stemmer: bool = True
    document_ids: tuple[str, ...] = ()
    document_contexts: tuple[FrozenDocumentContext, ...] = ()
    union_rrf_prune_limit: int | None = None
    query_expansion_profile: str | None = None
    phase7_fusion_profile: Phase7FusionProfile | None = None
    frozen_rerank_batch_size: int | None = None
    freeze_rerank_threads: bool = False
    frozen_rerank_threads: int | None = None

    @property
    def indexed_document_ids(self) -> tuple[str, ...]:
        """Return the documents whose stable IDs form this frozen corpus."""

        return self.document_ids or (self.document_id,)

    @property
    def document_context_by_id(self) -> dict[str, dict[str, str]]:
        """Return trusted metadata in the shape consumed by reranking."""

        return {
            item.document_id: {
                "document_title": item.document_title,
                "document_role": item.document_role,
            }
            for item in self.document_contexts
        }


# Package-owned values let every adapter validate the same corpus identity without
# reading host artifacts or accepting field-by-field environment overrides.
PHASE7_RETRIEVAL_CONTRACT = FrozenRetrievalContract(
    document_id="atv320-installation-manual-en-nve41289-09-c181b4d7f11b",
    document_ids=(
        "atv320-installation-manual-en-nve41289-09-c181b4d7f11b",
        "atv320-programming-manual-en-nve41295-06-f5e9bb48167a",
    ),
    document_contexts=(
        FrozenDocumentContext(
            document_id="atv320-installation-manual-en-nve41289-09-c181b4d7f11b",
            document_title=("Altivar Machine ATV320 Variable Speed Drives Installation Manual"),
            document_role="installation",
        ),
        FrozenDocumentContext(
            document_id="atv320-programming-manual-en-nve41295-06-f5e9bb48167a",
            document_title=("Altivar Machine ATV320 Variable Speed Drives Programming Manual"),
            document_role="programming",
        ),
    ),
    chunk_count=2753,
    chunk_ids_sha256="2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d",
    dense_collection="industrial_manual_phase7_dense_v1",
    hybrid_collection="industrial_manual_phase7_hybrid_v1",
    bm25_avg_len=81.33599709407919,
    rrf_k=40,
    dense_candidate_limit=60,
    sparse_candidate_limit=40,
    union_rrf_prune_limit=30,
    query_expansion_profile=QUERY_EXPANSION_PROFILE,
    phase7_fusion_profile=PHASE7_CALIBRATION_FUSION_PROFILE,
    frozen_rerank_batch_size=8,
    freeze_rerank_threads=True,
    frozen_rerank_threads=None,
)


def retrieval_contract_for(profile: RetrievalProfile) -> FrozenRetrievalContract:
    """Return the complete immutable contract for one supported runtime profile."""

    if profile != "phase7":
        raise ValueError(f"Unsupported retrieval profile: {profile}")
    return PHASE7_RETRIEVAL_CONTRACT
