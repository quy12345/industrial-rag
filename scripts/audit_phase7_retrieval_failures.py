"""Audit Phase 7 calibration retrieval misses without opening held-out queries.

The output is intentionally sanitized: it records calibration IDs, ranks, corpus
integrity and document distributions, but excludes query strings and chunk text.
No cross-encoder or generation provider is initialized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.candidate_audit import dense_results_to_candidates
from app.config import Settings
from app.evaluation import direct_evidence_rank, load_frozen_chunks, phrase_matches
from app.evaluation_e2e import score_expected_answer_fact
from app.hybrid_retrieval import create_sparse_embedding_model, sparse_search
from app.phase7 import (
    Phase7DatasetItem,
    dataset_sha256,
    read_phase7_dataset,
    validate_phase7_datasets,
    write_json_atomic,
)
from app.retrieval import (
    create_embedding_model,
    create_qdrant_client,
    dense_search,
    get_embedding_dimension,
    get_indexed_chunk_ids,
)
from app.retrieval_runtime import PHASE7_RETRIEVAL_CONTRACT, validate_frozen_runtime

AUDIT_IDS = (
    "phase7_calibration_004",
    "phase7_calibration_005",
    "phase7_calibration_010",
)
SEARCH_LIMIT = 200

# Calibration-only diagnostic rewrites. They are fixed before executing the audit,
# are never applied to held-out rows, and are identified by name only in artifacts.
QUERY_VARIANTS = {
    "phase7_calibration_004": {
        "original": None,
        "english_translation": "What must be done to prevent unintended motor rotation?",
        "technical_identifiers": "block motor shaft prevent unintended rotation",
    },
    "phase7_calibration_005": {
        "original": None,
        "technical_terms": "installation work verification check",
    },
    "phase7_calibration_010": {
        "original": None,
        "english_translation": "Which menu groups can the MODE key switch between?",
        "technical_identifiers": "MODE key Reference Monitoring Configuration menu groups",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/eval/phase7/calibration.jsonl")
    )
    parser.add_argument("--test", type=Path, default=Path("data/eval/phase7/test.jsonl"))
    parser.add_argument(
        "--chunks", type=Path, default=Path("artifacts/phase7/frozen-chunks.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/phase-7-calibration-004-010-audit.json"),
    )
    args = parser.parse_args()

    calibration = read_phase7_dataset(args.calibration)
    test = read_phase7_dataset(args.test)
    chunks = load_frozen_chunks(args.chunks)
    validation = validate_phase7_datasets(calibration, test, chunks)
    items = {item.id: item for item in calibration}
    selected = [items[item_id] for item_id in AUDIT_IDS]
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}

    settings = _phase7_settings(Settings())
    client = create_qdrant_client(settings)
    validate_frozen_runtime(
        client,
        collection_names=(settings.qdrant_collection, settings.qdrant_hybrid_collection),
        contract=PHASE7_RETRIEVAL_CONTRACT,
    )
    dense_ids = _indexed_ids(client, settings.qdrant_collection)
    hybrid_ids = _indexed_ids(client, settings.qdrant_hybrid_collection)
    dense_model = create_embedding_model(
        settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    if get_embedding_dimension(dense_model) != PHASE7_RETRIEVAL_CONTRACT.dense_dimension:
        raise RuntimeError("Dense model dimension differs from the frozen Phase 7 contract.")
    sparse_model = create_sparse_embedding_model(
        settings.sparse_model,
        settings.embedding_cache_dir,
        disable_stemmer=settings.bm25_disable_stemmer,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
    )

    rows = [
        _audit_item(
            item,
            by_chunk_id=by_chunk_id,
            dense_ids=dense_ids,
            hybrid_ids=hybrid_ids,
            client=client,
            settings=settings,
            dense_model=dense_model,
            sparse_model=sparse_model,
        )
        for item in selected
    ]
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": list(AUDIT_IDS),
        "calibration_dataset_sha256": dataset_sha256(calibration),
        "corpus": validation["corpus"],
        "search_limit": SEARCH_LIMIT,
        "cross_encoder_loaded": False,
        "provider_calls": 0,
        "held_out_queries_executed": 0,
        "rows": rows,
        "sanitization": {
            "query_text": "excluded",
            "chunk_text": "excluded",
            "answer_text": "not applicable",
        },
    }
    write_json_atomic(args.output, payload)
    print(f"Phase 7 calibration miss audit PASS: {args.output}")
    return 0


def _audit_item(
    item: Phase7DatasetItem,
    *,
    by_chunk_id: dict[str, Any],
    dense_ids: set[str],
    hybrid_ids: set[str],
    client: Any,
    settings: Settings,
    dense_model: Any,
    sparse_model: Any,
) -> dict[str, Any]:
    qrels = [by_chunk_id[chunk_id] for chunk_id in item.relevant_chunk_ids]
    combined_evidence = "\n".join(chunk.text for chunk in qrels)
    fact_coverage = (
        [
            score_expected_answer_fact(fact, combined_evidence)
            for fact in item.expected_answer_facts
        ]
        if item.scenario == "en_to_en"
        else []
    )
    variants: list[dict[str, Any]] = []
    for variant_id, configured_query in QUERY_VARIANTS[item.id].items():
        query = item.question if configured_query is None else configured_query
        dense = dense_results_to_candidates(
            dense_search(
                client,
                query,
                collection_name=settings.qdrant_collection,
                vector_name=settings.dense_vector_name,
                embedding_model=dense_model,
                limit=SEARCH_LIMIT,
            )
        )
        sparse = sparse_search(
            client,
            query,
            collection_name=settings.qdrant_hybrid_collection,
            sparse_vector_name=settings.sparse_vector_name,
            sparse_embedding_model=sparse_model,
            limit=SEARCH_LIMIT,
        )
        variants.append(
            {
                "variant_id": variant_id,
                "dense_qrel_rank_at_200": direct_evidence_rank(
                    dense, set(item.relevant_chunk_ids)
                ),
                "sparse_qrel_rank_at_200": direct_evidence_rank(
                    sparse, set(item.relevant_chunk_ids)
                ),
                "dense_document_distribution_top20": _document_distribution(dense[:20]),
                "sparse_document_distribution_top20": _document_distribution(sparse[:20]),
                "dense_dominant_document_top20": _dominant_document(dense[:20]),
                "sparse_dominant_document_top20": _dominant_document(sparse[:20]),
            }
        )
    return {
        "id": item.id,
        "expected_document_ids": item.expected_document_ids,
        "qrel_count": len(qrels),
        "all_qrels_in_dense_collection": set(item.relevant_chunk_ids).issubset(dense_ids),
        "all_qrels_in_hybrid_collection": set(item.relevant_chunk_ids).issubset(hybrid_ids),
        "all_expected_phrases_in_qrels": all(
            any(phrase_matches(chunk.text, phrase) for chunk in qrels)
            for phrase in item.expected_phrases
        ),
        "answer_fact_source_support": (
            all(result["deterministic_matched"] for result in fact_coverage)
            if item.scenario == "en_to_en"
            else "cross_lingual_requires_reviewed_source_phrase"
        ),
        "fact_qrel_coverage": [
            {
                "id": result["id"],
                "type": result["type"],
                "deterministic_matched": result["deterministic_matched"],
                "strict_phrase_matched": result["strict_phrase_matched"],
                "max_alias_token_recall": result["max_alias_token_recall"],
            }
            for result in fact_coverage
        ],
        "qrel_chunk_diagnostics": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_numbers": chunk.page_numbers,
                "content_type": chunk.content_type,
                "heading_count": len(chunk.headings),
                "character_count": len(chunk.text),
                "expected_phrase_match": any(
                    phrase_matches(chunk.text, phrase) for phrase in item.expected_phrases
                ),
            }
            for chunk in qrels
        ],
        "query_variants": variants,
    }


def _phase7_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_collection": PHASE7_RETRIEVAL_CONTRACT.dense_collection,
            "qdrant_hybrid_collection": PHASE7_RETRIEVAL_CONTRACT.hybrid_collection,
            "bm25_avg_len": PHASE7_RETRIEVAL_CONTRACT.bm25_avg_len,
        }
    )


def _indexed_ids(client: Any, collection_name: str) -> set[str]:
    return set().union(
        *(
            get_indexed_chunk_ids(
                client, collection_name=collection_name, document_id=document_id
            )
            for document_id in PHASE7_RETRIEVAL_CONTRACT.document_ids
        )
    )


def _document_distribution(candidates: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(candidate.document_id for candidate in candidates).items()))


def _dominant_document(candidates: list[Any]) -> str | None:
    counts = Counter(candidate.document_id for candidate in candidates)
    return (
        min(counts, key=lambda document_id: (-counts[document_id], document_id))
        if counts
        else None
    )


if __name__ == "__main__":
    raise SystemExit(main())
