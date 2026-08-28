"""Read-only real-model smoke for the active frozen ATV320 retrieval runtime."""

from __future__ import annotations

import argparse
import json

from app.config import Settings, get_settings, resolve_retrieval_runtime
from app.domain.retrieval import QueryRetriever
from app.domain.retrieval_contracts import ATV320_RETRIEVAL_CONTRACT, FrozenRetrievalContract


def main() -> int:
    args = _build_parser().parse_args()
    settings, contract = resolve_retrieval_runtime(get_settings())
    retriever = _build_query_retriever(settings, contract=contract)
    result = retriever.retrieve(args.question, document_id=args.document_id)
    print(
        json.dumps(
            {
                "contract_id": contract.contract_id,
                "contract_chunk_count": contract.chunk_count,
                "retriever": type(retriever).__name__,
                "candidate_count": len(result.candidates),
                "top_chunk_id": result.candidates[0].chunk_id if result.candidates else None,
                "retrieval_ms": result.retrieval_ms,
                "rerank_ms": result.rerank_ms,
            },
            sort_keys=True,
        )
    )
    return 0


def _build_query_retriever(
    settings: Settings,
    *,
    contract: FrozenRetrievalContract,
) -> QueryRetriever:
    from app.composition.retrieval import build_query_retriever

    return build_query_retriever(settings, contract=contract)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="What tasks are covered by the ATV320 Installation Manual?",
    )
    parser.add_argument(
        "--document-id",
        default=ATV320_RETRIEVAL_CONTRACT.document_ids[0],
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
