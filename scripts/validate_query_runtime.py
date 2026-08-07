"""Read-only real-model smoke for the frozen Phase 6 retrieval runtime."""

from __future__ import annotations

import argparse
import json

from app.config import get_settings
from app.retrieval_runtime import PHASE6_RETRIEVAL_CONTRACT, build_query_retriever


def main() -> int:
    args = _build_parser().parse_args()
    retriever = build_query_retriever(get_settings())
    result = retriever.retrieve(args.question, document_id=args.document_id)
    print(
        json.dumps(
            {
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="Which algorithm detects anomalous sensor data?",
    )
    parser.add_argument(
        "--document-id",
        default=PHASE6_RETRIEVAL_CONTRACT.document_id,
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
