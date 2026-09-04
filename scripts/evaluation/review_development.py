"""Print local-only development failure evidence without calling a provider."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.infrastructure.corpus_artifacts import load_frozen_chunks
from evaluation.dataset import read_dataset
from evaluation.development_review import DevelopmentReviewError, build_development_review

DEVELOPMENT_PATH = Path("data/eval/atv320/development-v1.jsonl")
FROZEN_CHUNKS_PATH = Path("artifacts/phase7/frozen-chunks.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require_canonical_development_path(args.development)
    try:
        artifact = _read_artifact(args.artifact)
        review = build_development_review(
            dataset=read_dataset(args.development),
            chunks=load_frozen_chunks(args.chunks),
            artifact=artifact,
            item_ids=args.item_id,
            repeat_index=args.repeat_index,
        )
    except (DevelopmentReviewError, OSError, ValueError) as exc:
        raise SystemExit(f"Development review failed: {exc}") from None
    _print_utf8_json(review)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT_PATH)
    parser.add_argument("--chunks", type=Path, default=FROZEN_CHUNKS_PATH)
    parser.add_argument(
        "--item-id",
        action="append",
        required=True,
        help="Development item ID; repeat the option to review a bounded cohort.",
    )
    parser.add_argument("--repeat-index", type=int, choices=(1, 2, 3), default=1)
    return parser.parse_args(argv)


def _require_canonical_development_path(path: Path) -> None:
    if path.resolve() != DEVELOPMENT_PATH.resolve():
        raise SystemExit(
            "Development review accepts only the canonical development-v1 dataset path."
        )


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DevelopmentReviewError(f"Malformed E2E artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise DevelopmentReviewError("The E2E artifact root must be an object.")
    return payload


def _print_utf8_json(payload: Mapping[str, Any]) -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
