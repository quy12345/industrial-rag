"""Describe continuous Ragas scores against private human labels (assessment v2)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.infrastructure.corpus_artifacts import write_json_atomic
from evaluation.human_review import HumanReviewError, validate_scorecard_agreement


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        artifact = _read_json(args.artifact)
        labels = _read_jsonl(args.labels)
        result = validate_scorecard_agreement(artifact, labels)
        write_json_atomic(args.output, result)
    except HumanReviewError as exc:
        raise SystemExit(f"Scorecard agreement validation failed: {exc}") from None
    print(f"Sanitized scorecard agreement written: {args.output}")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanReviewError("Unable to read sanitized scorecard artifact.") from exc
    if not isinstance(payload, dict):
        raise HumanReviewError("Sanitized scorecard artifact must be an object.")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanReviewError("Unable to read private human labels.") from exc
    if any(not isinstance(row, dict) for row in payload):
        raise HumanReviewError("Every human-label row must be an object.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
