"""Offline guards for the replacement held-out-v2 runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_phase7_heldout_v2 import (
    PROVIDER_APPROVAL_TOKEN,
    _require_private_path,
    _validate_egress_approval,
)


def test_heldout_v2_requires_exact_egress_approval() -> None:
    _validate_egress_approval(PROVIDER_APPROVAL_TOKEN)
    with pytest.raises(SystemExit, match="approval"):
        _validate_egress_approval("APPROVE")


def test_heldout_v2_rejects_historic_or_public_input_paths() -> None:
    _require_private_path(Path("data/eval/phase7/private-heldout-v2/heldout-v2.jsonl"))
    with pytest.raises(SystemExit, match="must remain"):
        _require_private_path(Path("data/eval/phase7/test.jsonl"))
