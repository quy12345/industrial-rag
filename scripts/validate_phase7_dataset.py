"""Compatibility entry point for Phase 7 dataset validation."""

from scripts.evaluation.validate_phase7_dataset import _build_parser as _build_parser
from scripts.evaluation.validate_phase7_dataset import main as main

if __name__ == "__main__":
    raise SystemExit(main())
