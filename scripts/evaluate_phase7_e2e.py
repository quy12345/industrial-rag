"""Supported shim for the canonical Phase 7 E2E evaluation command."""

from scripts.evaluation.evaluate_phase7_e2e import main as main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
