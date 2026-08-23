"""Compatibility entry point for the provider-free Phase 7 retrieval closure."""

from scripts.evaluation.evaluate_phase7_retrieval_closure import main as main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
