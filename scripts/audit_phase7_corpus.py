"""Compatibility entry point for the supported Phase 7 corpus audit."""

from scripts.operations.audit_phase7_corpus import SOURCES as SOURCES
from scripts.operations.audit_phase7_corpus import _build_parser as _build_parser
from scripts.operations.audit_phase7_corpus import main as main

if __name__ == "__main__":
    raise SystemExit(main())
