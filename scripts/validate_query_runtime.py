"""Compatibility entry point for the supported Phase 7 retrieval smoke."""

from scripts.operations.validate_query_runtime import _build_parser as _build_parser
from scripts.operations.validate_query_runtime import main as main

if __name__ == "__main__":
    raise SystemExit(main())
