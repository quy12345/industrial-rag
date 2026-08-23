"""Compatibility entry point for the supported Phase 7 query smoke."""

from scripts.operations.query_smoke import DEFAULT_OUTPUT as DEFAULT_OUTPUT
from scripts.operations.query_smoke import SCENARIOS as SCENARIOS
from scripts.operations.query_smoke import _atomic_write as _atomic_write
from scripts.operations.query_smoke import _build_parser as _build_parser
from scripts.operations.query_smoke import _git_commit as _git_commit
from scripts.operations.query_smoke import _runtime_versions as _runtime_versions
from scripts.operations.query_smoke import main as main

if __name__ == "__main__":
    raise SystemExit(main())
