"""Compatibility entry point for the supported Phase 7 indexing command."""

from scripts.operations.index_phase7_corpus import DEFAULT_INPUTS as DEFAULT_INPUTS
from scripts.operations.index_phase7_corpus import _git_commit as _git_commit
from scripts.operations.index_phase7_corpus import _parser as _parser
from scripts.operations.index_phase7_corpus import _positive_int as _positive_int
from scripts.operations.index_phase7_corpus import _version as _version
from scripts.operations.index_phase7_corpus import _write_manifest as _write_manifest
from scripts.operations.index_phase7_corpus import main as main

if __name__ == "__main__":
    raise SystemExit(main())
