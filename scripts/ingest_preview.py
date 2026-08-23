"""Compatibility entry point for the supported ingestion preview command."""

from scripts.operations.ingest_preview import _build_parser as _build_parser
from scripts.operations.ingest_preview import _build_pdf_plan as _build_pdf_plan
from scripts.operations.ingest_preview import (
    _configure_output_encoding as _configure_output_encoding,
)
from scripts.operations.ingest_preview import _non_negative_int as _non_negative_int
from scripts.operations.ingest_preview import _parse_page_range as _parse_page_range
from scripts.operations.ingest_preview import _positive_int as _positive_int
from scripts.operations.ingest_preview import _print_previews as _print_previews
from scripts.operations.ingest_preview import _print_summary as _print_summary
from scripts.operations.ingest_preview import main as main

if __name__ == "__main__":
    raise SystemExit(main())
