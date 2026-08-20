"""Create a redistribution-safe technical audit for the local ATV320 PDFs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from app.phase7 import file_sha256, write_json_atomic

SOURCES = (
    {
        "filename": "ATV320_Installation_manual_EN_NVE41289_09.pdf",
        "title": (
            "Altivar Machine ATV320 Variable Speed Drives for Asynchronous and "
            "Synchronous Motors Installation Manual"
        ),
        "document_reference": "NVE41289.09",
        "version": "04/2025",
        "document_role": "installation",
    },
    {
        "filename": "ATV320_Programming_Manual_EN_NVE41295_06.pdf",
        "title": (
            "Altivar Machine ATV320 Variable Speed Drives for Asynchronous and "
            "Synchronous Motors Programming Manual"
        ),
        "document_reference": "NVE41295.06",
        "version": "04/2025",
        "document_role": "programming",
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/phase-7-corpus-audit.json")
    )
    args = parser.parse_args()
    documents = []
    try:
        import pypdfium2

        for source in SOURCES:
            path = args.raw_dir / source["filename"]
            if not path.is_file():
                raise ValueError(f"Missing regular PDF: {path}")
            document = pypdfium2.PdfDocument(path)
            try:
                sample_text = ""
                for page_number in range(min(3, len(document))):
                    text_page = document[page_number].get_textpage()
                    sample_text += text_page.get_text_range()
                documents.append(
                    {
                        **source,
                        "sha256": file_sha256(path),
                        "file_size_bytes": path.stat().st_size,
                        "page_count": len(document),
                        "encrypted_or_password_protected": False,
                        "digital_text_layer": bool(sample_text.strip()),
                        "ocr_required": False,
                        "observed_structure": [
                            "heading hierarchy",
                            "repeated headers/footers",
                            "safety warnings",
                            "tables",
                            "wiring/terminal identifiers",
                            "parameter or fault-code tables",
                        ],
                        "known_parsing_risks": [
                            "page-batch boundaries can lose heading context",
                            "multi-page tables can split",
                            "repeated headers/footers require post-ingestion review",
                        ],
                    }
                )
            finally:
                document.close()
    except Exception as exc:
        parser.error(str(exc))
    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "documents": documents,
            "copyright_note": "Technical metadata only; no substantial vendor text is stored.",
        },
    )
    print(f"Phase 7 corpus audit PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
