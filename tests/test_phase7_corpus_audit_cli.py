"""Offline contract tests for the supported Phase 7 corpus audit command."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.audit_phase7_corpus as compatibility_cli
from scripts.operations import audit_phase7_corpus as audit_cli


def test_corpus_audit_shim_and_parser_preserve_cli_contract() -> None:
    assert compatibility_cli.main is audit_cli.main
    assert compatibility_cli._build_parser is audit_cli._build_parser
    assert compatibility_cli.SOURCES is audit_cli.SOURCES

    args = audit_cli._build_parser().parse_args([])

    assert args.raw_dir == Path("data/raw")
    assert args.output == Path("artifacts/metrics/phase-7-corpus-audit.json")
    assert [source["document_role"] for source in audit_cli.SOURCES] == [
        "installation",
        "programming",
    ]


def test_corpus_audit_uses_fake_local_pdfs_and_writes_sanitized_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "audit.json"
    for source in audit_cli.SOURCES:
        (tmp_path / source["filename"]).write_bytes(b"fake-pdf")

    opened_documents: list[FakeDocument] = []

    class FakeTextPage:
        @staticmethod
        def get_text_range() -> str:
            return "digital technical text"

    class FakePage:
        @staticmethod
        def get_textpage() -> FakeTextPage:
            return FakeTextPage()

    class FakeDocument:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.closed = False
            opened_documents.append(self)

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            self.closed = True

    written: list[tuple[Path, dict[str, object]]] = []
    args = SimpleNamespace(raw_dir=tmp_path, output=output)
    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=FakeDocument))
    monkeypatch.setattr(
        audit_cli,
        "_build_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(
        audit_cli,
        "write_json_atomic",
        lambda path, payload: written.append((path, payload)),
    )

    assert audit_cli.main() == 0
    assert len(opened_documents) == 2
    assert all(document.closed for document in opened_documents)
    assert written[0][0] == output
    documents = written[0][1]["documents"]
    assert isinstance(documents, list)
    assert [document["page_count"] for document in documents] == [4, 4]
    assert all(document["digital_text_layer"] is True for document in documents)
    assert all("sample_text" not in document for document in documents)
    assert capsys.readouterr().out == f"Phase 7 corpus audit PASS: {output}\n"


def test_corpus_audit_preserves_missing_pdf_argparse_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=object))
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_phase7_corpus.py", "--raw-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as caught:
        audit_cli.main()

    assert caught.value.code == 2
    assert "Missing regular PDF" in capsys.readouterr().err
