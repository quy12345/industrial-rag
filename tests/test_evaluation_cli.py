"""Offline contract tests for supported evaluation commands."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.dataset import DatasetValidationError
from scripts.evaluation import validate_dataset as validation_cli


def test_dataset_validation_parser_preserves_cli_contract() -> None:
    args = validation_cli._build_parser().parse_args([])

    assert args.calibration == Path("data/eval/phase7/calibration.jsonl")
    assert args.test == Path("data/eval/phase7/test.jsonl")
    assert args.chunks == Path("artifacts/phase7/frozen-chunks.jsonl")
    assert args.output == Path("artifacts/metrics/phase-7-dataset-validation.json")


def test_dataset_validation_cli_coordinates_injected_offline_helpers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        calibration=Path("calibration.jsonl"),
        test=Path("test.jsonl"),
        chunks=Path("chunks.jsonl"),
        output=Path("report.json"),
    )
    reads: list[Path] = []
    written: list[tuple[Path, object]] = []
    calibration = [object()]
    test = [object()]
    chunks = [object()]
    report = {"status": "valid"}

    monkeypatch.setattr(
        validation_cli,
        "_build_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(
        validation_cli,
        "read_dataset",
        lambda path: reads.append(path) or (calibration if path == args.calibration else test),
    )
    monkeypatch.setattr(validation_cli, "load_frozen_chunks", lambda path: chunks)
    monkeypatch.setattr(
        validation_cli,
        "validate_dataset_splits",
        lambda actual_calibration, actual_test, actual_chunks: (
            report
            if (actual_calibration, actual_test, actual_chunks) == (calibration, test, chunks)
            else pytest.fail("CLI changed validation argument order")
        ),
    )
    monkeypatch.setattr(
        validation_cli,
        "write_json_atomic",
        lambda path, payload: written.append((path, payload)),
    )

    assert validation_cli.main() == 0
    assert reads == [args.calibration, args.test]
    assert written == [(args.output, report)]
    assert capsys.readouterr().out == "ATV320 dataset validation PASS: report.json\n"


def test_dataset_validation_cli_preserves_argparse_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_dataset.py"])
    monkeypatch.setattr(
        validation_cli,
        "read_dataset",
        lambda path: (_ for _ in ()).throw(DatasetValidationError("invalid annotations")),
    )

    with pytest.raises(SystemExit) as caught:
        validation_cli.main()

    assert caught.value.code == 2
    assert "validate_dataset.py: error: invalid annotations" in capsys.readouterr().err
