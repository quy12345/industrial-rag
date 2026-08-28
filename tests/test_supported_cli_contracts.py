"""Public invocation characterization for every supported repository CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]

SUPPORTED_CLI_OPTIONS = [
    ("scripts.operations.audit_corpus", {"--raw-dir", "--output"}),
    (
        "scripts.evaluation.evaluate_e2e",
        {
            "--dataset",
            "--calibration",
            "--test",
            "--chunks",
            "--manifest",
            "--top-k",
            "--provider-approval-token",
            "--max-queries",
            "--item-id",
            "--checkpoint",
            "--output",
        },
    ),
    (
        "scripts.evaluation.evaluate_retrieval",
        {"--calibration", "--test", "--chunks", "--output"},
    ),
    (
        "scripts.operations.index_corpus",
        {
            "--page-batch-size",
            "--chunker",
            "--dense-collection",
            "--hybrid-collection",
            "--chunks-output",
            "--frozen-chunks",
            "--manifest-output",
            "--preview-only",
            "--verify-reindex",
        },
    ),
    (
        "scripts.operations.ingest_preview",
        {
            "--limit",
            "--output",
            "--page-start",
            "--page-end",
            "--batch-size",
            "--preview-chars",
        },
    ),
    ("scripts.operations.query_smoke", {"--output"}),
    (
        "scripts.evaluation.validate_dataset",
        {"--calibration", "--test", "--chunks", "--output"},
    ),
    ("scripts.operations.validate_query_runtime", {"--document-id"}),
]

REMOVED_CLI_MODULES = (
    "scripts.audit_phase7_corpus",
    "scripts.index_phase7_corpus",
    "scripts.ingest_preview",
    "scripts.query_smoke",
    "scripts.validate_query_runtime",
    "scripts.evaluate_phase7_e2e",
    "scripts.evaluate_phase7_retrieval_closure",
    "scripts.validate_phase7_dataset",
    "scripts.operations.audit_phase7_corpus",
    "scripts.operations.index_phase7_corpus",
    "scripts.evaluation.evaluate_phase7_e2e",
    "scripts.evaluation.evaluate_phase7_retrieval_closure",
    "scripts.evaluation.validate_phase7_dataset",
)


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "API_AUTH_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "BASELINE_COMMIT",
    ):
        environment.pop(name, None)
    environment["PYTHONUTF8"] = "1"
    return environment


def _run_cli(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=REPOSITORY_ROOT,
        env=_offline_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("module", [module for module, _ in SUPPORTED_CLI_OPTIONS])
def test_supported_cli_import_does_not_load_model_runtime(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module(sys.argv[1]); "
                "raise SystemExit('onnxruntime' in sys.modules)"
            ),
            module,
        ],
        cwd=REPOSITORY_ROOT,
        env=_offline_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Importing {module} loaded the ONNX model runtime before CLI argument parsing.\n"
        f"{result.stderr}"
    )


@pytest.mark.parametrize(("module", "expected_options"), SUPPORTED_CLI_OPTIONS)
def test_supported_cli_help_preserves_the_public_option_surface(
    module: str,
    expected_options: set[str],
) -> None:
    result = _run_cli(module, "--help")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.startswith("usage:")
    assert expected_options <= set(result.stdout.split())


@pytest.mark.parametrize(
    ("module", "arguments", "expected_error"),
    [
        (
            "scripts.operations.index_corpus",
            ("--chunker", "invalid"),
            "invalid choice",
        ),
        (
            "scripts.operations.ingest_preview",
            (),
            "the following arguments are required: input",
        ),
        (
            "scripts.evaluation.evaluate_e2e",
            ("--dataset", "calibration"),
            "the following arguments are required: --provider-approval-token",
        ),
    ],
)
def test_supported_cli_invalid_arguments_fail_before_integration_access(
    module: str,
    arguments: tuple[str, ...],
    expected_error: str,
) -> None:
    result = _run_cli(module, *arguments)

    assert result.returncode == 2
    assert expected_error in result.stderr


@pytest.mark.parametrize("module", REMOVED_CLI_MODULES)
def test_removed_cli_modules_fail_instead_of_silently_redirecting(module: str) -> None:
    result = _run_cli(module, "--help")

    assert result.returncode == 1
    assert f"No module named {module}" in result.stderr
