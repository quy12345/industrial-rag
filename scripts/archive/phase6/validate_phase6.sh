#!/bin/sh
set -eu

python -m pip install --user -e ".[dev]"
python -m ruff check .
python -m pytest -q --basetemp /tmp/industrial-rag-phase6-pytest
