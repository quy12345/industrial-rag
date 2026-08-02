# Industrial Technical Manual RAG

Phase 1 project scaffold for a future RAG question-answering system over industrial technical manuals in PDF and DOCX format.

## Status

**Phase 1 — Project Scaffold**

This phase includes a minimal FastAPI application, a health endpoint, settings loaded with `pydantic-settings`, a unit test, Ruff configuration, Docker support, Qdrant in Docker Compose, and GitHub Actions CI.

Document ingestion, parsing, retrieval, embeddings, LLM generation, citations, and evaluation are roadmap items and are not implemented yet.

## Architecture overview

The planned system will eventually contain document ingestion and parsing, retrieval backed by Qdrant, answer generation, citations, and evaluation. The current API is intentionally limited to the health endpoint; Qdrant is provisioned for later phases but is not accessed by the Python application.

## Repository structure

```text
app/                 FastAPI application and configuration
scripts/             Future command-line scripts
data/                Raw and evaluation data directories
tests/               Pytest tests
artifacts/           Metrics and figures generated later
.github/workflows/   CI workflow
Dockerfile           API container image
docker-compose.yml   API and Qdrant services
pyproject.toml       Project metadata and tool configuration
```

## Run locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Check the health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

## Test and lint

```bash
pytest
ruff check .
```

## Docker Compose

Copy `.env.example` to `.env` if you want to customize local environment values, then run:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`, and Qdrant is exposed on ports `6333` and `6334`. The API does not connect to Qdrant in Phase 1.

## Roadmap

1. Add document ingestion and PDF/DOCX parsing.
2. Add chunking, embeddings, and Qdrant retrieval.
3. Add LLM answer generation and source citations.
4. Add retrieval and answer evaluation.
5. Improve API, Docker deployment, and CI coverage.
