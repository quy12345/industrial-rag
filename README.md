# Industrial Technical Manual RAG

## Status

**Phase 2 — Document ingestion preview**

This phase adds a CLI-only ingestion preview for PDF and DOCX technical manuals. It uses Docling's native `DocumentConverter` and `HierarchicalChunker` to inspect structure-aware chunks and normalize their metadata into JSON-serializable Pydantic models.

## Implemented

- FastAPI scaffold and `GET /api/v1/health`
- Typed settings, pytest, Ruff, Docker Compose, and CI
- Docling PDF/DOCX conversion
- Native structure-aware chunking with `HierarchicalChunker`
- Deterministic document and chunk IDs
- Page provenance, heading breadcrumbs, conservative content types, and metadata normalization
- Terminal chunk previews and optional UTF-8 JSONL output

The API still exposes only the health endpoint. This phase does not write chunks to Qdrant, create embeddings, run retrieval, call an LLM, or add LangChain.

## Repository structure

```text
app/                 FastAPI application, models, and ingestion logic
scripts/             CLI ingestion preview and future evaluation placeholder
data/                Raw and evaluation data directories
tests/               Pytest tests
artifacts/           Generated metrics, figures, and local preview output
.github/workflows/   CI workflow
Dockerfile           API container image
docker-compose.yml   API and Qdrant services
pyproject.toml       Project metadata and tool configuration
```

## Local setup

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

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Ingestion preview

Place a PDF or DOCX file under `data/raw/`, then run:

```bash
python scripts/ingest_preview.py data/raw/manual.pdf
```

Useful options:

```bash
python scripts/ingest_preview.py data/raw/manual.pdf --limit 10
python scripts/ingest_preview.py data/raw/manual.pdf --limit 10 --output artifacts/ingestion-preview.jsonl
python scripts/ingest_preview.py data/raw/manual.pdf --batch-size 8 --limit 3
```

- `--limit` controls how many chunk previews are printed; `0` prints no chunk details.
- `--preview-chars` defaults to `500` and limits terminal text per preview.
- `--output` overwrites the target with one `DocumentChunk` JSON object per UTF-8 line and creates missing parent directories.
- `--page-start` and `--page-end` select an inclusive PDF page range and must be used together.
- `--batch-size` splits a PDF page range into smaller, sequential conversion runs. A value of `8` is a conservative starting point for the current development environment, not a universal optimum.

Each normalized chunk has this schema:

```json
{
  "chunk_id": "manual-a4f832bd71c2_p18_c0003",
  "document_id": "manual-a4f832bd71c2",
  "filename": "manual.pdf",
  "text": "...",
  "page_numbers": [18],
  "headings": ["Troubleshooting"],
  "content_type": "text",
  "metadata": {
    "source_path": "data/raw/manual.pdf",
    "file_extension": ".pdf",
    "chunk_index": 3,
    "character_count": 842
  }
}
```

Docling page numbers are copied directly from provenance without adding or subtracting one. If provenance or headings are absent, the corresponding lists remain empty. Content type is only marked as `table`, `list`, or `code` when Docling item labels provide evidence; otherwise it is `text`, `mixed`, or `unknown`.

First runs may take longer because Docling can initialize local model assets. Scanned PDFs without a text layer may require OCR configuration. Table extraction and metadata availability depend on the source document structure.

## Large PDF handling

Docling can accumulate native memory while preprocessing many pages in one conversion run. For digital PDFs with an existing text layer, this project disables OCR and uses memory-conscious Docling batch settings. Page-range batching reduces peak memory by creating a new converter for each range and processing ranges sequentially.

Every range must return Docling `SUCCESS`. The ingestion stops on `PARTIAL_SUCCESS` or `FAILURE`, and JSONL output is written atomically only after all requested ranges have completed. This prevents a partial conversion from looking like a complete artifact.

Windows PowerShell example:

```powershell
python scripts/ingest_preview.py data/raw/manual.pdf `
  --page-start 1 `
  --page-end 21 `
  --batch-size 8 `
  --limit 3 `
  --output artifacts/manual-batched.jsonl
```

Batch boundaries are page-aligned, but heading context is not reconstructed across ranges and multi-page tables are not merged across a boundary. Scanned PDFs still require an OCR-enabled mode, which is outside this Phase 2 patch. Suitable batch size depends on available RAM and PDF complexity. Some native backends may still require process isolation even when converters are not reused.

## Test and lint

```bash
pytest
ruff check .
```

## Docker Compose

```bash
docker compose config
docker compose up --build
```

The API is available at `http://localhost:8000`; Qdrant is exposed on ports `6333` and `6334`. The Python application does not connect to Qdrant in Phase 2.

## Roadmap

1. Add embeddings and Qdrant ingestion.
2. Add dense and hybrid retrieval.
3. Add reranking, LLM generation, and citations.
4. Add retrieval and answer evaluation.
