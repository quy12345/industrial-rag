# Industrial Technical Manual RAG

## Status

**Phase 3A — Dense indexing and retrieval**

This phase adds multilingual dense embeddings, Qdrant indexing, and ranked dense similarity search on top of the existing batched Docling ingestion flow. It does not generate final answers or call an LLM.

## Implemented

- FastAPI scaffold and `GET /api/v1/health`
- Typed settings, pytest, Ruff, Docker Compose, and CI
- Docling PDF/DOCX conversion
- Native structure-aware chunking with `HierarchicalChunker`
- Deterministic document and chunk IDs
- Page provenance, heading breadcrumbs, conservative content types, and metadata normalization
- Terminal chunk previews and optional UTF-8 JSONL output
- FastEmbed multilingual passage and query embeddings
- A shared Qdrant collection with the named cosine vector `dense`
- Deterministic UUIDv5 point IDs and citation-ready payload metadata
- Document indexing, safe re-index replacement, and dense search with document filtering
- CLI commands for indexing and search, with Qdrant in-memory unit tests

The API still exposes only the health endpoint. LangChain, LLM generation, answer citations, abstention, query APIs, sparse retrieval, BM25, RRF, reranking, and retrieval evaluation metrics are not implemented.

## Repository structure

```text
app/                 FastAPI application, ingestion, models, and dense retrieval
scripts/             CLI ingestion preview, dense indexing, and dense search
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

## Dense indexing and retrieval

Start Qdrant:

```powershell
docker compose up -d qdrant
```

Index the 21-page manual with memory-conscious PDF batching:

```powershell
python scripts/index_document.py data/raw/manual.pdf `
  --page-start 1 `
  --page-end 21 `
  --page-batch-size 8
```

If Docling reports `std::bad_alloc` while Docker Desktop is using part of the available RAM, retry with `--page-batch-size 4`. Smaller page batches reduce peak memory, but additional page-range boundaries can change the number of structure-aware chunks; do not assume different batch sizes produce identical chunk counts.

Run dense search:

```powershell
python scripts/search_dense.py `
  "Thuật toán nào được dùng để phát hiện bất thường?" `
  --limit 5
```

Filter results to one indexed document when needed:

```powershell
python scripts/search_dense.py `
  "ODA-MD hoạt động ở đâu trong mạng?" `
  --document-id manual-77d5dae4c2c5 `
  --limit 5
```

The default model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The first real indexing or search run needs internet access to download its public model assets; unit tests use a deterministic fake model and never download weights. Heading breadcrumbs are added only to embedding input, while the original chunk text remains unchanged in Qdrant payloads.

Re-indexing the same document removes its old points only after parsing and all embeddings succeed, then upserts deterministic replacement IDs. Other documents are not deleted. This MVP sequence is not a database transaction, so an upsert failure after deletion can temporarily leave that document incomplete.

Dense similarity scores are ranking signals, not probabilities. No default score threshold is claimed to be optimal; threshold tuning belongs to later retrieval evaluation. Dense results are ranked source chunks, not final RAG answers.

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

The API is available at `http://localhost:8000`; Qdrant is exposed on ports `6333` and `6334`. Host-side CLIs use `http://localhost:6333`, while Docker Compose overrides the API container host to `http://qdrant:6333`. The health endpoint remains a liveness check and does not contact Qdrant.

## Roadmap

1. Review and evaluate Phase 3A dense retrieval behavior.
2. Add an API and final answer contract in a later phase.
3. Add hybrid retrieval and reranking only after retrieval evaluation.
4. Add LLM generation, citations, and abstention in a later phase.
