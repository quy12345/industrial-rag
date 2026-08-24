# R10 — Ingestion and content identity ownership

## 1. Goal and scope

R10 removes implementation from the root `app.ingestion` and `app.content_identity` modules. Stable
content identity now belongs to the domain; Docling orchestration and JSONL output have explicit
infrastructure owners. Supported ingestion commands import those owners directly.

The move preserves ingestion, chunking, stable IDs, content fingerprints, CLI output, and all failure
behavior. It does not parse the source manuals or run indexing.

## 2. Position in the system

```text
CLI input path
    |
    +--> infrastructure.ingestion.docling   conversion adapter
    +--> infrastructure.ingestion.pipeline normalization coordinator
    +--> infrastructure.ingestion.jsonl     optional atomic output
                         |
                         v
                 domain.documents.DocumentChunk

candidate text -> domain.content_identity -> evidence/reranking/dataset validation
```

`app.ingestion` and `app.content_identity` remain compatibility facades until R12, but active code no
longer depends on them.

## 3. Relevant background concepts

- A **canonical owner** is the only module containing an implementation.
- A **compatibility facade** exposes old imports by identity without duplicating behavior.
- An **infrastructure pipeline** coordinates an external parser and maps its SDK-shaped output into
  domain records.
- An **atomic write** creates a complete temporary file and replaces the destination only after
  serialization succeeds.
- A **stable content fingerprint** hashes exact normalized content and is not a semantic similarity
  score.

## 4. Input, output, and contracts

`ingest_document` accepts the same PDF/DOCX path, optional inclusive page range, optional PDF batch
size, and `hierarchical|hybrid` chunker. It returns the same ordered `DocumentChunk` list.

Document and chunk IDs retain their filename/content hash, NFKC normalization, page ordering,
heading normalization, occurrence index, and fixed digest lengths. Metadata keys and content-type
classification are unchanged.

`write_chunks_jsonl` retains UTF-8 newline-delimited Pydantic output and same-directory atomic
replacement. On failure, an existing complete destination remains untouched and the temporary file
is removed.

`normalize_evidence_content` retains NFKC, case folding, whitespace collapse, and punctuation.
`evidence_content_fingerprint` retains SHA-256 over that exact UTF-8 normalized value.

## 5. Step-by-step data flow

1. The inbound command validates CLI arguments without loading Docling.
2. The pipeline validates the input path and supported extension.
3. For PDFs, the Docling adapter supplies the page count and the domain policy creates batches.
4. The lazy Docling adapter converts each requested range.
5. The pipeline extracts text, provenance pages, heading breadcrumbs, and conservative content type.
6. Domain helpers assign the stable document ID, occurrence-aware chunk ID, and global chunk index.
7. The caller receives ordered canonical `DocumentChunk` objects.
8. Optional JSONL output writes every serialized chunk before replacing the destination.
9. Separately, evidence/reranking consumers normalize content and compute exact fingerprints through
   the domain owner.

## 6. Responsibilities of changed files

| Path | Responsibility after R10 |
| --- | --- |
| [`app/domain/content_identity.py`](../../app/domain/content_identity.py) | Pure normalization and SHA-256 evidence identity. |
| [`app/infrastructure/ingestion/docling.py`](../../app/infrastructure/ingestion/docling.py) | Lazy Docling/PDFium conversion and SDK error mapping. |
| [`app/infrastructure/ingestion/pipeline.py`](../../app/infrastructure/ingestion/pipeline.py) | Input validation, batching, conversion orchestration, and raw-chunk normalization. |
| [`app/infrastructure/ingestion/jsonl.py`](../../app/infrastructure/ingestion/jsonl.py) | Atomic normalized-chunk JSONL output. |
| Historical `app/ingestion.py` (removed in R12B) | Temporary public re-exports only. |
| Historical `app/content_identity.py` (removed in R12B) | Temporary public re-exports only. |
| [`scripts/operations/ingest_preview.py`](../../scripts/operations/ingest_preview.py) | Thin preview adapter using canonical owners. |
| [`scripts/operations/index_phase7_corpus.py`](../../scripts/operations/index_phase7_corpus.py) | Explicit integration adapter using canonical ingestion owners. |

Reranking, evidence selection, and Phase 7 dataset validation change only their content-identity
import path.

## 7. Important symbols and why they exist

- `DocumentChunk` is the persisted normalized domain record.
- `build_document_id` and `build_chunk_id` preserve corpus identity.
- `ingest_document` owns one complete conversion/normalization execution.
- `_append_normalized_chunks` keeps occurrence counters and indexes document-wide across batches.
- `get_pdf_page_count` and `_convert_document` remain lazy adapter seams used by offline fakes.
- `write_chunks_jsonl` isolates the only optional output side effect.
- `normalize_evidence_content` defines exact-equivalence normalization.
- `evidence_content_fingerprint` provides deterministic cross-document deduplication identity.

## 8. Before-and-after structure

```text
Before R10
  app.ingestion.py          295 lines: orchestration + normalization + JSONL + facade
  app.content_identity.py    20 lines: domain implementation at package root
  supported scripts -> root compatibility modules

After R10
  infrastructure/ingestion/pipeline.py  conversion and normalization
  infrastructure/ingestion/jsonl.py     atomic output
  domain/content_identity.py            pure content identity
  app.ingestion.py                       27-line facade
  app.content_identity.py                 8-line facade
  supported scripts -> canonical owners
```

The temporary architecture debt set falls from five edges to the two retrieval-composition edges
reserved for R11/R12.

## 9. Design decisions and trade-offs

The normalized-chunk pipeline is infrastructure rather than application code because it interprets
Docling-shaped objects and directly coordinates that adapter. Stable document records and hashes
remain in the domain because they do not depend on Docling.

JSONL output is separate from conversion so persistence failures cannot obscure parsing ownership.
The root facade no longer exposes private implementation helpers; existing supported interfaces
remain, while private monkeypatch targets are intentionally not treated as public contracts.

Archived scripts keep their historical root imports until R12. No duplicate compatibility pipeline
or silent fallback is introduced.

## 10. Tests and the behavior each test protects

| Test area | Protected behavior |
| --- | --- |
| Stable IDs | Exact document/chunk hashes, Unicode normalization, pages, and duplicates. |
| Path/range validation | Supported types, inclusive ranges, PDF-only batching, and messages. |
| Chunk normalization | Text, pages, headings, types, metadata, order, and global indexes. |
| Conversion failures | Partial/failed conversion is rejected without incomplete output. |
| JSONL | Exact schema, UTF-8, atomic replacement, and temporary-file cleanup. |
| Import laziness | Importing the facade does not load Docling or PDFium. |
| Content identity | Exact normalization string, hard-coded fingerprint, and facade identity. |
| CLI/indexing tests | Parser/output/error and integration safety behavior. |
| Architecture matrix | Canonical consumers cannot return to either root facade. |

## 11. Commands and expected results

Focused validation:

```powershell
python -m ruff check app/domain/content_identity.py app/infrastructure/ingestion `
  app/ingestion.py app/content_identity.py scripts/operations tests/test_ingestion.py `
  tests/test_evidence_selection.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_ingestion.py tests/test_phase7_index_cli.py `
  tests/test_reranking.py tests/test_evidence_selection.py tests/test_phase7.py `
  tests/test_architecture_boundaries.py
```

Final validation:

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Expected: no source-manual parsing, indexing, Qdrant, provider, or model access.

## 12. Small usage example

Canonical ingestion code uses explicit owners:

```python
from pathlib import Path

from app.infrastructure.ingestion.jsonl import write_chunks_jsonl
from app.infrastructure.ingestion.pipeline import ingest_document

chunks = ingest_document(Path("manual.pdf"), page_range=(1, 8))
write_chunks_jsonl(Path("chunks.jsonl"), chunks)
```

The old `from app.ingestion import ingest_document` import resolves to the same function during the
compatibility window.

## 13. Common failures and debugging

- A stable-ID failure means normalization or occurrence counting changed; do not update the golden
  value during this refactor.
- If Docling loads during import, inspect the adapter for module-level SDK imports.
- If batch indexes restart, occurrence/index state was moved inside the conversion loop.
- If an interrupted JSONL write replaces the destination, the temporary-file boundary moved.
- If a CLI imports `app.ingestion`, the supported adapter has regressed to compatibility ownership.
- If a fingerprint changes, compare the normalized hard-coded characterization string first.

## 14. Current limitations

- Root ingestion and content-identity facades remain until the R12 hard cut.
- The pipeline still accepts SDK-shaped `Any` values at the isolated Docling boundary.
- Actual source-manual parsing and indexing remain explicit integration operations.
- Retrieval composition still uses `app.retrieval` and `app.retrieval_runtime`; R11 owns that cleanup.
- Parsing, chunking, embedding, collection, and retrieval behavior are intentionally unchanged.

## 15. Self-check questions

1. Why does content identity belong to the domain?
2. Why is normalized-chunk mapping infrastructure-specific in this repository?
3. Which state must remain outside the per-page-batch loop?
4. How does atomic JSONL writing protect an existing complete file?
5. What proves that compatibility imports and canonical imports share identity?
6. Why must importing ingestion remain model- and SDK-lazy?

## 16. Interview summary

R10 turns two ambiguous root modules into explicit compatibility facades. The Docling pipeline,
atomic JSONL writer, stable domain records, and content fingerprint each have one owner. Supported
commands and runtime/evaluation consumers use those owners directly. Golden IDs, exact fingerprints,
failure tests, CLI tests, and the dependency matrix prove that the structural move did not change
corpus or runtime behavior.

## 17. Validation results and proposed commit

Validation receipts:

| Check | Result |
| --- | --- |
| Pre-change characterization, Python 3.11.15 | PASS — 120 tests |
| Post-change focused Ruff | PASS |
| Post-change focused pytest, Python 3.11.15 | PASS — 122 tests |
| Full Ruff | PASS |
| Full pytest, Python 3.11.15 | PASS — 407 tests, 1 warning |
| Docker Compose configuration | PASS |
| Protected data and artifacts | PASS — all 86 files unchanged |
| E2E source pins and held-out v2 worktree | PASS — unchanged |
| Local Markdown links and inline source paths | PASS |
| `git diff --check` and 14-file scope | PASS |

Proposed Conventional Commit after review:

```text
refactor: clarify ingestion and content identity ownership
```

## 18. Status

`COMPLETE`
