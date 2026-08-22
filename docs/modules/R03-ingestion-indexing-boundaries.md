# R03 — Ingestion and indexing boundaries

## 1. Goal and scope

R03 separates stable document identity, document parsing, and Qdrant indexing infrastructure while
preserving the frozen Phase 7 corpus and index semantics. This document is shared by all R03 slices.

R03A and R03B are implemented. R03A moves `DocumentChunk` and stable identity policies into the
domain, moves Docling/PDFium access behind an infrastructure adapter, and retains `app.ingestion` as
a compatibility facade and ingestion coordinator. R03B separates Qdrant client construction, dense
embedding/indexing, and dense manifest I/O while retaining `app.retrieval` as a compatibility facade.
Hybrid and CLI restructuring remain future R03 slices.

## 2. Position in the system

```text
PDF / DOCX path
      ↓
app.ingestion compatibility facade and coordinator
      ├── app.domain.documents (record, identity, page-batch policy)
      └── app.infrastructure.ingestion.docling (Docling/PDFium adapter)
      ↓
list[DocumentChunk]
      ↓
app.retrieval compatibility facade
      ├── app.infrastructure.qdrant.client
      ├── app.infrastructure.qdrant.dense
      └── app.infrastructure.qdrant.manifests
      ↓
Qdrant points / dense manifest
```

R03A and R03B change dependency ownership only. Validation uses in-memory Qdrant; no production
collection operation or corpus indexing is executed.

## 3. Relevant background concepts

A domain record represents information meaningful to the application without depending on the SDK
that produced it. `DocumentChunk` therefore belongs with stable document identity rather than HTTP
response schemas.

An infrastructure adapter contains SDK-specific construction and error interpretation. Docling and
PDFium remain lazy imports so importing the application does not load document models or native PDF
libraries.

A compatibility facade preserves established imports while responsibility moves incrementally.
Callers may still use `app.ingestion` and `app.models.DocumentChunk`; both expose the canonical domain
objects by identity rather than copied implementations.

## 4. Input, output, and contracts

`build_document_id(Path)` returns the normalized filename slug plus the first 12 hexadecimal
characters of the file-content SHA-256 digest.

`build_chunk_id(document_id, page_numbers, headings, text, occurrence_index)` returns the existing
page-prefixed ID with a 16-character SHA-256 suffix. NFKC text/headings normalization, newline
normalization, sorted unique pages, and duplicate occurrence remain unchanged.

`ingest_document(...)` continues to accept PDF/DOCX paths, optional inclusive PDF page ranges,
optional PDF batch size, and the `hierarchical|hybrid` chunker selection. It returns chunks in
conversion order and rejects empty or incomplete results.

`DocumentChunk` retains the exact Pydantic fields and JSON representation:

```text
chunk_id, document_id, filename, text, page_numbers, headings, content_type, metadata
```

`index_chunks(...)` retains the same arguments and returns the number of upserted points. It embeds
all batches before inspecting or mutating a collection, upserts every new batch before deleting stale
document points, and preserves exact UUID5, named-vector, and payload schemas.

`write_index_manifest(...)` and `validate_index_manifest(...)` preserve the atomic JSON contract and
fail-closed mismatch messages. `create_qdrant_client(Settings)` preserves its reachability check and
sanitized `RetrievalError`.

## 5. Step-by-step data flow

1. `app.ingestion.validate_input_path` validates existence, file type, and extension.
2. PDF ranges are validated against `docling.get_pdf_page_count`; DOCX rejects range/batch options.
3. The canonical domain helper hashes the source filename/content into a document ID.
4. The coordinator calls the Docling adapter once per selected range.
5. The adapter validates `ConversionStatus` and rejects partial, failed, or unexpected results.
6. The coordinator extracts text, pages, headings, and conservative content type from raw chunks.
7. Document-wide occurrence counts distinguish identical normalized chunks without changing order.
8. The canonical domain helper creates each exact chunk ID.
9. Normalized `DocumentChunk` records retain global chunk indexes and stable source metadata.
10. Optional JSONL output remains an atomic temporary-file replacement in `app.ingestion`.
11. Dense indexing builds all embedding text and vectors before any Qdrant mutation.
12. Each chunk ID maps to the unchanged namespace UUID5 and exact citation-ready payload.
13. The adapter creates or validates the named cosine-vector collection.
14. Existing point IDs are read per document.
15. New batches are upserted with `wait=True`.
16. Stale points are deleted only after all upserts succeed.
17. Dense manifest writes use a same-directory temporary file followed by atomic replacement.
18. `app.retrieval` exposes the canonical adapter symbols while retaining dense search for R04.

## 6. Responsibilities of changed files

- [`app/domain/documents.py`](../../app/domain/documents.py) owns `DocumentChunk`, ingestion errors,
  supported document types, stable IDs, canonical keys, and pure page batching.
- [`app/infrastructure/ingestion/docling.py`](../../app/infrastructure/ingestion/docling.py) owns lazy
  Docling/PDFium imports, conversion, status validation, and SDK error normalization.
- [`app/ingestion.py`](../../app/ingestion.py) remains the compatibility facade and coordinates file
  validation, conversion ranges, normalization, occurrence tracking, and atomic JSONL output.
- [`app/models.py`](../../app/models.py) explicitly re-exports the canonical `DocumentChunk` while
  retaining existing API and retrieval models.
- [`tests/test_ingestion.py`](../../tests/test_ingestion.py) protects exact IDs, facade identity,
  import laziness, conversion behavior, normalization, ordering, and atomic output.
- [`tests/test_architecture_boundaries.py`](../../tests/test_architecture_boundaries.py) prevents
  domain imports from infrastructure/external SDKs and prevents Qdrant adapters from importing old
  compatibility facades.
- [`app/errors.py`](../../app/errors.py) owns the canonical `RetrievalError` shared by extracted
  infrastructure and compatibility callers.
- [`app/infrastructure/qdrant/client.py`](../../app/infrastructure/qdrant/client.py) owns Qdrant
  client construction and endpoint error sanitization.
- [`app/infrastructure/qdrant/dense.py`](../../app/infrastructure/qdrant/dense.py) owns dense model
  construction, embedding text, UUID5/payload mapping, collection validation, and safe indexing.
- [`app/infrastructure/qdrant/manifests.py`](../../app/infrastructure/qdrant/manifests.py) owns the
  atomic dense-index manifest contract.
- [`app/retrieval.py`](../../app/retrieval.py) remains a compatibility facade and retains dense search
  until the R04 retrieval boundary is implemented.
- [`tests/test_retrieval.py`](../../tests/test_retrieval.py) additionally protects exact dense UUID,
  payload/filter/manifest snapshots, mutation ordering, and facade identity.

Package `__init__.py` files mark the new infrastructure boundary without eager imports.

## 7. Important symbols and why they exist

- `DocumentChunk`: canonical cross-layer chunk record.
- `IngestionError`: stable failure type shared by the coordinator and adapter.
- `build_document_id`: persisted document identity algorithm.
- `build_chunk_id`: persisted chunk identity algorithm including duplicate occurrence.
- `canonical_chunk_key`: normalized identity input used by occurrence counting and hashing.
- `build_page_batches`: deterministic inclusive range policy.
- `convert_document`: concrete lazy Docling adapter operation.
- `validate_conversion_result`: fail-closed mapping from Docling status to `IngestionError`.
- `_convert_document` and `_validate_conversion_result` in `app.ingestion`: temporary compatibility
  aliases that preserve existing integration/test seams.
- `RetrievalError`: one canonical infrastructure failure type re-exported by `app.retrieval`.
- `create_qdrant_client`: explicit client adapter with no construction during module import.
- `build_embedding_text`: unchanged passage representation consumed by dense and hybrid embedding.
- `build_point_id`: persisted namespace UUID5 mapping from chunk IDs to Qdrant point IDs.
- `index_chunks`: dense indexing transaction ordering without rollback claims.
- `write_index_manifest` / `validate_index_manifest`: atomic persistence and fail-closed validation.

## 8. Before-and-after structure

Before R03A:

```text
app/models.py       HTTP/retrieval models + DocumentChunk
app/ingestion.py    identity + batching + Docling/PDFium + normalization + JSONL
```

After R03A and R03B:

```text
app/domain/documents.py                  chunk record + stable identity policies
app/infrastructure/ingestion/docling.py  Docling/PDFium adapter
app/ingestion.py                         compatible coordinator/facade + JSONL
app/models.py                            explicit compatibility export + remaining models
app/infrastructure/qdrant/client.py      Qdrant client construction
app/infrastructure/qdrant/dense.py       dense embedding/indexing infrastructure
app/infrastructure/qdrant/manifests.py   dense manifest persistence/validation
app/retrieval.py                         compatibility exports + dense search pending R04
```

## 9. Design decisions and trade-offs

- Stable algorithms were moved byte-for-byte in behavior; no ID migration is introduced.
- Pydantic remains the record implementation because JSON serialization is an established public
  contract. Replacing it with a dataclass would be unrelated churn.
- `app.ingestion` remains a facade so supported CLIs and archived Phase 6 tools do not break during
  the staged cleanup.
- Existing private aliases are retained because current offline tests and integrations monkeypatch
  the conversion seam. R03D or R07 may remove them only after consumers use an explicit port.
- Normalization stays in the coordinator for this slice because it maps SDK-shaped raw chunks into
  domain records. Introducing a larger parser abstraction is deferred until there is a concrete
  second adapter.
- Atomic JSONL remains in the facade until manifest/file infrastructure is addressed with indexing.
- Dense indexing helpers use public names inside infrastructure; old private names remain aliases in
  `app.retrieval` only because `app.hybrid_retrieval` still consumes them. R03C owns that cleanup.
- `RetrievalError` moves to `app.errors` so adapters do not import the compatibility facade or create
  a circular dependency. `app.retrieval.RetrievalError` is the same class object.
- Dense search deliberately stays in `app.retrieval`; moving its ranking path belongs to R04.
- The existing operation ordering is preserved, including its limitation: a partial multi-batch
  upsert can leave new points, but stale deletion never begins after an upsert failure.

## 10. Tests and protected behavior

| Test area | Protected behavior |
|---|---|
| exact document ID | ASCII slug and 12-character content digest |
| Unicode/duplicate golden IDs | NFKC/newline/page normalization and occurrence suffix |
| facade identity | old imports reference canonical domain objects |
| subprocess import check | no eager Docling or PDFium load |
| input/range/batch tests | supported files and inclusive page semantics |
| conversion status tests | partial/failure output is rejected with bounded details |
| normalization tests | text, pages, headings, type, global index, and order |
| JSONL tests | UTF-8 schema and atomic failure behavior |
| architecture boundary | domain has no infrastructure or external SDK dependency |
| exact dense snapshots | embedding text, UUID5, payload, filter, and manifest shape |
| dense safety tests | embed-before-mutate and upsert-before-stale-delete |
| collection tests | named cosine vector creation and strict compatibility validation |
| facade identity | old dense imports point to canonical client/index/manifest adapters |
| Qdrant boundary | extracted adapters do not import retrieval compatibility modules |

The pre-refactor characterization suite passed before production files moved.

## 11. Commands and expected results

Validation is the evidence that a structural refactor preserved the established contract:

- **Characterization tests** record exact pre-refactor outputs such as stable IDs, ordering, error
  mapping, and serialization. A changed output is reviewed instead of being accepted silently.
- **Focused tests** provide fast feedback for the responsibility being moved and make failures easy
  to localize.
- **Full tests** detect indirect regressions in consumers such as retrieval, evaluation, API, and
  scripts that still use `DocumentChunk` or the compatibility facade.
- **Ruff** catches invalid imports, undefined names, and structural hygiene issues that runtime tests
  may not execute.
- **Architecture tests** enforce dependency direction, including the rule that domain code cannot
  import infrastructure or external SDKs.
- **Import-laziness tests** prove that a normal application import does not load Docling/PDFium or
  trigger heavy external initialization.
- **Markdown-link and diff checks** ensure the learning document points to real files and the patch
  contains no whitespace damage.
- **Offline Python 3.11 execution** confirms behavior on the repository target interpreter while
  preventing accidental provider, model-download, or Qdrant access.

These checks reduce refactoring risk; they do not prove every possible runtime scenario. Their
confidence is limited to the contracts represented by the tests, which is why missing behavior is
characterized before risky code is moved.

Focused R03A checks:

```text
python -m ruff check app/domain app/infrastructure app/ingestion.py app/models.py tests/test_ingestion.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_ingestion.py tests/test_architecture_boundaries.py
```

Focused R03B checks:

```text
python -m ruff check app/errors.py app/retrieval.py app/infrastructure/qdrant tests/test_retrieval.py tests/test_architecture_boundaries.py
python -m pytest -q tests/test_retrieval.py tests/test_hybrid_retrieval.py tests/test_retrieval_runtime.py tests/test_bootstrap.py tests/test_health.py tests/test_architecture_boundaries.py
```

Slice-completion checks:

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

All tests must remain offline. These commands do not parse the source manuals, construct models,
connect to Qdrant, write points, or run evaluation.

## 12. Small usage example

Canonical domain imports:

```python
from app.domain.documents import DocumentChunk, build_chunk_id

chunk_id = build_chunk_id("manual", [2, 1], ["Safety"], "Disconnect power", 0)
chunk = DocumentChunk(
    chunk_id=chunk_id,
    document_id="manual",
    filename="manual.pdf",
    text="Disconnect power",
    page_numbers=[1, 2],
    headings=["Safety"],
)
```

Existing `from app.ingestion import build_chunk_id` and
`from app.models import DocumentChunk` imports remain valid and resolve to the same objects.

## 13. Common failures and debugging

- An exact-ID failure means canonical input encoding or normalization drifted; do not update the
  golden vector during a refactor.
- If import laziness fails, inspect the infrastructure module for module-level Docling/PDFium imports.
- If monkeypatched ingestion tests call the real SDK, verify the coordinator calls its local
  `_convert_document` compatibility alias.
- A different JSON field set usually means `DocumentChunk` was duplicated rather than re-exported.
- If batch chunk indexes restart at zero, occurrence/index state was moved inside the range loop.
- Domain boundary failures mean an adapter or SDK type leaked into `app/domain`.

## 14. Current limitations

- R03C has not isolated hybrid/sparse indexing or removed private dense-helper imports.
- R03D has not thinned supported ingestion/index CLI composition.
- `app.ingestion` still coordinates normalization and atomic JSONL for compatibility.
- Dense search remains in `app.retrieval` for R04; the module is therefore still a transitional
  facade rather than a finished adapter boundary.
- Dense multi-batch upsert has no rollback for already successful new batches; this existing behavior
  is documented, not changed during structural refactoring.
- Long-chunk/table behavior is intentionally unchanged and belongs outside Round 1.

## 15. Self-check questions

1. Which identity fields are normalized before a chunk ID is hashed?
2. Why is duplicate occurrence included in the stable ID?
3. Why does `app.models.DocumentChunk` remain available?
4. What guarantees that importing ingestion does not initialize Docling?
5. Why must stale deletion happen only after every dense upsert succeeds?
6. Which responsibilities remain for R03C and R03D?

## 16. Interview summary

R03A extracted persisted document identity and the chunk record into a framework-neutral domain
module, then placed Docling/PDFium behind a lazy infrastructure adapter. R03B separated dense
embedding, Qdrant indexing, client creation, and manifest I/O. Compatibility facades keep current
callers stable. Exact golden identities and payloads, failure-ordering tests, object-identity checks,
and offline in-memory Qdrant tests demonstrate that the moves change dependency direction rather
than persisted behavior.

## 17. Validation results and proposed commit

Current R03A/R03B validation results:

```text
Pre-refactor ingestion characterization             PASS — 25 tests
R03A focused ingestion/architecture suite           PASS — 30 tests
R03B pre-refactor dense characterization            PASS — 25 tests
R03B focused dense/hybrid/runtime suite              PASS — 56 tests, 1 warning
Focused Ruff                                        PASS
Full Ruff                                           PASS
Full pytest Python 3.11.15                          PASS — 347 tests, 1 warning
Markdown links (12 local targets) / git diff check  PASS
```

Proposed commit after user review:

```text
refactor: separate document ingestion from Docling
```

## 18. Status

`IN_PROGRESS` — R03A and R03B are implemented and their focused/full validation passes. R03C–R03D
are intentionally not implemented in this slice.
