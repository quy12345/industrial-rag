# Phase 7 corpus walkthrough — ATV320 source audit and freeze

Status: **in progress; no Phase 7 qrels or benchmark metrics are approved yet.**

Phase 7 replaces neither frozen development collection. It uses two separate Schneider Electric
ATV320 manuals and separate collections:

```text
industrial_manual_phase7_dense_v1
industrial_manual_phase7_hybrid_v1
```

The protected Phase 3–6 collections remain `industrial_manual_chunks` and
`industrial_manual_chunks_v2`. The Phase 7 index CLI rejects either protected name before it can
create or write a collection.

## Source provenance

| Role | Reference | Version | Pages | SHA-256 |
|---|---|---|---:|---|
| Installation | NVE41289.09 | 04/2025 | 194 | `c181b4d7…42eaebab` |
| Programming | NVE41295.06 | 04/2025 | 460 | `f5e9bb48…71d6d7d5` |

The full metadata and official Schneider URLs are in [data/sources.yaml](../data/sources.yaml).
The PDFs are intentionally local-only and ignored by Git because their redistribution rights have
not been established. `retrieved_at` is recorded transparently as unknown for the pre-existing local
files; the audit date is 2026-08-07.

Both files are unencrypted PDF 1.7 files with a usable digital text layer. OCR is disabled. They
contain heading hierarchies, warning content, wiring and terminal identifiers, and parameter/fault
tables. Parsing risks retained from earlier phases are page-batch heading discontinuity, repeated
headers/footers, and multi-page tables splitting across batches.

## Commands

Run technical metadata/text-layer audit without copying vendor text into an artifact:

```powershell
docker compose --profile tools run --rm --no-deps -v "${PWD}:/workspace" -w /workspace ingestion `
  python -m scripts.audit_phase7_corpus
```

Preview/freeze chunks first. The selected candidate profile is 64 pages per Docling batch; it is a
parsing-resource choice, not a retrieval-tuning parameter:

```powershell
docker compose --profile tools run --rm --no-deps -v "${PWD}:/workspace" -w /workspace ingestion `
  python -m scripts.index_phase7_corpus --preview-only --page-batch-size 64
```

After human review of the preview, index only the new collections and verify a deterministic second
indexing pass:

```powershell
docker compose up -d qdrant
docker compose --profile tools run --rm --no-deps -v "${PWD}:/workspace" -w /workspace ingestion `
  python -m scripts.index_phase7_corpus --page-batch-size 64 --verify-reindex
```

This writes ignored runtime evidence only:

```text
artifacts/phase7/frozen-chunks.jsonl
artifacts/metrics/phase-7-corpus-audit.json
artifacts/metrics/phase-7-corpus-manifest.json
```

The manifest stores source hashes, document IDs, chunk count, sorted chunk-ID hash, ingestion
profile, model/index contract, collection names, versions, and creation commit. It contains no raw
manual text or vectors.
