# B01 — Reproducible Phase 7 ingestion

## 1. Goal and scope

B01 makes a clean ingestion-image rebuild reproduce the frozen Phase 7 parsing profile. It pins the
Docling, PDF parsing, and tokenization stack that created the accepted corpus plus the ONNX runtime
recorded by frozen evaluation, aligns the supported CLI default and Compose input path with that
profile, makes slow dependency downloads resumable, and prevents a known Docling warning from
echoing manual contents.

It does not rebuild an image, parse the PDFs, replace frozen artifacts, re-index Qdrant, or change
retrieval behavior. Those expensive or mutating operations remain manual and approval-gated.

## 2. Position in the system

```text
two local ATV320 PDFs
        ↓
Docling conversion adapter
        ↓
HybridChunker in 64-page batches
        ↓
normalized DocumentChunk records
        ↓
preview JSONL + corpus manifest
        ↓
separately guarded Qdrant indexing
```

B01 stops at the preview/manifest boundary during automated validation.

## 3. Relevant background concepts

- A **transitive dependency** is installed because another package requires it. A compatible version
  range can still select a newer transitive release on a later clean build.
- **Corpus identity** is the document IDs, chunk count, and SHA-256 of the sorted chunk IDs. It is
  sensitive to parsing and chunking boundaries even when the source PDFs are unchanged.
- A **reproduction profile** includes tool versions and operational parameters, not only source code.
- A **BuildKit cache mount** persists pip's validated HTTP/download cache outside an image layer, so
  a failed build can resume without adding cache files to the final image or repository.
- **Warning sanitization** keeps a useful diagnostic while removing the embedded manual text. It does
  not suppress unrelated warnings or alter chunk values.

## 4. Input, output, and contracts

The accepted input contract is:

- `data/raw/ATV320_Installation_manual_EN_NVE41289_09.pdf`;
- `data/raw/ATV320_Programming_Manual_EN_NVE41295_06.pdf`;
- Docling `2.117.0`, docling-core `2.90.0`, docling-ibm-models `3.13.3`, and
  docling-parse `7.10.0`;
- Hugging Face Hub `1.26.0`, Transformers `5.14.1`, Tokenizers `0.22.2`, and semchunk `3.2.5`;
- pypdfium2 `5.12.1`;
- ONNX Runtime `1.28.0` for the FastEmbed/Jina execution environment;
- `HybridChunker` with page batch size `64` and OCR disabled.

The frozen output contract remains:

- `2753` chunks;
- document IDs unchanged;
- chunk-ID hash
  `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.

No API, retrieval, evidence, citation, provider, artifact schema, or collection contract changes.

## 5. Step-by-step data flow

1. The retrieval base bootstraps pip `26.2.1` with ten network retries, a 120-second socket timeout,
   and a persistent BuildKit cache.
2. Once upgraded, both large dependency steps additionally allow ten resume attempts. All three
   installs share the cache while retaining package hash verification.
3. The ingestion image installs the exact baseline parsing, tokenization, and ONNX Runtime versions
   from `pyproject.toml`.
4. Compose mounts the host `data/raw` directory at `/app/data/raw`, matching the container working
   directory and the CLI's relative default inputs.
5. The CLI resolves the two default PDF paths and selects batch size `64` plus hybrid chunking.
6. The pipeline calculates deterministic document identity and page batches.
7. The lazy Docling adapter converts and chunks each page batch.
8. Known oversized-heading warnings are counted; their embedded body text is discarded from output.
9. Other warning categories are re-emitted unchanged.
10. Normalization and stable-ID functions receive the same Docling chunks as before.
11. Preview output is written only to the caller-selected path.

## 6. Responsibilities of changed files

| Path | B01 responsibility |
| --- | --- |
| [`pyproject.toml`](../../pyproject.toml) | Exact ingestion and inference dependency versions. |
| [`Dockerfile`](../../Dockerfile) | Resumable pip version and shared BuildKit download cache for ingestion builds. |
| [`docker-compose.yml`](../../docker-compose.yml) | Runtime-only read-only PDF mount matching CLI defaults. |
| [`scripts/operations/index_phase7_corpus.py`](../../scripts/operations/index_phase7_corpus.py) | Supported Phase 7 profile default. |
| [`app/infrastructure/ingestion/docling.py`](../../app/infrastructure/ingestion/docling.py) | Lazy Docling conversion and safe SDK-warning boundary. |
| [`tests/test_phase7_index_cli.py`](../../tests/test_phase7_index_cli.py) | Dependency, mount, parser, and indexing safety contracts. |
| [`tests/test_ingestion.py`](../../tests/test_ingestion.py) | Stable ingestion behavior and warning sanitization. |

No disposable repository script or comparison artifact is introduced.

## 7. Important symbols and why they exist

- `PHASE7_PAGE_BATCH_SIZE` names the accepted batch-size contract instead of hiding it in argparse.
- The exact `pip==26.2.1` build step supplies resume behavior for slow dependency downloads.
- `industrial-rag-pip` names the BuildKit cache shared by pip bootstrap, retrieval, and ingestion
  install steps.
- `_OVERSIZED_METADATA_WARNING_PREFIX` identifies the exact Docling diagnostic known to embed manual
  content.
- `_chunk_document` materializes the same chunk iterator while reducing repeated sensitive warnings
  to one count-only warning.
- `DEFAULT_INPUTS` stays relative to `/app`, so the same paths work locally and inside Compose.

## 8. Before-and-after structure

```text
Before
  docling>=2.117,<2.118 -> docling-core could drift
  FastEmbed range        -> onnxruntime could drift
  pip 24 + no cache      -> truncated wheel lost the whole dependency step
  CLI default batch 16  != frozen profile batch 64
  /data/raw mount       != /app/data/raw default path
  Docling warning       -> full chunk text in terminal

After
  baseline Docling + PDF parser + tokenizer stack pinned from build provenance
  onnxruntime==1.28.0
  pip 26.2.1 + retries/resume + shared BuildKit cache + hash verification
  CLI default batch 64
  /app/data/raw read-only mount
  one count-only warning; chunks unchanged
```

## 9. Design decisions and trade-offs

Only versions recovered from frozen ingestion/evaluation evidence and the retained BuildKit history
for the baseline image are added as direct pins. Build `qre4mri8ev8zjrtz4uekklyhv` at revision
`43bf9768ea30cc5204fd3fdc00dac3cfde23d161` records the exact parser and tokenizer versions used
immediately before the accepted corpus was generated. A repository-wide lockfile would be a larger
dependency-management change and is not required to fix the confirmed identity drift. The active
frozen corpus remains authoritative because a live provider-free A/B run found no retrieval-quality
gain from the docling-core `2.92.0` chunks. ONNX Runtime `1.28.0` is also recorded in the frozen
calibration, held-out, snapshot, and CPU benchmark artifacts; allowing FastEmbed to resolve the newly
released `1.29.0` caused a separate wheel-integrity failure.

The first B01 reproduction with only Docling and docling-core pinned produced the correct count but
replaced nine stable IDs. The rebuilt image had drifted to docling-ibm-models `3.14.0`, docling-parse
`7.16.0`, pypdfium2 `5.13.0`, Transformers `5.16.1`, Tokenizers `0.23.1`, and Hugging Face Hub
`1.28.0`. The retained baseline build records `3.13.3`, `7.10.0`, `5.12.1`, `5.14.1`, `0.22.2`, and
`1.26.0` respectively, so B01 pins those evidence-backed values instead of copying versions from the
host virtual environment.

The warning boundary is narrow: only the exact oversized metadata warning is sanitized. Token-length
and other diagnostics remain visible because hiding them could conceal a real ingestion problem.

The pip cache is a builder cache, not an image layer or repository artifact. Hash mismatches still
fail the build; B01 improves resume/retry behavior instead of trusting incomplete bytes.

## 10. Tests and the behavior each test protects

| Test | Protected behavior |
| --- | --- |
| `test_indexing_parser_preserves_supported_contract` | Default inputs, batch `64`, hybrid mode, and safety flags. |
| `test_runtime_dependencies_preserve_frozen_versions` | Exact parser, tokenizer, chunker, and ONNX runtime versions in their supported extras. |
| `test_ingestion_compose_mount_matches_default_input_paths` | Container defaults can see the two PDFs without positional overrides. |
| `test_ingestion_build_uses_resumable_pip_and_shared_cache` | Pinned modern pip, explicit retry budgets, three shared cached install steps, and no old no-cache retrieval installs. |
| `test_docling_chunk_warnings_do_not_echo_manual_content` | Chunk list unchanged, known raw text removed, unrelated diagnostic retained. |
| Existing ingestion tests | Stable IDs, batching, normalization, status failures, lazy imports, and atomic JSONL. |
| Existing indexing tests | Protected targets, mutation ordering, and per-document verification. |

All automated tests use fakes or local files; they do not parse the manuals or access Qdrant.

## 11. Commands and expected results

Lightweight validation run by the agent:

```powershell
python -m ruff check .
python -m pytest -q --basetemp=.pytest-tmp-b01
docker compose config --quiet
git diff --check
```

The reproduction acceptance check is intentionally left to the user because the ingestion image
build and Docling conversion are slow:

```powershell
$chunkCheckDir = Join-Path ([System.IO.Path]::GetTempPath()) "industrial-rag-b01-check"
New-Item -ItemType Directory -Path $chunkCheckDir -Force | Out-Null

docker compose --progress=plain build ingestion
docker run --rm --entrypoint python industrial-rag-ingestion:local -c `
  "from importlib.metadata import version; names=('docling','docling-core','docling-ibm-models','docling-parse','pypdfium2','transformers','tokenizers','huggingface-hub','semchunk','onnxruntime'); print(' | '.join(f'{name}={version(name)}' for name in names))"
docker compose run --rm --no-deps `
  --volume "${chunkCheckDir}:/tmp/chunk-check" `
  ingestion python -m scripts.operations.index_phase7_corpus `
  --preview-only `
  --chunks-output /tmp/chunk-check/frozen-chunks.jsonl `
  --manifest-output /tmp/chunk-check/manifest.json
```

This command relies on the canonical defaults: two inputs, batch `64`, and hybrid chunking. It writes
outside the repository's frozen artifacts and performs no Qdrant write.

## 12. Small usage example

After the manual run, compare only the stable contract fields:

```powershell
$baseline = Get-Content artifacts/metrics/phase-7-corpus-manifest.json -Raw | ConvertFrom-Json
$fresh = Get-Content (Join-Path $chunkCheckDir "manifest.json") -Raw | ConvertFrom-Json

[pscustomobject]@{
  SameCount = $baseline.total_chunk_count -eq $fresh.total_chunk_count
  SameChunkIdHash = $baseline.chunk_set.chunk_ids_sha256 -eq $fresh.chunk_set.chunk_ids_sha256
  FreshDocling = $fresh.runtime_versions.docling
  FreshDoclingCore = $fresh.runtime_versions.'docling-core'
}
```

Expected acceptance result: both booleans are `True`; the package gate reports the exact versions
listed in section 4 and ONNX Runtime `1.28.0`.

## 13. Common failures and debugging

- `Input document does not exist`: inspect `docker compose config`; the ingestion mount target must
  be `/app/data/raw`.
- `invalid spec: :/tmp/chunk-check`: the PowerShell variable is empty in the current shell; rerun its
  assignment before the Compose command.
- Fresh docling-core reports `2.92.0`: the old ingestion image was reused; rebuild the ingestion
  target after changing `pyproject.toml`.
- pip reports expected hash `85f8e840...` for ONNX Runtime `1.29.0`: the resolver is using the stale
  unpinned dependency graph; rebuild after the B01 ONNX `1.28.0` pin is present. Do not disable pip's
  hash verification.
- pip reports expected hash `89cd4683...` for NumPy `2.4.6`: the wheel download was truncated (the
  log stopped at `15.2/16.9 MB`). Use the B01 pip `26.2.1`/BuildKit-cache Dockerfile and rerun the
  normal build; do not add the received hash or disable verification.
- Bootstrap pip stops with `ReadTimeoutError`: this happens before resumable pip is installed. The
  bootstrap step must itself use the shared cache, ten standard retries, and the 120-second timeout.
- Count/hash still differs: stop before indexing. Preserve both manifests and inspect installed
  versions, source hashes, batch size, and chunker.
- Count is `2753` but hash is `fc9d9112...`: only the top-level Docling packages were pinned. The
  retained baseline build proves that six parser/tokenizer dependencies also drifted; rebuild after
  the complete B01 ingestion profile is present in `pyproject.toml`.
- `[transformers] Token indices sequence length ...`: `HybridChunker` first calls the Hugging Face
  tokenizer to count an unsplit hierarchical block, then compares that count with its chunk budget
  and passes oversized text to semchunk. The full sequence is not sent through the Transformer model
  on this code path, so the logger's generic "indexing errors" suffix does not mean this preview
  failed or that the produced chunk was corrupted. Keep this content-free diagnostic visible; stop
  only if conversion fails or a separate post-chunk check proves an emitted chunk exceeds its budget.

## 14. Current limitations

- B01 pins the packages on the parsing/chunking path but does not introduce a complete
  cross-platform lockfile for unrelated application dependencies.
- The current ingestion image is intentionally not size-optimized. Local inspection reports
  `9.67 GB` in `docker image ls`, a `5.76 GB` ingestion dependency layer, and `5.7 GB` of installed
  Python packages. CUDA/NVIDIA libraries (`2.724 GB`), Torch (`1.131 GB`), and Triton (`691 MB`)
  dominate that footprint even though the Compose container reports `cuda_available=False`.
- Conversion is also not lifecycle-optimized. The two manuals contain 654 pages, which become 12
  conversion ranges at the frozen 64-page batch size. The current loop creates a new
  `DocumentConverter` and `HybridChunker` for every range, repeatedly loading model and tokenizer
  state.
- CPU-only Torch, targeted `docling-slim` extras, persistent model caches, dependency-layer
  reordering, and converter/chunker reuse are candidates for a separate optimization module. They
  are excluded from B01 because dependency binaries and converter lifecycle can affect parsed output;
  each candidate must preserve the frozen chunk count and ID hash before adoption.
- The first cache-enabled build must still download each dependency once; later retries reuse valid
  responses. Docker builder pruning removes this external cache.
- Exact reproduction still depends on the same PDFs, Python/container platform, and Docling runtime.
- The expensive real conversion is not part of pytest or automated agent validation.
- Existing collections and frozen artifacts are not migrated or overwritten.

## 15. Self-check questions

1. Why can an unchanged direct Docling version still produce different chunks?
2. Which dependency versions and operational values define the accepted ingestion profile?
3. Why does Compose mount raw PDFs at `/app/data/raw`?
4. Which warnings are sanitized and which remain visible?
5. Why must a failed reproduction stop before Qdrant indexing?

## 16. Interview summary

B01 fixes reproducibility defects at the infrastructure boundary. A clean image selected compatible
but newer parser/tokenizer packages and changed nine stable chunk IDs, then another build resolved a
new ONNX Runtime wheel outside the frozen evaluation environment. Slow builds also received truncated
wheels. The fix pins the evidence-backed baseline stack, uses pip's resumable downloads with an
external BuildKit cache, makes CLI and Compose defaults agree with the frozen profile, and sanitizes a
verbose SDK warning without changing chunks. Offline tests protect these contracts; the expensive
end-to-end conversion remains an explicit manual acceptance step.

## 17. Validation results and proposed commit

| Check | Result |
| --- | --- |
| Pre-change focused pytest | PASS — 33 tests using a workspace basetemp |
| Post-change focused pytest | PASS — 36 tests |
| First user ingestion build | FAILED safely before Docling — downloaded ONNX Runtime `1.29.0` wheel did not match PyPI hash |
| Frozen runtime evidence | PASS — evaluation artifacts record ONNX Runtime `1.28.0` |
| Second user ingestion build | FAILED safely — ONNX pin selected `1.28.0`; NumPy `2.4.6` stopped at `15.2/16.9 MB` and failed its official hash |
| Third user ingestion build | FAILED before dependency install — pip bootstrap timed out at `0.1/1.8 MB`; no file/cache lock was present |
| File/cache lock diagnosis | PASS — source files allow exclusive open and BuildKit reports no running build |
| Fourth user ingestion build | PASS — image contains Docling `2.117.0`, docling-core `2.90.0`, and ONNX Runtime `1.28.0` |
| First complete user preview | PARTIAL — `2753` chunks, but hash `fc9d9112...`; nine IDs replaced |
| Frozen image provenance | PASS — retained build history identifies all six drifted parser/tokenizer dependencies |
| Post-provenance focused Ruff/pytest | PASS — 37 tests |
| Post-provenance full Ruff/pytest | PASS — 386 tests, 1 unrelated dependency warning |
| Post-provenance TOML, `pip check`, Compose, and diff checks | PASS |
| Post build-hardening focused pytest | PASS — 37 tests |
| Focused Ruff | PASS |
| Full Ruff | PASS |
| Full pytest | PASS — 386 tests, 1 unrelated dependency warning |
| Local package metadata | PASS — Docling `2.117.0`, docling-core `2.90.0`, ONNX Runtime `1.28.0` |
| Local dependency consistency (`pip check`) | PASS |
| Docker Compose configuration | PASS — exit `0`; sandbox could not read the user Docker config |
| Final user-run ingestion rebuild/chunk reproduction | PASS — `2753` chunks and frozen hash `2a972de9...` reproduced exactly |
| Diff, protected data/artifact scope, and six local links | PASS |

Proposed Conventional Commit after acceptance:

```text
fix: make phase7 ingestion reproducible
```

## 18. Status

`COMPLETE` — offline checks pass and the final user-run clean rebuild reproduces both the frozen
`2753` chunk count and complete chunk-ID hash.
