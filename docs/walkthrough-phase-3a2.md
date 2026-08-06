# Phase 3A.2 walkthrough

This guide reproduces the dense-baseline and Docker-stabilization checks without deleting the
Qdrant volume or using a broad Docker prune command.

## What Phase 3A.2 closes

- Direct-evidence retrieval evaluation uses stable `relevant_chunk_ids`, never page-only matches.
- The retrieval development set has 30 factual queries: 15 Vietnamese and 15 English.
- The frozen evaluation chunk set is `artifacts/manual-batched.jsonl` with 99 chunks and ID hash
  `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`.
- The API Docker target has FastEmbed/Qdrant retrieval dependencies but no Docling.
- The Docling ingestion target is profile-gated and shares a FastEmbed model-cache volume with API.

This is a **retrieval development set**, not the held-out Phase 6 evaluation suite. Do not tune a
model and report it as a final result on this same set.

## 1. Local environment

Use CPython 3.11 when available. Confirm every command resolves to the same interpreter:

```powershell
python --version
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m pytest
```

The committed CI workflow uses Python 3.11. If the existing `.venv` has another Python version,
leave it untouched and create a separate `.venv311` with a known CPython 3.11 executable.

## 2. Validate qrels without external services

This check reads the dataset and frozen artifact, confirms every direct-evidence ID exists, and
checks each expected phrase/page diagnostic against its qrel chunk.

```powershell
@'
from pathlib import Path
from app.evaluation import load_evaluation_cases, load_frozen_chunks, validate_cases_against_chunks

cases = load_evaluation_cases(Path("data/eval/dense_smoke.jsonl"))
chunks = load_frozen_chunks(Path("artifacts/manual-batched.jsonl"))
validate_cases_against_chunks(cases, chunks)
print(f"qrels={len(cases)}, frozen_chunks={len(chunks)}")
'@ | python -
```

Expected result: `qrels=30, frozen_chunks=99`.

## 3. Start Qdrant and verify Compose

```powershell
docker compose config
docker compose --profile tools config
docker compose up -d qdrant
curl.exe http://localhost:6333/
```

The returned Qdrant version must be `1.18.3`. The `qdrant_storage` named volume persists the
existing collection and is never removed by the project commands.

## 4. Re-index the frozen manual safely

Use the known memory-safe page batching profile:

```powershell
python scripts/index_document.py data/raw/manual.pdf `
  --page-start 1 `
  --page-end 21 `
  --page-batch-size 4
```

Expected output includes:

```text
Document ID: manual-77d5dae4c2c5
Chunks indexed: 99
Vector name: dense
Embedding dimension: 384
```

Run the same command a second time. The collection count must remain 99; re-indexing upserts the
new stable point IDs first and only then removes stale IDs for this document.

## 5. Run the direct-evidence baseline

```powershell
python scripts/evaluate.py `
  --chunks artifacts/manual-batched.jsonl `
  --limit 20 `
  --output artifacts/metrics/dense-baseline.json
```

The evaluator first rejects a mismatched manifest, a qrel missing from the frozen set, or an index
whose document chunk IDs differ from the frozen 99 IDs. One warmup query is excluded from latency.
The reported warm latency includes query embedding, Qdrant round trip, and result mapping.

The 2026-08-05 run produced:

| Metric | Result |
|---|---:|
| Hit@1 | 0.167 |
| Hit@3 | 0.367 |
| Hit@5 | 0.400 |
| MRR@5 | 0.269 |
| MRR@20 | 0.298 |
| Average latency | 22.45 ms |
| p50 latency | 15.14 ms |
| p95 latency | 37.77 ms |
| Failure cases | 18 / 30 |

All six bilingual critical labels are currently outside top 5 except `dense_002`, which ranks 2.
This is the deliberately frozen dense baseline for Phase 4; it is not a reason to alter the model,
chunk set, or qrels in Phase 3A.2.

## 6. Build and run the API image

```powershell
docker compose build api --progress plain
docker run --rm industrial-rag-api:local `
  python -c "import importlib.util; assert importlib.util.find_spec('docling') is None"
docker compose up -d api
curl.exe http://localhost:8000/api/v1/health
docker image ls industrial-rag-api:local
```

The checked API image was 544 MB according to `docker image ls`, contains no Docling, and returned:

```json
{"status":"ok","service":"industrial-rag","version":"0.1.0"}
```

Run `docker compose build api` again without source/dependency changes. BuildKit should report the
application layers as `CACHED`; `docker compose up -d api` then reuses `industrial-rag-api:local`
instead of creating a new image.

## 7. Use ingestion on demand

```powershell
docker compose --profile tools build ingestion --progress plain
docker compose --profile tools run --rm ingestion `
  python -c "import docling; print(docling.__version__)"
docker compose --profile tools run --rm ingestion `
  python scripts/index_document.py /data/raw/manual.pdf `
  --page-start 1 --page-end 21 --page-batch-size 4
```

`ingestion` is not started by `docker compose up -d qdrant api`. It mounts `data/raw` read-only,
mounts `artifacts` for generated reports, and shares `fastembed_cache` at `/models/fastembed`.
Model weights are downloaded at runtime into that volume, never during `docker build`.

The ingestion target also installs `libxcb1`, `libgl1`, and `libglib2.0-0t64`. These Debian runtime
libraries provide `libxcb.so.1`, `libGL.so.1`, and `libgthread-2.0.so.0`, which Docling's PDF/image
pipeline needs but `python:3.11-slim` does not include.

## 8. Docker storage and safe diagnosis

Inspect storage before taking any cleanup action:

```powershell
docker system df
docker buildx du
docker image ls --all
```

The 2026-08-05 inspection found a 544 MB API image, a 265 MB Qdrant image, a 482.6 MB persistent
Qdrant volume, and 12.83 GB reclaimable BuildKit cache from older/partial builds. Build cache is
separate from the API image and is the main cause of disk pressure.

Do not run `docker system prune`, `docker volume prune`, or any command that removes
`industrial-rag_qdrant_storage`. If space must be reclaimed, inspect the cache first and use a
time-bounded `docker builder prune` only after confirming no active build is using it.

## 9. Known validation status

- API build, no-Docling import check, API health, Qdrant 1.18.3, two safe re-indexes, baseline
  evaluation, ingestion image build, and containerized batch-4 ingestion all passed.
- The checked ingestion image is 9.57 GB. Running it with `/data/raw/manual.pdf` produced 99 chunks
  and left the Qdrant collection at 99 points.
- The host `.venv` used for the recorded run is Python 3.13.5; host CPython 3.11 validation remains
  pending until a usable 3.11 interpreter is installed or selected.
