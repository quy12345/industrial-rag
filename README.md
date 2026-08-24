# Industrial Technical Manual RAG

Behavior-preserving modular monolith for answering questions from two Schneider Electric ATV320
technical manuals. The system combines dense and BM25 sparse retrieval, deterministic fusion,
multilingual reranking, an evidence gate, structured generation, and trusted citations.

## Current status

Round 1 portfolio cleanup is complete. The active product surface uses only the installation and
programming manuals, frozen as 2,753 chunks. The former single `manual.pdf` corpus and its tools are
historical archive material; their Qdrant collections are preserved but are not valid runtime
targets.

The cleanup changed ownership and dependency boundaries, not retrieval or answer behavior. It did
not re-index data, mutate collections, call a provider, download models, or rerun an evaluation.
Historical measurements and the engineering sequence live in
[PROJECT_JOURNEY](docs/PROJECT_JOURNEY.md) and the [walkthroughs](docs/walkthrough-phase-7.md).

Held-out v2 is an exposed regression benchmark. It must not be described as unseen or used for
tuning. Its historical sanitized result is provenance, not a current supported workflow.

## Runtime contract

| Item | Active value |
| --- | --- |
| Python | 3.11 |
| Retrieval profile | `phase7` |
| Corpus | ATV320 installation + programming manuals |
| Frozen chunks | 2,753 |
| Chunk-ID hash | `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d` |
| Dense collection | `industrial_manual_phase7_dense_v1` |
| Hybrid collection | `industrial_manual_phase7_hybrid_v1` |
| API | FastAPI on port 8000 |
| Demo UI | Streamlit on localhost port 8501 |

The immutable retrieval values are owned by `PHASE7_RETRIEVAL_CONTRACT` and
`PHASE7_CALIBRATION_FUSION_PROFILE`. Environment settings select that contract; partial profile
overrides are rejected.

## Architecture

```text
Browser
  -> Streamlit (HTTP only)
  -> FastAPI adapters
  -> application QueryService
  -> domain contracts and policies
  -> Qdrant / embedding / reranker / generation adapters

Evaluation
  -> public application/domain interfaces

Production runtime  -X->  evaluation or scripts
```

The query safety order is fixed:

```text
retrieve -> rerank -> select evidence -> evidence gate
         -> generate -> validate source IDs -> build trusted citations
```

- `app/api/`, `scripts.operations`/`scripts.evaluation`, and `ui/` are inbound adapters.
- `app/application/` coordinates use cases.
- `app/domain/` owns framework-free contracts and policies.
- `app/infrastructure/` owns external storage, model, corpus, and provider concerns.
- `evaluation/` owns offline schemas, replay, and metrics.
- `scripts/operations/` and `scripts/evaluation/` own supported CLI implementations.
- `scripts/archive/` preserves unsupported historical workflows.

See [CODEBASE](docs/CODEBASE.md), the
[target architecture](docs/portfolio-cleanup/02-target-architecture.md), and
[ADR-004](docs/adr/ADR-004-evaluation-and-script-lifecycle.md) for responsibility details.

## Installation

Create one Python 3.11 environment and install the extras needed for local development:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set exactly one generation provider in `.env`:

```dotenv
GENERATION_PROVIDER=openai
OPENAI_API_KEY=...
```

or:

```dotenv
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=...
```

Do not commit `.env`. Model weights are loaded lazily at runtime and are not baked into unit tests.

## Quickstart

The frozen Phase 7 collections must already exist before readiness or query execution. Starting
Compose does not re-index them.

```powershell
docker compose up -d qdrant api ui
docker compose ps
```

Open:

- API health: `http://localhost:8000/api/v1/health`
- API readiness: `http://localhost:8000/api/v1/ready`
- Streamlit: `http://localhost:8501`

The UI communicates only through FastAPI. A query can call the configured external generation
provider, so use it only after approving data egress for the selected provider.

Stop without deleting named volumes:

```powershell
docker compose down
```

Never add `-v` merely to restart the stack: the named Qdrant and model-cache volumes are part of the
preserved local state.

## Supported commands

Supported commands use their canonical ownership paths:

| Command | Purpose | External effects |
| --- | --- | --- |
| `python -m scripts.operations.audit_phase7_corpus --help` | Audit the two source manuals | Reads local PDFs when executed |
| `python -m scripts.operations.index_phase7_corpus --help` | Guarded Phase 7 indexing | Mutates Qdrant only when explicitly executed |
| `python -m scripts.operations.ingest_preview --help` | Preview document ingestion | Reads input and may write requested output |
| `python -m scripts.operations.query_smoke --help` | Bounded end-to-end query smoke | Uses Qdrant, models, and configured provider |
| `python -m scripts.operations.validate_query_runtime --help` | Read-only runtime validation | Uses Qdrant and local models; no provider |
| `python -m scripts.evaluation.validate_phase7_dataset --help` | Offline dataset validation | Provider-free and Qdrant-free |
| `python -m scripts.evaluation.evaluate_phase7_retrieval_closure --help` | Provider-free retrieval closure | Uses Qdrant and local reranker |
| `python -m scripts.evaluation.evaluate_phase7_e2e --help` | Pinned resumable E2E evaluator | Provider use requires explicit approval |

R13 removed all eight top-level command shims. Operational and evaluation commands now run directly
through their ownership packages. E2E provenance uses artifact schema v6 and source identity v2 over
canonical owners. See
[scripts/README.md](scripts/README.md) before using integration or archived tools.

Archived commands have no top-level compatibility path and are unsupported. They are retained for
provenance, not as recommended workflows.

## Validation

The default suite is offline: it uses fakes or in-memory Qdrant and must not call providers, download
models, or connect to production services.

```powershell
python -m ruff check .
python -m pytest -q
docker compose config --quiet
git diff --check
```

Round 1 closure passed 384 tests with one known third-party Starlette/TestClient deprecation
warning. Test counts are not a public contract; behavior coverage and a passing full suite are the
acceptance criteria.

Real Qdrant/model/provider checks are separate integration actions. Never infer that they passed
from the offline suite.

## Public contracts preserved by Round 1

- FastAPI paths, schemas, status/error mapping, request IDs, and optional bearer authentication.
- Stable document/chunk IDs, manifest identity, and deterministic Qdrant point IDs.
- Dense/sparse candidate limits, weighted RRF, reserves, reranking, and tie behavior.
- Candidate, reranked, selected-evidence, and generation-context boundaries.
- Evidence gate, abstention, source-ID validation, and trusted citation construction.
- Streamlit's HTTP-only boundary and no automatic POST retry.
- Frozen Phase 7 collections, corpus data, dataset schemas, artifact hashes, and source pins.
- Explicit failure behavior: there is no silent fallback to another profile or pipeline.

## Known limitations

- The Jina reranker is licensed CC-BY-NC-4.0 and is slow on CPU; commercial licensing and latency
  work are unresolved.
- OCR quality, complex tables, and multi-page continuity are not fully solved.
- Held-out v2 is exposed regression evidence, not an unseen benchmark and not a tuning source.
- Provider privacy approval is required before sending questions or manual evidence externally.
- Current authentication is optional bearer-token protection; rate limiting, TLS termination,
  multi-user isolation, and production deployment are outside this prototype.
- Round 2 may consider model/tuning/performance work, but none starts automatically after this
  documentation closure.

## Further reading

- [Round 1 index](docs/portfolio-cleanup/00-index.md)
- [Round 1 roadmap and completion record](docs/portfolio-cleanup/03-round-1-roadmap.md)
- [R07 module document](docs/modules/R07-evaluation-scripts-documentation.md)
- [Engineering journey](docs/PROJECT_JOURNEY.md)
- [Streamlit walkthrough](docs/walkthrough-streamlit-demo.md)
