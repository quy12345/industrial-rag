# Industrial Technical Manual RAG

A portfolio-ready retrieval-augmented generation system for two Schneider Electric ATV320 manuals.
It parses technical PDFs with Docling, indexes stable chunks in Qdrant, combines dense and sparse
retrieval, reranks candidates, gates weak evidence, and returns answers with validated citations.

## What works

- Two-manual ingestion with deterministic document and chunk identities.
- Dense and BM25 retrieval, rank-only fusion, cross-encoder reranking, and evidence selection.
- FastAPI query and health endpoints with explicit error mapping and safe abstention.
- Provider adapters behind one grounded-generation contract.
- A Streamlit demo that communicates with the system only through FastAPI.
- Offline unit and characterization tests plus versioned evaluation artifacts.

## Architecture

```text
FastAPI / CLI / Streamlit
          |
          v
  Application services
          |
          v
 Domain contracts and policies
          ^
          |
Infrastructure adapters

Evaluation ---> public application/domain interfaces
Production -X-> evaluation or scripts
```

The production modular monolith lives in `app/`. Offline benchmark contracts live in
`evaluation/`; supported command adapters live in `scripts/operations/` and
`scripts/evaluation/`. See [Architecture](docs/ARCHITECTURE.md) for ownership and runtime flows.

## Quickstart

Python 3.11 and Docker Compose are required. Copy the example settings, add a provider key only if
you intend to generate answers, and start the current index-backed demo:

```powershell
Copy-Item .env.example .env
docker compose up -d qdrant api ui
```

- API documentation: <http://localhost:8000/docs>
- API liveness: <http://localhost:8000/api/v1/health>
- API readiness: <http://localhost:8000/api/v1/ready>
- Streamlit: <http://localhost:8501>

The API does not silently create or replace collections. Follow
[Operations](docs/OPERATIONS.md) before running ingestion, integration checks, or evaluation.

## Supported commands

```text
python -m scripts.operations.audit_corpus
python -m scripts.operations.index_corpus
python -m scripts.operations.ingest_preview
python -m scripts.operations.query_smoke
python -m scripts.operations.validate_query_runtime
python -m scripts.evaluation.validate_dataset
python -m scripts.evaluation.evaluate_retrieval
python -m scripts.evaluation.evaluate_e2e
```

Use `--help` before an operational or evaluation command. Some commands initialize local models,
access Qdrant, write artifacts, or call a provider; their side effects and approval gates are listed
in [scripts/README.md](scripts/README.md).

## Offline validation

These repository checks do not call a provider, download a model, or write Qdrant:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
docker compose config --quiet
git diff --check
```

## Evaluation snapshot

The latest sealed held-out-v2 regression run evaluated all 45 items against the unchanged frozen
corpus. Generation used `gemini-3.5-flash-lite`; semantic evaluation used Ragas `0.4.3` with
`openai/gpt-5.6-luna` through OpenRouter. Abstained responses were excluded from semantic averages
instead of being scored as zero.

| Evaluation signal | Result |
| --- | ---: |
| Candidate recall over 30 answerable items | 0.9000 |
| Deterministic fact accuracy over 27 answered items | 0.7778 |
| Abstention precision / recall | 0.8333 / 1.0000 |
| Referentially valid citation IDs | 1.0000 |
| Wrong-document citations | 2 |
| Ragas Faithfulness | 1.0000 |
| Ragas Answer Relevancy | 0.8204 |
| Ragas Context Precision without reference | 0.9753 |
| Semantic eligible / excluded / failed | 27 / 18 / 0 |

The semantic run completed successfully, but the existing deterministic quality gate remained
`FAIL` because candidate recall, fact accuracy, abstention precision, and wrong-document citation
requirements were not all met. This distinction is intentional: Ragas measures whether an answer is
supported by the supplied contexts and relevant to the question; it does not replace corpus qrels,
expected facts, citation-document checks, or abstention tests. The sanitized local v8 artifact has
SHA-256 `c975ef742c44bc73a2986ed622b3d26d7054b8d900214a955b5ce461a3bb260f`.

See [Operations](docs/OPERATIONS.md) for the reproducible command and
[Project Journey](docs/PROJECT_JOURNEY.md) for interpretation. Held-out v2 is exposed regression
evidence and must not be used for further tuning.

## Frozen corpus identity

The active product contract contains 2,753 chunks from the two ATV320 PDFs. The ordered chunk-ID
hash is:

```text
2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d
```

The Qdrant collection names still contain `phase7` because they are immutable physical storage
identities. That historical label is not a runtime profile selector and must not be renamed without
an explicit collection migration.

## Limitations

- The current Jina reranker is slow on CPU and has non-commercial licensing constraints.
- OCR, multi-page table continuity, and calibrated semantic support remain open engineering work.
- Evaluation results are regression evidence for this corpus, not general production-quality claims.
- The exposed held-out v2 split must not be used for further tuning.

For the development history and preserved bug-to-fix evidence, read
[Project Journey](docs/PROJECT_JOURNEY.md).
