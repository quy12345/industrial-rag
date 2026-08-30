# Operations

This is the canonical runbook for local Docker, supported commands, ingestion reproduction, and
troubleshooting. Architecture details belong in [ARCHITECTURE.md](ARCHITECTURE.md).

## Safe local setup

```powershell
Copy-Item .env.example .env
docker compose config --quiet
docker compose up -d qdrant api ui
```

The existing named Qdrant volume and all four known collections must be preserved. Do not use
`docker compose down -v`, collection recreation, or cleanup commands that remove named volumes.

Useful read-only checks:

```powershell
docker compose ps
docker compose logs api --tail 100
Invoke-RestMethod http://localhost:8000/api/v1/health
Invoke-RestMethod http://localhost:8000/api/v1/ready
```

## Offline repository validation

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
docker compose config --quiet
git diff --check
```

The test suite uses fake providers, fake models, and in-memory Qdrant. Passing it is not evidence that
a local model cache, real provider credential, or external Qdrant instance is available.

## Supported command inventory

Always inspect `--help` first. Commands are adapters; reusable logic is owned by `app/` or
`evaluation/`.

| Command | Purpose | Important side effects |
|---|---|---|
| `scripts.operations.audit_corpus` | Compare active corpus identity with its contract | Reads local artifacts/files |
| `scripts.operations.index_corpus` | Preview or explicitly index the two PDFs | Docling/models; writes output and, without preview mode, Qdrant |
| `scripts.operations.ingest_preview` | Inspect deterministic parsing/chunking output | Docling/model cache; writes preview files |
| `scripts.operations.query_smoke` | Run a bounded end-to-end query diagnostic | Qdrant/models and possibly a provider, according to flags |
| `scripts.operations.validate_query_runtime` | Validate current retrieval runtime | Qdrant and local retrieval/reranker models |
| `scripts.evaluation.validate_dataset` | Validate the legacy or optimization dataset pair | Reads frozen evaluation files only |
| `scripts.evaluation.evaluate_retrieval` | Produce current retrieval evaluation | Qdrant and local models; writes a versioned artifact |
| `scripts.evaluation.evaluate_e2e` | Run approval-gated generation evaluation | May call a real provider and write a checkpoint/artifact |

Historical commands under `scripts/archive/` are unsupported provenance snapshots. Do not infer that
they are safe because they remain in Git; some can mutate Qdrant, overwrite artifacts, or call a
provider.

## Dataset validation suites

`scripts.evaluation.validate_dataset` defaults to the legacy `calibration`/`test` pair. When the
separately approved optimization inputs are present, select the new pair explicitly:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.validate_dataset `
  --suite optimization `
  --output artifacts/metrics/atv320-optimization-dataset-validation.json
```

The optimization contract requires 120 development items (80 answerable, 40 unanswerable) and 60
confirmation items (40 answerable, 20 unanswerable). Within each answerability class, each split is
balanced equally between English and Vietnamese. Every optimization item has a non-blank
`fact_family_id` (the field remains optional for legacy items). Each of the nine answerable question
types has a minimum quota of eight development items and four confirmation items; the pair must be
disjoint by item ID, normalized exact question, and fact family. The validator also checks
answerable qrels, expected documents, pages, phrases, and the frozen corpus as it does for the
legacy suite.

Use opaque lowercase item IDs, `fact_family_id` values, and expected-fact IDs (letters, digits, `.`,
`_`, or `-`; at most 64 characters), never question or fact sentences. Sanitized v9 rows retain
these IDs so later paired and clustered comparisons can group translations/paraphrases and diagnose
fact outcomes without persisting the underlying question or expected fact.

This command reads both payloads plus the frozen chunks and validates their structure; it does not
validate manifest path containment, authorize tuning, or authorize provider egress. Its sanitized
JSON output includes both dataset hashes and the frozen corpus identity, so the reviewed output can
serve as the hash manifest without a one-off generator. Pair validation emits that sealing report
only when all 180 optimization records are explicitly `approved`; single-split validation remains
available through `evaluation.dataset.validate_dataset` while annotations are still drafts.
Development
is the inspectable optimization surface. A development E2E run reads only that split and requires
its manifest to carry the sealed confirmation hash; it does not open the confirmation payload. E2E
requires the confirmation dataset and manifest to remain under
`data/eval/atv320/private-confirmation-v1/`, uses separate confirmation approval tokens, and rejects
partial `--item-id` or `--max-queries` runs.

Development permits `--item-id` or `--max-queries` diagnostics only when both `--output` and
`--checkpoint` are explicit, preventing a partial trace from replacing or blocking the default full
development artifact.

Sealed held-out v2 likewise rejects `--item-id` and `--max-queries`; checkpoint resume is the only
supported way to continue an interrupted full sealed run.

Optimization E2E is also provider- and cost-bearing. After the datasets, manifests, and egress have
been explicitly approved, use the dataset-specific generator token below; add the independent
semantic token only when `--semantic-judge ragas` is requested.

| Dataset | Generator approval token | Optional semantic approval token |
| --- | --- | --- |
| development | `APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS` | `APPROVE ATV320 DEVELOPMENT OPENROUTER RAGAS EGRESS` |
| confirmation | `APPROVE ATV320 CONFIRMATION GENERATOR EGRESS` | `APPROVE ATV320 CONFIRMATION OPENROUTER RAGAS EGRESS` |

```powershell
# Inspectable development run; --item-id is allowed only for a sanitized diagnostic.
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_e2e `
  --dataset development `
  --provider-approval-token "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS"

# Full sealed confirmation run; partial-selection flags are rejected.
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_e2e `
  --dataset confirmation `
  --provider-approval-token "APPROVE ATV320 CONFIRMATION GENERATOR EGRESS"
```

## Approved held-out regression run

Held-out v2 is exposed regression evidence, not an unseen benchmark. Run it only after the user has
explicitly approved provider data egress and cost. The exact dataset-specific token prevents a
calibration approval from authorizing held-out traffic accidentally. Confirm that the interpreter is
Python 3.11 before running:

```powershell
.\.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:2] == (3, 11)"
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_e2e `
  --dataset heldout-v2 `
  --provider-approval-token "APPROVE PHASE 7 HELDOUT PROVIDER EGRESS"
```

`heldout-v2` is the sealed private regression split. It is intentionally distinct from the public
`test` split even though both contain 45 records. The command requires both its dataset and manifest
to remain under `data/eval/phase7/private-heldout-v2/`, then validates their hash, the frozen corpus,
collection identities, runtime source identity, and configured provider before completing.

The current evaluator declares artifact schema v9. If the command above is approved and executed,
its default paths are `artifacts/metrics/atv320-heldout-v2-e2e-v9-checkpoint.jsonl` and
`artifacts/metrics/atv320-heldout-v2-e2e-v9.json`. V9 records a sanitized retrieval trace because
the E2E adapter explicitly opts in; production builders default to `capture_trace=False` and the
production bootstrap does not opt in. The artifact includes stage counts, IDs, ranks, and allowlisted
role signals, but still excludes raw questions, evidence, prompts, answers, and provider responses.
Reviewed expected-fact text is also excluded; only fact IDs, matcher outcomes, counts, and positions
are persisted.

Schema availability is not execution evidence. The latest executed held-out result remains the v8
checkpoint at `artifacts/metrics/atv320-heldout-v2-e2e-v8-checkpoint.jsonl` and final artifact at
`artifacts/metrics/atv320-heldout-v2-e2e-v8.json`; no v9 held-out run is claimed. Existing v7 and v8
evidence remain immutable. A v8 checkpoint cannot resume under schema v9/source identity v5. Never
use any held-out result to tune the current runtime.

## Optional Ragas semantic evaluation

Semantic scoring runs in a separate Python 3.11 evaluator image. It contains retrieval, generation,
Ragas, and the OpenAI-compatible SDK, but no Docling and no PDF mount. The default E2E mode is
`none`, so normal runs neither import Ragas nor construct an OpenRouter judge.

Build and verify only this smaller evaluator target:

```powershell
docker compose --progress=plain build evaluation
docker compose run --rm --no-deps evaluation python -c "import sys; from importlib.metadata import version; assert sys.version_info[:2] == (3, 11); assert version('ragas') == '0.4.3'; assert version('openai') == '2.53.0'; assert version('langchain-community') == '0.3.31'; from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference; print('semantic imports OK')"
docker compose run --rm --no-deps evaluation python -m scripts.evaluation.evaluate_e2e --help
```

Set `OPENROUTER_API_KEY=sk-or-v1-...` in the untracked `.env`; do not put the key in a command, log,
artifact, or chat message. The base URL and model slugs are frozen in the evaluator rather than
configurable through `.env`.

Ragas uses OpenRouter model `openai/gpt-5.6-luna` with low reasoning effort for Faithfulness, Answer
Relevancy, and Context Precision without reference. Answer Relevancy uses
`openai/text-embedding-3-small`. Both request types are pinned to the OpenAI upstream, require full
parameter support, and disable provider fallback. Requests set
`store=false`, use a 60-second timeout with at most one retry, and do not use a persistent Ragas
cache. The adapter also sets `RAGAS_DO_NOT_TRACK=true`; the approval token authorizes OpenRouter and
its pinned OpenAI upstream, not Ragas usage telemetry. `store=false` prevents application-requested
storage but does not promise zero retention; both OpenRouter and upstream data policies still apply.

The evaluator pins `langchain-community==0.3.31` because Ragas 0.4.3 imports a legacy VertexAI
module at package import time. Unbounded resolution to `langchain-community==0.4.2` breaks even this
OpenAI-compatible evaluator before any metric runs. Keep this transitive pin until Ragas publishes
and the project explicitly adopts a release with the corrected import.

Semantic egress requires a second dataset-specific approval token. The generation token authorizes
Gemini only; the judge token authorizes OpenRouter plus the pinned OpenAI upstream only. Run the same
first five calibration items twice with separate outputs before any held-out execution:

```powershell
docker compose up -d qdrant
docker compose run --rm evaluation python -m scripts.evaluation.evaluate_e2e `
  --dataset calibration `
  --max-queries 5 `
  --semantic-judge ragas `
  --provider-approval-token "APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS" `
  --judge-provider-approval-token "APPROVE ATV320 CALIBRATION OPENROUTER RAGAS EGRESS" `
  --checkpoint artifacts/metrics/atv320-ragas-calibration-preflight-a-checkpoint.jsonl `
  --output artifacts/metrics/atv320-ragas-calibration-preflight-a.json
```

Repeat with suffix `b`, then compare finite scores, EN/VI behavior, latency, and run-to-run variance.
Review the same five non-held-out questions locally through the current demo for human sanity; do not
copy raw question, answer, or evidence into logs or artifacts. Stop if Luna produces invalid
structured output or clearly unreliable judgments. Do not fall back to a different model.

After that review is accepted, the held-out command adds these arguments to the approved regression
run shown above:

```text
--semantic-judge ragas
--judge-provider-approval-token "APPROVE ATV320 HELDOUT OPENROUTER RAGAS EGRESS"
```

Schema-v9 checkpoints fail closed if the schema, source identity, judge model, embedding model,
metric configuration, or input policy differs. Only a fully answered item with all three finite
semantic scores is checkpointed. Abstentions are recorded as not applicable and excluded from
semantic denominators. Deterministic quality gates remain the only source of exit code `0` or `2`.

## Reproduce ingestion without indexing

The ingestion image is intentionally heavier than the API image because Docling and its document
models are isolated from normal query serving. Building and conversion can take a long time. The
user should run those steps explicitly; normal validation does not build the image or re-chunk PDFs.

Use an external temporary directory and preview mode:

```powershell
$chunkCheckDir = Join-Path ([System.IO.Path]::GetTempPath()) "industrial-rag-ingestion-check"
New-Item -ItemType Directory -Path $chunkCheckDir -Force | Out-Null

docker compose --progress=plain build ingestion

docker compose run --rm --no-deps `
  --volume "${chunkCheckDir}:/tmp/chunk-check" `
  ingestion python -m scripts.operations.index_corpus `
  --preview-only `
  --chunks-output /tmp/chunk-check/frozen-chunks.jsonl `
  --manifest-output /tmp/chunk-check/manifest.json
```

Preview mode must produce exactly 2,753 chunks and the ordered chunk-ID hash:

```text
2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d
```

Compare the generated manifest with `artifacts/metrics/phase-7-corpus-manifest.json`. Exact
reproduction also requires the Docling transitive versions pinned by the ingestion dependency set.
Do not accept equal counts as proof when the chunk-ID hash differs.

## Troubleshooting

### `invalid spec: :/tmp/chunk-check`

The PowerShell variable before the colon is empty, often because a new shell was opened. Re-run the
assignment and verify it before Compose:

```powershell
$chunkCheckDir
Test-Path -LiteralPath $chunkCheckDir
```

### Input document does not exist

Compose mounts repository PDFs at `/data/raw`, while a local default such as `data/raw/...` is
relative to the container working directory. Use the command defaults from the current Compose
image or pass the two explicit `/data/raw/<filename>.pdf` paths shown by:

```powershell
docker compose run --rm --no-deps ingestion sh -lc "ls -lh /data/raw"
```

### Docling/Transformers sequence-length warnings

Hybrid chunking may inspect a structure whose headings, captions, or table text exceed a tokenizer's
nominal 512-token window. The warning alone does not prove an indexing failure: the chunker can split
or ignore oversized metadata. Treat a raised exception, changed count, changed stable-ID hash, or a
failed contract test as the failure signal. Current logging sanitizes warning payloads so full manual
content is not copied into operational logs.

### Docling version drift

Matching the top-level `docling` version is insufficient. A change in `docling-core` previously kept
the count at 2,753 but changed nine stable chunk IDs. Rebuild from the locked dependency set, inspect
installed versions, and require the frozen hash before indexing.

```powershell
docker run --rm --entrypoint python industrial-rag-ingestion:local -c "from importlib.metadata import version; print(version('docling'), version('docling-core'), version('onnxruntime'))"
```

### Pip hash mismatch or truncated package download

Do not disable hash checking. A cached or interrupted wheel can have the right name and wrong bytes.
Retry the build after removing only the failed build cache entry or allow Docker/pip to re-download
the package. Never prune named Qdrant volumes as part of build-cache cleanup.

### Slow first query

Dense, sparse, and reranker models are large and may download or initialize on the first integration
run. This is expected outside unit tests. A successful runtime diagnostic must report the active
`contract_id`; warnings are not a substitute for checking its exit code and JSON result.

## Operational safety

- Do not run provider evaluation without explicit data-egress approval.
- Do not use exposed held-out v2 for tuning.
- Do not re-index or mutate collections merely to validate a refactor.
- Do not commit PDFs, `.env`, model caches, raw benchmark payloads, or generated temporary previews.
- A missing model, provider, collection, or source path is an error; never switch pipelines silently.
