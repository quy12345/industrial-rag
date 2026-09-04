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

Generation configuration uses one credential/model contract plus a declared OpenAI-compatible
endpoint profile. The active profiles are:

| Profile | API dialect | Endpoint ownership |
| --- | --- | --- |
| `openai-responses` | Responses API | Direct OpenAI |
| `gemini-chat` | Chat Completions | Direct Gemini OpenAI-compatible endpoint |
| `openrouter-chat` | Chat Completions | OpenRouter with required parameters and no fallback |

The default `.env.example` keeps the current Gemini selection. Use only generic `GENERATION_*`
settings; the retired `GEMINI_*` and shared `OPENAI_*` generation settings are not compatibility
aliases. A local `.env` must therefore be migrated explicitly. For example:

```dotenv
GENERATION_ENDPOINT_PROFILE=openrouter-chat
GENERATION_API_KEY=
GENERATION_MODEL=deepseek/deepseek-v4-flash-0731
GENERATION_REASONING=disabled
GENERATION_TEMPERATURE=0
GENERATION_STORE=false
```

Endpoint profiles own base URLs, API dialects, and provider-specific reasoning parameters. Model
IDs remain configuration. Unsupported profile/reasoning combinations and `GENERATION_STORE=true`
fail closed before provider I/O. `OPENROUTER_API_KEY` remains the separate optional Ragas judge
credential; do not confuse it with `GENERATION_API_KEY` even when both calls use OpenRouter.
For OpenRouter, `disabled` explicitly sends reasoning effort `none`; use `omit` for a non-thinking
model whose providers reject the reasoning parameter entirely.

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
`_`, or `-`; at most 64 characters), never question or fact sentences. Sanitized v14 rows retain
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

## Provider-free retrieval variants

Q02 preregisters exactly six development variants; the CLI accepts no arbitrary weights, reserve
counts, or glossary terms:

| Variant | Query profile | Refill after same-document dedup | Dense/sparse reserves |
| --- | --- | --- | --- |
| `baseline` | `vi_technical_glossary_v1` | no | 5 / 24 |
| `refill` | `vi_technical_glossary_v1` | yes | 5 / 24 |
| `query-expansion-v2` | `vi_technical_glossary_v2` | no | 5 / 24 |
| `refill-query-expansion-v2` | `vi_technical_glossary_v2` | yes | 5 / 24 |
| `dense-reserve-d20-s5` | `vi_technical_glossary_v1` | no | 20 / 5 |
| `sparse-reserve-d5-s20` | `vi_technical_glossary_v1` | no | 5 / 20 |

The v2 lexical policy affects sparse search only. It matches complete Vietnamese phrases across
NFC, NFD, and conservative accentless input while retaining the original query, identifiers,
numbers, and units. Dense retrieval and the cross-encoder continue to receive the original query.

After all 180 optimization records have been human-reviewed, approved, and sealed, start Qdrant and
run each variant against the full development split. For example:

```powershell
docker compose up -d qdrant
docker compose run --rm evaluation python -m scripts.evaluation.evaluate_retrieval `
  --dataset development `
  --retrieval-variant refill
```

The command uses Qdrant plus local embedding/reranker models but makes zero generator or judge calls
and never writes a collection. Each variant makes separate trace-only dense and sparse probes to
rank 100; normal fusion still consumes only dense top 60 and sparse top 40, the reranker still sees
at most 30 candidates, and probe latency is excluded from the runtime-quality comparison. Schema-v5
output and checkpoint names include the variant ID, and a checkpoint from another variant or the old
schema fails closed. The v5 evaluator additionally applies the canonical evidence selector with a
fixed diagnostic `top_k=5`. Its artifact retains stable chunk/document IDs, page numbers, heading
counts, and exact-duplicate alternatives, but excludes the question, chunk text, headings, filename,
and unlisted metadata. Strict wrong-document rates remain diagnostics; reviewed authoritative
alternatives explain repeated manual content. Development gates require complete coverage of every
metrics. This fixed evaluation boundary does not reduce the API's supported `top_k` range of 1–10.
The command labels its result `retrieval gates PASS/FAIL`: reviewed-authority coverage is gated;
strict document identity remains diagnostic. Any metric movement is an evaluation-contract
correction, not a production quality claim.
Non-baseline variants are rejected for the legacy suite, and the command has no
confirmation or held-out dataset choice. The current
repository does not yet contain the reviewed development/confirmation payloads, so no variant run,
winner, or quality improvement is claimed by this implementation alone.

Use the trace to check the raw ceiling first: qrel absence from both diagnostic dense top 100 and
sparse top 100 means refill, RRF reserves, and lexical fusion cannot recover it. Stop this tuning
path and prepare a separately approved shadow model/index proposal; do not mutate the current
collection.

Sealed held-out v2 likewise rejects `--item-id` and `--max-queries`; checkpoint resume is the only
supported way to continue an interrupted full sealed run.

## Current E2E Ragas scorecard

Schema v15 supports one E2E workflow for `development`, sealed `confirmation`, and exposed
`heldout-v2`. It always runs the configured generator plus Ragas Factual Correctness,
Faithfulness, and Context Recall. Generator egress and judge egress require separate approvals.

```powershell
docker compose run --rm evaluation python -m scripts.evaluation.evaluate_e2e `
  --dataset development `
  --provider-approval-token "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS" `
  --judge-provider-approval-token "APPROVE ATV320 DEVELOPMENT OPENROUTER RAGAS SCORECARD EGRESS" `
  --ragas-diagnostics
```

The four canonical outcomes are answer utility, unsafe-answer rate, citation faithfulness, and
final-context recall. Optional Answer Relevancy and Context Precision without reference are
diagnostics only. The scorecard has no composite score or thresholds: v15 reports
`BASELINE_ONLY`, and a valid complete run exits zero regardless of score.

Default outputs are `artifacts/metrics/atv320-{dataset}-e2e-v15.json` and
`artifacts/metrics/atv320-{dataset}-e2e-v15-checkpoint.jsonl`. They exclude raw questions,
answers, reviewed references, evidence, prompts, provider output, judge reasoning, and retrieval
traces. V14 checkpoints fail closed. E2E fixes the evidence boundary at five and uses
`capture_trace=False`; use `scripts.evaluation.evaluate_retrieval` for deep stage diagnostics.

Ragas uses OpenRouter model `openai/gpt-5.6-luna`, low reasoning, OpenAI-only routing, no fallback,
`store=false`, a 60-second timeout, one retry, and no persistent cache. The optional diagnostics
alone initialize `openai/text-embedding-3-small`. `OPENROUTER_API_KEY` remains in the untracked
`.env`. The evaluator image is Python 3.11 and contains no Docling.

## Human validation of the scorecard

Before any full development baseline, run the fixed 40-item metadata-selected cohort. This sends
30 answerable and 10 unanswerable EN/VI-balanced cases to both the configured generator and Ragas,
so it requires fresh approval for both egress paths:

```powershell
docker compose run --rm evaluation python -m scripts.evaluation.evaluate_e2e \`
  --dataset development \`
  --review-cohort judge-validation-v1 \`
  --local-review-output artifacts/local-review/judge-validation-v1/review.md \`
  --checkpoint artifacts/metrics/atv320-judge-validation-v1-e2e-v15-checkpoint.jsonl \`
  --output artifacts/metrics/atv320-judge-validation-v1-e2e-v15.json \`
  --provider-approval-token "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS" \`
  --judge-provider-approval-token "APPROVE ATV320 DEVELOPMENT OPENROUTER RAGAS SCORECARD EGRESS" \`
  --ragas-diagnostics
```

The private directory contains `review.md` and `labels.jsonl`. It is Git-ignored and contains raw
questions, answers, reviewed facts, citations, cited chunks, and final generation evidence. Never
commit, upload, paste into CI, or share these files. Replace applicable `null` values in
`labels.jsonl` with `true` or `false`; leave only genuinely non-applicable fields as `null`.

Validate agreement without provider access:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.validate_scorecard \`
  --artifact artifacts/metrics/atv320-judge-validation-v1-e2e-v15.json \`
  --labels artifacts/local-review/judge-validation-v1/labels.jsonl \`
  --output artifacts/metrics/atv320-judge-validation-v1-agreement.json
```

The sanitized agreement artifact contains only hashes, applicable counts, TP/TN/FP/FN,
disagreement IDs, and acceptance status. Each rubric must reach at least 85%, all applicable labels
must be complete, correctness/faithfulness need at least 20 answered-answerable samples, and no
human-confirmed unsafe false answer may be missed. Exit code is zero for PASS and two for a valid
FAIL. If validation fails, stop and diagnose categories; do not tune a prompt or threshold to
individual review items.

After agreement passes and only after a separate approval, run full development once using the
current E2E command with `--ragas-diagnostics`. Do not run confirmation or held-out in E04 and do
not derive thresholds from that same baseline.

## Historical evaluator contracts

Schema v8 is the latest executed held-out evidence. Schemas v9-v14 recorded completed retrieval,
fact-matcher, provenance, generation-variant, and hybrid-judge experiments. E04 hard-cut those
active commands and modules after their durable lessons moved to `PROJECT_JOURNEY.md`. The original
commands and artifacts remain discoverable in Git history; they are not supported operations and
cannot resume under schema v15/source identity v11.

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
