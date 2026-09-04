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
`_`, or `-`; at most 64 characters), never question or fact sentences. Sanitized v13 rows retain
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
at most 30 candidates, and probe latency is excluded from the runtime-quality comparison. Schema-v4
output and checkpoint names include the variant ID, and a checkpoint from another variant or the old
schema fails closed. The v4 evaluator additionally applies the canonical evidence selector with a
fixed diagnostic `top_k=5`. Its artifact retains stable chunk/document IDs, page numbers, heading
counts, and exact-duplicate alternatives, but excludes the question, chunk text, headings, filename,
and unlisted metadata. Strict wrong-document rates remain authoritative; parallel
exact-content-equivalent rates explain repeated manual content without relaxing those strict
metrics. This fixed evaluation boundary does not reduce the API's supported `top_k` range of 1–10.
The command labels its result `retrieval gates PASS/FAIL`: schema-v4 evidence metrics are diagnostic
and do not affect the exit code until promotion criteria have been selected on reviewed development
data.
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

Optimization E2E is also provider- and cost-bearing. After the datasets, manifests, and egress have
been explicitly approved, use the dataset-specific generator token below. Semantic and hybrid
fact judging each require their own independent token.

| Dataset | Generator approval token | Optional semantic approval token | Optional fact-judge token |
| --- | --- | --- | --- |
| development | `APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS` | `APPROVE ATV320 DEVELOPMENT OPENROUTER RAGAS EGRESS` | `APPROVE ATV320 DEVELOPMENT OPENROUTER FACT JUDGE EGRESS` |
| confirmation | `APPROVE ATV320 CONFIRMATION GENERATOR EGRESS` | `APPROVE ATV320 CONFIRMATION OPENROUTER RAGAS EGRESS` | `APPROVE ATV320 CONFIRMATION OPENROUTER FACT JUDGE EGRESS` |

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

## Development generation variants

The E2E evaluator supports two preregistered generation contracts on the inspectable development
split:

- `baseline-v1` uses the current answer-level structured contract.
- `claim-coverage-v1` asks for separately sourced answer claims, validates their ephemeral source
  labels, and renders them back to the existing answer/citation contract.

Use `--repeat-count 1` for a single diagnostic run or `--repeat-count 3` to compare provider
stability. Run both variants with the same repeat count; for example, a three-repeat comparison is:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_e2e `
  --dataset development `
  --generation-variant baseline-v1 `
  --repeat-count 3 `
  --provider-approval-token "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS"

.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_e2e `
  --dataset development `
  --generation-variant claim-coverage-v1 `
  --repeat-count 3 `
  --provider-approval-token "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS"
```

`claim-coverage-v1` is development-only. Three-repeat runs also require the complete development
split, reject `--item-id` and `--max-queries`, and cannot be combined with Ragas
(`--semantic-judge ragas`). Calibration, test, confirmation, and held-out evaluation remain
restricted to `baseline-v1` with `--repeat-count 1`.

The evaluator processes all development items for repeat 1 before repeat 2 and repeat 3. It writes
each completed `(item_id, repeat_index)` in that repeat-major order. Resume is fail-closed: the
checkpoint must be an exact prefix of the same dataset order and must match the variant, repeat
count, schema, source identity, resolved evidence `top_k`/gate, context/citation limits, and other
run identity fields. Repeated metrics are reported per run rather than pooled across all records.

Schema-v13 default paths encode the full experiment identity:

| Variant and repeats | Artifact | Checkpoint |
| --- | --- | --- |
| baseline, once | `artifacts/metrics/atv320-development-baseline-v1-r1-e2e-v13.json` | `artifacts/metrics/atv320-development-baseline-v1-r1-e2e-v13-checkpoint.jsonl` |
| claim coverage, once | `artifacts/metrics/atv320-development-claim-coverage-v1-r1-e2e-v13.json` | `artifacts/metrics/atv320-development-claim-coverage-v1-r1-e2e-v13-checkpoint.jsonl` |
| baseline, three repeats | `artifacts/metrics/atv320-development-baseline-v1-r3-e2e-v13.json` | `artifacts/metrics/atv320-development-baseline-v1-r3-e2e-v13-checkpoint.jsonl` |
| claim coverage, three repeats | `artifacts/metrics/atv320-development-claim-coverage-v1-r3-e2e-v13.json` | `artifacts/metrics/atv320-development-claim-coverage-v1-r3-e2e-v13-checkpoint.jsonl` |

These switches exist only in the evaluation CLI. They do not change the production composition
root, FastAPI request/response schema, evidence gate, citation validation, or default generator
behavior. This runbook records supported commands only; it does not claim that either variant has
been executed or improved quality.

## Local development failure review

The sanitized E2E artifact cannot contain raw questions, answers, expected facts, or evidence text.
To diagnose a bounded set of development failures, join that artifact with the local development
dataset and frozen chunks and print the result only to the current terminal:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.review_development `
  --artifact artifacts/metrics/q04-qwen-development-baseline-v1-r1-e2e-v11.json `
  --item-id dvi015 `
  --item-id dvi024 `
  --item-id dvi071
```

The command performs no Qdrant, model, or provider access and offers no output-file option. It
validates the development dataset and frozen-corpus hashes before revealing local text, rejects
non-development artifacts, and labels automated conclusions as diagnostic hints requiring human
confirmation. The original generated answer remains unavailable because the sanitized artifact
never stored it; a later targeted provider rerun is a new sample, not recovery of the old output.
Do not redirect this command into a tracked file or paste its output into logs, issues, or commits.

After provider-free review has isolated cases that genuinely require generation diagnosis, the E2E
adapter can rerun a bounded cohort with the current `baseline-v1` generator. The recommended mode
writes a private Markdown report with each question, new answer, expected-fact verdicts, citations,
and only the cited generation evidence:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_e2e `
  --dataset development `
  --review-item-id dvi031 `
  --review-item-id dvi065 `
  --provider-approval-token "APPROVE ATV320 TARGETED DEVELOPMENT REVIEW EGRESS" `
  --local-review-output artifacts/local-review/targeted-review.md `
  --checkpoint artifacts/metrics/local-targeted-review-checkpoint.jsonl `
  --output artifacts/metrics/local-targeted-review.json
```

This is an explicit provider call and therefore requires fresh user approval for the named IDs,
configured model, and estimated cost before execution. At most 20 unique development IDs are
accepted. The mode requires `baseline-v1`, one repeat, no Ragas, new explicit checkpoint/output
paths, and a new or empty checkpoint. The artifact and checkpoint remain sanitized; raw content is
written only to a new `.md` file below the Git-ignored `artifacts/local-review/` directory. The
command refuses paths outside that directory and refuses to overwrite an existing report. Omit
`--local-review-output` only when an ephemeral terminal JSON payload is explicitly preferred.
Because provider output can vary, this is a new diagnostic sample rather than a reconstruction of
the original full-run answer. Do not use this mode with confirmation or held-out data, commit or
paste the private report into public logs, or select a favorable retry after repeated runs.

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

The current evaluator declares artifact schema v13. If the command above is approved and executed,
its default paths are
`artifacts/metrics/atv320-heldout-v2-baseline-v1-r1-e2e-v13-checkpoint.jsonl` and
`artifacts/metrics/atv320-heldout-v2-baseline-v1-r1-e2e-v13.json`. V13 records a sanitized
retrieval trace because the E2E adapter explicitly opts in; production builders default to
`capture_trace=False` and the production bootstrap does not opt in. The artifact includes stage
counts, IDs, ranks, and allowlisted role signals, but still excludes raw questions, evidence,
prompts, answers, and provider responses.
Reviewed expected-fact text is also excluded; only fact IDs, matcher outcomes, counts, and positions
are persisted.

Schema availability is not execution evidence. The latest executed held-out result remains the v8
checkpoint at `artifacts/metrics/atv320-heldout-v2-e2e-v8-checkpoint.jsonl` and final artifact at
`artifacts/metrics/atv320-heldout-v2-e2e-v8.json`; no v9-v13 held-out run is claimed.
Existing v7 and v8 evidence remain immutable. A v12-or-earlier checkpoint cannot resume under
schema v13/source identity v9. Never use any held-out result to tune the current runtime.

## Optional hybrid fact evaluation

The default `--fact-evaluator deterministic-v3` is offline. It matches exact numeric/unit and
identifier facts, then applies reviewed clause-local text alternatives with decimal-safe splitting,
bounded English inflections, EN/VI function-word tolerance, and negation guards. Its result is both
the deterministic and final verdict.

`--fact-evaluator hybrid-luna-v1` keeps deterministic positives, sends all unresolved text facts
for one item in a single request to OpenRouter model `openai/gpt-5.6-luna`, and treats exact
numeric/identifier misses as authoritative failures. The judge receives only the question, generated
answer, and reviewed unresolved fact contracts—never retrieved evidence. It must return exactly one
`present`, `absent`, or `contradicted` verdict per requested fact ID. Missing, duplicate, extra,
or invalid verdicts stop the run without checkpointing the incomplete item.

The route is pinned to OpenAI upstream with provider fallback disabled, low reasoning,
`store=false`, a 60-second timeout, and one retry. Set `OPENROUTER_API_KEY` only in the untracked
`.env`. A development preflight additionally requires both independent approvals:

```powershell
docker compose run --rm evaluation python -m scripts.evaluation.evaluate_e2e `
  --dataset development `
  --max-queries 5 `
  --fact-evaluator hybrid-luna-v1 `
  --provider-approval-token "APPROVE ATV320 DEVELOPMENT GENERATOR EGRESS" `
  --fact-judge-approval-token "APPROVE ATV320 DEVELOPMENT OPENROUTER FACT JUDGE EGRESS" `
  --checkpoint artifacts/metrics/atv320-fact-judge-development-preflight-checkpoint.jsonl `
  --output artifacts/metrics/atv320-fact-judge-development-preflight.json
```

Do not run this command until the named items, egress, model, and cost have fresh approval. Generator,
Ragas, and fact-judge tokens are non-interchangeable. Schema-v13 artifacts persist only fact IDs,
deterministic/final outcomes, resolution, verdict categories, counts, and latency. They exclude
question, answer, aliases, judge reasoning, provider bodies, and keys. Deterministic and final
accuracy are reported separately; the unchanged `0.85` quality gate uses final accuracy only in
hybrid mode. Hybrid evaluation cannot be combined with a three-repeat generation run.

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
only the currently selected generation endpoint; the judge token independently authorizes
OpenRouter plus the pinned OpenAI upstream. Run the same first five calibration items twice with
separate outputs before any held-out execution:

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

Schema-v13 checkpoints fail closed if the schema, source identity, endpoint profile, generation
model, generation variant, repeat
schedule, judge model, embedding model,
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
