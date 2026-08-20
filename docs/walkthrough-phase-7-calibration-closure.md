# Phase 7 calibration closure walkthrough

## Outcome

The 2026-08-11 closure began as `PARTIAL`. It improves evaluator correctness, seals calibration from
held-out reads, separates full reranker output from actual LLM evidence, and removes exact
cross-document duplicate evidence. The separately approved calibration 005 diagnostic is complete
and stable across three generations over one fixed evidence bundle. The 2026-08-17 provider-free
continuation moves calibration 010 direct evidence into actual top 5, and the later three-run Gemini
calibration passes all technical gates.

Held-out status is `BLOCKED_GOVERNANCE`. Historical tracked documentation mirrored held-out content,
and the old calibration CLI loaded both JSONL files. The current code closes those paths, but changing
the present revision cannot erase statistical exposure in Git history.

No qrel, chunk, embedding model, Jina model, Qdrant collection, volume, public Query API, or Docker
image was changed. Exactly three approved provider generations were made for calibration 005; no
held-out run, re-index, image build, or prune was performed.

## Replacement held-out v2

The historic split remains blocked forever as an unseen benchmark. A separate
45-row replacement draft (30 answerable, 15 unanswerable) now lives under the
Git-ignored local directory `data/eval/phase7/private-heldout-v2/`. Its qrels
and expected phrases validated against the frozen 2,753-chunk corpus; it has
not been frozen or executed. The generic freezer
`scripts/freeze_phase7_heldout_v2.py` requires the exact human dataset token,
keeps both the approved JSONL and its manifest in that private directory, and
does not open calibration or historic held-out data. Git ignore reduces
repository exposure only; the workspace owner must apply OS-level ACLs if a
stronger access boundary is needed.

The sanitized one-shot v2 artifact reports candidate recall `0.900`, Hit@5
`0.800`, MRR@5 `0.725`, deterministic fact accuracy `0.786` (28 answered
answerable rows), valid citation IDs `1.000`, two wrong-document citations,
and abstention precision/recall `0.882`/`1.000`. It therefore does not meet the
calibration release targets. It is a final observation for this v2 dataset, not
a new tuning signal; any corrective iteration needs a fresh sealed benchmark.
The user-built ingestion image was only inspected: Python 3.11.15 and Docling/LangChain imports pass.
Docker storage accounting reports 9.66 GB total, 9.515 GB unique, while image metadata reports
3,255,679,310 bytes; image-size optimization remains a separate follow-up.

## Sealed calibration flow

```text
calibration-v3.jsonl only
  -> single-split schema/qrel validation against frozen 2,753 chunks
  -> held-out SHA-256 read from manifest only
  -> read-only Qdrant contract validation
  -> dense@60 + expanded sparse@40 -> weighted RRF/reserves -> Jina
  -> rank-only role prior
  -> exact cross-document duplicate evidence selection
  -> actual top-5 evidence gate
  -> provider generation only after a new exact approval token
  -> referential citation validation
  -> typed fact scoring and sanitized schema-v5 artifact
```

`scripts/evaluate_phase7_e2e.py` no longer validates calibration by loading the test split. A poisoned
held-out path unit test proves that calibration invokes only the active calibration loader. The run
identity includes dataset/corpus/runtime/prompt/evaluator hashes, provider model and base-URL host,
reasoning effort, explicit Gemini temperature `0`, token/timeout/retry/store settings, Python/library
versions, top-k, and correction count. Any mismatch invalidates an existing checkpoint.

Held-out mode currently exits with `BLOCKED_GOVERNANCE` before reading its dataset or contacting a
provider.

## Fact evaluator v2

The headline remains `deterministic_fact_accuracy`. `strict_phrase_accuracy` and token coverage stay
diagnostic.

Text facts now use NFKC/casefold token positions and a finite inflection relation for ASCII lexical
tokens of at least four characters: `s`, `es`, `ed`, `d`, and `ing`. This generic rule makes
`block`/`blocked` and `contact`/`contacts` pass. It is not applied to identifiers or numeric-unit
facts, and it is not arbitrary prefix matching: `MODE` does not match `model`, `IP65` does not match
`IP650`, and `rEF` does not match a larger identifier.

Negation is evaluated around matched spans inside clauses. A positive complete clause wins over a
historical negated clause; a direct local negation fails; multiple local negations are `ambiguous`
and fail the headline. English and Vietnamese negation boundaries are tested. Sanitized traces store
matched alternatives/positions, match mode, negation positions/distance, and polarity—never the raw
answer.

Historical calibration 005 cannot be reconstructed because v4 intentionally omitted raw answers.
The new diagnostic script retrieves/reranks once, freezes one evidence bundle, and makes exactly
three provider attempts against it. Raw answers may be written only beneath ignored
`artifacts/private-debug/`; the metrics artifact excludes question, answer, evidence, prompt, and
provider response.

The approved 2026-08-11 diagnostic completed all three attempts. Every attempt cited `S1`; every
typed-fact result matched with `polarity=positive`, exact match mode, and no local negation. Because
the evidence manifest was fixed and retrieval/reranking each executed once, this is not a best-run
selection. It classifies the current 005 path as evaluator closure, not a generation failure or
provider-instability case. The historical v4 answer remains unknowable by design.

## Actual evidence selection and citations

`QueryExecution` now has three distinct sets:

- `candidate_pool`: pre-rerank pool, up to 30.
- `candidates`: full post-rerank ordered pool.
- `evidence_candidates`: deterministic post-dedup top-k sent to the LLM.

The selector groups only exact NFKC/case/whitespace content repeated across different documents. If
the query-derived role matches one source, that source represents the group; otherwise the best
existing rank wins. Equivalent chunk/document IDs and original ranks remain in metadata. It does not
collapse same-document duplicates or near-duplicates, hard-filter a document, or read expected
document IDs/qrels.

The prompt also asks for the smallest sufficient source set and the highest-ranked source when two
sources repeat one fact. Citation validation stays referential and does not become evaluation-aware.

## Provider-free 010 closure result

The historical snapshot-v2 run proved that raw CE/RRF rank fusion and whole-chunk label counting were
insufficient. Snapshot v3 reran the same frozen Qdrant/Jina path in the existing Python 3.11 image and
adds sanitized relation-scoped feature counts. It stores IDs, component/cross-encoder ranks, finite
scores, document roles, exact-content hashes and integer counts; it stores no question or chunk text
and reports zero provider calls and zero held-out executions.

The finite grid still tests RRF-rank multipliers `0`, `0.25`, `0.5`, `1`, and `2` with offsets `10`,
`20`, and `40`. If those and historical `list_completeness_v1` fail, exactly one generic relation
fallback is tested. It requires list + key/button + switch/change + technical-ID cues from the query,
then counts bracketed label/code targets only after that relation in the same candidate clause.

| Metric | Result |
|---|---:|
| Candidate recall | 12/12 = 1.000 |
| Hit@5 | 12/12 = 1.000 |
| MRR@5 | 0.892 |
| EN / VI Hit@5 | 6/6 / 6/6 |
| Wrong-document top-1 | 0/12 |
| Best replay wrong-document top-5 | 4/60 = 0.067 |
| Calibration 010 pre-fallback / actual evidence rank | 6 / 5 |

The old fallback failed because it counted `PIN code` and unrelated menu labels across whole chunks.
The scoped rule compares only the direct MODE-key switch clause: the conditional candidate lists two
targets after “switch”, while the direct candidate lists three. It reorders only ranks 5–10 and moves
010 from 6 to 5. The same rule leaves 009 at rank 2 and passes all VI/EN and contamination gates.

The active offset-40 runtime plus exact cross-document evidence selection measures wrong-document
evidence `4/60 = 0.067`, candidate recall `12/12`, Hit@5 `12/12`, MRR@5 `0.892`, and 010 at actual
rank 5. The one-pass real closure measured rerank p95 `11.178 s`; reranker CPU latency remains a
production limitation, but the subsequent three-run provider stability gate passes.

## Commands and artifacts

Executed provider-free commands:

```powershell
python -m ruff check .
python -m pytest -q --basetemp .pytest-tmp-closure-full1
docker compose config --quiet

docker compose --profile tools run --rm --no-deps `
  -v "${PWD}:/workspace" -w /workspace ingestion `
  python -m scripts.create_phase7_reranker_snapshot

python -m scripts.calibrate_phase7_role_prior
python -m scripts.generate_phase7_calibration_closure_readiness
```

Generated ignored artifacts:

| Artifact | Meaning |
|---|---|
| `phase-7-reranker-snapshot-v2.json` | One sanitized real-Jina replay source |
| `phase-7-role-prior-ablation-v2.json` | Finite rank grid plus one registered fallback |
| `phase-7-reranker-snapshot-v3.json` | Sanitized real-Jina snapshot with relation-scoped counts |
| `phase-7-relation-list-ablation-v1.json` | PASS replay: 010 rank 5 and all retrieval gates pass |
| `phase-7-contamination-closure-v5.json` | PASS real Qdrant/Jina runtime closure |
| `phase-7-heldout-readiness-v2.json` | Technical gates and governance block |
| `phase-7-calibration-005-diagnostic-v1.json` | Completed: 3/3 positive fact matches on one fixed evidence bundle |
| `phase-7-calibration-e2e-v5-run-{1,2,3}.json` | Completed: three independent sanitized Gemini calibration runs |
| `phase-7-calibration-stability-v1.json` | PASS worst-run aggregation: 12/12 facts in every run |

The diagnostic and stability artifacts exist after explicit approvals. The three V5 calibration runs
have identical frozen retrieval signatures; all three pass deterministic facts `12/12`, citation ID
validity `100%`, zero unsupported/wrong-document citations, and abstention precision/recall `1.000`.
Historical artifacts are not overwritten.

Final read-only validation passed the frozen runtime checker for all collections:

- `industrial_manual_chunks`: 99 points.
- `industrial_manual_chunks_v2`: 99 points.
- Phase 3--6 stable-ID hash:
  `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`.
- `industrial_manual_phase7_dense_v1`: 2,753 points.
- `industrial_manual_phase7_hybrid_v1`: 2,753 points.
- Phase 7 stable-ID hash:
  `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.

FastEmbed 0.8.0 emitted its known MiniLM mean-pooling compatibility warning during snapshot creation.
The version and existing indexed behavior remain frozen; this closure did not pin back to 0.5.1,
change pooling, or re-index. Compose also reported an existing orphan container named
`phase7-dev-validation`; it was not removed because cleanup was outside scope.

Canonical Python 3.11.15 container validation passed Ruff and `286` tests with the single known
Starlette/TestClient warning. The local `.venv` reports Python 3.13.5 and was used only for targeted
checks; it was not treated as canonical, deleted, or recreated.
Buildx history showed the latest ingestion build as `Completed`; no active build job remained. The
resident `com.docker.build` process is Docker Desktop's background service, not an active build.

## What must happen next

1. Review and commit the closure implementation; Codex does not commit automatically.
2. Treat 005 as closed for the current evaluator/provider path: 3/3 fixed-evidence attempts passed;
   do not reinterpret the unavailable historical v4 output.
3. Technical calibration is closed: all provider-free and three-run stability gates pass. Do not
   rerun it merely to seek a better metric.
4. Resolve governance with a new access-controlled final set or explicitly stop calling the existing
   held-out set unseen. Until then, held-out execution remains blocked regardless of technical pass.

After approval, use distinct checkpoints and outputs; never reuse a completed checkpoint as another
independent run:

```powershell
python -m scripts.evaluate_phase7_e2e --dataset calibration `
  --provider-approval-token "APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS" `
  --checkpoint artifacts/metrics/phase-7-calibration-e2e-v5-run-1-checkpoint.jsonl `
  --output artifacts/metrics/phase-7-calibration-e2e-v5-run-1.json

# Repeat with run-2 and run-3 in both path arguments.
python -m scripts.aggregate_phase7_calibration_stability `
  --run artifacts/metrics/phase-7-calibration-e2e-v5-run-1.json `
  --run artifacts/metrics/phase-7-calibration-e2e-v5-run-2.json `
  --run artifacts/metrics/phase-7-calibration-e2e-v5-run-3.json
```
