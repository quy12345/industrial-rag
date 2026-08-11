# Phase 7 calibration closure walkthrough

## Outcome

The 2026-08-11 closure is intentionally `PARTIAL`, not a release pass. It improves evaluator
correctness, seals calibration from held-out reads, separates full reranker output from actual LLM
evidence, and removes exact cross-document duplicate evidence. It does not make calibration 010
direct evidence enter top 5, and calibration 005 still needs a separately approved diagnostic.

Held-out status is `BLOCKED_GOVERNANCE`. Historical tracked documentation mirrored held-out content,
and the old calibration CLI loaded both JSONL files. The current code closes those paths, but changing
the present revision cannot erase statistical exposure in Git history.

No qrel, chunk, embedding model, Jina model, Qdrant collection, volume, public Query API, or Docker
image was changed. No provider call, held-out run, re-index, image build, or prune was performed.
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

The real Qdrant/Jina snapshot-v2 run completed in 84.7 seconds in the existing Python 3.11 ingestion
image. It stores IDs, component/cross-encoder ranks, finite scores, document roles, exact-content
hashes, and sanitized list-feature counts. It stores no question or chunk text and reports zero
provider calls and zero held-out executions.

The finite grid tested RRF-rank multipliers `0`, `0.25`, `0.5`, `1`, and `2` with offsets `10`, `20`,
and `40`. The strongest provider-free profile preserves:

| Metric | Result |
|---|---:|
| Candidate recall | 12/12 = 1.000 |
| Hit@5 | 11/12 = 0.917 |
| MRR@5 | 0.875 |
| EN / VI Hit@5 | 6/6 / 5/6 |
| Wrong-document top-1 | 0/12 |
| Best replay wrong-document top-5 | 4/60 = 0.067 |
| Calibration 010 full rank / actual evidence rank | 6 / not present |

No rank-only profile moves 010 to top 5. The one pre-registered fallback,
`list_completeness_v1`, activates only for query-derived bilingual list intent and reorders ranks
5–10 by query-identifier coverage, bracketed label/code-pair count, and original rank. It also fails:
rank 5 contains `MODE` and more generic label/code pairs than the qrel at rank 6. The fallback is not
activated, and no query-ID/qrel/expected-fact-specific rule was added.

The active offset-20 runtime plus exact cross-document evidence selection measures wrong-document
evidence `7/60 = 0.117`, candidate recall `12/12`, Hit@5 `11/12`, MRR@5 `0.875`, and 010 outside
actual top 5. Therefore technical readiness remains false.

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
| `phase-7-heldout-readiness-v2.json` | Technical gates and governance block |
| `phase-7-calibration-005-diagnostic-v1.json` | Pending separate 005 provider approval |
| `phase-7-calibration-e2e-v5-run-{1,2,3}.json` | Pending three independent approved runs |
| `phase-7-calibration-stability-v1.json` | Pending worst-run aggregation |

The diagnostic and stability artifacts do not exist until their provider runs are explicitly
approved. Historical artifacts are not overwritten.

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

Canonical Python 3.11.15 container validation passed Ruff and `279` tests with the single known
Starlette/TestClient warning. The local `.venv` reports Python 3.13.5 and also passed 279 tests, but
it was treated only as a secondary check and was not deleted or recreated.
Buildx history showed the latest ingestion build as `Completed`; no active build job remained. The
resident `com.docker.build` process is Docker Desktop's background service, not an active build.

## What must happen next

1. Review and commit the closure implementation; Codex does not commit automatically.
2. Approve 005 separately with `APPROVE PHASE 7 CALIBRATION 005 DIAGNOSTIC EGRESS` if sending that
   question and fixed excerpts to the configured provider is acceptable.
3. Do not select the best of three outputs. Classify 005 as evaluator, generation, ambiguity, or
   provider instability from all attempts.
4. If technical gates can be resolved without qrel/model/dataset gaming, grant the separate full-run
   token `APPROVE PHASE 7 CALIBRATION V5 STABILITY EGRESS` and create three independent outputs.
5. Aggregate them with `scripts.aggregate_phase7_calibration_stability`; the worst run must reach
   11/12 facts and every hard citation/abstention gate.
6. Resolve governance with a new access-controlled final set or explicitly stop calling the existing
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
