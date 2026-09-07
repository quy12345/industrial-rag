# Engineering journey

This document preserves the project's compact development narrative and its bug-to-fix evidence.
It does not replace current architecture or operational instructions; see
[ARCHITECTURE.md](ARCHITECTURE.md) and [OPERATIONS.md](OPERATIONS.md).

Historical measurements below describe the exact artifacts that produced them. They are not claims
about a new run, and the exposed held-out v2 split remains regression-only evidence.

## Milestones

| Milestone | Representative commit | Durable outcome |
|---|---|---|
| Project foundation | `b5549b8` | FastAPI health, settings, tests, Ruff, Compose, and CI |
| Structured ingestion | `71da1aa` | Docling parsing, page batching, structure-aware chunks, atomic JSONL |
| Dense retrieval | `7677913` | Multilingual MiniLM embeddings and Qdrant dense search |
| Stable identity | `01df8f6` | Content-stable chunk IDs, UUIDv5 point IDs, safer re-indexing |
| Direct-evidence evaluation | `43bf976` | Qrels tied to exact evidence chunks instead of pages |
| Sparse and fusion retrieval | `51ead18`, `9718926` | BM25 vectors, rank-only RRF, and candidate-pool audit |
| Cross-encoder reranking | `07c074b` | Bounded union reranking and strict output validation |
| Grounded query API | `e3b3704` | Evidence gate, provider ports, citation validation, correction, abstention |
| ATV320 corpus | `8267c4b` and later commits | Two-manual corpus, frozen evaluation data, retrieval closure, regression artifacts |
| Round 1 cleanup | R00-R07 commits | Modular ownership, configuration/composition boundaries, thin adapters, archives |
| Surface simplification | R08-R13 commits | Canonical imports and direct supported command owners; legacy facades removed |
| Reproducible ingestion | `d88a2b2` | Locked transitive document stack and recovered exact 2,753-chunk identity |
| Active naming cleanup | `e94b862` to `39ddc13` | Generic runtime names, ATV320 contract, current evaluation package, phase-free CLI paths |

## Why the architecture evolved

Dense retrieval was a useful starting point, but direct evidence established the first meaningful
evaluation contract: a page match is not the same as retrieving the chunk that supports an answer.
Stable chunk identity then connected parsing, Qdrant payloads, qrels, citations, and regression
artifacts.

Sparse retrieval improved exact technical terms, while dense retrieval contributed evidence sparse
search missed. Raw scores were not merged because cosine and BM25 scales are incomparable; rank-only
reciprocal-rank fusion preserved their ordering signals. Candidate audit showed that reranking cannot
recover evidence absent from its input pool, which led to a bounded dense/sparse union before the
cross-encoder.

The query layer added a fixed safety order:

```text
retrieve -> rerank -> select evidence -> evidence gate -> generate
         -> validate source labels -> build trusted citations or abstain
```

The provider can reference supplied source labels, but it cannot invent trusted citation metadata.
The application reconstructs citations from retrieved chunks and fails safely after one invalid
citation-correction attempt.

The ATV320 corpus later expanded the system to two real manuals and 2,753 frozen chunks. Historical
calibration artifacts remain useful evidence, but completed research commands were archived rather
than treated as permanent product APIs. Round 1 then separated production, evaluation, and scripts;
the later naming cleanup removed phase terminology from active semantics without migrating frozen
data or physical Qdrant collections.

## Historical evaluation checkpoint

The frozen calibration retrieval artifact reports candidate recall `12/12`, Hit@5 `11/12`, MRR@5
`0.875`, and zero wrong-document top-1 results for its recorded source identity. It also records the
remaining weakness: one relevant result at rank 6 and wrong-document candidates in the final top 5.
These values are historical artifact evidence, not a fresh run performed by this documentation
change.

The one-shot held-out-v2 workflow was executed only after explicit approval and was then archived.
Because its payload and results now exist in the repository, it is exposed regression evidence and
must not guide new tuning.

## Bug and correction ledger

Each entry uses the same review format: **Symptom -> Root cause -> Fix -> Proof -> Commit/artifact**.

### Same-page qrels inflated retrieval quality

**Symptom** -> A result could count as correct while returning an unrelated chunk from the right
page. **Root cause** -> Early qrels used page identity as a proxy for evidence. **Fix** -> Freeze exact
direct-evidence chunk IDs and score only those IDs. **Proof** -> Deterministic qrel tests distinguish
same-page distractors from relevant chunks. **Commit/artifact** -> `43bf976` and the frozen direct
qrels.

### Sparse evidence disappeared before reranking

**Symptom** -> Exact technical evidence found by sparse search did not always reach the reranker.
**Root cause** -> A fused cutoff could demote a candidate unique to one retrieval channel. **Fix** ->
Audit candidate pools, preserve bounded dense/sparse coverage, and rerank the explicit union.
**Proof** -> Golden candidate-order and coverage-policy tests plus the recorded retrieval calibration.
**Commit/artifact** -> `9718926`, `07c074b`, and the retrieval-closure artifact.

### Answers could outlive weak evidence or invalid citations

**Symptom** -> Generation alone could produce a fluent answer with insufficient support or unknown
source labels. **Root cause** -> Evidence sufficiency, structured generation, and citation trust were
not separate decisions. **Fix** -> Gate before generation, validate every returned source ID, allow one
correction attempt, then abstain. **Proof** -> Query-service tests cover blocked generation, corrected
citations, repeated invalid labels, and safe abstention. **Commit/artifact** -> `e3b3704` and current
application characterization tests.

### Configuration and source identity drifted

**Symptom** -> A diagnostic could appear comparable while using partially different runtime settings
or source files. **Root cause** -> Frozen behavior was duplicated across environment selectors and
artifact provenance lists. **Fix** -> Centralize the ATV320 retrieval/fusion contract, reject partial
overrides, and version evaluation source identity. **Proof** -> Configuration, composition, and
provenance tests fail closed on mismatches. **Commit/artifact** -> R02, ADR-005, E2E source identity
v3.

### Compatibility facades left stale imports

**Symptom** -> Multiple `app.*` and top-level script paths appeared authoritative after ownership had
moved. **Root cause** -> Temporary compatibility modules outlived their migration window. **Fix** ->
Move consumers to canonical owners, version provenance, then hard-cut facades and command shims.
**Proof** -> Architecture/import inventory tests require old paths to fail and prevent private
cross-script imports. **Commit/artifact** -> R12-R13 and ADR-005.

### Docling transitive drift changed nine chunk IDs

**Symptom** -> A fresh run still produced 2,753 chunks but its ordered ID hash differed in nine
positions. **Root cause** -> `docling` matched while a transitive `docling-core` version differed.
**Fix** -> Lock the complete ingestion dependency set used by the frozen corpus. **Proof** -> A new
preview reproduced both count `2753` and hash
`2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.
**Commit/artifact** -> `d88a2b2` and `artifacts/metrics/phase-7-corpus-manifest.json`.

### Pip reported a hash mismatch during the ingestion build

**Symptom** -> Docker failed with an expected/received wheel SHA-256 mismatch. **Root cause** -> A
downloaded package payload was incomplete or inconsistent with the locked file; it was not evidence
that repository pins should be relaxed. **Fix** -> Preserve hash checking and retry only the failed
build/download cache path. **Proof** -> The rebuilt image reports the locked Docling stack and later
reproduces the frozen chunk hash. **Commit/artifact** -> reproducible-ingestion build evidence and
`d88a2b2`.

### Compose rejected or misresolved the ingestion mount

**Symptom** -> Compose returned `invalid spec: :/tmp/chunk-check`, or ingestion could not find the PDF.
**Root cause** -> The PowerShell mount variable was empty in a new shell, while the container PDF path
was `/data/raw` rather than a working-directory-relative path. **Fix** -> Initialize and verify the
temporary directory, use a quoted volume specification, and use the container mount paths. **Proof**
-> Preview mode completed and wrote its outputs outside the repository. **Commit/artifact** ->
[OPERATIONS.md](OPERATIONS.md) reproduction recipe.

### A warning copied a large manual excerpt into logs

**Symptom** -> Docling's oversized-heading warning included a long technical-manual passage. **Root
cause** -> A dependency formatted the full chunk object into its warning. **Fix** -> Sanitize known
document warnings at the infrastructure boundary while preserving a concise warning and failure
signals. **Proof** -> `tests/test_ingestion.py` verifies that manual content is absent while the
warning type remains observable. **Commit/artifact** -> `d88a2b2`.

### Equal chunk counts hid a non-reproducible corpus

**Symptom** -> Baseline and fresh runs both reported 2,753 chunks, yet the stable-ID hashes differed.
**Root cause** -> Count-only validation ignored content-boundary and metadata changes. **Fix** -> Treat
document hashes, runtime versions, count, and ordered chunk-ID hash as one reproduction contract.
**Proof** -> The final run matched all 2,753 IDs and the frozen hash exactly. **Commit/artifact** ->
`d88a2b2` and the corpus manifest.

### Retrieval evaluation hashed deleted source paths

**Symptom** -> Evaluation provenance referred to three root modules removed during the canonical-owner
migration. **Root cause** -> Source identity was not updated when compatibility facades were deleted.
**Fix** -> Hash `evaluation/e2e.py`, `app/domain/policies/query_analysis.py`, and
`app/composition/retrieval.py`, then require every mapped path to exist. **Proof** -> Offline tests
verify path existence, deterministic hashes, and source identity v3. **Commit/artifact** -> `e3bcc4b`
and the v7 E2E artifact contract.

### Public test was mistaken for the sealed held-out-v2 split

**Symptom** -> A fresh `--dataset test` run produced a different dataset hash and materially lower
retrieval metrics than the historical held-out-v2 artifact. **Root cause** -> The current CLI exposed
the public 45-row test split, while historical held-out v2 lived under a separate private root with a
different sealed manifest; equal row counts hid the identity mismatch. **Fix** -> Add an explicit
`heldout-v2` choice, require its dataset and manifest to remain under the private root, validate the
sealed `c91cf3e0...` dataset hash, and use separate v7 output/checkpoint paths. **Proof** -> The fresh
Python 3.11 v7 artifact matches all 45 historical per-query retrieval records exactly (canonical
retrieval-record SHA-256 `889556304e1be22845cffabfd98ab629d1f04f27ad759c8888f49412a29a342f`).
Generation metrics varied, while corpus identity and aggregate retrieval metrics remained identical.
**Commit/artifact** -> `artifacts/metrics/atv320-heldout-v2-e2e-v7.json`; implementation is grouped
under the Conventional Commit `feat: enable sealed heldout v2 regression runs`.

### Ragas semantic evaluation needed gateway-specific async wiring

**Symptom** -> The first semantic preflight rejected a synchronous OpenAI client, then OpenRouter
reported no eligible Luna endpoint even though the OpenAI upstream was available. **Root cause** ->
Ragas Collections metrics await their LLM and embedding clients, while OpenRouter's strict parameter
routing did not advertise `max_completion_tokens` for that upstream. **Fix** -> Use `AsyncOpenAI`,
make the embedding adapter asynchronous, send the supported `max_tokens` request parameter, and keep
OpenAI-only routing with fallback disabled. **Proof** -> Two five-item EN/VI calibration runs and one
45-item held-out-v2 run completed with Ragas `0.4.3`, Luna, and zero semantic failures. The held-out
run scored Faithfulness `1.0`, Answer Relevancy `0.8204`, and Context Precision without reference
`0.9753` across 27 eligible answers; 18 abstentions were excluded rather than scored as zero.
Dataset, corpus, runtime profile, and every candidate/rerank/evidence boundary matched v7. The
artifact SHA-256 is `c975ef742c44bc73a2986ed622b3d26d7054b8d900214a955b5ce461a3bb260f`.
**Commit/artifact** -> `artifacts/metrics/atv320-heldout-v2-e2e-v8.json`; proposed Conventional Commit
`feat: add reproducible ragas semantic evaluation`.

### Repeated evidence lost an intermediate source identity

**Symptom** -> Same-document exact duplicates were collapsed before reranking, then a later
cross-document collapse retained only the representative and the other manual's chunk ID; an
intermediate page/source ID disappeared from diagnostics. **Root cause** -> The evidence selector
rebuilt `equivalent_chunk_ids` from immediate candidates instead of flattening provenance produced
by the earlier deduplication boundary. It also reported duplicate groups that were outside the final
generation context. **Fix** -> Carry compact, text-free provenance through both boundaries, flatten
it deterministically by chunk ID, and expose diagnostics only for selected evidence. **Proof** ->
Offline characterization tests cover nested provenance, selected-only groups, unchanged candidate
ordering, and the public `top_k=10` behavior; the provider-free schema-v4 evaluator records strict
and exact-content-equivalent evidence outcomes without raw content. **Commit/artifact** -> proposed
commits `fix: preserve complete evidence provenance` and `feat: trace evidence selection boundaries`;
no integration artifact is claimed yet.

### Claim-level generation did not improve the Qwen development baseline

**Symptom** -> A five-item preflight suggested that a claim-level structured answer might preserve
more required facts, but that sample was too small to select a generation contract. **Root cause** ->
`claim-coverage-v1` can validate the claims and source labels that the model emits, but it cannot
know which reviewed expected facts the model omitted. Its stricter per-claim support rule also made
Qwen more conservative, especially for Vietnamese and cross-document questions. **Fix/decision** ->
Reject promotion of `claim-coverage-v1` and retain `baseline-v1` only as the current control. This is
not a quality approval: the baseline still misses the development candidate-recall, Hit@5,
fact-completeness, answerable-coverage, and wrong-document-citation promotion targets. Do not run confirmation
or held-out evaluation from this result; investigate retrieval coverage, evidence provenance, and
the reviewed development failure cohorts first. **Proof** -> One full, sanitized, single-repeat A/B
run used the same 120 development items, Qwen model, corpus identity, and identical retrieval and
evidence fingerprints. Baseline produced 46 complete answers out of 80 answerable items, `64.81%`
fact coverage, four false abstentions, and three false answers. Claim coverage produced 45 complete
answers, `62.35%` fact coverage, seven false abstentions, and two false answers. Both had `80.00%`
candidate recall, `76.25%` Hit@5, two wrong-document citations, no unsupported citation IDs, and
`100%` referentially valid citation IDs. Paired complete-answer outcomes favored baseline on four
items and claim coverage on three (`McNemar p=1.0`); a 10,000-sample fact-family clustered bootstrap
estimated the claim-coverage minus baseline complete-answer difference at `-1.25` percentage points
with a 95% interval of `[-7.89, +5.36]`. This single repeat provides no basis to promote the experimental
variant, and it cannot declare the retained baseline production-ready or superior to a generator that
has not received the same full evaluation. **Commit/artifact** -> local ignored artifacts
`q04-qwen-development-baseline-v1-r1-e2e-v11.json` (SHA-256 `c8795667ade92b5e36b5b16ae42ba66121f568c3468a770a581b201948dc1ebb`)
and `q04-qwen-development-claim-coverage-v1-r1-e2e-v11.json` (SHA-256
`99f57d0523174105491295cc70e5aa5b709bf72acc4dc1048b0ccbe54d813241`); Qwen non-thinking endpoint
support is commit `e034469`.

### Deterministic fact matching understated reviewed multilingual answers

**Symptom** -> The 120-item development baseline answered 76 questions, but the v2 matcher passed
only 105 of 151 reviewed facts; 21 of 46 misses still had alias-token recall of at least `0.8`, and
human review confirmed false negatives for ESC behavior and a Vietnamese protective-cover
procedure. A separate TBR case exposed the opposite risk: seeing a parameter and `19.2` did not
prove the unasked relation “default.” **Root cause** -> Fact matching lived inside the E2E
orchestrator, split clauses at decimal points, treated EN/VI function words as technical evidence,
and had no bounded semantic adjudication path for valid paraphrases. One reviewed annotation also
required a default value outside the question's scope. **Fix** -> Isolate deterministic matching in
`evaluation/facts.py`; version decimal-safe, function-word-tolerant, clause-local matching as v3;
keep exact numeric/identifier checks authoritative; and add an approval-gated, one-call-per-item
Luna fallback for unresolved text facts. Preserve deterministic and final verdicts separately in
schema v13, with no raw evaluation content in artifacts. **Proof** -> Synthetic offline tests cover
decimal `19.2`, EN/VI function-word variation, reviewed Vietnamese word order, negation,
relation-negative controls, batched fallback routing, exact verdict IDs, sanitization, approval
separation, and v12 checkpoint rejection. The local development annotations and manifest were
revalidated against the frozen corpus. No live Luna result or quality improvement is claimed until
the separately approved preflight runs. **Commit/artifact** -> E02 implementation pending review;
future evidence must use a sanitized schema-v13 artifact.

### Strict provenance treated reviewed duplicate evidence as an error

**Symptom** -> Development diagnostics reported a wrong-document citation even when the cited chunk
was a reviewed authoritative alternative or a frozen-corpus exact-content duplicate. Conversely, a
citation from the expected manual could pass without supporting the reviewed fact, and multi-source
qrels were treated as one flat OR list. **Root cause** -> Release gates used document identity and
flat qrel membership rather than reviewed fact-level evidence requirements. **Fix** -> Model
canonical and explicitly reviewed alternative chunks as OR choices within an evidence requirement,
combine requirements with AND, and classify citation provenance as canonical, reviewed alternative,
exact-content equivalent, or unreviewed. Schema v14/v5 gates authoritative coverage and rejects an
unreviewed citation; strict document/qrel metrics remain diagnostics. Numeric facts also distinguish
only the reviewed formatting equivalence (`20`, `20.0`, `20,0`). **Proof** -> Offline tests cover
cross-manual reviewed alternatives, unreviewed same-manual chunks, exact duplicates, partial
multi-source coverage, and checkpoint version rejection. No new development, confirmation, or
held-out evaluation has been run, so no changed score is claimed. **Commit/artifact** ->
`fix: correct deterministic fact evaluation contracts`, `feat: model reviewed authoritative evidence`,
and `fix: align evaluation gates with authoritative coverage`.

### Rule-heavy evaluation obscured the product outcomes

**Symptom** -> Schema v14 grew to about 26.3 MB and mixed retrieval traces, strict provenance,
typed fact matching, generation experiments, and release gates in one E2E artifact. **Root cause**
-> Each investigation added another permanent rule and payload instead of keeping deep diagnostics
separate from the small set of user-facing quality outcomes. **Fix** -> E04 replaces the active E2E
contract with a compact schema-v15 Ragas scorecard for answer utility, unsafe-answer rate, citation
faithfulness, and final-context recall. Retrieval traces remain in the provider-free retrieval
evaluator. A fixed metadata-selected 40-item cohort and private human labels must reach 85% judge
agreement before any full development baseline. **Proof** -> Offline tests validate scorecard
arithmetic, raw-content exclusion, the 30/10 EN/VI cohort, private review resume, and confusion
matrices. The completed live cohort produced a 95,622-byte sanitized artifact; subsequent human
review found 24.14% answer-correctness agreement, 68.97% citation-support agreement, and 100%
agreement for both context completeness and abstention. **Commit/artifact** -> implementation
commits `refactor: simplify end-to-end evaluation around ragas` and
`feat: add human scorecard validation`; sanitized agreement artifact
`artifacts/metrics/atv320-judge-validation-v1-agreement.json`.

### Custom outcome names hid the underlying Ragas metrics

**Symptom** -> The schema-v16 artifact exposed `answer_utility`, `citation_faithfulness`, and
`final_context_recall`, so readers could reasonably mistake project-specific aggregation for metric
names defined by Ragas. **Root cause** -> The scorecard combined judge scores and deterministic
abstention logic behind a four-outcome presentation layer. **Fix** -> Schema v17 reports Factual
Correctness, Faithfulness, Context Recall, Answer Relevancy, and Context Precision without reference
under their exact Ragas names. Answerable-response, unsafe-answer, and citation-reference validity
remain plainly labeled deterministic rates in a separate group; `answer_utility` is removed instead
of renamed. **Proof** -> Offline scorecard tests verify the exact metric inventory, applicability
populations, EN/VI aggregation, citation-validity counting, and absence of the legacy `outcomes`
field. The completed v16 development artifact remains immutable historical evidence; the subsequent
20-case critical v17 run is a failure-selected diagnostic, not a new full-development result.
**Commit/artifact** -> proposed `refactor: report standard ragas metric names`;
`artifacts/metrics/atv320-development-critical-e2e-v17.json`.

### Atomic fact aliases were lossy Ragas references

**Symptom** -> In the 40-item schema-v15 judge-validation cohort, human agreement was 24.14% for
answer correctness and 68.97% for citation support, despite 100% agreement for context completeness
and abstention. A representative correct answer preserved the manual's “before handling” condition,
while its reference alias retained only “the product has sufficiently cooled down.” **Root cause**
-> Matcher aliases were rendered as bullet fragments and reused as if they were complete reference
answers; this discarded conditions and relations before Ragas claim decomposition. **Fix** ->
Schema v16 gives each answerable item an independent natural reference answer with a separate human
review status, retains atomic facts for audit, and refuses provider egress until selected references
are approved. Generated answers are not reused as ground truth. **Proof** -> Characterization tests
cover the complete cooling/handling relation, missing approval fails closed, and all 80 EN/VI
development references were independently approved in the private review pack. **Commit/artifact** -> implementation pending commit; the
historical sanitized agreement artifact is
`artifacts/metrics/atv320-judge-validation-v1-agreement.json`.

### Cited headings and fixed-answer judge replay (Batch A)

**Symptom** -> Critical review v17 found correct table answers whose scope was carried by headings,
and human-correct answers with low or zero F1. **Root cause** -> Faithfulness received bare chunk
text, while human assessment interpreted `FC == 1` as a correctness verdict. Regenerating answers
would confound evaluation-input correction with generator variance. **Fix** -> V18/source identity
14 uses verified title/heading/full text for cited final evidence only; Context Recall retains the
exact bounded generator context. Development-only judge replay imports fixed answers, and future
private review runs capture identity-bound inputs before judging. Assessment v2 reports continuous
FC by human labels and diagnostic support tables with `REVIEW_REQUIRED`, not overall PASS/FAIL.
**Proof** -> Offline validation imported all 20 original v17 review inputs against dataset/corpus
without generator or judge calls. Synthetic tests cover source mismatch before egress, fixed-answer
resume after judge failure, preservation of edited labels, and sanitized outputs. **Commit/artifact**
-> implementation pending review/commit; source artifact
`artifacts/metrics/atv320-development-critical-e2e-v17.json`, SHA-256
`ec475e7ee3ffcca92292a4d91a5954a4a0b829745df2b31e20f1e0840e6cb7a1`.

Offline validation: Ruff passed; 465 pytest tests passed on the local Python 3.13.5 interpreter
(one existing Starlette/httpx deprecation warning). Compose configuration and `git diff --check`
passed; 15 local Markdown links resolved. Compose also warned that the sandbox could not read the
user Docker config. Frozen chunk and development dataset hashes, plus the v17 source artifact hash, remain
unchanged. Production/application, UI, dataset and Docker configuration files were not edited.

The subsequently approved evaluation-only rebuild completed with image ID
`sha256:50e40e461411446da5f6ec43540e95b814ecb2a569b2a0be0bd9759a972f6e67`.
Python 3.11.16 validated Ragas 0.4.3/OpenAI 2.53.0, all five Collections imports, adapter
construction using a dummy key, and both E2E/assessment CLI help commands. In a network-disabled
container, 438 tests passed with one Starlette/AnyIO deprecation warning. The 27 tests in
`tests/test_ingestion.py` were excluded because this image deliberately has no Docling; the first
full collection attempt failed for that missing dependency, not an ingestion behavior failure.
Pytest's existing pure-Python tooling and required test fixtures were mounted read-only for this
check, without adding test dependencies to the image. No credentials, PDFs or Qdrant volumes were
mounted; no provider scoring, model download, ingestion build or runtime-service restart occurred.

Two separately approved live v18 replays then judged the same 20 legacy inputs without generation,
Qdrant, retrieval models or optional diagnostics. Each completed all 20 cases and 43 core metric
calls. R1 artifact SHA-256 is
`6183008e6e321400aa14fd63aae24076e6c71ce86eb6d46ffa015a4952fb2b36`; R2 is
`0ec03f7e62984c2f9734c6e27a4538dbc469bc575003cb60dd571f550e3e4945`.
Both retained all v17 retrieval/evidence fingerprints and passed raw-content scans. Exact provider
billing was unavailable from the adapter and must not be inferred from stored generation usage.

Context Recall was exactly stable: mean `0.6134`, with all 17 applicable item scores unchanged
across v17/R1/R2. The five human-identified heading-dependent cases (`dvi005`, `dvi006`, `dvi008`,
`dvi010`, `dvi068`) improved in mean Faithfulness from `0.1500` in v17 to `0.9000` and `1.0000` in
R1/R2. Overall Faithfulness moved from `0.5229` to `0.7473` and `0.8883`. This supports the cited
heading input correction, not a production quality claim. Faithfulness still showed judge variance:
`dvi012` scored `0.0` then `1.0`, while `dvi006` scored `0.5` then `1.0`.

Factual Correctness, whose answer/reference inputs did not change, averaged `0.6162` and `0.5585`
versus v17's `0.5392`. Seven of 13 applicable items changed between R1 and R2; mean absolute delta
was `0.2115`, with `dvi032` moving from `0.0` to `1.0` and `dvi021` from `0.67` to `0.0`.
Therefore it remains a noisy continuous diagnostic and cannot serve as a binary correctness gate.
Context misses and all behavior outcomes remained unchanged; replay cannot repair retrieval or
generation failures. Private label files for this critical pack still contain null templates, so no
assessment-v2 human aggregation is claimed from them.

### Development retrieval variant diagnosis (Batch B)

**Symptom** -> The critical cohort contained answerable misses whose authoritative evidence was
absent from the final top five, but aggregate E2E scores could not identify the responsible stage.
**Root cause** -> A separately approved, provider-free schema-v5 run over all 80 answerable
development items located the first incomplete authoritative boundary: 10 items at bounded
candidate selection, four at the normal dense/sparse component limits, two at raw cross-encoder
top five, two at the role/list policy, and one (`dvi045`) outside both diagnostic component top
100 lists. Sixty-one items remained complete. Same-document deduplication, pruning, and the final
evidence selector introduced no first losses. **Fix** -> All six preregistered variants were
compared pairwise without changing the production default. Refill changed no coverage;
query-expansion v2 rescued only `dvi028`; the dense-reserve variant regressed; and the
sparse-reserve variant rescued `dvi024` plus candidate coverage for `dvi044`, but only one final
evidence miss and fewer than the four-item shortlist minimum. The decision is therefore to keep
the baseline and not promote a retrieval variant. **Proof** -> Each sanitized artifact contains 80
rows, the same dataset/corpus identities, zero provider/held-out/confirmation calls, and no raw
question, answer, fact alias, or evidence text. Complete candidate/evidence rates were
`0.8125/0.7625` for baseline, `0.8250/0.7750` for query-expansion v2, `0.8000/0.7250` for
dense reserve, and `0.8375/0.7750` for sparse reserve. `dvi026` had five highly repetitive final
chunks (pairwise token Jaccard up to `0.965`) while its qrel remained outside the candidate budget.
For `dvi071` and the Installation facet of `dvi076`, authoritative evidence ranked in raw
cross-encoder top three but the role policy moved it below top five. **Commit/artifact** -> no code
commit and no production promotion; ignored local artifacts
`artifacts/metrics/atv320-development-retrieval-*-v5.json`, with baseline SHA-256
`ee5c156d6d8a30f16c00497f36e8a0c35e1426f74465e66b9a8355caa9e03145` and final sparse-reserve
SHA-256 `ea3d9d6767b4f81244d7cf246ef96c14971614870caafec4bbcf74bd0123d1db`.

The separately approved Batch C screen completed without a production promotion. Keep
evaluation-contract correction, stochastic judge/generator variance, and validated production
improvement as distinct claims.

### Bounded scope-aware generation experiment (Batch C)

**Symptom** -> Human review found answers that accepted an unsupported premise, mixed the right
identifier with the wrong purpose, overstated missing evidence as absence from the manual, or
assembled a safety procedure without all necessary conditions. **Root cause** -> The production
prompt strongly enforces grounding and source IDs but does not state these scope and partial-answer
rules explicitly. **Fix** -> One generic `scope-aware-v1` prompt was tested in the evaluator on the
20-case critical cohort. It contained no development IDs or case-specific terminology, and its
policy/hash was isolated from production checkpoints. **Proof** -> All 20 retrieval/evidence
fingerprints matched the production v17 source, but behavior did not improve: answerable response
remained `13/17`; the same four answerable items abstained; and all three unanswerable items were
still answered (`unsafe_answer_rate=1.0`). Factual Correctness fell to `0.4862`, compared with
`0.5585-0.6162` in the two fixed-answer production replays; Faithfulness `0.8141` remained inside
observed judge variation, and Context Recall was effectively unchanged at `0.6176`. The candidate
therefore failed the preregistered critical screen and was not run on full development. Its prompt,
selector, tests, and current-operation documentation were removed instead of leaving rejected
experiment code active. **Commit/artifact** -> no production commit or promotion; ignored sanitized
artifact SHA-256 `3ea36362352afe112490a5fbdff4e93f44144f240d7e1f6fca8b63d980f5e587` and private review pack
under `artifacts/local-review/development-critical-scope-aware-v1-v18/`.

### Deterministic protected-scope guard (Q06)

**Symptom** -> The generator combined nearby but distinct credential, pairing, QR, and product-data
facts into unsupported relations. **Root cause** -> Citation identity alone did not require the
protected technical scope named in a question to occur in the rendered final evidence. **Fix** -> A
small bilingual, boundary-safe registry checks only questions with at least two protected concepts,
first over the exact rendered final evidence and then over validated cited sources. The new
development-only scope replay validates historical live artifact/input identities separately from
the target policy and records no raw content. **Proof** -> Offline policy, truncation, citation, and
replay tests cover the behavior; no provider, Qdrant, or held-out run occurred. A scope pass remains
lexical presence only and is not an entailment verdict. **Commit/artifact** -> `571d51c`;
development scope replay is pending explicit execution approval.

## Portfolio conclusions

- Correct evaluation labels are as important as the retrieval implementation.
- Candidate recall and ranking quality are separate failure modes.
- Stable identities make refactoring and reproducible evidence possible.
- Frozen product identity should be concrete (`ATV320`); reusable behavior should be generic.
- Archives preserve provenance without expanding the supported product surface.
- Documentation is now owned by a small set of living documents instead of per-step logs.
