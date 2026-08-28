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

## Portfolio conclusions

- Correct evaluation labels are as important as the retrieval implementation.
- Candidate recall and ranking quality are separate failure modes.
- Stable identities make refactoring and reproducible evidence possible.
- Frozen product identity should be concrete (`ATV320`); reusable behavior should be generic.
- Archives preserve provenance without expanding the supported product surface.
- Documentation is now owned by a small set of living documents instead of per-step logs.
