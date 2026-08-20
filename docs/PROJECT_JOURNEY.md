# Industrial Technical Manual RAG — engineering journey

This is a reconstruction from repository history and measured artifacts. It distinguishes historical
measurements from current contracts rather than rewriting old results.

## The problem

The project answers factual questions from technical manuals while keeping retrieval evidence,
citations, and abstention explicit. A useful answer is not enough: the system must show which chunk
supports it, avoid inventing unsupported sources, and be reproducible when a document is re-indexed.

## Evolution by phase

| Phase | Key commit | Engineering decision |
|---|---|---|
| 1 | `b5549b8` | Established FastAPI health, settings, pytest/Ruff, Compose, and CI. |
| 2 | `71da1aa` | Added Docling parsing, structure-aware chunks, PDF batches, and atomic JSONL. |
| 3A | `7677913` | Added multilingual MiniLM dense embeddings and Qdrant dense search. |
| 3A.1 | `01df8f6` | Made chunk IDs content-stable, point IDs UUIDv5, and re-indexing safe. |
| 3A.2 | `43bf976` | Hardened direct-evidence qrels/evaluation and separated API/ingestion Docker dependencies. |
| 4 | `51ead18` | Added BM25 sparse vectors and client-side RRF in a separate v2 collection. |
| 4.1 | `9718926` | Audited candidate pools instead of assuming hybrid was best. |
| 5 | `07c074b` | Compared three Jina reranking pools with strict model-output validation. |
| 6 | `e3b3704` | Added query API, evidence gate, structured generation, citation validation, abstention, and Gemini/OpenAI routing. |
| 7 | `8267c4b` + working tree | Adds a real industrial corpus, calibration closure, sealed held-out evaluation, and portfolio-grade hardening. |

## Phase 7 checkpoint

Phase 7 does not overwrite the 99-chunk development corpus. It indexes 2,753 frozen chunks from
two ATV320 manuals in `industrial_manual_phase7_dense_v1` and
`industrial_manual_phase7_hybrid_v1`; their stable-ID hash is
`2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.
The first 20-row calibration exposed two evaluation lessons before the 45-row held-out set was run:
English evidence phrases cannot score Vietnamese generated answers, and one logical evidence block
can have multiple exact duplicate chunk IDs. Dataset v2 therefore separates reviewed answer facts
from qrel-validation phrases and expands only exact-content equivalents. All 42 answerable rows were
then source-reviewed, all 65 records approved, and the final v2 hashes frozen. The held-out outputs
remain unseen.

Dataset-v2 calibration improved direct retrieval over v1 but exposed two separate problems. Strict
contiguous phrase scoring marked calibration 002/008 wrong despite complete answer-fact token
coverage, while 004/005/006/010 were absent from the 20/20 candidate pool. Phase 7.4 therefore keeps
strict phrase accuracy only as a diagnostic and introduces deterministic typed fact scoring. A
sanitized rescore of the same provider outputs changes 6/12 strict matches to 8/12 deterministic
matches without editing expected phrases or qrels.

The retrieval closure does not widen the cross-encoder to 80 candidates. It retrieves dense@60 and
expanded sparse@40, augments only query terms through a fixed bilingual technical glossary, then uses
weighted rank-only RRF `k=40`, dense@5/sparse@24 coverage reserves, and a soft query-role prior within
the same 30-candidate budget. Canonical Python 3.11 calibration reaches candidate recall 12/12,
Hit@5 11/12, MRR@5 0.875 and zero wrong-document top-1 results. It remains `PARTIAL`: calibration 010
is rank 6 and wrong-document candidates still occupy 0.267 of final top-5 slots. The fresh provider
E2E run requires explicit data-egress approval; held-out remains sealed.

## What changed in the architecture

The first retrieval design was dense-only. Its important correction was that relevance must be a
stable direct-evidence chunk ID, never merely a matching page. This made Hit@k and MRR meaningful
and let later sparse, RRF, and reranker strategies be compared against the same frozen input.

Qdrant stores vectors plus trusted payload, not generated answers. The payload keeps chunk ID,
document ID, filename, pages, headings, content type, and text. Deterministic UUIDv5 point IDs make
the index repeatable; safe indexing embeds/upserts first and only then removes stale points for the
same document. That is why a failed re-index does not erase another document or the prior corpus.

Phase 4 did not merge raw cosine and BM25 scores: their scales are incomparable. It stored sparse
and dense vectors in a separate collection and used rank-only reciprocal-rank fusion. Candidate-pool
audit then found sparse was stronger for exact terms but dense contributed evidence absent from sparse.

Phase 5 therefore tested sparse, RRF-hybrid, and dense∪sparse rather than declaring hybrid a default.
The union reranker was best on the frozen development set: Hit@5 `0.767`, MRR@5 `0.546`, and
candidate recall `0.933`, with 3/3 critical bilingual intents in top 5. Its CPU p95 was about
11.9 seconds, so the metric result is not a production latency claim.

Phase 6 reused that retrieval behavior in the public query API. The safety order is fixed:

```text
retrieve → rerank → evidence gate → generate → validate source IDs → build trusted citations
```

The LLM receives labeled evidence but never controls citation metadata. One correction retry can
repair invalid source labels without retrieval drift; a second failure becomes a safe abstention.
Gemini uses its OpenAI-compatible endpoint and OpenAI uses Responses, but both pass through the same
schema/evidence/citation contract.

## Lessons useful in an interview

- Evaluation design is part of the product. Same-page matching inflated early retrieval metrics;
  direct qrels fixed that correctness bug.
- Candidate recall and ranking quality answer different questions. A reranker cannot recover a chunk
  absent from its candidate pool.
- Stable identities are the bridge between ingestion, Qdrant, qrels, citations, and regression tests.
- Dependency split matters operationally: the API image has retrieval+LLM dependencies but no Docling;
  ingestion is on-demand and heavy.
- A citation ID can be valid yet not semantically support every claim. Phase 6 guarantees the first;
  Phase 7 evaluates the second on a held-out set.

## Known limits and next decision points

The 30-query Vietnamese research-paper corpus is development/regression evidence only. The Jina model
is CC-BY-NC-4.0 and slow on CPU, so it is suitable for the current non-commercial demo but unresolved
for commercial deployment. OCR, multi-page-table continuity, calibrated abstention, semantic citation
coverage, authentication/rate limiting, provider privacy approval, and a representative industrial
corpus remain open work. Phase 7 must report a separate ATV320 held-out result without tuning on it.
