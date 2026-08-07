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
| 7 | working tree | Adds a real industrial corpus, held-out evaluation, and portfolio-grade hardening. |

## Phase 7 checkpoint

Phase 7 does not overwrite the 99-chunk development corpus. It indexes 2,753 frozen chunks from
two ATV320 manuals in `industrial_manual_phase7_dense_v1` and
`industrial_manual_phase7_hybrid_v1`; their stable-ID hash is
`2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`.
The 20-row calibration and 45-row held-out sets are approved and hash-locked. The E2E evaluator
scores qrel-only retrieval, phrase presence, citations, abstention, and latency without persisting
raw provider content. A corpus owner must separately authorize provider data egress before the real
benchmark runs.

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
