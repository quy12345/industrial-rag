# Phase 7 dataset review — pending human approval

This review is deliberately separate from the Phase 3A.2 30-query development set. It will use the
two frozen ATV320 manuals only. No held-out metric is reported before approval.

## Review gate

The generated datasets must satisfy these fixed counts before approval:

| Dataset | Answerable | Unanswerable | Total | Allowed use |
|---|---:|---:|---:|---|
| Calibration | 12 | 8 | 20 | Configure evidence/abstention policy only |
| Held-out test | 30 | 15 | 45 | One frozen final benchmark only |

The held-out answerable rows will cover installation/safety/wiring/specification, programming/
parameters/diagnostics, and cross-document selection. Vietnamese and English questions are recorded
explicitly; at least ten held-out answerable rows must be Vietnamese query to English evidence.

Each row will show the following compact review fields after the corpus preview has finished:

```text
ID | question | language/scenario/type | answerable | expected document/page
direct-evidence chunk IDs | expected phrases | human grounding note | review status
```

Ground truth is stable `relevant_chunk_ids` only. Page and phrase data check annotation quality but
never make an unrelated same-page result a retrieval hit. Unanswerable rows contain no qrels and
must state why the requested fact is absent from both manuals.

## Validator and approval procedure

After the frozen chunk export and draft JSONL files exist, run:

```powershell
python -m scripts.validate_phase7_dataset `
  --chunks artifacts/phase7/frozen-chunks.jsonl
```

The validator checks strict JSONL/Pydantic schema, unique IDs/questions, calibration/test separation,
fixed distributions, qrel existence, correct document ownership, phrase evidence, page diagnostics,
scenario coverage, and deterministic hashes. It does not call Qdrant, a model, or an LLM.

All new rows begin with `review_status: needs_human_review`. Once the review table, corpus manifest,
and validator output are complete, this document will list every row and request the exact approval:

```text
APPROVE PHASE 7 DATASET
```

Until that response is received, calibration, end-to-end provider runs, held-out evaluation, and
threshold/prompt changes are prohibited.
