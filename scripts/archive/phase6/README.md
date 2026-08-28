# Phase 6 historical reranking utilities

This directory preserves the Phase 4 retrieval evaluation and Phase 5 reranking experiment that
used the frozen Phase 6 corpus. It is retained for historical reproducibility and is not part of
the active Phase 7 runtime.

The utilities keep their original arguments, validation rules, output format, and artifact names.
Only their Python module paths changed:

```powershell
python -m scripts.archive.phase6.search_reranked --help
python -m scripts.archive.phase6.evaluate_reranking --help
python -m scripts.archive.phase6.evaluate --help
python -m scripts.archive.phase6.audit_candidate_pools --help
python -m scripts.archive.phase6.generate_phase5_readiness --help
python -m scripts.archive.phase6.search_dense --help
python -m scripts.archive.phase6.search_hybrid --help
python -m scripts.archive.phase6.index_document --help
python -m scripts.archive.phase6.index_hybrid --help
```

`rerank_runtime.py` validates the historical 99-chunk manifest and Phase 6 retrieval contract.
`search_reranked.py` reads the historical dense artifact before an interactive reranked search.
`evaluate_reranking.py` evaluates the three historical candidate strategies or rebuilds their
comparison artifact.

`evaluate.py` produces the historical dense, sparse, and hybrid retrieval metrics.
`audit_candidate_pools.py` measures whether those candidate pools contain direct evidence.
`generate_phase5_readiness.py` validates their frozen identities and writes the historical handoff.
`search_dense.py` and `search_hybrid.py` query the legacy Phase 6 collections directly.
`index_document.py` and `index_hybrid.py` preserve the legacy write paths for those collections.
`validate_phase6.sh` preserves the old dependency-install and validation recipe; it must not be run
as part of the active test workflow.

`generate_phase5_readiness.py` is an unsupported historical snapshot. In particular, a missing or
invalid frozen-chunk file can surface an uncaught `EvaluationError`. This known behavior will not be
fixed as part of R00 because the handoff is retired and is not an active product contract.

The archived search adapters are also unsupported. They bypass the active Phase 7 profile resolver
and require explicit historical configuration. Never use them as a Phase 7 smoke test.

The archived indexing adapters are unsupported and mutating. They can create collections, upsert
points, remove stale points for an indexed document, and overwrite historical manifests. Do not run
them without separate explicit approval; R00 never uses them to restore the retired corpus.

These commands can initialize a real cross-encoder and can access Qdrant unless a no-model path
such as `--comparison-only` is selected. Do not run them as unit tests, use them for new portfolio
claims, or point them at either Phase 7 collection. Never recreate or delete a collection to make
a historical check pass.
