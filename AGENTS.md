# Industrial Technical Manual RAG

- Target Python is 3.11. Use `python -m ruff check .` and `python -m pytest -q` with one interpreter.
- `app/` contains ingestion, retrieval, hybrid/RRF and evaluation code; `scripts/` contains explicit
  integration CLIs; `tests/` must remain offline with fake models/in-memory Qdrant.
- Frozen evaluation contract: 99 chunks for `manual-77d5dae4c2c5`, chunk-ID hash
  `bac72ba44aa76ee5ee0220ca62f84c81efef54b76f2c8b566f4c1f3cf293b2be`, and 30 direct-evidence qrels.
  Never alter qrels/chunks to improve a metric.
- Preserve Qdrant collections v1/v2 and the named volumes. Never recreate/delete them, prune Docker,
  or claim a metric without an executed command/artifact.
- Real Qdrant/model checks use explicit manual or integration commands; model downloads never belong
  in unit tests or Docker builds.
- Update README and `docs/` when a public retrieval contract changes. Do not commit, push, or merge
  unless the user explicitly asks.
