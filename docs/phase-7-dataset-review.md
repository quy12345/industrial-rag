# Phase 7 dataset review receipt

This tracked file is intentionally metadata-only. It must not mirror calibration questions,
held-out questions, answer facts, expected phrases, qrels, pages, or evidence text.

## Frozen receipt

- Active calibration: `data/eval/phase7/calibration-v3.jsonl`
- Calibration rows: 20 (12 answerable, 8 unanswerable)
- Calibration SHA-256: `8e3e7798ec632022485352e917d2b4ce2907744e673ef64c4ad369d6ed5cfa46`
- Held-out rows: 45 (30 answerable, 15 unanswerable)
- Sealed held-out SHA-256 from the approved manifest:
  `68c9c52e745a7616a869a2f55024964501dfb0cb537bbe6ff91dac5fbcae3c54`
- Corpus chunks: 2,753
- Corpus stable-ID SHA-256:
  `2a972de9cfb551dd1d71dc9cb591d75071ad772d7d26519501539cad33e2f56d`
- Approval state: the active calibration-v3 dataset was approved previously; this receipt does not
  grant provider egress or held-out execution approval.

## Governance status

Historical versions of this tracked document mirrored held-out content, and the historical E2E CLI
loaded both JSONL splits even in calibration mode. Removing those details from the current revision
does not erase Git history or restore statistical secrecy.

Therefore the current held-out set is `BLOCKED_GOVERNANCE` for any claim that it is an unseen final
test. A trustworthy final metric requires one of these explicit decisions:

1. Create a new access-controlled held-out set that calibration tooling and tracked documentation
   cannot read.
2. Keep the current set but explicitly downgrade it to a non-unseen diagnostic set and avoid final
   generalization claims.

The sealed calibration CLI now loads and validates exactly one split. Calibration mode obtains only
the held-out hash from the manifest and never opens the held-out JSONL path. Held-out mode remains
separately token-gated.

## Review boundary

- Do not add row-level dataset content back to this file.
- Do not place provider outputs, raw questions, answers, prompts, excerpts, or secrets in tracked
  artifacts.
- Generated private diagnostics belong only under ignored `artifacts/private-debug/`.
- Never tune runtime behavior from held-out output.
