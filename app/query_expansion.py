"""Deterministic Vietnamese technical-term augmentation for Phase 7 calibration.

This is deliberately a small lexical transform, not a translation model. Terms
describe query intent and identifiers; they never contain expected answers, qrel
IDs, document IDs, pages, or held-out-specific rules.
"""

from __future__ import annotations

import unicodedata

QUERY_EXPANSION_PROFILE = "vi_technical_glossary_v1"

TECHNICAL_QUERY_GLOSSARY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tránh", ("avoid", "prevent")),
    ("trục", ("shaft",)),
    ("động cơ", ("motor",)),
    ("quay", ("rotate", "rotation")),
    ("phím", ("key",)),
    ("chuyển", ("switch", "navigate")),
    ("nhóm", ("group",)),
    ("nguồn điện", ("power", "supply")),
    ("điện áp", ("voltage",)),
    ("đầu cực", ("terminal",)),
    ("thiết bị bảo vệ", ("protective", "equipment")),
    ("lắp", ("install", "installed")),
    ("tham chiếu", ("reference",)),
    ("giám sát", ("monitoring",)),
    ("cấu hình", ("configuration",)),
)


def augment_vietnamese_technical_query(query: str) -> tuple[str, tuple[str, ...]]:
    """Append deterministic English technical tokens and report matched rules."""

    normalized = unicodedata.normalize("NFKC", query).casefold().strip()
    if not normalized:
        raise ValueError("Query expansion requires a non-empty query.")
    terms: list[str] = []
    matched_rules: list[str] = []
    for source, expansions in TECHNICAL_QUERY_GLOSSARY:
        if source not in normalized:
            continue
        matched_rules.append(source)
        for term in expansions:
            if term.casefold() not in normalized and term not in terms:
                terms.append(term)
    if not terms:
        return query.strip(), tuple(matched_rules)
    return f"{query.strip()} {' '.join(terms)}", tuple(matched_rules)
