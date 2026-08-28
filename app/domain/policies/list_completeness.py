"""Bounded list-completeness policies over query and candidate text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.domain.policies.fusion import FusionPolicyError
from app.domain.policies.query_roles import normalize_query_text, query_contains_phrase
from app.domain.retrieval import RetrievalCandidate

LIST_COMPLETENESS_PROFILE = "phase7_list_completeness_v1"
RELATION_LIST_COMPLETENESS_PROFILE = "phase7_relation_list_completeness_v1"
_LIST_INTENT_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("which_groups", ("which groups", "what groups", "which categories", "what categories")),
    ("which_menus", ("which menus", "what menus", "which menu groups", "what menu groups")),
    ("vi_groups", ("nhung nhom nao", "cac nhom nao", "nhom menu nao")),
)
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(r"(?<![\w])[A-Z][A-Z0-9_-]{2,}(?![\w])")
_BRACKETED_LABEL_CODE_PATTERN = re.compile(
    r"\[([^\]\n]{1,80})\]\s*([A-Za-z][A-Za-z0-9_.-]{1,15})(?=\s|[-,.;:)]|$)"
)
_RELATION_LIST_KEY_CUES = ("key", "button", "phim")
_RELATION_LIST_SWITCH_CUES = (
    "switch",
    "switch between",
    "change between",
    "toggle between",
    "chuyen",
    "chuyen giua",
)
_CANDIDATE_SWITCH_PATTERN = re.compile(
    r"(?<![\w])(?:switch(?:es|ed|ing)?|toggl(?:e|es|ed|ing)|chang(?:e|es|ed|ing))(?![\w])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ListIntentInference:
    """Query-only signal for bounded list-completeness ordering."""

    enabled: bool
    cue_ids: tuple[str, ...]
    technical_identifiers: tuple[str, ...]


@dataclass(frozen=True)
class RelationListInference:
    """Query-only signal for a key/identifier relation that requests a list."""

    enabled: bool
    cue_ids: tuple[str, ...]
    technical_identifiers: tuple[str, ...]


def infer_list_intent(query: str) -> ListIntentInference:
    """Infer a generic bilingual list request without evaluation-label access."""

    normalized = normalize_query_text(query)
    cue_ids = tuple(
        identifier
        for identifier, phrases in _LIST_INTENT_CUES
        if any(query_contains_phrase(normalized, phrase) for phrase in phrases)
    )
    identifiers = tuple(
        sorted(
            {
                match.group(0).casefold()
                for match in _TECHNICAL_IDENTIFIER_PATTERN.finditer(
                    unicodedata.normalize("NFKC", query)
                )
            }
        )
    )
    return ListIntentInference(bool(cue_ids), cue_ids, identifiers)


def infer_relation_list_intent(query: str) -> RelationListInference:
    """Require list, key/button, switch/change, and technical-ID cues from the query."""

    list_intent = infer_list_intent(query)
    normalized = normalize_query_text(query)
    has_key = any(query_contains_phrase(normalized, cue) for cue in _RELATION_LIST_KEY_CUES)
    has_switch = any(query_contains_phrase(normalized, cue) for cue in _RELATION_LIST_SWITCH_CUES)
    cue_ids = (
        *list_intent.cue_ids,
        *(("key_or_button",) if has_key else ()),
        *(("switch_relation",) if has_switch else ()),
    )
    enabled = bool(
        list_intent.enabled
        and list_intent.technical_identifiers
        and has_key
        and has_switch
    )
    return RelationListInference(enabled, cue_ids, list_intent.technical_identifiers)


def list_completeness_features(
    text: str, *, technical_identifiers: tuple[str, ...]
) -> dict[str, int]:
    """Return bounded structural counts derived only from query and candidate text."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    pairs = {
        (" ".join(label.casefold().split()), code.casefold())
        for label, code in _BRACKETED_LABEL_CODE_PATTERN.findall(text)
    }
    identifier_matches = sum(
        re.search(rf"(?<![\w]){re.escape(identifier)}(?![\w])", normalized) is not None
        for identifier in technical_identifiers
    )
    return {
        "query_identifier_match_count": identifier_matches,
        "bracketed_label_code_pair_count": len(pairs),
    }


def relation_list_completeness_features(
    text: str, *, technical_identifiers: tuple[str, ...]
) -> dict[str, int]:
    """Count only targets after a key/identifier switch relation in one clause.

    This deliberately ignores unrelated labels elsewhere in the chunk. It uses
    query-derived identifiers but no qrels, expected facts, pages, or documents.
    """

    normalized_text = unicodedata.normalize("NFKC", text)
    clauses = re.split(r"(?<=[.!?;])\s+|\n+", normalized_text)
    key_anchor_count = 0
    relation_match_count = 0
    maximum_target_count = 0
    for clause in clauses:
        anchor_ends: list[int] = []
        for identifier in technical_identifiers:
            escaped = re.escape(identifier)
            patterns = (
                rf"(?<![\w]){escaped}(?![\w])(?:\s*\([^)]{{1,12}}\))?\s+(?:key|button)(?![\w])",
                rf"(?<![\w])(?:key|button)\s+{escaped}(?![\w])",
            )
            for pattern in patterns:
                anchor_ends.extend(
                    match.end() for match in re.finditer(pattern, clause, re.IGNORECASE)
                )
        if not anchor_ends:
            continue
        key_anchor_count += 1
        relation_targets: list[int] = []
        for anchor_end in anchor_ends:
            relation = _CANDIDATE_SWITCH_PATTERN.search(clause, anchor_end)
            if relation is None:
                continue
            relation_match_count += 1
            tail = clause[relation.end() :]
            pairs = {
                (" ".join(label.casefold().split()), code.casefold())
                for label, code in _BRACKETED_LABEL_CODE_PATTERN.findall(tail)
            }
            relation_targets.append(len(pairs))
        if relation_targets:
            maximum_target_count = max(maximum_target_count, *relation_targets)
    return {
        "query_key_anchor_match_count": key_anchor_count,
        "scoped_relation_match_count": relation_match_count,
        "scoped_relation_target_count": maximum_target_count,
    }


def apply_list_completeness_fallback(
    candidates: list[RetrievalCandidate], *, query: str
) -> list[RetrievalCandidate]:
    """Apply the registered ranks-5-to-10 list fallback from raw runtime inputs."""

    inference = infer_list_intent(query)
    enriched = []
    for candidate in candidates:
        metadata = dict(candidate.metadata)
        metadata.update(
            list_completeness_features(
                candidate.text,
                technical_identifiers=inference.technical_identifiers,
            )
        )
        metadata["list_intent_cue_ids"] = inference.cue_ids
        metadata["query_technical_identifiers"] = inference.technical_identifiers
        enriched.append(candidate.model_copy(update={"metadata": metadata}))
    return apply_list_completeness_from_metadata(enriched, enabled=inference.enabled)


def apply_list_completeness_from_metadata(
    candidates: list[RetrievalCandidate], *, enabled: bool
) -> list[RetrievalCandidate]:
    """Replay the fallback from sanitized structural feature counts."""

    if not enabled:
        return list(candidates)
    ranks = [candidate.rerank_rank for candidate in candidates]
    if any(rank is None or rank <= 0 for rank in ranks) or set(ranks) != set(
        range(1, len(candidates) + 1)
    ):
        raise FusionPolicyError(
            "List completeness requires unique, contiguous one-based rerank ranks."
        )
    if len(candidates) < 5:
        return list(candidates)
    ordered = sorted(candidates, key=lambda item: (item.rerank_rank or 2**31, item.chunk_id))
    window = ordered[4:10]
    for candidate in window:
        for field in ("query_identifier_match_count", "bracketed_label_code_pair_count"):
            value = candidate.metadata.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FusionPolicyError(
                    "List completeness requires non-negative sanitized feature counts."
                )
    window.sort(
        key=lambda candidate: (
            -int(candidate.metadata["query_identifier_match_count"]),
            -int(candidate.metadata["bracketed_label_code_pair_count"]),
            int(candidate.rerank_rank or 2**31),
            candidate.chunk_id,
        )
    )
    reordered = [*ordered[:4], *window, *ordered[10:]]
    result: list[RetrievalCandidate] = []
    for rank, candidate in enumerate(reordered, start=1):
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "pre_list_completeness_rank": candidate.rerank_rank,
                "pre_list_completeness_rank_score": candidate.score,
                "list_completeness_profile": LIST_COMPLETENESS_PROFILE,
            }
        )
        result.append(
            candidate.model_copy(
                update={"metadata": metadata, "rerank_rank": rank, "score": 1 / (100 + rank)}
            )
        )
    return result


def apply_relation_list_completeness_fallback(
    candidates: list[RetrievalCandidate], *, query: str
) -> list[RetrievalCandidate]:
    """Apply relation-scoped list completeness from raw runtime inputs."""

    inference = infer_relation_list_intent(query)
    enriched: list[RetrievalCandidate] = []
    for candidate in candidates:
        metadata = dict(candidate.metadata)
        metadata.update(
            relation_list_completeness_features(
                candidate.text,
                technical_identifiers=inference.technical_identifiers,
            )
        )
        metadata["relation_list_intent_cue_ids"] = inference.cue_ids
        metadata["relation_list_query_identifiers"] = inference.technical_identifiers
        enriched.append(candidate.model_copy(update={"metadata": metadata}))
    return apply_relation_list_completeness_from_metadata(
        enriched,
        enabled=inference.enabled,
    )


def apply_relation_list_completeness_from_metadata(
    candidates: list[RetrievalCandidate], *, enabled: bool
) -> list[RetrievalCandidate]:
    """Replay relation-scoped list completeness in the bounded ranks 5--10 window."""

    if not enabled:
        return list(candidates)
    ranks = [candidate.rerank_rank for candidate in candidates]
    if any(rank is None or rank <= 0 for rank in ranks) or set(ranks) != set(
        range(1, len(candidates) + 1)
    ):
        raise FusionPolicyError(
            "Relation-list completeness requires unique, contiguous one-based rerank ranks."
        )
    if len(candidates) < 5:
        return list(candidates)
    ordered = sorted(candidates, key=lambda item: (item.rerank_rank or 2**31, item.chunk_id))
    window = ordered[4:10]
    fields = (
        "query_key_anchor_match_count",
        "scoped_relation_match_count",
        "scoped_relation_target_count",
    )
    for candidate in window:
        for field in fields:
            value = candidate.metadata.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FusionPolicyError(
                    "Relation-list completeness requires non-negative sanitized feature counts."
                )
    window.sort(
        key=lambda candidate: (
            -int(candidate.metadata["scoped_relation_match_count"]),
            -int(candidate.metadata["scoped_relation_target_count"]),
            -int(candidate.metadata["query_key_anchor_match_count"]),
            int(candidate.rerank_rank or 2**31),
            candidate.chunk_id,
        )
    )
    reordered = [*ordered[:4], *window, *ordered[10:]]
    result: list[RetrievalCandidate] = []
    for rank, candidate in enumerate(reordered, start=1):
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "pre_relation_list_rank": candidate.rerank_rank,
                "pre_relation_list_rank_score": candidate.score,
                "relation_list_completeness_profile": RELATION_LIST_COMPLETENESS_PROFILE,
            }
        )
        result.append(
            candidate.model_copy(
                update={"metadata": metadata, "rerank_rank": rank, "score": 1 / (200 + rank)}
            )
        )
    return result
