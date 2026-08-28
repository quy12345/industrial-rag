"""Deterministic query-role inference from query text only."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

QueryRole = Literal["installation", "programming", "neutral"]
RoleConfidence = Literal["strong", "weak", "neutral"]
QUERY_ROLE_PROFILE = "phase7_query_role_v2"


@dataclass(frozen=True)
class QueryRoleCue:
    """One bilingual query-only cue; IDs are safe to record in artifacts."""

    identifier: str
    role: Literal["installation", "programming"]
    strength: Literal["strong", "weak"]
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class QueryRoleInference:
    """Auditable role inference derived solely from the query."""

    role: QueryRole
    confidence: RoleConfidence
    cue_ids: tuple[str, ...]
    installation_cues: tuple[str, ...]
    programming_cues: tuple[str, ...]


# Generic bilingual technical cues only. These contain no dataset IDs, qrels,
# expected pages, expected documents, or answer facts.
QUERY_ROLE_CUES: tuple[QueryRoleCue, ...] = (
    QueryRoleCue("safety", "installation", "strong", ("safety", "an toan")),
    QueryRoleCue(
        "prevent_rotation",
        "installation",
        "strong",
        ("prevent", "tranh", "rotate", "rotation", "shaft", "truc"),
    ),
    QueryRoleCue("electrical", "installation", "strong", ("electrical", "dien", "power", "nguon")),
    QueryRoleCue(
        "installation",
        "installation",
        "strong",
        ("install", "installed", "installing", "installation", "lap dat"),
    ),
    QueryRoleCue(
        "wiring", "installation", "strong", ("wiring", "wire", "terminal", "dau day", "dau cuc")
    ),
    QueryRoleCue(
        "contacts",
        "installation",
        "weak",
        ("contact", "tiep diem", "run", "lenh chay", "motor", "dong co"),
    ),
    QueryRoleCue("protection", "installation", "weak", ("protection", "protective", "bao ve")),
    QueryRoleCue("menu", "programming", "strong", ("mode", "menu", "configuration", "cau hinh")),
    QueryRoleCue("parameter", "programming", "strong", ("parameter", "tham so", "programming")),
    QueryRoleCue(
        "monitoring", "programming", "strong", ("monitoring", "giam sat", "reference", "tham chieu")
    ),
    QueryRoleCue("fault", "programming", "weak", ("fault", "loi")),
)


def infer_query_role(query: str) -> QueryRoleInference:
    """Infer a conservative role using normalized token/phrase boundaries."""

    normalized = normalize_query_text(query)
    installation: list[str] = []
    programming: list[str] = []
    installation_strengths: list[str] = []
    programming_strengths: list[str] = []
    cue_ids: list[str] = []
    for cue in QUERY_ROLE_CUES:
        if not any(query_contains_phrase(normalized, phrase) for phrase in cue.phrases):
            continue
        cue_ids.append(cue.identifier)
        if cue.role == "installation":
            installation.append(cue.identifier)
            installation_strengths.append(cue.strength)
        else:
            programming.append(cue.identifier)
            programming_strengths.append(cue.strength)
    if installation and not programming:
        role: QueryRole = "installation"
        confidence: RoleConfidence = "strong" if "strong" in installation_strengths else "weak"
    elif programming and not installation:
        role = "programming"
        confidence = "strong" if "strong" in programming_strengths else "weak"
    else:
        role = "neutral"
        confidence = "neutral"
    return QueryRoleInference(
        role=role,
        confidence=confidence,
        cue_ids=tuple(cue_ids),
        installation_cues=tuple(installation),
        programming_cues=tuple(programming),
    )


def normalize_query_text(value: str) -> str:
    """Normalize bilingual query text for deterministic boundary matching."""

    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.replace("\u0111", "d")
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def query_contains_phrase(query: str, phrase: str) -> bool:
    """Return whether normalized query tokens contain one complete phrase."""

    normalized_phrase = normalize_query_text(phrase)
    if not normalized_phrase:
        return False
    tokens = query.split()
    phrase_tokens = normalized_phrase.split()
    width = len(phrase_tokens)
    return any(tokens[index : index + width] == phrase_tokens for index in range(len(tokens)))
