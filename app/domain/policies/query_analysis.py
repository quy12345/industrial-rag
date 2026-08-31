"""Deterministic query-only lexical analysis policies."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

QUERY_EXPANSION_PROFILE = "vi_technical_glossary_v1"
QUERY_EXPANSION_PROFILE_V2 = "vi_technical_glossary_v2"

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


@dataclass(frozen=True)
class QueryExpansionRule:
    """One stable, query-only Vietnamese-to-English lexical rule."""

    identifier: str
    source_phrases: tuple[str, ...]
    target_terms: tuple[str, ...]


# This profile is registered before development evaluation. Its rules contain only
# general vocabulary from the two ATV320 manuals and the approved development
# taxonomy; they contain no qrels, pages, document IDs, or expected answer facts.
TECHNICAL_QUERY_GLOSSARY_V2: tuple[QueryExpansionRule, ...] = (
    QueryExpansionRule("safety", ("an toàn",), ("safety",)),
    QueryExpansionRule("hazard", ("nguy hiểm", "mối nguy"), ("hazard",)),
    QueryExpansionRule("warning", ("cảnh báo",), ("warning",)),
    QueryExpansionRule(
        "disconnect_power",
        ("ngắt nguồn điện", "ngắt nguồn", "ngắt điện"),
        ("disconnect", "power"),
    ),
    QueryExpansionRule("grounding", ("nối đất", "tiếp địa"), ("grounding", "ground")),
    QueryExpansionRule("prevention", ("tránh", "ngăn chặn"), ("avoid", "prevent")),
    QueryExpansionRule(
        "protective_equipment",
        ("thiết bị bảo vệ",),
        ("protective", "equipment"),
    ),
    QueryExpansionRule("shaft", ("trục",), ("shaft",)),
    QueryExpansionRule("motor", ("động cơ",), ("motor",)),
    QueryExpansionRule("rotation", ("quay", "xoay"), ("rotation",)),
    QueryExpansionRule("mounting", ("lắp đặt", "lắp"), ("installation", "mounting")),
    QueryExpansionRule("clearance", ("khoảng cách", "khoảng hở"), ("clearance",)),
    QueryExpansionRule("position", ("vị trí",), ("position",)),
    QueryExpansionRule("wiring", ("đấu dây", "đi dây"), ("wiring", "wire")),
    QueryExpansionRule("terminal", ("đầu cực", "cọc đấu dây"), ("terminal",)),
    QueryExpansionRule("cable", ("cáp",), ("cable",)),
    QueryExpansionRule("conductor", ("dây dẫn",), ("conductor", "wire")),
    QueryExpansionRule("power_supply", ("nguồn điện", "nguồn cấp"), ("power", "supply")),
    QueryExpansionRule("voltage", ("điện áp",), ("voltage",)),
    QueryExpansionRule("current", ("dòng điện",), ("current",)),
    QueryExpansionRule("navigation_key", ("phím",), ("key",)),
    QueryExpansionRule("switch", ("chuyển đổi", "chuyển"), ("switch",)),
    QueryExpansionRule("group", ("nhóm",), ("group",)),
    QueryExpansionRule("parameter", ("tham số",), ("parameter",)),
    QueryExpansionRule(
        "settings",
        ("cài đặt", "thiết lập", "cấu hình"),
        ("setting", "configuration"),
    ),
    QueryExpansionRule("code", ("mã tham số", "mã"), ("code",)),
    QueryExpansionRule("value", ("giá trị",), ("value",)),
    QueryExpansionRule(
        "factory_default",
        ("cài đặt gốc", "thiết lập gốc", "mặc định nhà máy"),
        ("factory", "setting", "default"),
    ),
    QueryExpansionRule("reference", ("tham chiếu",), ("reference",)),
    QueryExpansionRule("monitoring", ("giám sát",), ("monitoring",)),
    QueryExpansionRule("fault", ("lỗi", "sự cố"), ("fault",)),
    QueryExpansionRule("cause", ("nguyên nhân",), ("cause",)),
    QueryExpansionRule("remedy", ("biện pháp khắc phục", "khắc phục"), ("remedy",)),
    QueryExpansionRule("reset", ("đặt lại", "khởi tạo lại"), ("reset",)),
    QueryExpansionRule("table", ("bảng",), ("table",)),
    QueryExpansionRule("rating", ("định mức",), ("rating",)),
    QueryExpansionRule("maximum", ("tối đa",), ("maximum",)),
    QueryExpansionRule("minimum", ("tối thiểu",), ("minimum",)),
    QueryExpansionRule("step", ("bước",), ("step",)),
    QueryExpansionRule("sequence", ("trình tự", "thứ tự"), ("sequence",)),
    QueryExpansionRule("procedure", ("quy trình",), ("procedure",)),
    QueryExpansionRule("before", ("trước khi",), ("before",)),
    QueryExpansionRule("after", ("sau khi",), ("after",)),
)

_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_ACCENTLESS_VIETNAMESE_CUES: tuple[str, ...] = (
    "an toan",
    "bao nhieu",
    "cau hinh",
    "dau cuc",
    "dien ap",
    "dong co",
    "giam sat",
    "khong",
    "la gi",
    "ma loi",
    "nguon dien",
    "nguyen nhan",
    "nhu the nao",
    "nhung",
    "nhom nao",
    "phai",
    "phim",
    "sau khi",
    "tham so",
    "thiet bi",
    "truoc khi",
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


def augment_vietnamese_technical_query_v2(query: str) -> tuple[str, tuple[str, ...]]:
    """Append registered corpus vocabulary without rewriting the original query."""

    if not query.strip():
        raise ValueError("Query expansion requires a non-empty query.")
    query_tokens = _query_tokens(query)
    if not _is_probably_vietnamese(query_tokens):
        return query, ()

    terms: list[str] = []
    matched_rule_ids: list[str] = []
    for rule in TECHNICAL_QUERY_GLOSSARY_V2:
        if not any(_contains_phrase(query_tokens, phrase) for phrase in rule.source_phrases):
            continue
        matched_rule_ids.append(rule.identifier)
        for term in rule.target_terms:
            if _contains_phrase(query_tokens, term) or term in terms:
                continue
            terms.append(term)
    if not terms:
        return query, tuple(matched_rule_ids)
    separator = "" if query[-1].isspace() else " "
    return f"{query}{separator}{' '.join(terms)}", tuple(matched_rule_ids)


def expand_query_for_profile(query: str, *, profile: str) -> tuple[str, tuple[str, ...]]:
    """Apply one explicit lexical profile and reject unknown behavior identities."""

    if profile == QUERY_EXPANSION_PROFILE:
        return augment_vietnamese_technical_query(query)
    if profile == QUERY_EXPANSION_PROFILE_V2:
        return augment_vietnamese_technical_query_v2(query)
    raise ValueError(f"Unsupported query expansion profile: {profile}")


def _query_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    return tuple(_WORD_PATTERN.findall(normalized))


def _contains_phrase(query_tokens: tuple[str, ...], phrase: str) -> bool:
    phrase_tokens = _query_tokens(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(query_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        all(
            _tokens_match(
                query_token,
                phrase_token,
                allow_uppercase_accent_fold=width > 1,
            )
            for query_token, phrase_token in zip(
                query_tokens[index : index + width],
                phrase_tokens,
                strict=True,
            )
        )
        for index in range(len(query_tokens) - width + 1)
    )


def _tokens_match(
    query_token: str,
    phrase_token: str,
    *,
    allow_uppercase_accent_fold: bool,
) -> bool:
    if query_token.casefold() == phrase_token.casefold():
        return True
    if _has_diacritic(query_token):
        return False
    if _looks_like_identifier(query_token) and not (
        allow_uppercase_accent_fold and query_token.isalpha() and query_token.isupper()
    ):
        return False
    return _fold_diacritics(query_token) == _fold_diacritics(phrase_token)


def _is_probably_vietnamese(query_tokens: tuple[str, ...]) -> bool:
    for rule in TECHNICAL_QUERY_GLOSSARY_V2:
        for phrase in rule.source_phrases:
            phrase_tokens = _query_tokens(phrase)
            if any(_has_diacritic(token) for token in phrase_tokens) and _contains_exact_phrase(
                query_tokens, phrase_tokens
            ):
                return True
    folded_tokens = tuple(_fold_diacritics(token) for token in query_tokens)
    return any(_contains_folded_phrase(folded_tokens, cue) for cue in _ACCENTLESS_VIETNAMESE_CUES)


def _contains_exact_phrase(
    query_tokens: tuple[str, ...], phrase_tokens: tuple[str, ...]
) -> bool:
    width = len(phrase_tokens)
    folded_phrase = tuple(token.casefold() for token in phrase_tokens)
    return bool(width) and any(
        tuple(token.casefold() for token in query_tokens[index : index + width])
        == folded_phrase
        for index in range(len(query_tokens) - width + 1)
    )


def _contains_folded_phrase(query_tokens: tuple[str, ...], phrase: str) -> bool:
    phrase_tokens = tuple(_fold_diacritics(token) for token in _query_tokens(phrase))
    width = len(phrase_tokens)
    return bool(width) and any(
        query_tokens[index : index + width] == phrase_tokens
        for index in range(len(query_tokens) - width + 1)
    )


def _has_diacritic(token: str) -> bool:
    normalized = token.casefold()
    decomposed = unicodedata.normalize("NFD", normalized)
    return "đ" in normalized or any(
        unicodedata.combining(character) for character in decomposed
    )


def _looks_like_identifier(token: str) -> bool:
    letters = [character for character in token if character.isalpha()]
    if not letters:
        return False
    return (
        any(character.isdigit() for character in token)
        or (len(letters) > 1 and all(character.isupper() for character in letters))
        or (not token.islower() and not token.istitle())
    )


def _fold_diacritics(token: str) -> str:
    decomposed = unicodedata.normalize("NFKD", token).casefold()
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).replace("đ", "d")
