"""Compatibility facade for deterministic technical query analysis."""

from app.domain.policies.query_analysis import (
    QUERY_EXPANSION_PROFILE,
    TECHNICAL_QUERY_GLOSSARY,
    augment_vietnamese_technical_query,
)

__all__ = [
    "QUERY_EXPANSION_PROFILE",
    "TECHNICAL_QUERY_GLOSSARY",
    "augment_vietnamese_technical_query",
]
