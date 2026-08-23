"""Compatibility facade for sanitized offline replay utilities."""

from evaluation.replay import Phase7ReplayError as Phase7ReplayError
from evaluation.replay import replay_role_prior as replay_role_prior
from evaluation.replay import snapshot_candidates_to_retrieval as snapshot_candidates_to_retrieval

__all__ = [
    "Phase7ReplayError",
    "replay_role_prior",
    "snapshot_candidates_to_retrieval",
]
