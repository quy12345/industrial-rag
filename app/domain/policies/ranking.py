"""Compatibility facade for deterministic ranking policies.

Canonical owners are ``fusion``, ``list_completeness``, and ``query_roles``.
This facade remains until the versioned R12 source-identity migration.
"""

from app.domain.policies.fusion import (
    PHASE7_CALIBRATION_FUSION_PROFILE,
    Phase7FusionProfile,
    Phase7OptimizationError,
    PostRerankConfidenceMode,
    apply_role_aware_rank_fusion,
    fuse_weighted_rrf,
    phase7_fusion_profile_grid,
    phase7_profile_from_mapping,
    select_coverage_preserving_candidates,
)
from app.domain.policies.list_completeness import (
    LIST_COMPLETENESS_PROFILE,
    RELATION_LIST_COMPLETENESS_PROFILE,
    ListIntentInference,
    RelationListInference,
    apply_list_completeness_fallback,
    apply_list_completeness_from_metadata,
    apply_relation_list_completeness_fallback,
    apply_relation_list_completeness_from_metadata,
    infer_list_intent,
    infer_relation_list_intent,
    list_completeness_features,
    relation_list_completeness_features,
)
from app.domain.policies.query_roles import (
    QUERY_ROLE_CUES,
    QUERY_ROLE_PROFILE,
    QueryRole,
    QueryRoleCue,
    QueryRoleInference,
    RoleConfidence,
    infer_query_role,
)

__all__ = [
    "LIST_COMPLETENESS_PROFILE",
    "PHASE7_CALIBRATION_FUSION_PROFILE",
    "QUERY_ROLE_CUES",
    "QUERY_ROLE_PROFILE",
    "RELATION_LIST_COMPLETENESS_PROFILE",
    "ListIntentInference",
    "Phase7FusionProfile",
    "Phase7OptimizationError",
    "PostRerankConfidenceMode",
    "QueryRole",
    "QueryRoleCue",
    "QueryRoleInference",
    "RelationListInference",
    "RoleConfidence",
    "apply_list_completeness_fallback",
    "apply_list_completeness_from_metadata",
    "apply_relation_list_completeness_fallback",
    "apply_relation_list_completeness_from_metadata",
    "apply_role_aware_rank_fusion",
    "fuse_weighted_rrf",
    "infer_list_intent",
    "infer_query_role",
    "infer_relation_list_intent",
    "list_completeness_features",
    "phase7_fusion_profile_grid",
    "phase7_profile_from_mapping",
    "relation_list_completeness_features",
    "select_coverage_preserving_candidates",
]
