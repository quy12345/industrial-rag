"""Static dependency characterization for the pre-cleanup application graph."""

from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).parents[1] / "app"
EVALUATION_ROOT = Path(__file__).parents[1] / "evaluation"
UI_ROOT = Path(__file__).parents[1] / "ui"

PRODUCTION_RUNTIME_ROOTS = {
    "app.application.indexing_service",
    "app.application.query_service",
    "app.application.reranking_service",
    "app.bootstrap",
    "app.main",
    "app.retrieval_runtime",
}

DOMAIN_FORBIDDEN_IMPORT_ROOTS = {
    "docling",
    "fastapi",
    "fastembed",
    "langchain",
    "openai",
    "qdrant_client",
    "streamlit",
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(APP_ROOT.parent).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module in {"app", "evaluation"}:
                imports.update(
                    f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*"
                )
            elif node.module.startswith(("app.", "evaluation.")):
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name in {"app", "evaluation"}
                or alias.name.startswith(("app.", "evaluation."))
            )
    return imports


def _import_graph() -> dict[str, set[str]]:
    paths = sorted(
        [*APP_ROOT.rglob("*.py"), *EVALUATION_ROOT.rglob("*.py"), *UI_ROOT.rglob("*.py")]
    )
    modules = {_module_name(path): path for path in paths}
    return {
        module: {dependency for dependency in _local_imports(path) if dependency in modules}
        for module, path in modules.items()
    }


def _reachable_modules(graph: dict[str, set[str]], root: str) -> set[str]:
    reachable: set[str] = set()
    pending = [root]
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(graph.get(module, ()))
    return reachable


def test_production_runtime_cannot_reach_evaluation() -> None:
    graph = _import_graph()
    violations: set[tuple[str, str, str]] = set()
    for root in PRODUCTION_RUNTIME_ROOTS:
        for source in _reachable_modules(graph, root):
            for dependency in graph[source]:
                if dependency == "app.evaluation" or dependency.startswith(
                    ("app.evaluation.", "evaluation.")
                ) or dependency == "evaluation":
                    violations.add((root, source, dependency))

    assert violations == set()


def test_inbound_adapters_are_not_imported_by_other_application_modules() -> None:
    graph = _import_graph()
    inbound_modules = {
        "app.main",
        "app.api",
        "app.api.app",
        "app.api.auth",
        "app.api.dependencies",
        "app.api.health",
        "app.api.query",
        "ui",
        "ui.api_client",
        "ui.config",
        "ui.state",
        "ui.streamlit_app",
    }
    unexpected = {
        (source, dependency)
        for source, dependencies in graph.items()
        if source not in inbound_modules
        for dependency in dependencies
        if dependency in inbound_modules
    }

    assert unexpected == set()


def test_domain_does_not_depend_on_infrastructure_or_external_sdks() -> None:
    graph = _import_graph()
    infrastructure_dependencies = {
        (source, dependency)
        for source, dependencies in graph.items()
        if source == "app.domain" or source.startswith("app.domain.")
        for dependency in dependencies
        if dependency == "app.infrastructure" or dependency.startswith("app.infrastructure.")
    }
    external_sdk_imports: set[tuple[str, str]] = set()
    for path in sorted((APP_ROOT / "domain").rglob("*.py")):
        source = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                if name.split(".", maxsplit=1)[0] in DOMAIN_FORBIDDEN_IMPORT_ROOTS:
                    external_sdk_imports.add((source, name))

    assert infrastructure_dependencies == set()
    assert external_sdk_imports == set()


def test_qdrant_infrastructure_does_not_import_compatibility_facades() -> None:
    graph = _import_graph()
    unexpected = {
        (source, dependency)
        for source, dependencies in graph.items()
        if source == "app.infrastructure.qdrant"
        or source.startswith("app.infrastructure.qdrant.")
        for dependency in dependencies
        if dependency in {"app.retrieval", "app.hybrid_retrieval"}
    }

    assert unexpected == set()


def test_hybrid_facade_does_not_import_private_dense_facade_helpers() -> None:
    path = APP_ROOT / "hybrid_retrieval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    private_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "app.retrieval"
        for alias in node.names
        if alias.name.startswith("_")
    }

    assert private_imports == set()


def test_application_services_have_no_adapter_or_evaluation_dependency() -> None:
    graph = _import_graph()
    application_modules = {
        module for module in graph if module.startswith("app.application.")
    }
    for module in application_modules:
        dependencies = graph[module]
        assert not {
            dependency
            for dependency in dependencies
            if dependency == "app.evaluation"
            or dependency.startswith("app.evaluation.")
            or dependency == "evaluation"
            or dependency.startswith("evaluation.")
            or dependency == "app.infrastructure"
            or dependency.startswith("app.infrastructure.")
        }


def test_fastapi_modules_use_one_explicit_dependency_seam() -> None:
    graph = _import_graph()

    dependencies = graph["app.api.dependencies"]
    assert "app.application.query_service" in dependencies
    assert "app.bootstrap" in dependencies
    assert "app.config" in dependencies

    for module in ("app.api.app", "app.api.auth", "app.api.health", "app.api.query"):
        assert "app.api.dependencies" in graph[module]
        assert "app.bootstrap" not in graph[module]
        assert "app.config" not in graph[module]
        assert "app.query_service" not in graph[module]

    assert "app.api.health" in graph["app.api.app"]
    assert "app.api.query" in graph["app.api.app"]


def test_query_runtime_uses_shared_public_contracts_not_the_models_facade() -> None:
    graph = _import_graph()

    for module in (
        "app.api.query",
        "app.application.query_service",
        "app.domain.citations",
    ):
        assert "app.contracts.query" in graph[module]

    assert "app.models" not in graph["app.api.query"]
    assert "app.contracts.query" in graph["app.models"]
    assert "app.models" not in graph["app.contracts.query"]


def test_streamlit_is_an_http_only_adapter_over_shared_query_contracts() -> None:
    graph = _import_graph()

    for module in ("ui.api_client", "ui.state", "ui.streamlit_app"):
        app_dependencies = {
            dependency for dependency in graph[module] if dependency.startswith("app.")
        }
        assert app_dependencies == {"app.contracts.query"}

    assert "app.models" not in graph["ui.api_client"]
    assert "app.models" not in graph["ui.state"]
    assert "app.models" not in graph["ui.streamlit_app"]


def test_grounded_query_consumers_use_canonical_generation_contracts_and_prompt_policy() -> None:
    graph = _import_graph()
    service = graph["app.application.query_service"]

    assert "app.domain.generation" in service
    assert "app.application.generation_prompt" in service
    assert "app.generation" not in service
    assert "app.domain.citations" in service
    assert "app.citations" not in service
    assert "app.application.query_service" in graph["app.query_service"]
    assert "app.domain.citations" in graph["app.citations"]
    assert "app.domain.generation" in graph["app.domain.citations"]
    assert "app.generation" not in graph["app.domain.citations"]


def test_query_runtime_uses_canonical_domain_evidence_policy() -> None:
    graph = _import_graph()

    assert "app.domain.evidence" in graph["app.application.query_service"]
    assert "app.evidence_selection" not in graph["app.application.query_service"]
    assert "app.domain.evidence" in graph["app.bootstrap"]
    assert "app.domain.evidence" in graph["app.evidence_selection"]


def test_query_service_depends_on_domain_retrieval_port_not_runtime_adapter() -> None:
    graph = _import_graph()

    service = graph["app.application.query_service"]
    assert "app.domain.retrieval" in service
    assert "app.retrieval_runtime" not in service
    assert "app.config" not in service
    assert "app.domain.retrieval" in graph["app.retrieval_runtime"]
    assert "app.application.query_service" in graph["app.bootstrap"]
    assert "app.query_service" not in graph["app.bootstrap"]


def test_generation_adapter_is_composed_without_compatibility_or_application_dependencies() -> None:
    graph = _import_graph()
    adapter = graph["app.infrastructure.generation.langchain_structured"]

    assert "app.domain.generation" in adapter
    assert "app.generation" not in adapter
    assert "app.application.generation_prompt" not in adapter
    assert "app.infrastructure.generation.langchain_structured" in graph["app.bootstrap"]
    assert "app.application.generation_prompt" in graph["app.bootstrap"]
    assert "app.generation" not in graph["app.bootstrap"]


def test_reranking_service_uses_domain_candidate_assembly_not_evaluation_audit() -> None:
    graph = _import_graph()

    service = graph["app.application.reranking_service"]
    assert "app.domain.retrieval" in service
    assert "app.candidate_audit" not in service
    assert "app.evaluation" not in service


def test_runtime_uses_canonical_domain_rrf_policy() -> None:
    graph = _import_graph()

    assert "app.domain.policies.fusion" in graph["app.application.reranking_service"]
    assert "app.domain.policies.fusion" in graph["app.hybrid_retrieval"]


def test_runtime_uses_canonical_domain_query_analysis_policy() -> None:
    graph = _import_graph()

    assert "app.domain.policies.query_analysis" in graph["app.retrieval_runtime"]
    assert "app.query_expansion" not in graph["app.retrieval_runtime"]
    assert "app.domain.policies.query_analysis" in graph["app.domain.retrieval_contracts"]


def test_runtime_uses_canonical_domain_ranking_policy() -> None:
    graph = _import_graph()

    for module in (
        "app.application.reranking_service",
        "app.domain.evidence",
        "app.domain.retrieval_contracts",
    ):
        assert "app.domain.policies.ranking" in graph[module]
        assert "app.phase7_optimization" not in graph[module]


def test_runtime_uses_canonical_dense_search_adapter() -> None:
    graph = _import_graph()

    service = graph["app.application.reranking_service"]
    assert "app.infrastructure.qdrant.dense" in graph["app.retrieval_runtime"]
    assert "app.infrastructure.qdrant.dense" in graph["app.hybrid_retrieval"]
    assert "app.infrastructure.qdrant.dense" not in service
    assert "app.retrieval" not in service
    assert "app.retrieval" not in graph["app.hybrid_retrieval"]


def test_runtime_uses_canonical_sparse_search_adapter() -> None:
    graph = _import_graph()

    for module in (
        "app.retrieval_runtime",
        "app.hybrid_retrieval",
    ):
        assert "app.infrastructure.qdrant.hybrid" in graph[module]
    assert "app.infrastructure.qdrant.hybrid" not in graph[
        "app.application.reranking_service"
    ]
    assert "app.hybrid_retrieval" not in graph["app.application.reranking_service"]
    assert "app.hybrid_retrieval" not in graph["app.retrieval_runtime"]


def test_runtime_composes_canonical_reranking_service_not_evaluation_facade() -> None:
    graph = _import_graph()

    assert "app.application.reranking_service" in graph["app.retrieval_runtime"]
    assert "app.reranking" not in graph["app.retrieval_runtime"]


def test_retrieval_evaluation_is_owned_outside_production_package() -> None:
    graph = _import_graph()

    assert "evaluation.retrieval" in graph["app.evaluation"]
    assert "app.domain.documents" in graph["evaluation.retrieval"]
    assert "app.infrastructure.corpus_artifacts" in graph["evaluation.retrieval"]
    assert "app.models" not in graph["evaluation.retrieval"]
    assert "app.evaluation" not in graph["evaluation.retrieval"]


def test_sanitized_replay_is_owned_outside_production_package() -> None:
    graph = _import_graph()

    assert "evaluation.replay" in graph["app.phase7_replay"]
    assert "app.domain.retrieval" in graph["evaluation.replay"]
    assert "app.domain.policies.ranking" in graph["evaluation.replay"]
    assert "app.models" not in graph["evaluation.replay"]
    assert "app.phase7_optimization" not in graph["evaluation.replay"]
    assert "app.phase7_replay" not in graph["evaluation.replay"]


def test_phase7_dataset_and_corpus_artifact_ownership_are_separate() -> None:
    graph = _import_graph()

    assert "evaluation.phase7_dataset" in graph["app.phase7"]
    assert "app.infrastructure.corpus_artifacts" in graph["app.phase7"]
    assert "evaluation.retrieval" in graph["evaluation.phase7_dataset"]
    assert "app.domain.documents" in graph["evaluation.phase7_dataset"]
    assert "app.infrastructure.corpus_artifacts" in graph["evaluation.phase7_dataset"]
    assert "app.evaluation" not in graph["evaluation.phase7_dataset"]
    assert "app.models" not in graph["evaluation.phase7_dataset"]
    assert "app.phase7" not in graph["evaluation.phase7_dataset"]
    assert "app.domain.retrieval_contracts" in graph["app.infrastructure.corpus_artifacts"]
    assert "app.domain.documents" in graph["app.infrastructure.corpus_artifacts"]
    assert not {
        dependency
        for dependency in graph["app.infrastructure.corpus_artifacts"]
        if dependency == "evaluation" or dependency.startswith("evaluation.")
    }


def test_supported_indexing_command_does_not_import_phase7_dataset_facade() -> None:
    imports = _local_imports(
        APP_ROOT.parent / "scripts" / "operations" / "index_phase7_corpus.py"
    )

    assert "app.infrastructure.corpus_artifacts" in imports
    assert "app.evaluation" not in imports
    assert "app.phase7" not in imports
    assert not {
        dependency
        for dependency in imports
        if dependency == "evaluation" or dependency.startswith("evaluation.")
    }


def test_dataset_validation_cli_uses_canonical_evaluation_interfaces() -> None:
    canonical_path = (
        APP_ROOT.parent / "scripts" / "evaluation" / "validate_phase7_dataset.py"
    )
    imports = _local_imports(canonical_path)

    assert "app.infrastructure.corpus_artifacts" in imports
    assert "evaluation.phase7_dataset" in imports
    assert "evaluation.retrieval" in imports
    assert "app.evaluation" not in imports
    assert "app.phase7" not in imports

    compatibility_path = APP_ROOT.parent / "scripts" / "validate_phase7_dataset.py"
    tree = ast.parse(
        compatibility_path.read_text(encoding="utf-8"),
        filename=str(compatibility_path),
    )
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules == {"scripts.evaluation.validate_phase7_dataset"}


def test_corpus_audit_cli_uses_canonical_artifact_infrastructure() -> None:
    canonical_path = APP_ROOT.parent / "scripts" / "operations" / "audit_phase7_corpus.py"
    imports = _local_imports(canonical_path)

    assert imports == {"app.infrastructure.corpus_artifacts"}
    assert "app.phase7" not in imports

    compatibility_path = APP_ROOT.parent / "scripts" / "audit_phase7_corpus.py"
    tree = ast.parse(
        compatibility_path.read_text(encoding="utf-8"),
        filename=str(compatibility_path),
    )
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules == {"scripts.operations.audit_phase7_corpus"}


def test_cross_encoder_adapter_depends_on_domain_port_and_stays_lazy_at_runtime_edge() -> None:
    graph = _import_graph()

    assert "app.domain.reranking" in graph["app.infrastructure.models.reranker"]
    assert "app.reranking" not in graph["app.infrastructure.models.reranker"]
    assert "app.domain.reranking" in graph["app.retrieval_runtime"]
    assert "app.infrastructure.models.reranker" in graph["app.retrieval_runtime"]
