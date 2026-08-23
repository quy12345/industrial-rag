"""Static dependency characterization for the pre-cleanup application graph."""

from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).parents[1] / "app"

# These are baseline debt, not approved dependencies. R04 owns their removal when
# retrieval and reranking responsibilities are separated from evaluation helpers.
KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS = {
    ("app.reranking", "app.evaluation"): "R04",
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


def _app_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "app":
                imports.update(
                    f"app.{alias.name}" for alias in node.names if alias.name != "*"
                )
            elif node.module.startswith("app."):
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("app."))
    return imports


def _import_graph() -> dict[str, set[str]]:
    paths = sorted(APP_ROOT.rglob("*.py"))
    modules = {_module_name(path): path for path in paths}
    return {
        module: {dependency for dependency in _app_imports(path) if dependency in modules}
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


def test_active_runtime_evaluation_imports_match_owned_baseline_debt() -> None:
    graph = _import_graph()
    active_runtime = _reachable_modules(graph, "app.main")
    actual = {
        (source, dependency)
        for source in active_runtime
        for dependency in graph[source]
        if dependency == "app.evaluation" or dependency.startswith("app.evaluation.")
    }

    assert actual == set(KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS)
    assert set(KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS.values()) == {"R04"}


def test_inbound_adapters_are_not_imported_by_other_application_modules() -> None:
    graph = _import_graph()
    inbound_modules = {
        "app.main",
        "app.api",
        "app.api.app",
        "app.api.auth",
        "app.api.query",
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


def test_indexing_application_service_has_no_adapter_or_evaluation_dependency() -> None:
    graph = _import_graph()
    dependencies = graph["app.application.indexing_service"]

    assert not {
        dependency
        for dependency in dependencies
        if dependency == "app.evaluation"
        or dependency.startswith("app.evaluation.")
        or dependency == "app.infrastructure"
        or dependency.startswith("app.infrastructure.")
    }


def test_reranking_uses_domain_candidate_assembly_not_evaluation_audit() -> None:
    graph = _import_graph()

    assert "app.domain.retrieval" in graph["app.reranking"]
    assert "app.candidate_audit" not in graph["app.reranking"]


def test_runtime_uses_canonical_domain_rrf_policy() -> None:
    graph = _import_graph()

    assert "app.domain.policies.fusion" in graph["app.reranking"]
    assert "app.domain.policies.fusion" in graph["app.hybrid_retrieval"]


def test_runtime_uses_canonical_domain_query_analysis_policy() -> None:
    graph = _import_graph()

    assert "app.domain.policies.query_analysis" in graph["app.retrieval_runtime"]
    assert "app.query_expansion" not in graph["app.retrieval_runtime"]
    assert "app.domain.policies.query_analysis" in graph["app.domain.retrieval_contracts"]


def test_runtime_uses_canonical_domain_ranking_policy() -> None:
    graph = _import_graph()

    for module in ("app.reranking", "app.evidence_selection", "app.domain.retrieval_contracts"):
        assert "app.domain.policies.ranking" in graph[module]
        assert "app.phase7_optimization" not in graph[module]


def test_runtime_uses_canonical_dense_search_adapter() -> None:
    graph = _import_graph()

    assert "app.infrastructure.qdrant.dense" in graph["app.reranking"]
    assert "app.infrastructure.qdrant.dense" in graph["app.hybrid_retrieval"]
    assert "app.retrieval" not in graph["app.reranking"]
    assert "app.retrieval" not in graph["app.hybrid_retrieval"]


def test_runtime_uses_canonical_sparse_search_adapter() -> None:
    graph = _import_graph()

    for module in ("app.reranking", "app.retrieval_runtime", "app.hybrid_retrieval"):
        assert "app.infrastructure.qdrant.hybrid" in graph[module]
    assert "app.hybrid_retrieval" not in graph["app.reranking"]
    assert "app.hybrid_retrieval" not in graph["app.retrieval_runtime"]
