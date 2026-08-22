"""Static dependency characterization for the pre-cleanup application graph."""

from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).parents[1] / "app"

# These are baseline debt, not approved dependencies. R04 owns their removal when
# retrieval and reranking responsibilities are separated from evaluation helpers.
KNOWN_ACTIVE_RUNTIME_EVALUATION_IMPORTS = {
    ("app.candidate_audit", "app.evaluation"): "R04",
    ("app.reranking", "app.evaluation"): "R04",
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
