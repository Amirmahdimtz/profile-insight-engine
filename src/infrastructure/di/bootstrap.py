from __future__ import annotations

import ast
import importlib
import threading
from pathlib import Path
from typing import Iterable


_bootstrap_lock = threading.Lock()
_bootstrapped = False
_EXCLUDED_SEGMENTS = {"di", "dtos", "models", "scripts", "alembic", "versions", "res", "sso"}


def _is_inject_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "inject"
    if isinstance(node, ast.Attribute):
        return node.attr == "inject"
    return False


def _contains_inject_provider(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_is_inject_decorator(decorator) for decorator in node.decorator_list):
            return True
    return False


def _discover_module_names(
    package_name: str,
    package_path: Iterable[str],
) -> tuple[str, ...]:
    """Discover only modules that declare DI providers under package roots."""

    module_names: set[str] = set()
    for root_value in package_path:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            relative_parts = list(path.relative_to(root).with_suffix("").parts)
            if any(part in _EXCLUDED_SEGMENTS for part in relative_parts):
                continue
            if not _contains_inject_provider(path):
                continue
            if relative_parts[-1] == "__init__":
                relative_parts.pop()
            module_names.add(".".join((package_name, *relative_parts)))
    return tuple(sorted(module_names))


def _bootstrap_package(package_name: str) -> None:
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name == package_name:
            return
        raise

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    for module_name in _discover_module_names(package_name, package_path):
        importlib.import_module(module_name)


def bootstrap_di() -> None:
    """Discover @inject providers in architecture order exactly once."""

    global _bootstrapped
    with _bootstrap_lock:
        if _bootstrapped:
            return
        for package_name in ("src.infrastructure", "src.core", "src.application"):
            _bootstrap_package(package_name)
        _bootstrapped = True
