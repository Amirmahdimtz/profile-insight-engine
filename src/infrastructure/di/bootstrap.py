from __future__ import annotations

import importlib
import pkgutil
import threading


_bootstrap_lock = threading.Lock()
_bootstrapped = False
_EXCLUDED_SEGMENTS = {"di", "dtos", "models", "scripts", "alembic", "versions", "res", "sso"}


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

    for module_info in pkgutil.walk_packages(package_path, package.__name__ + "."):
        relative_parts = module_info.name.split(".")[2:]
        if any(part in _EXCLUDED_SEGMENTS for part in relative_parts):
            continue
        importlib.import_module(module_info.name)


def bootstrap_di() -> None:
    """Discover @inject providers in architecture order exactly once."""

    global _bootstrapped
    with _bootstrap_lock:
        if _bootstrapped:
            return
        for package_name in ("src.infrastructure", "src.core", "src.application"):
            _bootstrap_package(package_name)
        _bootstrapped = True
