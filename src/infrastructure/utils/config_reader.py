from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.infrastructure.di.inject import inject


class ConfigurationError(ValueError):
    """Raised when required application configuration is missing or invalid."""


@inject
class ConfigReader:
    __di_singleton__ = True

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path is not None else self._default_config_path()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"configuration file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"configuration file is invalid YAML: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ConfigurationError("configuration root must be an object")
        self._values = dict(payload)

    @staticmethod
    def _default_config_path() -> Path:
        res_root = Path(__file__).resolve().parents[2] / "host" / "res"
        filename = (
            "appsettings.development.yaml"
            if os.environ.get("env_type", "").strip().lower() == "development"
            else "appsettings.yaml"
        )
        return res_root / filename

    def get(self, key: str) -> Any:
        if not isinstance(key, str) or not key or key != key.strip():
            raise ConfigurationError("configuration key must be a non-empty trimmed string")
        value: Any = self._values
        for segment in key.split("."):
            if not isinstance(value, Mapping) or segment not in value:
                raise ConfigurationError(f"missing configuration key: {key}")
            value = value[segment]
        return value

    def get_positive_int(self, key: str) -> int:
        value = self.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"configuration key '{key}' must be a positive integer")
        return value

    def get_non_empty_string_list(self, key: str) -> tuple[str, ...]:
        value = self.get(key)
        if not isinstance(value, list) or not value:
            raise ConfigurationError(f"configuration key '{key}' must be a non-empty list")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item or item != item.strip():
                raise ConfigurationError(
                    f"configuration key '{key}' must contain non-empty trimmed strings"
                )
            normalized.append(item)
        if len(normalized) != len(set(normalized)):
            raise ConfigurationError(f"configuration key '{key}' must not contain duplicates")
        return tuple(normalized)
