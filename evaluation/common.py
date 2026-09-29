from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.core.profile_analysis.contracts import (
    ContractValidationError,
    EvidenceType,
    ThemeType,
    validate_observable_claim,
)


DATASET_SCHEMA_VERSION = "1.0.0"
LABEL_SCHEMA_VERSION = "1.0.0"
PREDICTION_SCHEMA_VERSION = "1.0.0"
BENCHMARK_REPORT_SCHEMA_VERSION = "1.0.0"
EVALUATOR_VERSION = "phase2-evaluator-1.0.0"
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EvaluationValidationError(ValueError):
    """Raised when a Phase 2 evaluation contract is invalid."""


class DatasetSplit(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


class DatasetSlice(str, Enum):
    PERSIAN_HEAVY = "persian_heavy"
    ENGLISH_HEAVY = "english_heavy"
    MIXED_PERSIAN_ENGLISH = "mixed_persian_english"
    TEXT_HEAVY = "text_heavy"
    LOW_QUALITY = "low_quality"
    SCREENSHOT = "screenshot"
    MEME = "meme"
    GROUP = "group"
    NO_TEXT = "no_text"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class EvaluationLabelType(str, Enum):
    OBJECT = EvidenceType.OBJECT.value
    SCENE = EvidenceType.SCENE.value
    OCR_TEXT = EvidenceType.OCR_TEXT.value
    ACTIVITY = EvidenceType.ACTIVITY.value
    ENVIRONMENT = EvidenceType.ENVIRONMENT.value
    VISIBLE_INTEREST = ThemeType.VISIBLE_INTEREST.value
    BRAND = EvidenceType.BRAND.value
    TEAM = EvidenceType.TEAM.value
    RELIGIOUS_CONTENT = EvidenceType.RELIGIOUS_CONTENT.value
    RECURRING_CONTENT = ThemeType.RECURRING_CONTENT.value
    SOCIAL_CONTEXT = EvidenceType.SOCIAL_CONTEXT.value


class UnsupportedClaimCategory(str, Enum):
    PERSON_RELIGION = "person_religion"
    POLITICAL_ORIENTATION = "political_orientation"
    ETHNICITY = "ethnicity"
    MENTAL_HEALTH = "mental_health"
    SEXUAL_ORIENTATION = "sexual_orientation"
    INTELLIGENCE = "intelligence"
    HONESTY = "honesty"
    FAMILY_RELATIONSHIP = "family_relationship"
    PERSONALITY_TRAIT = "personality_trait"


class DataPolicy(str, Enum):
    CONSENTED_OR_AUTHORIZED = "consented_or_authorized"


class BenchmarkReportKind(str, Enum):
    BENCHMARK = "benchmark"
    INFRASTRUCTURE_SANITY_BASELINE = "infrastructure_sanity_baseline"


_JSON_SCALARS = (str, int, float, bool, type(None))


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_id(value: Any, field_name: str) -> str:
    value = _require_text(value, field_name)
    if not _ID_PATTERN.fullmatch(value):
        raise EvaluationValidationError(
            f"{field_name} must contain only letters, numbers, '.', '_' or '-' and start alphanumeric"
        )
    return value


def _require_version(value: Any, field_name: str) -> str:
    value = _require_text(value, field_name)
    if not _VERSION_PATTERN.fullmatch(value):
        raise EvaluationValidationError(f"{field_name} must use MAJOR.MINOR.PATCH numeric format")
    return value


def _require_probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{field_name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise EvaluationValidationError(f"{field_name} must be a finite number in [0, 1]")
    return result


def _require_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationValidationError(f"{field_name} must be a finite number")
    return result


def _require_non_negative_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvaluationValidationError(f"{field_name} must be a finite non-negative number")
    return result


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationValidationError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationValidationError(f"{field_name} must be a positive integer")
    return value


def _require_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EvaluationValidationError(f"{field_name} must be a sequence")
    return tuple(value)


def _normalize_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationValidationError(f"{field_name} must not contain NaN or Infinity")
        return value
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvaluationValidationError(f"{field_name} object keys must be strings")
            normalized[key] = _normalize_json_value(item, field_name)
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normalize_json_value(item, field_name) for item in value)
    raise EvaluationValidationError(f"{field_name} must be JSON-compatible")


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _to_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    return value


def deterministic_json(value: Any) -> str:
    return json.dumps(
        _to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _expect_mapping(payload: Any, model_name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise EvaluationValidationError(f"{model_name} must be an object")
    return payload


def _expect_keys(
    payload: Mapping[str, Any],
    model_name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    unknown = set(payload) - required - optional
    if unknown:
        raise EvaluationValidationError(
            f"{model_name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    missing = required - set(payload)
    if missing:
        raise EvaluationValidationError(
            f"{model_name} is missing required fields: {', '.join(sorted(missing))}"
        )


def _parse_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    if not isinstance(value, str):
        raise EvaluationValidationError(f"{field_name} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise EvaluationValidationError(f"{field_name} must be one of: {allowed}") from exc


def _validate_observable_label(label: str) -> str:
    label = _require_text(label, "label")
    try:
        validate_observable_claim(None, label)
    except ContractValidationError as exc:
        raise EvaluationValidationError(str(exc)) from exc
    return label


def _require_relative_path(value: Any, field_name: str) -> str:
    value = _require_text(value, field_name)
    if "\\" in value:
        raise EvaluationValidationError(f"{field_name} must use portable '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise EvaluationValidationError(f"{field_name} must be a safe relative POSIX path")
    return path.as_posix()


