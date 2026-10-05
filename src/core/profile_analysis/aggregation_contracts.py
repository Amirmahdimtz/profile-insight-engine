from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from src.core.profile_analysis.contracts import EvidenceType


class AggregationValidationError(ValueError):
    """Raised when Phase 7 cross-image aggregation data is invalid."""


class AggregationKind(str, Enum):
    OBJECT = EvidenceType.OBJECT.value
    SCENE = EvidenceType.SCENE.value
    OCR_TEXT = EvidenceType.OCR_TEXT.value
    ACTIVITY = EvidenceType.ACTIVITY.value
    ENVIRONMENT = EvidenceType.ENVIRONMENT.value
    TOPIC = EvidenceType.TOPIC.value
    BRAND = EvidenceType.BRAND.value
    TEAM = EvidenceType.TEAM.value
    RELIGIOUS_CONTENT = EvidenceType.RELIGIOUS_CONTENT.value
    SOCIAL_CONTEXT = EvidenceType.SOCIAL_CONTEXT.value
    OTHER_OBSERVABLE = EvidenceType.OTHER_OBSERVABLE.value
    SEMANTIC_THEME = "semantic_theme"


def normalize_aggregation_text(value: str) -> str:
    """Apply only exact, deterministic text identity normalization for aggregation."""

    if not isinstance(value, str):
        raise AggregationValidationError("aggregation text must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", value).split()).strip()
    if not normalized:
        raise AggregationValidationError("aggregation text must not be empty")
    return normalized


def aggregation_key(value: str) -> str:
    return normalize_aggregation_text(value).casefold()


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AggregationValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AggregationValidationError(f"{field_name} must be a positive integer")
    return value


def _require_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationValidationError(f"{field_name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise AggregationValidationError(f"{field_name} must be a finite number in [0, 1]")
    return result


def _require_optional_probability(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_probability(value, field_name)


def _normalize_text_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AggregationValidationError(f"{field_name} must be a sequence")
    result = tuple(_require_text(item, field_name) for item in value)
    if not result:
        raise AggregationValidationError(f"{field_name} must not be empty")
    if len(result) != len(set(result)):
        raise AggregationValidationError(f"{field_name} must contain unique values")
    return tuple(sorted(result, key=lambda item: (item.casefold(), item)))


@dataclass(frozen=True)
class AggregatedTheme:
    """Deterministic Phase 7 aggregate over normalized observable support signals."""

    kind: AggregationKind
    key: str
    label: str
    evidence_count: int
    unique_image_count: int
    image_universe_count: int
    image_coverage: float
    average_confidence: float | None
    max_confidence: float | None
    source_diversity: int
    cross_image_consistency: float
    sources: tuple[str, ...]
    supporting_image_ids: tuple[str, ...]
    supporting_signal_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AggregationKind):
            raise AggregationValidationError("aggregated_theme.kind must be an AggregationKind")
        normalized_key = aggregation_key(self.key)
        if normalized_key != self.key:
            raise AggregationValidationError(
                "aggregated_theme.key must already use canonical aggregation normalization"
            )
        object.__setattr__(self, "label", normalize_aggregation_text(self.label))
        evidence_count = _require_positive_int(self.evidence_count, "aggregated_theme.evidence_count")
        unique_image_count = _require_positive_int(
            self.unique_image_count, "aggregated_theme.unique_image_count"
        )
        image_universe_count = _require_positive_int(
            self.image_universe_count, "aggregated_theme.image_universe_count"
        )
        if unique_image_count > image_universe_count:
            raise AggregationValidationError(
                "aggregated_theme.unique_image_count must not exceed image_universe_count"
            )
        coverage = _require_probability(self.image_coverage, "aggregated_theme.image_coverage")
        expected_coverage = unique_image_count / image_universe_count
        if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            raise AggregationValidationError(
                "aggregated_theme.image_coverage must equal unique_image_count / image_universe_count"
            )
        average_confidence = _require_optional_probability(
            self.average_confidence, "aggregated_theme.average_confidence"
        )
        max_confidence = _require_optional_probability(
            self.max_confidence, "aggregated_theme.max_confidence"
        )
        if (average_confidence is None) != (max_confidence is None):
            raise AggregationValidationError(
                "average_confidence and max_confidence must both be null or both be present"
            )
        if (
            average_confidence is not None
            and max_confidence is not None
            and average_confidence > max_confidence
        ):
            raise AggregationValidationError(
                "average_confidence must not exceed max_confidence"
            )
        source_diversity = _require_positive_int(
            self.source_diversity, "aggregated_theme.source_diversity"
        )
        consistency = _require_probability(
            self.cross_image_consistency, "aggregated_theme.cross_image_consistency"
        )
        sources = _normalize_text_sequence(self.sources, "aggregated_theme.sources")
        image_ids = _normalize_text_sequence(
            self.supporting_image_ids, "aggregated_theme.supporting_image_ids"
        )
        signal_ids = _normalize_text_sequence(
            self.supporting_signal_ids, "aggregated_theme.supporting_signal_ids"
        )
        if source_diversity != len(sources):
            raise AggregationValidationError(
                "aggregated_theme.source_diversity must equal the number of sources"
            )
        if unique_image_count != len(image_ids):
            raise AggregationValidationError(
                "aggregated_theme.unique_image_count must equal supporting_image_ids"
            )
        if evidence_count != len(signal_ids):
            raise AggregationValidationError(
                "aggregated_theme.evidence_count must equal supporting_signal_ids"
            )
        object.__setattr__(self, "image_coverage", coverage)
        object.__setattr__(self, "average_confidence", average_confidence)
        object.__setattr__(self, "max_confidence", max_confidence)
        object.__setattr__(self, "cross_image_consistency", consistency)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "supporting_image_ids", image_ids)
        object.__setattr__(self, "supporting_signal_ids", signal_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "label": self.label,
            "evidence_count": self.evidence_count,
            "unique_image_count": self.unique_image_count,
            "image_universe_count": self.image_universe_count,
            "image_coverage": self.image_coverage,
            "average_confidence": self.average_confidence,
            "max_confidence": self.max_confidence,
            "source_diversity": self.source_diversity,
            "cross_image_consistency": self.cross_image_consistency,
            "sources": list(self.sources),
            "supporting_image_ids": list(self.supporting_image_ids),
            "supporting_signal_ids": list(self.supporting_signal_ids),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
