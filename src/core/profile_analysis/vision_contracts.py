from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.core.profile_analysis.contracts import Evidence


class VisionValidationError(ValueError):
    """Raised when provider-neutral visual data violates the Phase 5 contract."""


class VisionProviderError(RuntimeError):
    """Raised when the configured vision provider cannot produce valid structured output."""


class VisionEvidenceKind(str, Enum):
    SCENE = "scene"
    OBJECT = "object"
    ACTIVITY = "activity"
    TOPIC = "topic"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VisionValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisionValidationError(f"{field_name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise VisionValidationError(f"{field_name} must be a finite number in [0, 1]")
    return result


def _normalize_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise VisionValidationError(f"{field_name} must be a sequence")
    return tuple(value)


@dataclass(frozen=True)
class VisionObservation:
    kind: VisionEvidenceKind
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VisionEvidenceKind):
            raise VisionValidationError("vision_observation.kind must be a VisionEvidenceKind")
        object.__setattr__(self, "label", _require_text(self.label, "vision_observation.label"))
        object.__setattr__(
            self,
            "confidence",
            _require_probability(self.confidence, "vision_observation.confidence"),
        )


@dataclass(frozen=True)
class VisionCaption:
    text: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "vision_caption.text"))
        object.__setattr__(
            self,
            "confidence",
            _require_probability(self.confidence, "vision_caption.confidence"),
        )


@dataclass(frozen=True)
class VisionProviderResult:
    image_id: str
    observations: tuple[VisionObservation, ...]
    caption: VisionCaption | None
    provider: str
    provider_version: str
    model_id: str
    model_version: str
    config_version: str
    confidence_semantics: str

    def __post_init__(self) -> None:
        for field_name in (
            "image_id",
            "provider",
            "provider_version",
            "model_id",
            "model_version",
            "config_version",
            "confidence_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), f"provider_result.{field_name}"),
            )
        observations = _normalize_sequence(self.observations, "provider_result.observations")
        if not all(isinstance(item, VisionObservation) for item in observations):
            raise VisionValidationError(
                "provider_result.observations must contain VisionObservation values"
            )
        identities = [(item.kind.value, item.label.casefold()) for item in observations]
        if len(identities) != len(set(identities)):
            raise VisionValidationError("provider_result.observations must not contain duplicates")
        ordered = tuple(
            sorted(
                observations,
                key=lambda item: (item.kind.value, item.label.casefold(), item.label),
            )
        )
        object.__setattr__(self, "observations", ordered)
        if self.caption is not None and not isinstance(self.caption, VisionCaption):
            raise VisionValidationError("provider_result.caption must be VisionCaption or null")


@dataclass(frozen=True)
class VisionExtractionResult:
    image_id: str
    evidence: tuple[Evidence, ...]
    provider: str
    provider_version: str
    model_id: str
    model_version: str
    config_version: str
    confidence_semantics: str

    def __post_init__(self) -> None:
        for field_name in (
            "image_id",
            "provider",
            "provider_version",
            "model_id",
            "model_version",
            "config_version",
            "confidence_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), f"vision_result.{field_name}"),
            )
        evidence = _normalize_sequence(self.evidence, "vision_result.evidence")
        if not all(isinstance(item, Evidence) for item in evidence):
            raise VisionValidationError("vision_result.evidence must contain Evidence values")
        if any(item.image_id != self.image_id for item in evidence):
            raise VisionValidationError("vision_result.evidence must reference the same image_id")
        ids = [item.id for item in evidence]
        if len(ids) != len(set(ids)):
            raise VisionValidationError("vision_result.evidence ids must be unique")
        object.__setattr__(self, "evidence", tuple(evidence))
