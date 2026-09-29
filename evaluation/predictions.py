from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from evaluation.common import (
    PREDICTION_SCHEMA_VERSION, EvaluationLabelType, EvaluationValidationError,
    _expect_keys, _expect_mapping, _normalize_json_value, _parse_enum, _require_id,
    _require_probability, _require_sequence, _require_text, _require_version,
    _validate_observable_label, deterministic_json,
)

@dataclass(frozen=True)
class PredictedEvidenceLabel:
    type: EvaluationLabelType
    label: str
    confidence: float
    value: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, EvaluationLabelType):
            raise EvaluationValidationError(
                "prediction.label.type must be an EvaluationLabelType"
            )
        object.__setattr__(self, "label", _validate_observable_label(self.label))
        object.__setattr__(self, "value", _normalize_json_value(self.value, "prediction.label.value"))
        object.__setattr__(
            self,
            "confidence",
            _require_probability(self.confidence, "prediction.label.confidence"),
        )

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.type.value, self.label, deterministic_json(self.value)

    @classmethod
    def from_dict(cls, payload: Any) -> "PredictedEvidenceLabel":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"type", "label", "confidence"}, {"value"})
        return cls(
            type=_parse_enum(EvaluationLabelType, payload["type"], "prediction.label.type"),
            label=payload["label"],
            confidence=payload["confidence"],
            value=payload.get("value"),
        )


@dataclass(frozen=True)
class PredictedClaim:
    key: str
    label: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _require_text(self.key, "prediction.claim.key"))
        object.__setattr__(self, "label", _require_text(self.label, "prediction.claim.label"))
        object.__setattr__(
            self,
            "confidence",
            _require_probability(self.confidence, "prediction.claim.confidence"),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> "PredictedClaim":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"key", "label", "confidence"})
        return cls(
            key=payload["key"],
            label=payload["label"],
            confidence=payload["confidence"],
        )


@dataclass(frozen=True)
class BenchmarkPrediction:
    sample_id: str
    labels: tuple[PredictedEvidenceLabel, ...] = ()
    ocr_text: str | None = None
    claims: tuple[PredictedClaim, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _require_id(self.sample_id, "prediction.sample_id"))
        labels = _require_sequence(self.labels, "prediction.labels")
        if not all(isinstance(item, PredictedEvidenceLabel) for item in labels):
            raise EvaluationValidationError(
                "prediction.labels must contain PredictedEvidenceLabel values"
            )
        identities = [item.identity for item in labels]
        if len(identities) != len(set(identities)):
            raise EvaluationValidationError("prediction.labels must not contain duplicates")
        labels = tuple(sorted(labels, key=lambda item: item.identity))
        if self.ocr_text is not None and not isinstance(self.ocr_text, str):
            raise EvaluationValidationError("prediction.ocr_text must be a string or null")
        claims = _require_sequence(self.claims, "prediction.claims")
        if not all(isinstance(item, PredictedClaim) for item in claims):
            raise EvaluationValidationError(
                "prediction.claims must contain PredictedClaim values"
            )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "claims",
            tuple(sorted(claims, key=lambda item: (item.key, item.label, item.confidence))),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> "BenchmarkPrediction":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"sample_id"}, {"labels", "ocr_text", "claims"})
        labels = _require_sequence(payload.get("labels", ()), "prediction.labels")
        claims = _require_sequence(payload.get("claims", ()), "prediction.claims")
        return cls(
            sample_id=payload["sample_id"],
            labels=tuple(PredictedEvidenceLabel.from_dict(item) for item in labels),
            ocr_text=payload.get("ocr_text"),
            claims=tuple(PredictedClaim.from_dict(item) for item in claims),
        )


@dataclass(frozen=True)
class PredictionSet:
    schema_version: str
    dataset_id: str
    dataset_version: str
    predictions: tuple[BenchmarkPrediction, ...]

    def __post_init__(self) -> None:
        schema_version = _require_version(self.schema_version, "predictions.schema_version")
        if schema_version != PREDICTION_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"predictions.schema_version must be {PREDICTION_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "predictions.dataset_id"))
        object.__setattr__(
            self,
            "dataset_version",
            _require_version(self.dataset_version, "predictions.dataset_version"),
        )
        predictions = _require_sequence(self.predictions, "predictions.predictions")
        if not all(isinstance(item, BenchmarkPrediction) for item in predictions):
            raise EvaluationValidationError(
                "predictions.predictions must contain BenchmarkPrediction values"
            )
        sample_ids = [item.sample_id for item in predictions]
        if len(sample_ids) != len(set(sample_ids)):
            raise EvaluationValidationError("prediction sample_ids must be unique")
        object.__setattr__(
            self, "predictions", tuple(sorted(predictions, key=lambda item: item.sample_id))
        )

    def to_json(self) -> str:
        return deterministic_json(self)

    @classmethod
    def from_dict(cls, payload: Any) -> "PredictionSet":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"schema_version", "dataset_id", "dataset_version", "predictions"},
        )
        predictions = _require_sequence(payload["predictions"], "predictions.predictions")
        return cls(
            schema_version=payload["schema_version"],
            dataset_id=payload["dataset_id"],
            dataset_version=payload["dataset_version"],
            predictions=tuple(BenchmarkPrediction.from_dict(item) for item in predictions),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PredictionSet":
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise EvaluationValidationError("predictions contain invalid JSON") from exc
        return cls.from_dict(parsed)

