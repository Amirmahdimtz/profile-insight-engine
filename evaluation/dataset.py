from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.common import (
    DATASET_SCHEMA_VERSION, LABEL_SCHEMA_VERSION, DataPolicy, DatasetSlice, DatasetSplit,
    EvaluationLabelType, EvaluationValidationError, UnsupportedClaimCategory,
    _expect_keys, _expect_mapping, _normalize_json_value, _parse_enum, _require_id,
    _require_relative_path, _require_sequence, _require_version, _to_primitive,
    _validate_observable_label, deterministic_json,
)

@dataclass(frozen=True)
class EvaluationImageReference:
    image_id: str
    relative_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _require_id(self.image_id, "image.image_id"))
        object.__setattr__(self, "relative_path", _require_relative_path(self.relative_path, "image.relative_path"))

    @classmethod
    def from_dict(cls, payload: Any) -> "EvaluationImageReference":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"image_id", "relative_path"})
        return cls(image_id=payload["image_id"], relative_path=payload["relative_path"])


@dataclass(frozen=True)
class ExpectedEvidenceLabel:
    label_id: str
    type: EvaluationLabelType
    label: str
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_id", _require_id(self.label_id, "label.label_id"))
        if not isinstance(self.type, EvaluationLabelType):
            raise EvaluationValidationError("label.type must be an EvaluationLabelType")
        object.__setattr__(self, "label", _validate_observable_label(self.label))
        object.__setattr__(self, "value", _normalize_json_value(self.value, "label.value"))

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.type.value, self.label, deterministic_json(self.value)

    @classmethod
    def from_dict(cls, payload: Any) -> "ExpectedEvidenceLabel":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"label_id", "type", "label"}, {"value"})
        return cls(
            label_id=payload["label_id"],
            type=_parse_enum(EvaluationLabelType, payload["type"], "label.type"),
            label=payload["label"],
            value=payload.get("value"),
        )


@dataclass(frozen=True)
class UnsupportedClaimAnnotation:
    category: UnsupportedClaimCategory

    def __post_init__(self) -> None:
        if not isinstance(self.category, UnsupportedClaimCategory):
            raise EvaluationValidationError(
                "unsupported_claim.category must be an UnsupportedClaimCategory"
            )

    @classmethod
    def from_dict(cls, payload: Any) -> "UnsupportedClaimAnnotation":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"category"})
        return cls(
            category=_parse_enum(
                UnsupportedClaimCategory,
                payload["category"],
                "unsupported_claim.category",
            )
        )


@dataclass(frozen=True)
class EvaluationGroundTruth:
    labels: tuple[ExpectedEvidenceLabel, ...] = ()
    ocr_text: str | None = None
    unsupported_claims: tuple[UnsupportedClaimAnnotation, ...] = ()

    def __post_init__(self) -> None:
        labels = _require_sequence(self.labels, "ground_truth.labels")
        if not all(isinstance(item, ExpectedEvidenceLabel) for item in labels):
            raise EvaluationValidationError(
                "ground_truth.labels must contain ExpectedEvidenceLabel values"
            )
        label_ids = [item.label_id for item in labels]
        if len(label_ids) != len(set(label_ids)):
            raise EvaluationValidationError("ground_truth.label_id values must be unique")
        identities = [item.identity for item in labels]
        if len(identities) != len(set(identities)):
            raise EvaluationValidationError("ground_truth labels must not contain duplicate identities")
        labels = tuple(sorted(labels, key=lambda item: item.label_id))

        if self.ocr_text is not None and not isinstance(self.ocr_text, str):
            raise EvaluationValidationError("ground_truth.ocr_text must be a string or null")

        unsupported = _require_sequence(
            self.unsupported_claims, "ground_truth.unsupported_claims"
        )
        if not all(isinstance(item, UnsupportedClaimAnnotation) for item in unsupported):
            raise EvaluationValidationError(
                "ground_truth.unsupported_claims must contain UnsupportedClaimAnnotation values"
            )
        categories = [item.category for item in unsupported]
        if len(categories) != len(set(categories)):
            raise EvaluationValidationError(
                "ground_truth.unsupported_claim categories must be unique"
            )
        unsupported = tuple(sorted(unsupported, key=lambda item: item.category.value))

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "unsupported_claims", unsupported)

    @classmethod
    def from_dict(cls, payload: Any) -> "EvaluationGroundTruth":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, set(), {"labels", "ocr_text", "unsupported_claims"})
        labels = _require_sequence(payload.get("labels", ()), "ground_truth.labels")
        unsupported = _require_sequence(
            payload.get("unsupported_claims", ()), "ground_truth.unsupported_claims"
        )
        return cls(
            labels=tuple(ExpectedEvidenceLabel.from_dict(item) for item in labels),
            ocr_text=payload.get("ocr_text"),
            unsupported_claims=tuple(
                UnsupportedClaimAnnotation.from_dict(item) for item in unsupported
            ),
        )


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    split: DatasetSplit
    slices: tuple[DatasetSlice, ...]
    image: EvaluationImageReference
    ground_truth: EvaluationGroundTruth

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _require_id(self.sample_id, "sample.sample_id"))
        if not isinstance(self.split, DatasetSplit):
            raise EvaluationValidationError("sample.split must be a DatasetSplit")
        slices = _require_sequence(self.slices, "sample.slices")
        if not slices or not all(isinstance(item, DatasetSlice) for item in slices):
            raise EvaluationValidationError(
                "sample.slices must contain at least one DatasetSlice"
            )
        if len(slices) != len(set(slices)):
            raise EvaluationValidationError("sample.slices must be unique")
        object.__setattr__(self, "slices", tuple(sorted(slices, key=lambda item: item.value)))
        if not isinstance(self.image, EvaluationImageReference):
            raise EvaluationValidationError(
                "sample.image must be an EvaluationImageReference"
            )
        if not isinstance(self.ground_truth, EvaluationGroundTruth):
            raise EvaluationValidationError(
                "sample.ground_truth must be an EvaluationGroundTruth"
            )

    @classmethod
    def from_dict(cls, payload: Any) -> "EvaluationSample":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"sample_id", "split", "slices", "image", "ground_truth"})
        slices = _require_sequence(payload["slices"], "sample.slices")
        return cls(
            sample_id=payload["sample_id"],
            split=_parse_enum(DatasetSplit, payload["split"], "sample.split"),
            slices=tuple(
                _parse_enum(DatasetSlice, value, "sample.slices") for value in slices
            ),
            image=EvaluationImageReference.from_dict(payload["image"]),
            ground_truth=EvaluationGroundTruth.from_dict(payload["ground_truth"]),
        )


@dataclass(frozen=True)
class EvaluationDatasetManifest:
    schema_version: str
    dataset_id: str
    dataset_version: str
    label_schema_version: str
    data_policy: DataPolicy
    samples: tuple[EvaluationSample, ...]

    def __post_init__(self) -> None:
        schema_version = _require_version(self.schema_version, "manifest.schema_version")
        if schema_version != DATASET_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"manifest.schema_version must be {DATASET_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "manifest.dataset_id"))
        object.__setattr__(
            self, "dataset_version", _require_version(self.dataset_version, "manifest.dataset_version")
        )
        label_schema_version = _require_version(
            self.label_schema_version, "manifest.label_schema_version"
        )
        if label_schema_version != LABEL_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"manifest.label_schema_version must be {LABEL_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "label_schema_version", label_schema_version)
        if self.data_policy is not DataPolicy.CONSENTED_OR_AUTHORIZED:
            raise EvaluationValidationError(
                "manifest.data_policy must declare consented_or_authorized data"
            )

        samples = _require_sequence(self.samples, "manifest.samples")
        if not samples:
            raise EvaluationValidationError("manifest.samples must not be empty")
        if not all(isinstance(item, EvaluationSample) for item in samples):
            raise EvaluationValidationError(
                "manifest.samples must contain EvaluationSample values"
            )

        sample_ids: set[str] = set()
        image_ids: dict[str, EvaluationSample] = {}
        paths: dict[str, EvaluationSample] = {}
        for sample in samples:
            if sample.sample_id in sample_ids:
                raise EvaluationValidationError(
                    f"duplicate sample_id: {sample.sample_id}"
                )
            sample_ids.add(sample.sample_id)

            previous_by_id = image_ids.get(sample.image.image_id)
            if previous_by_id is not None:
                if previous_by_id.split is not sample.split:
                    raise EvaluationValidationError(
                        f"train/eval leakage for image_id: {sample.image.image_id}"
                    )
                raise EvaluationValidationError(
                    f"duplicate image_id: {sample.image.image_id}"
                )
            image_ids[sample.image.image_id] = sample

            previous_by_path = paths.get(sample.image.relative_path)
            if previous_by_path is not None:
                if previous_by_path.split is not sample.split:
                    raise EvaluationValidationError(
                        f"train/eval leakage for image reference: {sample.image.relative_path}"
                    )
                raise EvaluationValidationError(
                    f"duplicate image reference: {sample.image.relative_path}"
                )
            paths[sample.image.relative_path] = sample

        if not any(sample.split is DatasetSplit.EVAL for sample in samples):
            raise EvaluationValidationError(
                "manifest must contain at least one eval sample; an empty train split is allowed"
            )

        object.__setattr__(self, "samples", tuple(sorted(samples, key=lambda item: item.sample_id)))

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    def to_json(self) -> str:
        return deterministic_json(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_references(self, dataset_root: str | Path) -> str:
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            raise EvaluationValidationError("dataset_root must be an existing directory")

        content_entries: list[dict[str, str]] = []
        content_hashes: dict[str, EvaluationSample] = {}
        for sample in self.samples:
            candidate = (root / sample.image.relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise EvaluationValidationError(
                    f"image reference escapes dataset_root: {sample.image.relative_path}"
                ) from exc
            if not candidate.is_file():
                raise EvaluationValidationError(
                    f"image reference does not exist: {sample.image.relative_path}"
                )

            content_sha256 = self._file_sha256(candidate)
            previous = content_hashes.get(content_sha256)
            if previous is not None:
                if previous.split is not sample.split:
                    raise EvaluationValidationError(
                        "train/eval leakage for identical image content: "
                        f"{previous.image.relative_path}, {sample.image.relative_path}"
                    )
                raise EvaluationValidationError(
                    "duplicate image content: "
                    f"{previous.image.relative_path}, {sample.image.relative_path}"
                )
            content_hashes[content_sha256] = sample
            content_entries.append(
                {
                    "sample_id": sample.sample_id,
                    "image_id": sample.image.image_id,
                    "relative_path": sample.image.relative_path,
                    "sha256": content_sha256,
                }
            )

        payload = deterministic_json(content_entries).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, payload: Any) -> "EvaluationDatasetManifest":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"schema_version", "dataset_id", "dataset_version", "label_schema_version", "data_policy", "samples"},
        )
        samples = _require_sequence(payload["samples"], "manifest.samples")
        return cls(
            schema_version=payload["schema_version"],
            dataset_id=payload["dataset_id"],
            dataset_version=payload["dataset_version"],
            label_schema_version=payload["label_schema_version"],
            data_policy=_parse_enum(DataPolicy, payload["data_policy"], "manifest.data_policy"),
            samples=tuple(EvaluationSample.from_dict(item) for item in samples),
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvaluationDatasetManifest":
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise EvaluationValidationError("manifest contains invalid JSON") from exc
        return cls.from_dict(parsed)

