from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import (
    VisionEvidenceKind,
    VisionObservation,
    VisionProviderError,
    VisionProviderResult,
    VisionValidationError,
)
from src.core.profile_analysis.vision_provider import IVisionProvider
from src.infrastructure.utils.config_reader import ConfigReader


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VisionValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise VisionValidationError(f"{field_name} must be boolean")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VisionValidationError(f"{field_name} must be a positive integer")
    return value


def _require_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisionValidationError(f"{field_name} must be a finite number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise VisionValidationError(f"{field_name} must be a finite number in [0, 1]")
    return normalized


@dataclass(frozen=True)
class SiglipZeroShotVisionSettings:
    model_id: str
    model_revision: str
    config_version: str
    device: str
    local_files_only: bool
    minimum_score: float
    max_observations_per_kind: int
    confidence_semantics: str
    hypothesis_template: str

    def __post_init__(self) -> None:
        prefix = "vision.siglip_zero_shot_candidate"
        for field_name in (
            "model_id",
            "model_revision",
            "config_version",
            "device",
            "confidence_semantics",
            "hypothesis_template",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(
                    getattr(self, field_name),
                    f"{prefix}.{field_name}",
                ),
            )
        object.__setattr__(
            self,
            "local_files_only",
            _require_bool(
                self.local_files_only,
                f"{prefix}.local_files_only",
            ),
        )
        object.__setattr__(
            self,
            "minimum_score",
            _require_probability(
                self.minimum_score,
                f"{prefix}.minimum_score",
            ),
        )
        object.__setattr__(
            self,
            "max_observations_per_kind",
            _require_positive_int(
                self.max_observations_per_kind,
                f"{prefix}.max_observations_per_kind",
            ),
        )
        try:
            rendered = self.hypothesis_template.format("candidate")
        except (IndexError, KeyError, ValueError) as exc:
            raise VisionValidationError(
                f"{prefix}.hypothesis_template must contain one positional placeholder"
            ) from exc
        if rendered == self.hypothesis_template:
            raise VisionValidationError(
                f"{prefix}.hypothesis_template must contain one positional placeholder"
            )

    @classmethod
    def from_config(
        cls,
        config_reader: ConfigReader,
    ) -> "SiglipZeroShotVisionSettings":
        prefix = "vision.siglip_zero_shot_candidate"
        return cls(
            model_id=config_reader.get_non_empty_string(f"{prefix}.model_id"),
            model_revision=config_reader.get_non_empty_string(
                f"{prefix}.model_revision"
            ),
            config_version=config_reader.get_non_empty_string(
                f"{prefix}.config_version"
            ),
            device=config_reader.get_non_empty_string(f"{prefix}.device"),
            local_files_only=config_reader.get(f"{prefix}.local_files_only"),
            minimum_score=config_reader.get(f"{prefix}.minimum_score"),
            max_observations_per_kind=config_reader.get_positive_int(
                f"{prefix}.max_observations_per_kind"
            ),
            confidence_semantics=config_reader.get_non_empty_string(
                f"{prefix}.confidence_semantics"
            ),
            hypothesis_template=config_reader.get_non_empty_string(
                f"{prefix}.hypothesis_template"
            ),
        )


@dataclass(frozen=True)
class ZeroShotVisionScore:
    kind: VisionEvidenceKind
    label: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VisionEvidenceKind):
            raise VisionValidationError(
                "zero_shot_vision_score.kind must be a VisionEvidenceKind"
            )
        object.__setattr__(
            self,
            "label",
            _require_text(
                self.label,
                "zero_shot_vision_score.label",
            ),
        )
        object.__setattr__(
            self,
            "score",
            _require_probability(
                self.score,
                "zero_shot_vision_score.score",
            ),
        )


def _normalize_taxonomy(
    taxonomy: Mapping[VisionEvidenceKind, Sequence[str]],
) -> dict[VisionEvidenceKind, tuple[str, ...]]:
    if not isinstance(taxonomy, Mapping):
        raise VisionValidationError(
            "vision taxonomy must be a mapping"
        )
    normalized: dict[VisionEvidenceKind, tuple[str, ...]] = {}
    for kind in VisionEvidenceKind:
        values = taxonomy.get(kind)
        if (
            isinstance(values, (str, bytes, bytearray))
            or not isinstance(values, Sequence)
            or not values
        ):
            raise VisionValidationError(
                f"vision taxonomy must contain labels for {kind.value}"
            )
        labels = tuple(
            sorted(
                (
                    _require_text(
                        value,
                        f"vision taxonomy {kind.value} label",
                    )
                    for value in values
                ),
                key=lambda value: (value.casefold(), value),
            )
        )
        if len(labels) != len(set(label.casefold() for label in labels)):
            raise VisionValidationError(
                f"vision taxonomy {kind.value} labels must be unique"
            )
        normalized[kind] = labels
    unknown = set(taxonomy) - set(VisionEvidenceKind)
    if unknown:
        raise VisionValidationError(
            "vision taxonomy contains unsupported evidence kinds"
        )
    return normalized


class SiglipZeroShotVisionProvider(IVisionProvider):
    """Phase 10 benchmark candidate using local SigLIP2 zero-shot classification."""

    def __init__(
        self,
        config_reader: ConfigReader,
        taxonomy: Mapping[VisionEvidenceKind, Sequence[str]],
        settings: SiglipZeroShotVisionSettings | None = None,
    ):
        self._settings = settings or SiglipZeroShotVisionSettings.from_config(
            config_reader
        )
        self._taxonomy = _normalize_taxonomy(taxonomy)
        self._entries = tuple(
            (kind, label)
            for kind in VisionEvidenceKind
            for label in self._taxonomy[kind]
        )
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._provider_version: str | None = None
        self._resolved_model_version: str | None = None

    @property
    def taxonomy(self) -> Mapping[VisionEvidenceKind, tuple[str, ...]]:
        return dict(self._taxonomy)

    async def extract_async(
        self,
        image: CanonicalImage,
    ) -> VisionProviderResult:
        scores = await self.score_async(image)
        return self.result_from_scores(image.image_id, scores)

    async def score_async(
        self,
        image: CanonicalImage,
    ) -> tuple[ZeroShotVisionScore, ...]:
        if not isinstance(image, CanonicalImage):
            raise VisionValidationError("image must be a CanonicalImage")
        path = Path(image.storage_reference)
        if not path.is_file():
            raise VisionProviderError(
                "canonical image storage reference does not exist"
            )

        await self._ensure_loaded_async()
        async with self._inference_lock:
            try:
                raw_scores = await asyncio.to_thread(
                    self._score_image_sync,
                    path,
                )
            except VisionProviderError:
                raise
            except Exception as exc:
                raise VisionProviderError(
                    "siglip zero-shot vision inference failed"
                ) from exc

        return tuple(
            ZeroShotVisionScore(
                kind=kind,
                label=label,
                score=score,
            )
            for (kind, label), score in zip(
                self._entries,
                raw_scores,
            )
        )

    def result_from_scores(
        self,
        image_id: str,
        scores: Sequence[ZeroShotVisionScore],
        *,
        minimum_score: float | None = None,
    ) -> VisionProviderResult:
        image_id = _require_text(image_id, "image_id")
        threshold = (
            self._settings.minimum_score
            if minimum_score is None
            else _require_probability(
                minimum_score,
                "minimum_score",
            )
        )
        materialized = tuple(scores)
        if len(materialized) != len(self._entries):
            raise VisionValidationError(
                "siglip zero-shot score count does not match taxonomy"
            )
        identities = tuple(
            (item.kind, item.label)
            for item in materialized
        )
        if identities != self._entries:
            raise VisionValidationError(
                "siglip zero-shot scores do not match taxonomy ordering"
            )

        observations: list[VisionObservation] = []
        for kind in VisionEvidenceKind:
            ranked = sorted(
                (
                    item
                    for item in materialized
                    if item.kind is kind
                    and item.score >= threshold
                ),
                key=lambda item: (
                    -item.score,
                    item.label.casefold(),
                    item.label,
                ),
            )
            for item in ranked[
                : self._settings.max_observations_per_kind
            ]:
                observations.append(
                    VisionObservation(
                        kind=item.kind,
                        label=item.label,
                        confidence=item.score,
                    )
                )

        if (
            self._provider_version is None
            or self._resolved_model_version is None
        ):
            raise VisionProviderError(
                "siglip zero-shot vision runtime metadata is unavailable"
            )

        return VisionProviderResult(
            image_id=image_id,
            observations=tuple(observations),
            caption=None,
            provider="transformers_zero_shot",
            provider_version=self._provider_version,
            model_id=self._settings.model_id,
            model_version=self._resolved_model_version,
            config_version=self._settings.config_version,
            confidence_semantics=self._settings.confidence_semantics,
        )

    async def _ensure_loaded_async(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                await asyncio.to_thread(self._load_sync)
            except VisionProviderError:
                raise
            except Exception as exc:
                raise VisionProviderError(
                    "siglip zero-shot vision model initialization failed"
                ) from exc

    def _load_sync(self) -> None:
        try:
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
            from transformers import (
                AutoModelForZeroShotImageClassification,
                AutoProcessor,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise VisionProviderError(
                "siglip zero-shot vision runtime dependencies are not installed"
            ) from exc

        try:
            if self._settings.local_files_only:
                config_path = Path(
                    hf_hub_download(
                        repo_id=self._settings.model_id,
                        filename="config.json",
                        revision=self._settings.model_revision,
                        local_files_only=True,
                    )
                )
                model_source: str | Path = config_path.parent
                load_revision: str | None = None
            else:
                model_source = self._settings.model_id
                load_revision = self._settings.model_revision

            processor = AutoProcessor.from_pretrained(
                model_source,
                revision=load_revision,
                local_files_only=self._settings.local_files_only,
            )
            model = AutoModelForZeroShotImageClassification.from_pretrained(
                model_source,
                revision=load_revision,
                local_files_only=self._settings.local_files_only,
            )
            config_path = Path(
                hf_hub_download(
                    repo_id=self._settings.model_id,
                    filename="config.json",
                    revision=self._settings.model_revision,
                    local_files_only=True,
                )
            )
            snapshot_path = config_path.parent
            model.to(self._settings.device)
            model.eval()
            if (
                not callable(getattr(processor, "image_processor", None))
                and getattr(processor, "image_processor", None) is None
            ):
                raise VisionProviderError(
                    "siglip zero-shot processor has no image processor"
                )
            if getattr(processor, "tokenizer", None) is None:
                raise VisionProviderError(
                    "siglip zero-shot processor has no tokenizer"
                )
        except VisionProviderError:
            raise
        except Exception as exc:
            raise VisionProviderError(
                "siglip zero-shot vision model could not be loaded"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model
        self._provider_version = f"transformers-{transformers.__version__}"
        self._resolved_model_version = snapshot_path.name

    def _score_image_sync(
        self,
        image_path: Path,
    ) -> list[float]:
        if (
            self._torch is None
            or self._processor is None
            or self._model is None
        ):
            raise VisionProviderError(
                "siglip zero-shot vision model is not initialized"
            )

        texts = [
            self._settings.hypothesis_template.format(label)
            for _, label in self._entries
        ]
        try:
            with Image.open(image_path) as source:
                source.load()
                image = source.convert("RGB")
            try:
                image_inputs = self._processor.image_processor(
                    images=[image],
                    return_tensors="pt",
                )
                text_inputs = self._processor.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {
                    **dict(image_inputs),
                    **dict(text_inputs),
                }
                inputs = {
                    name: value.to(self._settings.device)
                    for name, value in inputs.items()
                }
                with self._torch.inference_mode():
                    outputs = self._model(**inputs)
                logits = getattr(outputs, "logits_per_image", None)
                shape = getattr(logits, "shape", ())
                if (
                    len(shape) != 2
                    or int(shape[0]) != 1
                    or int(shape[1]) != len(self._entries)
                ):
                    raise VisionProviderError(
                        "siglip zero-shot vision returned an unexpected score shape"
                    )
                values = (
                    self._torch.sigmoid(logits[0])
                    .detach()
                    .cpu()
                    .tolist()
                )
            finally:
                image.close()
        except VisionProviderError:
            raise
        except Exception as exc:
            raise VisionProviderError(
                "siglip zero-shot vision inference failed"
            ) from exc

        scores: list[float] = []
        for value in values:
            scores.append(
                _require_probability(
                    value,
                    "siglip zero-shot score",
                )
            )
        return scores
