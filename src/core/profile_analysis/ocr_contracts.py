from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.core.profile_analysis.contracts import Evidence


class OcrValidationError(ValueError):
    """Raised when provider-neutral OCR data violates the Phase 4 contract."""


class OcrProviderError(RuntimeError):
    """Raised when the configured OCR provider cannot produce valid OCR output."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OcrValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _require_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OcrValidationError(f"{field_name} must be a finite number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise OcrValidationError(f"{field_name} must be a finite number in [0, 1]")
    return normalized


def _normalize_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OcrValidationError(f"{field_name} must be a sequence")
    return tuple(value)


@dataclass(frozen=True)
class OcrRegion:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "width", "height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise OcrValidationError(f"region.{field_name} must be an integer")
        if self.x < 0 or self.y < 0:
            raise OcrValidationError("region.x and region.y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise OcrValidationError("region.width and region.height must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class OcrProviderBlock:
    order: int
    raw_text: str
    confidence: float
    region: OcrRegion | None = None

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise OcrValidationError("provider_block.order must be a non-negative integer")
        object.__setattr__(self, "raw_text", _require_text(self.raw_text, "provider_block.raw_text"))
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "provider_block.confidence"))
        if self.region is not None and not isinstance(self.region, OcrRegion):
            raise OcrValidationError("provider_block.region must be OcrRegion or null")


@dataclass(frozen=True)
class OcrProviderResult:
    image_id: str
    blocks: tuple[OcrProviderBlock, ...]
    provider: str
    provider_version: str
    model_id: str
    config_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _require_text(self.image_id, "provider_result.image_id"))
        object.__setattr__(self, "provider", _require_text(self.provider, "provider_result.provider"))
        object.__setattr__(self, "provider_version", _require_text(self.provider_version, "provider_result.provider_version"))
        object.__setattr__(self, "model_id", _require_text(self.model_id, "provider_result.model_id"))
        object.__setattr__(self, "config_version", _require_text(self.config_version, "provider_result.config_version"))
        blocks = _normalize_sequence(self.blocks, "provider_result.blocks")
        if not all(isinstance(item, OcrProviderBlock) for item in blocks):
            raise OcrValidationError("provider_result.blocks must contain OcrProviderBlock values")
        orders = [item.order for item in blocks]
        if len(orders) != len(set(orders)):
            raise OcrValidationError("provider_result.blocks must have unique order values")
        object.__setattr__(self, "blocks", tuple(sorted(blocks, key=lambda item: item.order)))


@dataclass(frozen=True)
class OcrTextBlock:
    order: int
    raw_text: str
    normalized_text: str
    confidence: float
    script_hint: str
    language_hint: str | None
    region: OcrRegion | None = None

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise OcrValidationError("ocr_block.order must be a non-negative integer")
        object.__setattr__(self, "raw_text", _require_text(self.raw_text, "ocr_block.raw_text"))
        object.__setattr__(self, "normalized_text", _require_text(self.normalized_text, "ocr_block.normalized_text"))
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "ocr_block.confidence"))
        object.__setattr__(self, "script_hint", _require_text(self.script_hint, "ocr_block.script_hint"))
        object.__setattr__(self, "language_hint", _require_optional_text(self.language_hint, "ocr_block.language_hint"))
        if self.region is not None and not isinstance(self.region, OcrRegion):
            raise OcrValidationError("ocr_block.region must be OcrRegion or null")


@dataclass(frozen=True)
class OcrExtractionResult:
    image_id: str
    blocks: tuple[OcrTextBlock, ...]
    evidence: tuple[Evidence, ...]
    provider: str
    provider_version: str
    model_id: str
    config_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _require_text(self.image_id, "ocr_result.image_id"))
        object.__setattr__(self, "provider", _require_text(self.provider, "ocr_result.provider"))
        object.__setattr__(self, "provider_version", _require_text(self.provider_version, "ocr_result.provider_version"))
        object.__setattr__(self, "model_id", _require_text(self.model_id, "ocr_result.model_id"))
        object.__setattr__(self, "config_version", _require_text(self.config_version, "ocr_result.config_version"))
        blocks = _normalize_sequence(self.blocks, "ocr_result.blocks")
        if not all(isinstance(item, OcrTextBlock) for item in blocks):
            raise OcrValidationError("ocr_result.blocks must contain OcrTextBlock values")
        orders = [item.order for item in blocks]
        if len(orders) != len(set(orders)):
            raise OcrValidationError("ocr_result.blocks must have unique order values")
        ordered_blocks = tuple(sorted(blocks, key=lambda item: item.order))
        object.__setattr__(self, "blocks", ordered_blocks)

        evidence = _normalize_sequence(self.evidence, "ocr_result.evidence")
        if not all(isinstance(item, Evidence) for item in evidence):
            raise OcrValidationError("ocr_result.evidence must contain Evidence values")
        if len(evidence) != len(ordered_blocks):
            raise OcrValidationError("ocr_result.evidence must contain one item per OCR block")
        if any(item.image_id != self.image_id for item in evidence):
            raise OcrValidationError("ocr_result.evidence must reference the same image_id")
        object.__setattr__(self, "evidence", tuple(evidence))

    @property
    def raw_text(self) -> str:
        return "\n".join(block.raw_text for block in self.blocks)

    @property
    def normalized_text(self) -> str:
        return "\n".join(block.normalized_text for block in self.blocks)
