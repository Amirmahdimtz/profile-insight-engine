from __future__ import annotations

import hashlib
import unicodedata

from src.core.profile_analysis.contracts import (
    ContractValidationError,
    Evidence,
    EvidenceType,
    validate_observable_claim,
)
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import (
    VisionEvidenceKind,
    VisionExtractionResult,
    VisionProviderResult,
    VisionValidationError,
)
from src.core.profile_analysis.vision_provider import IVisionProvider
from src.infrastructure.di.inject import inject


_TYPE_BY_KIND = {
    VisionEvidenceKind.SCENE: EvidenceType.SCENE,
    VisionEvidenceKind.OBJECT: EvidenceType.OBJECT,
    VisionEvidenceKind.ACTIVITY: EvidenceType.ACTIVITY,
    VisionEvidenceKind.TOPIC: EvidenceType.TOPIC,
}


def normalize_visual_text(value: str) -> str:
    if not isinstance(value, str):
        raise VisionValidationError("visual text must be a string")
    return " ".join(unicodedata.normalize("NFC", value).split()).strip()


def _evidence_id(image_id: str, evidence_type: EvidenceType, label: str, value: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase5-visual-evidence-v1\0")
    digest.update(image_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(evidence_type.value.encode("ascii"))
    digest.update(b"\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    return f"vision-{digest.hexdigest()[:24]}"


@inject
class EvidenceExtractionService:
    def __init__(self, vision_provider: IVisionProvider):
        self._vision_provider = vision_provider

    async def extract_async(self, image: CanonicalImage) -> VisionExtractionResult:
        if not isinstance(image, CanonicalImage):
            raise VisionValidationError("image must be a CanonicalImage")
        provider_result = await self._vision_provider.extract_async(image)
        return self.normalize_result(image, provider_result)

    def normalize_result(
        self,
        image: CanonicalImage,
        provider_result: VisionProviderResult,
    ) -> VisionExtractionResult:
        if not isinstance(image, CanonicalImage):
            raise VisionValidationError("image must be a CanonicalImage")
        if not isinstance(provider_result, VisionProviderResult):
            raise VisionValidationError("vision provider returned an invalid result type")
        if provider_result.image_id != image.image_id:
            raise VisionValidationError(
                "vision provider result image_id does not match CanonicalImage"
            )

        metadata_base: dict[str, object] = {
            "provider_version": provider_result.provider_version,
            "model_id": provider_result.model_id,
            "model_version": provider_result.model_version,
            "config_version": provider_result.config_version,
            "confidence_semantics": provider_result.confidence_semantics,
            "confidence_calibrated": False,
        }
        evidence: list[Evidence] = []
        seen: set[tuple[str, str]] = set()

        for observation in provider_result.observations:
            label = normalize_visual_text(observation.label)
            if not label:
                raise VisionValidationError(
                    "vision observation label became empty after normalization"
                )
            identity = (observation.kind.value, label.casefold())
            if identity in seen:
                raise VisionValidationError(
                    "vision provider output contains duplicate observations"
                )
            seen.add(identity)
            try:
                validate_observable_claim(None, label)
            except ContractValidationError as exc:
                raise VisionValidationError(
                    "vision provider output contains unsupported claim"
                ) from exc
            evidence_type = _TYPE_BY_KIND[observation.kind]
            metadata = dict(metadata_base)
            metadata["visual_kind"] = observation.kind.value
            evidence.append(
                Evidence(
                    id=_evidence_id(image.image_id, evidence_type, label, label),
                    image_id=image.image_id,
                    type=evidence_type,
                    label=label,
                    value=label,
                    confidence=observation.confidence,
                    source=provider_result.provider,
                    metadata=metadata,
                )
            )

        if provider_result.caption is not None:
            caption = normalize_visual_text(provider_result.caption.text)
            if not caption:
                raise VisionValidationError(
                    "vision caption became empty after normalization"
                )
            try:
                validate_observable_claim(None, caption)
            except ContractValidationError as exc:
                raise VisionValidationError(
                    "vision provider caption contains unsupported claim"
                ) from exc
            metadata = dict(metadata_base)
            metadata["visual_kind"] = "caption"
            evidence.append(
                Evidence(
                    id=_evidence_id(
                        image.image_id,
                        EvidenceType.OTHER_OBSERVABLE,
                        "factual_caption",
                        caption,
                    ),
                    image_id=image.image_id,
                    type=EvidenceType.OTHER_OBSERVABLE,
                    label="factual_caption",
                    value=caption,
                    confidence=provider_result.caption.confidence,
                    source=provider_result.provider,
                    metadata=metadata,
                )
            )

        evidence.sort(
            key=lambda item: (
                item.type.value,
                item.label.casefold(),
                str(item.value),
            )
        )
        return VisionExtractionResult(
            image_id=image.image_id,
            evidence=tuple(evidence),
            provider=provider_result.provider,
            provider_version=provider_result.provider_version,
            model_id=provider_result.model_id,
            model_version=provider_result.model_version,
            config_version=provider_result.config_version,
            confidence_semantics=provider_result.confidence_semantics,
        )
