from __future__ import annotations

import hashlib
import re
import unicodedata

from src.core.profile_analysis.contracts import Evidence, EvidenceType
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.ocr_contracts import (
    OcrExtractionResult,
    OcrProviderResult,
    OcrTextBlock,
    OcrValidationError,
)
from src.core.profile_analysis.ocr_provider import IOcrProvider
from src.infrastructure.di.inject import inject


_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")
_ARABIC_SCRIPT_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def normalize_ocr_text(value: str) -> str:
    """Apply minimal deterministic text cleanup while preserving the raw OCR text separately."""

    if not isinstance(value, str):
        raise OcrValidationError("OCR text must be a string")
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u202f", " ")
    normalized = normalized.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _is_arabic_script(char: str) -> bool:
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in _ARABIC_SCRIPT_RANGES)


def infer_script_and_language_hint(text: str) -> tuple[str, str | None]:
    arabic_letters = 0
    latin_letters = 0
    digits = 0
    other_letters = 0
    for char in text:
        if char.isdigit():
            digits += 1
        elif _is_arabic_script(char) and char.isalpha():
            arabic_letters += 1
        elif "LATIN" in unicodedata.name(char, "") and char.isalpha():
            latin_letters += 1
        elif char.isalpha():
            other_letters += 1

    if arabic_letters and latin_letters:
        return "mixed_arabic_latin", "mixed"
    if arabic_letters:
        return "arabic", "fa"
    if latin_letters:
        return "latin", "en"
    if digits and not other_letters:
        return "numeric", None
    return "other", None


def _evidence_id(image_id: str, block: OcrTextBlock) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase4-ocr-evidence-v1\0")
    digest.update(image_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(block.order).encode("ascii"))
    digest.update(b"\0")
    digest.update(block.raw_text.encode("utf-8"))
    return f"ocr-{digest.hexdigest()[:24]}"


@inject
class OcrEvidenceService:
    def __init__(self, ocr_provider: IOcrProvider):
        self._ocr_provider = ocr_provider

    async def extract_async(self, image: CanonicalImage) -> OcrExtractionResult:
        if not isinstance(image, CanonicalImage):
            raise OcrValidationError("image must be a CanonicalImage")
        provider_result = await self._ocr_provider.extract_async(image)
        if not isinstance(provider_result, OcrProviderResult):
            raise OcrValidationError("OCR provider returned an invalid result type")
        if provider_result.image_id != image.image_id:
            raise OcrValidationError("OCR provider result image_id does not match CanonicalImage")

        blocks: list[OcrTextBlock] = []
        evidence: list[Evidence] = []
        for provider_block in provider_result.blocks:
            normalized_text = normalize_ocr_text(provider_block.raw_text)
            if not normalized_text:
                continue
            script_hint, language_hint = infer_script_and_language_hint(normalized_text)
            block = OcrTextBlock(
                order=provider_block.order,
                raw_text=provider_block.raw_text,
                normalized_text=normalized_text,
                confidence=provider_block.confidence,
                script_hint=script_hint,
                language_hint=language_hint,
                region=provider_block.region,
            )
            metadata: dict[str, object] = {
                "block_order": block.order,
                "raw_text": block.raw_text,
                "script_hint": block.script_hint,
                "language_hint": block.language_hint,
                "provider_version": provider_result.provider_version,
                "model_id": provider_result.model_id,
                "config_version": provider_result.config_version,
            }
            if block.region is not None:
                metadata["region"] = block.region.to_dict()
            blocks.append(block)
            evidence.append(
                Evidence(
                    id=_evidence_id(image.image_id, block),
                    image_id=image.image_id,
                    type=EvidenceType.OCR_TEXT,
                    label="ocr_text",
                    value=block.normalized_text,
                    confidence=block.confidence,
                    source=provider_result.provider,
                    metadata=metadata,
                )
            )

        return OcrExtractionResult(
            image_id=image.image_id,
            blocks=tuple(blocks),
            evidence=tuple(evidence),
            provider=provider_result.provider,
            provider_version=provider_result.provider_version,
            model_id=provider_result.model_id,
            config_version=provider_result.config_version,
        )
