from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.ocr_contracts import OcrProviderResult


class IOcrProvider(ABC):
    """Provider-neutral OCR boundary for replaceable local OCR implementations."""

    @abstractmethod
    async def extract_async(self, image: CanonicalImage) -> OcrProviderResult:
        raise NotImplementedError
