from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import VisionProviderResult


class IVisionProvider(ABC):
    """Provider-neutral boundary for replaceable local visual-understanding implementations."""

    @abstractmethod
    async def extract_async(self, image: CanonicalImage) -> VisionProviderResult:
        raise NotImplementedError
