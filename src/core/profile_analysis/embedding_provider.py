from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.core.profile_analysis.embedding_contracts import EmbeddingBatchResult
from src.core.profile_analysis.image_contracts import CanonicalImage


class IEmbeddingProvider(ABC):
    """Provider-neutral boundary for replaceable local multimodal embedding runtimes."""

    @abstractmethod
    async def embed_images_async(self, images: Sequence[CanonicalImage]) -> EmbeddingBatchResult:
        raise NotImplementedError

    @abstractmethod
    async def embed_texts_async(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        raise NotImplementedError
