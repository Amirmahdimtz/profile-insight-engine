from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Sequence

from src.core.profile_analysis.contracts import ContractValidationError, validate_observable_claim
from src.core.profile_analysis.embedding_contracts import (
    EmbeddingBatchResult,
    EmbeddingValidationError,
    NearDuplicateGroup,
    SemanticRetrievalResult,
    SemanticThemeResult,
    ThemeSimilarityEvidence,
)
from src.core.profile_analysis.embedding_provider import IEmbeddingProvider
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


def _normalize_label(value: str) -> str:
    if not isinstance(value, str):
        raise EmbeddingValidationError("candidate theme labels must be strings")
    normalized = " ".join(unicodedata.normalize("NFC", value).split()).strip()
    if not normalized:
        raise EmbeddingValidationError("candidate theme labels must not be empty")
    try:
        validate_observable_claim(None, normalized)
    except ContractValidationError as exc:
        raise EmbeddingValidationError("candidate theme label violates observable-claim policy") from exc
    return normalized


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise EmbeddingValidationError("image and text embeddings must use the same dimension")
    value = sum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, value))


def _read_similarity(config: ConfigReader, key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbeddingValidationError(f"configuration key '{key}' must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise EmbeddingValidationError(f"configuration key '{key}' must be in [-1, 1]")
    return value


@inject
class SemanticThemeService:
    def __init__(self, embedding_provider: IEmbeddingProvider, config_reader: ConfigReader):
        self._embedding_provider = embedding_provider
        self._config_reader = config_reader

    async def analyze_async(
        self,
        images: Sequence[CanonicalImage],
        candidate_theme_labels: Sequence[str],
    ) -> SemanticThemeResult:
        images = tuple(images)
        if not images or not all(isinstance(image, CanonicalImage) for image in images):
            raise EmbeddingValidationError("images must contain CanonicalImage values")
        image_ids = [image.image_id for image in images]
        if len(image_ids) != len(set(image_ids)):
            raise EmbeddingValidationError("images must have unique image_id values")

        labels = tuple(_normalize_label(value) for value in candidate_theme_labels)
        if not labels:
            raise EmbeddingValidationError("candidate_theme_labels must not be empty")
        folded = [label.casefold() for label in labels]
        if len(folded) != len(set(folded)):
            raise EmbeddingValidationError("candidate_theme_labels must be unique after normalization")

        image_batch = await self._embedding_provider.embed_images_async(images)
        text_batch = await self._embedding_provider.embed_texts_async(labels)
        self._validate_batches(image_batch, text_batch, image_ids, labels)

        theme_threshold = _read_similarity(self._config_reader, "embedding.theme_similarity_threshold")
        duplicate_threshold = _read_similarity(
            self._config_reader, "embedding.near_duplicate_similarity_threshold"
        )
        retrieval_k = self._config_reader.get_positive_int("embedding.retrieval_k")

        similarities: list[ThemeSimilarityEvidence] = []
        retrieval: list[SemanticRetrievalResult] = []
        image_vectors = {vector.item_id: vector.values for vector in image_batch.vectors}
        text_vectors = {vector.item_id: vector.values for vector in text_batch.vectors}

        for label in labels:
            ranked = []
            for image_id in image_ids:
                score = _cosine(image_vectors[image_id], text_vectors[label])
                similarities.append(
                    ThemeSimilarityEvidence(
                        image_id=image_id,
                        theme_label=label,
                        similarity=score,
                        threshold=theme_threshold,
                    )
                )
                ranked.append((image_id, score))
            ranked.sort(key=lambda item: (-item[1], item[0]))
            limited = ranked[: min(retrieval_k, len(ranked))]
            retrieval.append(
                SemanticRetrievalResult(
                    query_label=label,
                    ranked_image_ids=tuple(item[0] for item in limited),
                    scores=tuple(item[1] for item in limited),
                )
            )

        groups = self._near_duplicate_groups(image_ids, image_vectors, duplicate_threshold)
        return SemanticThemeResult(
            theme_similarities=tuple(similarities),
            retrieval_results=tuple(retrieval),
            near_duplicate_groups=groups,
            provider=image_batch.provider,
            provider_version=image_batch.provider_version,
            model_id=image_batch.model_id,
            model_version=image_batch.model_version,
            config_version=image_batch.config_version,
            embedding_dimension=image_batch.dimension,
        )

    @staticmethod
    def _validate_batches(
        image_batch: EmbeddingBatchResult,
        text_batch: EmbeddingBatchResult,
        image_ids: Sequence[str],
        labels: Sequence[str],
    ) -> None:
        if not isinstance(image_batch, EmbeddingBatchResult) or not isinstance(text_batch, EmbeddingBatchResult):
            raise EmbeddingValidationError("embedding provider returned an invalid result type")
        if tuple(vector.item_id for vector in image_batch.vectors) != tuple(image_ids):
            raise EmbeddingValidationError("image embedding batch must preserve input order")
        if tuple(vector.item_id for vector in text_batch.vectors) != tuple(labels):
            raise EmbeddingValidationError("text embedding batch must preserve input order")
        identity_fields = ("provider", "provider_version", "model_id", "model_version", "config_version")
        if any(getattr(image_batch, field) != getattr(text_batch, field) for field in identity_fields):
            raise EmbeddingValidationError("image and text embeddings must come from the same provider/model/config")
        if image_batch.dimension != text_batch.dimension:
            raise EmbeddingValidationError("image and text embeddings must use the same dimension")

    @staticmethod
    def _near_duplicate_groups(
        image_ids: Sequence[str],
        vectors: dict[str, tuple[float, ...]],
        threshold: float,
    ) -> tuple[NearDuplicateGroup, ...]:
        parent = list(range(len(image_ids)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for left in range(len(image_ids)):
            for right in range(left + 1, len(image_ids)):
                if _cosine(vectors[image_ids[left]], vectors[image_ids[right]]) >= threshold:
                    union(left, right)

        members: dict[int, list[str]] = {}
        for index, image_id in enumerate(image_ids):
            members.setdefault(find(index), []).append(image_id)

        groups: list[NearDuplicateGroup] = []
        for image_group in members.values():
            if len(image_group) < 2:
                continue
            digest = hashlib.sha256()
            digest.update(b"phase6-near-duplicate-v1\0")
            for image_id in image_group:
                digest.update(image_id.encode("utf-8"))
                digest.update(b"\0")
            groups.append(
                NearDuplicateGroup(
                    group_id=f"near-duplicate-{digest.hexdigest()[:24]}",
                    image_ids=tuple(image_group),
                )
            )
        groups.sort(key=lambda group: group.image_ids)
        return tuple(groups)
