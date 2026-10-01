from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class EmbeddingValidationError(ValueError):
    """Raised when Phase 6 embedding or similarity data violates its contract."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EmbeddingValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbeddingValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EmbeddingValidationError(f"{field_name} must be a finite number")
    return result


def _require_similarity(value: object, field_name: str) -> float:
    result = _require_finite(value, field_name)
    if result < -1.0 or result > 1.0:
        raise EmbeddingValidationError(f"{field_name} must be in [-1, 1]")
    return result


@dataclass(frozen=True)
class EmbeddingVector:
    item_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _require_text(self.item_id, "embedding.item_id"))
        if isinstance(self.values, Sequence) and not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise EmbeddingValidationError("embedding.values must not be empty")
        values = tuple(_require_finite(value, "embedding.values") for value in self.values)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise EmbeddingValidationError("embedding.values must have non-zero norm")
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise EmbeddingValidationError("embedding.values must be L2-normalized")
        object.__setattr__(self, "values", values)

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: tuple[EmbeddingVector, ...]
    provider: str
    provider_version: str
    model_id: str
    model_version: str
    config_version: str

    def __post_init__(self) -> None:
        if isinstance(self.vectors, Sequence) and not isinstance(self.vectors, tuple):
            object.__setattr__(self, "vectors", tuple(self.vectors))
        if not self.vectors or not all(isinstance(item, EmbeddingVector) for item in self.vectors):
            raise EmbeddingValidationError("embedding batch must contain EmbeddingVector values")
        ids = [item.item_id for item in self.vectors]
        if len(ids) != len(set(ids)):
            raise EmbeddingValidationError("embedding batch item_id values must be unique")
        dimensions = {item.dimension for item in self.vectors}
        if len(dimensions) != 1:
            raise EmbeddingValidationError("embedding batch vectors must have one dimension")
        for name in ("provider", "provider_version", "model_id", "model_version", "config_version"):
            object.__setattr__(self, name, _require_text(getattr(self, name), f"embedding.{name}"))

    @property
    def dimension(self) -> int:
        return self.vectors[0].dimension


@dataclass(frozen=True)
class ThemeSimilarityEvidence:
    image_id: str
    theme_label: str
    similarity: float
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _require_text(self.image_id, "theme_similarity.image_id"))
        object.__setattr__(self, "theme_label", _require_text(self.theme_label, "theme_similarity.theme_label"))
        object.__setattr__(self, "similarity", _require_similarity(self.similarity, "theme_similarity.similarity"))
        object.__setattr__(self, "threshold", _require_similarity(self.threshold, "theme_similarity.threshold"))

    @property
    def matched(self) -> bool:
        return self.similarity >= self.threshold


@dataclass(frozen=True)
class SemanticRetrievalResult:
    query_label: str
    ranked_image_ids: tuple[str, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_label", _require_text(self.query_label, "retrieval.query_label"))
        if len(self.ranked_image_ids) != len(self.scores):
            raise EmbeddingValidationError("retrieval image ids and scores must have equal length")
        image_ids = tuple(_require_text(value, "retrieval.ranked_image_ids") for value in self.ranked_image_ids)
        if len(image_ids) != len(set(image_ids)):
            raise EmbeddingValidationError("retrieval.ranked_image_ids must be unique")
        scores = tuple(_require_similarity(value, "retrieval.scores") for value in self.scores)
        if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
            raise EmbeddingValidationError("retrieval.scores must be descending")
        object.__setattr__(self, "ranked_image_ids", image_ids)
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class NearDuplicateGroup:
    group_id: str
    image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _require_text(self.group_id, "near_duplicate.group_id"))
        image_ids = tuple(_require_text(value, "near_duplicate.image_ids") for value in self.image_ids)
        if len(image_ids) < 2 or len(image_ids) != len(set(image_ids)):
            raise EmbeddingValidationError("near_duplicate.image_ids must contain at least two unique ids")
        object.__setattr__(self, "image_ids", image_ids)


@dataclass(frozen=True)
class SemanticThemeResult:
    theme_similarities: tuple[ThemeSimilarityEvidence, ...]
    retrieval_results: tuple[SemanticRetrievalResult, ...]
    near_duplicate_groups: tuple[NearDuplicateGroup, ...]
    provider: str
    provider_version: str
    model_id: str
    model_version: str
    config_version: str
    embedding_dimension: int

    def __post_init__(self) -> None:
        collections = (
            ("theme_similarities", ThemeSimilarityEvidence),
            ("retrieval_results", SemanticRetrievalResult),
            ("near_duplicate_groups", NearDuplicateGroup),
        )
        for field_name, item_type in collections:
            value = getattr(self, field_name)
            if isinstance(value, Sequence) and not isinstance(value, tuple):
                value = tuple(value)
                object.__setattr__(self, field_name, value)
            if not isinstance(value, tuple) or not all(isinstance(item, item_type) for item in value):
                raise EmbeddingValidationError(
                    f"semantic_result.{field_name} must contain {item_type.__name__} values"
                )

        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise EmbeddingValidationError("embedding_dimension must be a positive integer")
        for name in ("provider", "provider_version", "model_id", "model_version", "config_version"):
            object.__setattr__(self, name, _require_text(getattr(self, name), f"semantic_result.{name}"))
