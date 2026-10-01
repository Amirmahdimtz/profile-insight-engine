from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from src.core.profile_analysis.embedding_contracts import (
    EmbeddingBatchResult,
    EmbeddingValidationError,
    EmbeddingVector,
)
from src.core.profile_analysis.embedding_provider import IEmbeddingProvider
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class EmbeddingProviderError(RuntimeError):
    """Sanitized runtime failure from the local embedding backend."""


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise EmbeddingValidationError(f"{field_name} must be boolean")
    return value


def _feature_tensor(value: Any) -> Any:
    pooler_output = getattr(value, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)) and value:
        candidate = value[1] if len(value) > 1 else value[0]
        if hasattr(candidate, "detach"):
            return candidate
    raise EmbeddingValidationError("embedding runtime returned an unsupported feature output")


def _normalize_rows(rows: Iterable[Iterable[object]], item_ids: Sequence[str]) -> tuple[EmbeddingVector, ...]:
    materialized = [tuple(row) for row in rows]
    if len(materialized) != len(item_ids):
        raise EmbeddingValidationError("embedding runtime output count does not match input count")
    vectors: list[EmbeddingVector] = []
    for item_id, row in zip(item_ids, materialized):
        numeric: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingValidationError("embedding runtime returned a non-numeric value")
            numeric.append(float(value))
        if not numeric:
            raise EmbeddingValidationError("embedding runtime returned an empty vector")
        norm = sum(value * value for value in numeric) ** 0.5
        if norm == 0.0:
            raise EmbeddingValidationError("embedding runtime returned a zero-norm vector")
        normalized = tuple(value / norm for value in numeric)
        vectors.append(EmbeddingVector(item_id=item_id, values=normalized))
    dimensions = {vector.dimension for vector in vectors}
    if len(dimensions) != 1:
        raise EmbeddingValidationError("embedding runtime returned inconsistent vector dimensions")
    return tuple(vectors)


@inject
class TransformersEmbeddingProvider(IEmbeddingProvider):
    """Local Hugging Face multimodal encoder; model remains configurable and benchmark-selected."""

    def __init__(
        self,
        config_reader: ConfigReader,
        model_id: str | None = None,
        model_revision: str | None = None,
    ):
        self._model_id = model_id or config_reader.get_non_empty_string("embedding.model_id")
        self._model_revision = model_revision or config_reader.get_non_empty_string("embedding.model_revision")
        if not isinstance(self._model_id, str) or not self._model_id or self._model_id != self._model_id.strip():
            raise EmbeddingValidationError("embedding model_id must be a non-empty trimmed string")
        if not isinstance(self._model_revision, str) or not self._model_revision or self._model_revision != self._model_revision.strip():
            raise EmbeddingValidationError("embedding model_revision must be a non-empty trimmed string")
        self._config_version = config_reader.get_non_empty_string("embedding.config_version")
        self._device = config_reader.get_non_empty_string("embedding.device")
        self._batch_size = config_reader.get_positive_int("embedding.batch_size")
        self._local_files_only = _require_bool(
            config_reader.get("embedding.local_files_only"),
            "embedding.local_files_only",
        )
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._resolved_model_version: str | None = None
        self._provider_version: str | None = None

    async def embed_images_async(self, images: Sequence[CanonicalImage]) -> EmbeddingBatchResult:
        images = tuple(images)
        if not images or not all(isinstance(image, CanonicalImage) for image in images):
            raise EmbeddingValidationError("images must contain CanonicalImage values")
        item_ids = tuple(image.image_id for image in images)
        if len(item_ids) != len(set(item_ids)):
            raise EmbeddingValidationError("images must have unique image_id values")
        await self._ensure_loaded_async()
        async with self._inference_lock:
            rows = await asyncio.to_thread(self._embed_images_sync, images)
        return self._result(item_ids, rows)

    async def embed_texts_async(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        texts = tuple(texts)
        if not texts:
            raise EmbeddingValidationError("texts must not be empty")
        normalized: list[str] = []
        for text in texts:
            if not isinstance(text, str) or not text or text != text.strip():
                raise EmbeddingValidationError("texts must contain non-empty trimmed strings")
            normalized.append(text)
        if len(normalized) != len(set(normalized)):
            raise EmbeddingValidationError("texts must be unique")
        await self._ensure_loaded_async()
        async with self._inference_lock:
            rows = await asyncio.to_thread(self._embed_texts_sync, tuple(normalized))
        return self._result(tuple(normalized), rows)

    async def _ensure_loaded_async(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                await asyncio.to_thread(self._load_sync)
            except EmbeddingProviderError:
                raise
            except Exception as exc:
                raise EmbeddingProviderError("embedding model initialization failed") from exc

    def _load_sync(self) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as exc:
            raise EmbeddingProviderError("embedding runtime dependencies are not installed") from exc
        try:
            processor = AutoProcessor.from_pretrained(
                self._model_id,
                revision=self._model_revision,
                local_files_only=self._local_files_only,
            )
            model = AutoModel.from_pretrained(
                self._model_id,
                revision=self._model_revision,
                local_files_only=self._local_files_only,
            )
            model.to(self._device)
            model.eval()
        except Exception as exc:
            raise EmbeddingProviderError("embedding model could not be loaded") from exc
        if not callable(getattr(model, "get_image_features", None)) or not callable(
            getattr(model, "get_text_features", None)
        ):
            raise EmbeddingProviderError(
                "configured embedding model does not expose image/text feature methods"
            )
        commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
        self._resolved_model_version = str(commit_hash or self._model_revision)
        self._provider_version = str(transformers.__version__)
        self._torch = torch
        self._processor = processor
        self._model = model

    def _embed_images_sync(self, images: Sequence[CanonicalImage]) -> list[list[float]]:
        rows: list[list[float]] = []
        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]
            opened: list[Image.Image] = []
            try:
                for item in batch:
                    path = Path(item.storage_reference)
                    if not path.is_file():
                        raise EmbeddingProviderError("canonical image storage reference is unavailable")
                    with Image.open(path) as source:
                        source.load()
                        opened.append(source.convert("RGB"))
                inputs = self._processor(images=opened, return_tensors="pt")
                inputs = {name: value.to(self._device) for name, value in inputs.items()}
                with self._torch.inference_mode():
                    features = _feature_tensor(self._model.get_image_features(**inputs))
                rows.extend(features.detach().cpu().tolist())
            except EmbeddingProviderError:
                raise
            except Exception as exc:
                raise EmbeddingProviderError("image embedding inference failed") from exc
            finally:
                for image in opened:
                    image.close()
        return rows

    def _embed_texts_sync(self, texts: Sequence[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                inputs = self._processor(text=list(batch), padding=True, return_tensors="pt")
                inputs = {name: value.to(self._device) for name, value in inputs.items()}
                with self._torch.inference_mode():
                    features = _feature_tensor(self._model.get_text_features(**inputs))
                rows.extend(features.detach().cpu().tolist())
            except Exception as exc:
                raise EmbeddingProviderError("text embedding inference failed") from exc
        return rows

    def _result(self, item_ids: Sequence[str], rows: Iterable[Iterable[object]]) -> EmbeddingBatchResult:
        if self._resolved_model_version is None or self._provider_version is None:
            raise EmbeddingProviderError("embedding model metadata is unavailable")
        return EmbeddingBatchResult(
            vectors=_normalize_rows(rows, item_ids),
            provider="transformers",
            provider_version=self._provider_version,
            model_id=self._model_id,
            model_version=self._resolved_model_version,
            config_version=self._config_version,
        )
