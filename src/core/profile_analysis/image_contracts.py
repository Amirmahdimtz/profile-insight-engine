from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ImageValidationError(ValueError):
    """Raised when a raw image fails the Phase 3 ingestion contract."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ImageValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True)
class RawImageInput:
    image_id: str
    filename: str
    declared_mime_type: str
    content: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _require_text(self.image_id, "image_id"))
        object.__setattr__(self, "filename", _require_text(self.filename, "filename"))
        object.__setattr__(
            self,
            "declared_mime_type",
            _require_text(self.declared_mime_type, "declared_mime_type").lower(),
        )
        if not isinstance(self.content, (bytes, bytearray, memoryview)):
            raise ImageValidationError("content must be bytes-like")
        object.__setattr__(self, "content", bytes(self.content))


@dataclass(frozen=True)
class CanonicalImage:
    image_id: str
    content_hash: str
    storage_reference: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    original_mime_type: str
    original_width: int
    original_height: int
    original_size_bytes: int
    was_resized: bool
    exif_orientation_applied: bool

    def __post_init__(self) -> None:
        _require_text(self.image_id, "canonical.image_id")
        if not _SHA256_PATTERN.fullmatch(self.content_hash):
            raise ImageValidationError("canonical.content_hash must be a lowercase SHA-256 digest")
        _require_text(self.storage_reference, "canonical.storage_reference")
        if self.mime_type != "image/png":
            raise ImageValidationError("canonical.mime_type must be image/png")
        _require_text(self.original_mime_type, "canonical.original_mime_type")
        for field_name in (
            "width",
            "height",
            "size_bytes",
            "original_width",
            "original_height",
            "original_size_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ImageValidationError(f"canonical.{field_name} must be a positive integer")


@dataclass(frozen=True)
class DuplicateImageMapping:
    duplicate_image_id: str
    canonical_image_id: str
    content_hash: str

    def __post_init__(self) -> None:
        _require_text(self.duplicate_image_id, "duplicate.duplicate_image_id")
        _require_text(self.canonical_image_id, "duplicate.canonical_image_id")
        if self.duplicate_image_id == self.canonical_image_id:
            raise ImageValidationError("duplicate image must reference a different canonical image")
        if not _SHA256_PATTERN.fullmatch(self.content_hash):
            raise ImageValidationError("duplicate.content_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ProcessedImageBatch:
    analysis_id: str
    images: tuple[CanonicalImage, ...]
    duplicates: tuple[DuplicateImageMapping, ...]
    storage_scope_reference: str

    def __post_init__(self) -> None:
        _require_text(self.analysis_id, "batch.analysis_id")
        _require_text(self.storage_scope_reference, "batch.storage_scope_reference")
        if isinstance(self.images, Sequence) and not isinstance(self.images, tuple):
            object.__setattr__(self, "images", tuple(self.images))
        if isinstance(self.duplicates, Sequence) and not isinstance(self.duplicates, tuple):
            object.__setattr__(self, "duplicates", tuple(self.duplicates))
        if not self.images or not all(isinstance(item, CanonicalImage) for item in self.images):
            raise ImageValidationError("batch.images must contain CanonicalImage values")
        if not all(isinstance(item, DuplicateImageMapping) for item in self.duplicates):
            raise ImageValidationError("batch.duplicates must contain DuplicateImageMapping values")

        image_ids = [item.image_id for item in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ImageValidationError("batch.images must have unique image_id values")
        images_by_id = {item.image_id: item for item in self.images}
        duplicate_ids = [item.duplicate_image_id for item in self.duplicates]
        if len(duplicate_ids) != len(set(duplicate_ids)):
            raise ImageValidationError("batch.duplicates must have unique duplicate_image_id values")
        for mapping in self.duplicates:
            duplicate = images_by_id.get(mapping.duplicate_image_id)
            canonical = images_by_id.get(mapping.canonical_image_id)
            if duplicate is None or canonical is None:
                raise ImageValidationError("batch.duplicates must reference images in the same batch")
            if duplicate.content_hash != mapping.content_hash or canonical.content_hash != mapping.content_hash:
                raise ImageValidationError("batch.duplicate content hash must match both referenced images")
            if duplicate.storage_reference != canonical.storage_reference:
                raise ImageValidationError("duplicate images must reuse the canonical storage reference")
            if image_ids.index(mapping.canonical_image_id) >= image_ids.index(mapping.duplicate_image_id):
                raise ImageValidationError("duplicate mapping must reference the first prior canonical image")
