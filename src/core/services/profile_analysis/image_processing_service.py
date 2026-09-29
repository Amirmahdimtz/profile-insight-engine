from __future__ import annotations

import asyncio
import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import PurePath
from typing import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from src.core.profile_analysis.contracts import ContractValidationError, ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import (
    CanonicalImage,
    DuplicateImageMapping,
    ImageValidationError,
    ProcessedImageBatch,
    RawImageInput,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.providers.image.local_image_storage import LocalImageStorage
from src.infrastructure.utils.config_reader import ConfigReader


_CANONICALIZATION_VERSION = b"profile-insight-canonical-v1"
_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_EXIF_ORIENTATION_TAG = 274


@dataclass(frozen=True)
class ImageProcessingSettings:
    max_images: int
    max_image_size_bytes: int
    max_image_pixels: int
    max_dimension: int
    allowed_mime_types: tuple[str, ...]

    @classmethod
    def from_config(cls, config_reader: ConfigReader) -> "ImageProcessingSettings":
        max_image_size_mb = config_reader.get_positive_int("profile_analysis.max_image_size_mb")
        allowed_mime_types = tuple(
            item.lower() for item in config_reader.get_non_empty_string_list("image.allowed_mime_types")
        )
        if len(allowed_mime_types) != len(set(allowed_mime_types)):
            raise ImageValidationError("image.allowed_mime_types must not contain duplicates")
        unknown_mime_types = sorted(set(allowed_mime_types) - set(_FORMAT_TO_MIME.values()))
        if unknown_mime_types:
            raise ImageValidationError(
                "image.allowed_mime_types contains unsupported values: "
                + ", ".join(unknown_mime_types)
            )
        return cls(
            max_images=config_reader.get_positive_int("profile_analysis.max_images"),
            max_image_size_bytes=max_image_size_mb * 1024 * 1024,
            max_image_pixels=config_reader.get_positive_int("profile_analysis.max_image_pixels"),
            max_dimension=config_reader.get_positive_int("image.max_dimension"),
            allowed_mime_types=allowed_mime_types,
        )


@dataclass(frozen=True)
class _VerifiedImageHeader:
    detected_mime: str
    width: int
    height: int


@dataclass(frozen=True)
class _CanonicalPayload:
    content: bytes
    content_hash: str
    width: int
    height: int
    original_mime_type: str
    original_width: int
    original_height: int
    original_size_bytes: int
    was_resized: bool
    exif_orientation_applied: bool


@inject
class ImageProcessingService:
    def __init__(self, config_reader: ConfigReader, image_storage: LocalImageStorage):
        self._settings = ImageProcessingSettings.from_config(config_reader)
        self._image_storage = image_storage

    async def process_async(
        self,
        request: ProfileAnalysisRequest,
        images: Sequence[RawImageInput],
    ) -> ProcessedImageBatch:
        if not isinstance(request, ProfileAnalysisRequest):
            raise ImageValidationError("request must be a ProfileAnalysisRequest")
        try:
            request.validate_image_count(self._settings.max_images)
        except ContractValidationError as exc:
            raise ImageValidationError(str(exc)) from exc

        if isinstance(images, (str, bytes, bytearray)) or not isinstance(images, Sequence):
            raise ImageValidationError("images must be a sequence of RawImageInput values")
        image_inputs = tuple(images)
        if len(image_inputs) != len(request.image_ids):
            raise ImageValidationError("raw image count must match request image_ids")
        if not all(isinstance(item, RawImageInput) for item in image_inputs):
            raise ImageValidationError("images must contain RawImageInput values")

        by_id: dict[str, RawImageInput] = {}
        for image_input in image_inputs:
            if image_input.image_id in by_id:
                raise ImageValidationError(f"duplicate raw image_id: {image_input.image_id}")
            by_id[image_input.image_id] = image_input
        if set(by_id) != set(request.image_ids):
            raise ImageValidationError("raw image_ids must exactly match request image_ids")

        scope_reference = await self._image_storage.create_scope_async(request.analysis_id)
        canonical_images: list[CanonicalImage] = []
        duplicate_mappings: list[DuplicateImageMapping] = []
        first_by_hash: dict[str, CanonicalImage] = {}

        try:
            for image_id in request.image_ids:
                raw_image = by_id[image_id]
                payload = await asyncio.to_thread(self._canonicalize, raw_image)
                first = first_by_hash.get(payload.content_hash)
                if first is None:
                    storage_reference = await self._image_storage.write_canonical_async(
                        scope_reference,
                        payload.content_hash,
                        payload.content,
                    )
                else:
                    storage_reference = first.storage_reference

                canonical = CanonicalImage(
                    image_id=image_id,
                    content_hash=payload.content_hash,
                    storage_reference=storage_reference,
                    mime_type="image/png",
                    width=payload.width,
                    height=payload.height,
                    size_bytes=len(payload.content),
                    original_mime_type=payload.original_mime_type,
                    original_width=payload.original_width,
                    original_height=payload.original_height,
                    original_size_bytes=payload.original_size_bytes,
                    was_resized=payload.was_resized,
                    exif_orientation_applied=payload.exif_orientation_applied,
                )
                canonical_images.append(canonical)
                if first is None:
                    first_by_hash[payload.content_hash] = canonical
                else:
                    duplicate_mappings.append(
                        DuplicateImageMapping(
                            duplicate_image_id=image_id,
                            canonical_image_id=first.image_id,
                            content_hash=payload.content_hash,
                        )
                    )
        except BaseException:
            await self._image_storage.cleanup_scope_async(scope_reference)
            raise

        return ProcessedImageBatch(
            analysis_id=request.analysis_id,
            images=tuple(canonical_images),
            duplicates=tuple(duplicate_mappings),
            storage_scope_reference=scope_reference,
        )

    async def release_async(self, batch: ProcessedImageBatch) -> None:
        if not isinstance(batch, ProcessedImageBatch):
            raise ImageValidationError("batch must be a ProcessedImageBatch")
        await self._image_storage.cleanup_scope_async(batch.storage_scope_reference)

    def _canonicalize(self, image_input: RawImageInput) -> _CanonicalPayload:
        content = image_input.content
        if not content:
            raise ImageValidationError(f"image '{image_input.image_id}' is blank")
        if len(content) > self._settings.max_image_size_bytes:
            raise ImageValidationError(
                f"image '{image_input.image_id}' exceeds configured file-size limit"
            )
        declared_mime = image_input.declared_mime_type
        if declared_mime not in self._settings.allowed_mime_types:
            raise ImageValidationError(
                f"image '{image_input.image_id}' has unsupported declared MIME type"
            )

        header = self._verify_image_bytes(image_input)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    orientation = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
                    orientation_applied = orientation in {2, 3, 4, 5, 6, 7, 8}
                    image.load()
                    canonical = ImageOps.exif_transpose(image)
                    canonical = self._to_canonical_rgb(canonical)
                    width, height = canonical.size
                    target_width, target_height = calculate_resized_dimensions(
                        width, height, self._settings.max_dimension
                    )
                    was_resized = (target_width, target_height) != (width, height)
                    if was_resized:
                        canonical = canonical.resize(
                            (target_width, target_height),
                            resample=Image.Resampling.LANCZOS,
                            reducing_gap=3.0,
                        )
                    canonical_bytes = self._encode_png(canonical)
                    content_hash = canonical_content_hash(canonical)
                    return _CanonicalPayload(
                        content=canonical_bytes,
                        content_hash=content_hash,
                        width=canonical.width,
                        height=canonical.height,
                        original_mime_type=header.detected_mime,
                        original_width=header.width,
                        original_height=header.height,
                        original_size_bytes=len(content),
                        was_resized=was_resized,
                        exif_orientation_applied=orientation_applied,
                    )
        except ImageValidationError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageValidationError(
                f"image '{image_input.image_id}' exceeds safe decoder limits"
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise ImageValidationError(f"image '{image_input.image_id}' is corrupt") from exc

    def _verify_image_bytes(self, image_input: RawImageInput) -> _VerifiedImageHeader:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image_input.content)) as image:
                    detected_format = (image.format or "").upper()
                    detected_mime = _FORMAT_TO_MIME.get(detected_format)
                    if detected_mime is None or detected_mime not in self._settings.allowed_mime_types:
                        raise ImageValidationError(
                            f"image '{image_input.image_id}' uses an unsupported decoded format"
                        )
                    if detected_mime != image_input.declared_mime_type:
                        raise ImageValidationError(
                            f"image '{image_input.image_id}' declared MIME does not match decoded format"
                        )
                    self._validate_filename_extension(image_input, detected_mime)
                    width, height = image.size
                    self._validate_dimensions(image_input.image_id, width, height)
                    if getattr(image, "n_frames", 1) != 1:
                        raise ImageValidationError(
                            f"image '{image_input.image_id}' must contain exactly one frame"
                        )
                    image.verify()
                    return _VerifiedImageHeader(detected_mime, width, height)
        except ImageValidationError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageValidationError(
                f"image '{image_input.image_id}' exceeds safe decoder limits"
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise ImageValidationError(f"image '{image_input.image_id}' is corrupt") from exc

    def _validate_dimensions(self, image_id: str, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ImageValidationError(f"image '{image_id}' has invalid dimensions")
        if width * height > self._settings.max_image_pixels:
            raise ImageValidationError(
                f"image '{image_id}' exceeds configured pixel-count limit"
            )

    @staticmethod
    def _validate_filename_extension(image_input: RawImageInput, detected_mime: str) -> None:
        suffix = PurePath(image_input.filename).suffix.lower()
        if not suffix:
            return
        expected_mime = _EXTENSION_TO_MIME.get(suffix)
        if expected_mime is None or expected_mime != detected_mime:
            raise ImageValidationError(
                f"image '{image_input.image_id}' filename extension does not match decoded format"
            )

    @staticmethod
    def _to_canonical_rgb(image: Image.Image) -> Image.Image:
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")

    @staticmethod
    def _encode_png(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
        return buffer.getvalue()


def calculate_resized_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    if min(width, height, max_dimension) <= 0:
        raise ImageValidationError("image dimensions and max_dimension must be positive")
    largest = max(width, height)
    if largest <= max_dimension:
        return width, height
    scale = max_dimension / largest
    return max(1, round(width * scale)), max(1, round(height * scale))


def canonical_content_hash(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise ImageValidationError("canonical hash requires RGB image data")
    digest = hashlib.sha256()
    digest.update(_CANONICALIZATION_VERSION)
    digest.update(b"\0RGB\0")
    digest.update(str(image.width).encode("ascii"))
    digest.update(b"x")
    digest.update(str(image.height).encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()
