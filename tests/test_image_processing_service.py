from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.core.profile_analysis.contracts import ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import ImageValidationError, RawImageInput
from src.core.services.profile_analysis.image_processing_service import ImageProcessingService
from src.infrastructure.providers.image.local_image_storage import LocalImageStorage
from src.infrastructure.utils.config_reader import ConfigReader


def _write_config(root: Path, *, max_images: int = 50, max_mb: int = 12, max_pixels: int = 40_000_000, max_dimension: int = 2048) -> Path:
    path = root / "appsettings.yaml"
    path.write_text(
        "\n".join(
            [
                "profile_analysis:",
                f"  max_images: {max_images}",
                f"  max_image_size_mb: {max_mb}",
                f"  max_image_pixels: {max_pixels}",
                "image:",
                f"  max_dimension: {max_dimension}",
                "  allowed_mime_types:",
                "    - image/jpeg",
                "    - image/png",
                "    - image/webp",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _image_bytes(fmt: str, size: tuple[int, int] = (32, 16), *, exif_orientation: int | None = None) -> bytes:
    image = Image.new("RGB", size, (20, 40, 60))
    buffer = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        kwargs["exif"] = exif
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


class ImageProcessingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_path = _write_config(root, max_dimension=16)
        self.storage = LocalImageStorage()
        self.storage._root = root / "storage"
        self.service = ImageProcessingService(ConfigReader(self.config_path), self.storage)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def _process_one(self, raw: RawImageInput):
        request = ProfileAnalysisRequest("analysis_1", (raw.image_id,))
        return await self.service.process_async(request, (raw,))

    async def test_png_is_validated_resized_and_saved_as_canonical_png(self):
        raw = RawImageInput("img_1", "photo.png", "image/png", _image_bytes("PNG", (40, 20)))
        batch = await self._process_one(raw)
        try:
            image = batch.images[0]
            self.assertEqual((image.width, image.height), (16, 8))
            self.assertTrue(image.was_resized)
            self.assertEqual(image.mime_type, "image/png")
            self.assertEqual(image.original_mime_type, "image/png")
            stored = Path(image.storage_reference)
            self.assertTrue(stored.is_file())
            with Image.open(stored) as canonical:
                self.assertEqual(canonical.mode, "RGB")
                self.assertEqual(canonical.size, (16, 8))
                self.assertFalse(canonical.getexif())
        finally:
            await self.service.release_async(batch)
        self.assertFalse(Path(batch.storage_scope_reference).exists())

    async def test_invalid_declared_mime_is_rejected_without_leaking_filename_or_bytes(self):
        raw = RawImageInput("img_1", "private-name.png", "application/octet-stream", b"TOP_SECRET_RAW_BYTES")
        with self.assertRaises(ImageValidationError) as ctx:
            await self._process_one(raw)
        message = str(ctx.exception)
        self.assertNotIn("private-name", message)
        self.assertNotIn("TOP_SECRET", message)

    async def test_declared_mime_must_match_decoded_format(self):
        raw = RawImageInput("img_1", "photo.jpg", "image/jpeg", _image_bytes("PNG"))
        with self.assertRaisesRegex(ImageValidationError, "declared MIME"):
            await self._process_one(raw)

    async def test_extension_spoofing_is_rejected_but_missing_extension_is_allowed(self):
        png = _image_bytes("PNG")
        with self.assertRaisesRegex(ImageValidationError, "filename extension"):
            await self._process_one(RawImageInput("img_1", "photo.jpg", "image/png", png))

        batch = await self._process_one(RawImageInput("img_1", "upload", "image/png", png))
        await self.service.release_async(batch)

    async def test_corrupt_image_is_rejected_and_failed_scope_is_removed(self):
        self.assertFalse(self.storage._root.exists())
        raw = RawImageInput("img_1", "photo.png", "image/png", b"\x89PNG\r\n\x1a\ncorrupt")
        with self.assertRaisesRegex(ImageValidationError, "corrupt"):
            await self._process_one(raw)
        if self.storage._root.exists():
            self.assertEqual(list(self.storage._root.iterdir()), [])

    async def test_blank_input_is_rejected(self):
        with self.assertRaisesRegex(ImageValidationError, "blank"):
            await self._process_one(RawImageInput("img_1", "photo.png", "image/png", b""))

    async def test_unsupported_decoded_format_is_rejected(self):
        bmp = _image_bytes("BMP")
        with self.assertRaisesRegex(ImageValidationError, "unsupported decoded format"):
            await self._process_one(RawImageInput("img_1", "photo", "image/png", bmp))

    async def test_file_size_limit_is_checked_before_decode(self):
        small_config = _write_config(Path(self.temp_dir.name), max_mb=1)
        service = ImageProcessingService(ConfigReader(small_config), self.storage)
        raw = RawImageInput("img_1", "photo.png", "image/png", b"x" * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ImageValidationError, "file-size limit"):
            await service.process_async(ProfileAnalysisRequest("analysis_2", ("img_1",)), (raw,))

    async def test_configured_pixel_limit_rejects_huge_dimensions(self):
        config = _write_config(Path(self.temp_dir.name), max_pixels=100, max_dimension=100)
        service = ImageProcessingService(ConfigReader(config), self.storage)
        raw = RawImageInput("img_1", "photo.png", "image/png", _image_bytes("PNG", (20, 20)))
        with self.assertRaisesRegex(ImageValidationError, "pixel-count limit"):
            await service.process_async(ProfileAnalysisRequest("analysis_3", ("img_1",)), (raw,))

    async def test_pillow_decompression_bomb_is_mapped_to_safe_validation_error(self):
        original_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 100
            raw = RawImageInput("img_1", "photo.png", "image/png", _image_bytes("PNG", (20, 20)))
            with self.assertRaisesRegex(ImageValidationError, "safe decoder limits"):
                await self._process_one(raw)
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit

    async def test_exif_orientation_is_applied_before_resize_and_exif_is_stripped(self):
        config = _write_config(Path(self.temp_dir.name), max_dimension=100)
        service = ImageProcessingService(ConfigReader(config), self.storage)
        raw = RawImageInput(
            "img_1",
            "photo.jpg",
            "image/jpeg",
            _image_bytes("JPEG", (40, 20), exif_orientation=6),
        )
        batch = await service.process_async(ProfileAnalysisRequest("analysis_4", ("img_1",)), (raw,))
        try:
            image = batch.images[0]
            self.assertEqual((image.original_width, image.original_height), (40, 20))
            self.assertEqual((image.width, image.height), (20, 40))
            self.assertTrue(image.exif_orientation_applied)
            with Image.open(image.storage_reference) as canonical:
                self.assertEqual(canonical.size, (20, 40))
                self.assertFalse(canonical.getexif())
        finally:
            await service.release_async(batch)

    async def test_preprocessing_hash_and_bytes_are_deterministic_across_runs(self):
        raw = RawImageInput("img_1", "photo.png", "image/png", _image_bytes("PNG", (41, 23)))
        first = await self.service.process_async(ProfileAnalysisRequest("analysis_a", ("img_1",)), (raw,))
        second = await self.service.process_async(ProfileAnalysisRequest("analysis_b", ("img_1",)), (raw,))
        try:
            self.assertEqual(first.images[0].content_hash, second.images[0].content_hash)
            self.assertEqual((first.images[0].width, first.images[0].height), (16, 9))
            self.assertEqual(Path(first.images[0].storage_reference).read_bytes(), Path(second.images[0].storage_reference).read_bytes())
        finally:
            await self.service.release_async(first)
            await self.service.release_async(second)

    async def test_duplicate_mapping_uses_first_request_image_and_single_canonical_file(self):
        content = _image_bytes("PNG", (12, 8))
        request = ProfileAnalysisRequest("analysis_dup", ("img_b", "img_a", "img_c"))
        inputs = (
            RawImageInput("img_a", "a.png", "image/png", _image_bytes("PNG", (10, 10))),
            RawImageInput("img_b", "b.png", "image/png", content),
            RawImageInput("img_c", "c.png", "image/png", content),
        )
        batch = await self.service.process_async(request, inputs)
        try:
            self.assertEqual([item.image_id for item in batch.images], ["img_b", "img_a", "img_c"])
            self.assertEqual(len(batch.duplicates), 1)
            duplicate = batch.duplicates[0]
            self.assertEqual(duplicate.duplicate_image_id, "img_c")
            self.assertEqual(duplicate.canonical_image_id, "img_b")
            by_id = {item.image_id: item for item in batch.images}
            self.assertEqual(by_id["img_b"].storage_reference, by_id["img_c"].storage_reference)
            files = list(Path(batch.storage_scope_reference).glob("*.png"))
            self.assertEqual(len(files), 2)
        finally:
            await self.service.release_async(batch)

    async def test_raw_image_ids_must_exactly_match_request(self):
        png = _image_bytes("PNG")
        request = ProfileAnalysisRequest("analysis_ids", ("img_1", "img_2"))
        inputs = (
            RawImageInput("img_1", "a.png", "image/png", png),
            RawImageInput("img_3", "b.png", "image/png", png),
        )
        with self.assertRaisesRegex(ImageValidationError, "exactly match"):
            await self.service.process_async(request, inputs)

    async def test_configured_image_count_limit_is_enforced(self):
        config = _write_config(Path(self.temp_dir.name), max_images=1)
        service = ImageProcessingService(ConfigReader(config), self.storage)
        png = _image_bytes("PNG")
        request = ProfileAnalysisRequest("analysis_count", ("img_1", "img_2"))
        inputs = (
            RawImageInput("img_1", "a.png", "image/png", png),
            RawImageInput("img_2", "b.png", "image/png", png),
        )
        with self.assertRaisesRegex(ImageValidationError, "max_images"):
            await service.process_async(request, inputs)


if __name__ == "__main__":
    unittest.main()
