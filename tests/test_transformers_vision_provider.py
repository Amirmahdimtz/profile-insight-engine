import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import VisionProviderError
from src.infrastructure.providers.vision.transformers_vision_provider import (
    TransformersVisionProvider,
    TransformersVisionSettings,
    _build_prefix_allowed_tokens_fn,
)
from src.infrastructure.utils.config_reader import ConfigReader





class _FakeTokenIds:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeGuide:
    instances = []

    def __init__(self, index):
        self.index = index
        self.advanced = []
        type(self).instances.append(self)

    def advance(self, *, token_id, return_tokens):
        self.advanced.append((token_id, return_tokens))

    def get_tokens(self):
        return [1, 4, 9]


class _OutOfRangeGuide(_FakeGuide):
    def get_tokens(self):
        return [9]


def _settings() -> TransformersVisionSettings:
    return TransformersVisionSettings(
        model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        model_revision="main",
        config_version="phase10-test",
        device="cpu",
        local_files_only=True,
        image_longest_edge=512,
        do_image_splitting=False,
        max_tokens=256,
        max_observations_per_kind=1,
        max_label_chars=32,
        max_caption_chars=80,
        confidence_semantics="model_self_reported_uncalibrated",
        structured_backend="outlines_core",
    )


def _image(path: Path) -> CanonicalImage:
    return CanonicalImage(
        image_id="img-1",
        content_hash="a" * 64,
        storage_reference=str(path),
        mime_type="image/png",
        width=10,
        height=10,
        size_bytes=10,
        original_mime_type="image/png",
        original_width=10,
        original_height=10,
        original_size_bytes=10,
        was_resized=False,
        exif_orientation_applied=False,
    )


class TransformersVisionProviderTests(
    unittest.IsolatedAsyncioTestCase
):
    def test_prefix_constraint_replays_generated_tokens_and_filters_vocab(self):
        _FakeGuide.instances.clear()
        callback = _build_prefix_allowed_tokens_fn(
            guide_type=_FakeGuide,
            index=object(),
            prompt_token_count=2,
            output_vocab_size=5,
        )
        allowed = callback(
            0,
            _FakeTokenIds([100, 101, 3]),
        )
        self.assertEqual(allowed, [1, 4])
        self.assertEqual(
            _FakeGuide.instances[-1].advanced,
            [(3, False)],
        )

    def test_prefix_constraint_fails_closed_when_no_model_token_is_valid(self):
        callback = _build_prefix_allowed_tokens_fn(
            guide_type=_OutOfRangeGuide,
            index=object(),
            prompt_token_count=2,
            output_vocab_size=5,
        )
        with self.assertRaisesRegex(
            VisionProviderError,
            "no valid model tokens",
        ):
            callback(
                0,
                _FakeTokenIds([100, 101]),
            )

    def test_generation_uses_prefix_constraint_instead_of_direct_logits_masking(self):
        import inspect

        source = inspect.getsource(
            TransformersVisionProvider._generate_structured_sync
        )
        self.assertIn("prefix_allowed_tokens_fn=", source)
        self.assertNotIn("logits_processor=", source)

    def test_candidate_is_not_di_bound_before_evidence_selection(self):
        self.assertFalse(
            getattr(
                TransformersVisionProvider,
                "__di_provider__",
                False,
            )
        )

    def test_single_tile_processor_settings_are_provider_local(self):
        provider = TransformersVisionProvider(
            object(),
            _settings(),
        )
        self.assertEqual(
            provider._processor_kwargs(),
            {
                "images_kwargs": {
                    "do_image_splitting": False,
                    "size": {"longest_edge": 512},
                    "max_image_size": {"longest_edge": 512},
                }
            },
        )

    async def test_result_uses_strict_existing_vision_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.png"
            path.write_bytes(b"canonical-image")
            provider = TransformersVisionProvider(
                object(),
                _settings(),
            )
            provider._provider_version = (
                "transformers-test;outlines-test"
            )
            provider._resolved_model_version = "snapshot-sha"
            structured = {
                "scenes": [],
                "objects": [
                    {
                        "label": "dog",
                        "confidence": 0.9,
                    }
                ],
                "activities": [],
                "topics": [],
                "caption": None,
            }
            with (
                patch.object(
                    provider,
                    "_ensure_loaded_async",
                    new=AsyncMock(),
                ),
                patch.object(
                    provider,
                    "_generate_structured_sync",
                    return_value=structured,
                ),
            ):
                result = await provider.extract_async(
                    _image(path)
                )
            self.assertEqual(result.provider, "transformers")
            self.assertEqual(
                result.model_version,
                "snapshot-sha",
            )
            self.assertEqual(len(result.observations), 1)

    async def test_contract_violation_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.png"
            path.write_bytes(b"canonical-image")
            provider = TransformersVisionProvider(
                object(),
                _settings(),
            )
            provider._provider_version = "test"
            provider._resolved_model_version = "snapshot"
            with (
                patch.object(
                    provider,
                    "_ensure_loaded_async",
                    new=AsyncMock(),
                ),
                patch.object(
                    provider,
                    "_generate_structured_sync",
                    return_value={
                        "scenes": [],
                        "objects": [],
                        "activities": [],
                        "topics": [],
                        "caption": None,
                        "raw_output": "sensitive",
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    VisionProviderError,
                    "invalid fields",
                ) as context:
                    await provider.extract_async(_image(path))
            self.assertNotIn(
                "sensitive",
                str(context.exception),
            )

    async def test_missing_canonical_file_is_sanitized(self):
        provider = TransformersVisionProvider(
            object(),
            _settings(),
        )
        image = _image(
            Path("Z:/definitely-missing-phase10-image.png")
        )
        with self.assertRaisesRegex(
            VisionProviderError,
            "storage reference",
        ):
            await provider.extract_async(image)


class TransformersVisionSettingsTests(unittest.TestCase):
    def test_repository_config_keeps_candidate_single_tile_and_offline(self):
        settings = TransformersVisionSettings.from_config(ConfigReader())
        self.assertEqual(settings.device, "cpu")
        self.assertTrue(settings.local_files_only)
        self.assertFalse(settings.do_image_splitting)
        self.assertEqual(settings.image_longest_edge, 512)
        self.assertEqual(settings.max_observations_per_kind, 1)
        self.assertEqual(settings.structured_backend, "outlines_core")

    def test_rejects_non_boolean_split_setting(self):
        values = _settings().__dict__ | {
            "do_image_splitting": 0
        }
        with self.assertRaisesRegex(
            Exception,
            "do_image_splitting",
        ):
            TransformersVisionSettings(**values)


if __name__ == "__main__":
    unittest.main()
