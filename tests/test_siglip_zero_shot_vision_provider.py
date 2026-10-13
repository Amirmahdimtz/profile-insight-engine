import inspect
import unittest

from src.core.profile_analysis.vision_contracts import (
    VisionEvidenceKind,
    VisionValidationError,
)
from src.infrastructure.providers.vision.siglip_zero_shot_vision_provider import (
    SiglipZeroShotVisionProvider,
    SiglipZeroShotVisionSettings,
    ZeroShotVisionScore,
    _normalize_taxonomy,
)
from src.infrastructure.utils.config_reader import ConfigReader


def _settings() -> SiglipZeroShotVisionSettings:
    return SiglipZeroShotVisionSettings(
        model_id="google/siglip2-base-patch16-224",
        model_revision="main",
        config_version="phase10-test",
        device="cpu",
        local_files_only=True,
        minimum_score=0.10,
        max_observations_per_kind=1,
        confidence_semantics=(
            "siglip_independent_sigmoid_score_uncalibrated"
        ),
        hypothesis_template="This is a photo of {}.",
    )


def _taxonomy():
    return {
        VisionEvidenceKind.SCENE: ("park", "office"),
        VisionEvidenceKind.OBJECT: ("dog", "car"),
        VisionEvidenceKind.ACTIVITY: ("running",),
        VisionEvidenceKind.TOPIC: ("football",),
    }


class SiglipZeroShotVisionSettingsTests(unittest.TestCase):
    def test_repository_config_is_offline_cpu_candidate(self):
        settings = SiglipZeroShotVisionSettings.from_config(
            ConfigReader()
        )
        self.assertEqual(
            settings.model_id,
            "google/siglip2-base-patch16-224",
        )
        self.assertEqual(settings.device, "cpu")
        self.assertTrue(settings.local_files_only)
        self.assertEqual(settings.minimum_score, 0.10)
        self.assertEqual(settings.max_observations_per_kind, 1)

    def test_rejects_invalid_probability_threshold(self):
        values = _settings().__dict__ | {
            "minimum_score": 1.5
        }
        with self.assertRaisesRegex(
            VisionValidationError,
            "minimum_score",
        ):
            SiglipZeroShotVisionSettings(**values)


class SiglipZeroShotVisionProviderTests(unittest.TestCase):
    def test_candidate_is_not_di_bound_before_evidence_selection(self):
        self.assertFalse(
            getattr(
                SiglipZeroShotVisionProvider,
                "__di_provider__",
                False,
            )
        )

    def test_taxonomy_allows_empty_categories_but_not_empty_universe(self):
        normalized = _normalize_taxonomy(
            {
                VisionEvidenceKind.OBJECT: ("dog",),
            }
        )
        self.assertEqual(
            normalized[VisionEvidenceKind.OBJECT],
            ("dog",),
        )
        self.assertEqual(
            normalized[VisionEvidenceKind.SCENE],
            (),
        )
        with self.assertRaisesRegex(
            VisionValidationError,
            "at least one label",
        ):
            _normalize_taxonomy({})

    def test_result_selects_highest_score_per_kind_above_threshold(self):
        provider = SiglipZeroShotVisionProvider(
            object(),
            _taxonomy(),
            _settings(),
        )
        provider._provider_version = "transformers-test"
        provider._resolved_model_version = "snapshot-test"
        scores = (
            ZeroShotVisionScore(
                VisionEvidenceKind.SCENE,
                "office",
                0.20,
            ),
            ZeroShotVisionScore(
                VisionEvidenceKind.SCENE,
                "park",
                0.80,
            ),
            ZeroShotVisionScore(
                VisionEvidenceKind.OBJECT,
                "car",
                0.05,
            ),
            ZeroShotVisionScore(
                VisionEvidenceKind.OBJECT,
                "dog",
                0.70,
            ),
            ZeroShotVisionScore(
                VisionEvidenceKind.ACTIVITY,
                "running",
                0.09,
            ),
            ZeroShotVisionScore(
                VisionEvidenceKind.TOPIC,
                "football",
                0.60,
            ),
        )
        ordered = tuple(
            sorted(
                scores,
                key=lambda item: (
                    item.kind.value,
                    item.label.casefold(),
                    item.label,
                ),
            )
        )
        result = provider.result_from_scores(
            "img-1",
            ordered,
        )
        self.assertIsNone(result.caption)
        self.assertEqual(
            {
                (item.kind.value, item.label)
                for item in result.observations
            },
            {
                ("scene", "park"),
                ("object", "dog"),
                ("topic", "football"),
            },
        )
        self.assertEqual(
            result.confidence_semantics,
            "siglip_independent_sigmoid_score_uncalibrated",
        )

    def test_runtime_uses_official_zero_shot_model_and_sigmoid_scores(self):
        load_source = inspect.getsource(
            SiglipZeroShotVisionProvider._load_sync
        )
        score_source = inspect.getsource(
            SiglipZeroShotVisionProvider._score_image_sync
        )
        self.assertIn(
            "AutoModelForZeroShotImageClassification",
            load_source,
        )
        self.assertIn(
            "local_files_only=self._settings.local_files_only",
            load_source,
        )
        self.assertIn("sigmoid(logits[0])", score_source)


if __name__ == "__main__":
    unittest.main()
