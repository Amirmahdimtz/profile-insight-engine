import math
import unittest

from src.core.profile_analysis.vision_contracts import (
    VisionCaption,
    VisionEvidenceKind,
    VisionObservation,
    VisionProviderResult,
    VisionValidationError,
)


class VisionContractTests(unittest.TestCase):
    def test_valid_provider_result_is_sorted_deterministically(self):
        result = VisionProviderResult(
            image_id="img-1",
            observations=(
                VisionObservation(VisionEvidenceKind.TOPIC, "Travel", 0.6),
                VisionObservation(VisionEvidenceKind.OBJECT, "Dog", 0.8),
            ),
            caption=VisionCaption("A dog is visible outdoors.", 0.7),
            provider="llama_cpp",
            provider_version="b123",
            model_id="candidate",
            model_version="v1",
            config_version="phase5-v1",
            confidence_semantics="model_self_reported_uncalibrated",
        )
        self.assertEqual(
            [(item.kind.value, item.label) for item in result.observations],
            [("object", "Dog"), ("topic", "Travel")],
        )

    def test_invalid_confidence_is_rejected(self):
        for value in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(VisionValidationError):
                    VisionObservation(VisionEvidenceKind.OBJECT, "Dog", value)

    def test_duplicate_observations_are_rejected_case_insensitively(self):
        with self.assertRaisesRegex(VisionValidationError, "duplicates"):
            VisionProviderResult(
                image_id="img-1",
                observations=(
                    VisionObservation(VisionEvidenceKind.OBJECT, "Dog", 0.8),
                    VisionObservation(VisionEvidenceKind.OBJECT, "dog", 0.7),
                ),
                caption=None,
                provider="llama_cpp",
                provider_version="b123",
                model_id="candidate",
                model_version="v1",
                config_version="phase5-v1",
                confidence_semantics="model_self_reported_uncalibrated",
            )


if __name__ == "__main__":
    unittest.main()
