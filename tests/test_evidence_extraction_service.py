import asyncio
import unittest

from src.core.profile_analysis.contracts import EvidenceType
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import (
    VisionCaption,
    VisionEvidenceKind,
    VisionObservation,
    VisionProviderResult,
    VisionValidationError,
)
from src.core.profile_analysis.vision_provider import IVisionProvider
from src.core.services.profile_analysis.evidence_extraction_service import (
    EvidenceExtractionService,
)


def _image(image_id: str = "img-1") -> CanonicalImage:
    return CanonicalImage(
        image_id=image_id,
        content_hash="a" * 64,
        storage_reference="unused-by-fake-provider",
        mime_type="image/png",
        width=100,
        height=100,
        size_bytes=100,
        original_mime_type="image/png",
        original_width=100,
        original_height=100,
        original_size_bytes=100,
        was_resized=False,
        exif_orientation_applied=False,
    )


class FakeVisionProvider(IVisionProvider):
    async def extract_async(
        self,
        image: CanonicalImage,
    ) -> VisionProviderResult:
        return VisionProviderResult(
            image_id=image.image_id,
            observations=(
                VisionObservation(
                    VisionEvidenceKind.SCENE,
                    "outdoor  park",
                    0.8,
                ),
                VisionObservation(
                    VisionEvidenceKind.OBJECT,
                    "dog",
                    0.9,
                ),
                VisionObservation(
                    VisionEvidenceKind.ACTIVITY,
                    "walking",
                    0.7,
                ),
                VisionObservation(
                    VisionEvidenceKind.TOPIC,
                    "pets",
                    0.6,
                ),
            ),
            caption=VisionCaption(
                "A dog is walking in a park.",
                0.75,
            ),
            provider="llama_cpp",
            provider_version="b123",
            model_id="candidate",
            model_version="v1",
            config_version="phase5-v1",
            confidence_semantics="model_self_reported_uncalibrated",
        )


class EvidenceExtractionServiceTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_maps_provider_result_to_standard_evidence_with_traceability(
        self,
    ):
        result = await EvidenceExtractionService(
            FakeVisionProvider()
        ).extract_async(_image())
        self.assertEqual(
            {item.type for item in result.evidence},
            {
                EvidenceType.SCENE,
                EvidenceType.OBJECT,
                EvidenceType.ACTIVITY,
                EvidenceType.TOPIC,
                EvidenceType.OTHER_OBSERVABLE,
            },
        )
        self.assertTrue(
            all(
                item.image_id == "img-1"
                for item in result.evidence
            )
        )
        self.assertTrue(
            all(
                item.source == "llama_cpp"
                for item in result.evidence
            )
        )
        for item in result.evidence:
            self.assertEqual(
                item.metadata["provider_version"],
                "b123",
            )
            self.assertEqual(
                item.metadata["model_id"],
                "candidate",
            )
            self.assertEqual(
                item.metadata["config_version"],
                "phase5-v1",
            )
            self.assertFalse(
                item.metadata["confidence_calibrated"]
            )

    async def test_mapping_is_deterministic(self):
        service = EvidenceExtractionService(FakeVisionProvider())
        first = await service.extract_async(_image())
        second = await service.extract_async(_image())
        self.assertEqual(
            [item.to_dict() for item in first.evidence],
            [item.to_dict() for item in second.evidence],
        )

    async def test_empty_valid_output_is_allowed(self):
        class EmptyProvider(IVisionProvider):
            async def extract_async(self, image):
                return VisionProviderResult(
                    image_id=image.image_id,
                    observations=(),
                    caption=None,
                    provider="llama_cpp",
                    provider_version="b123",
                    model_id="candidate",
                    model_version="v1",
                    config_version="phase5-v1",
                    confidence_semantics=(
                        "model_self_reported_uncalibrated"
                    ),
                )

        result = await EvidenceExtractionService(
            EmptyProvider()
        ).extract_async(_image())
        self.assertEqual(result.evidence, ())

    async def test_mismatched_image_id_is_rejected(self):
        class MismatchProvider(FakeVisionProvider):
            async def extract_async(self, image):
                result = await super().extract_async(image)
                return VisionProviderResult(
                    image_id="other",
                    observations=result.observations,
                    caption=result.caption,
                    provider=result.provider,
                    provider_version=result.provider_version,
                    model_id=result.model_id,
                    model_version=result.model_version,
                    config_version=result.config_version,
                    confidence_semantics=(
                        result.confidence_semantics
                    ),
                )

        with self.assertRaisesRegex(
            VisionValidationError,
            "image_id",
        ):
            await EvidenceExtractionService(
                MismatchProvider()
            ).extract_async(_image())

    async def test_sensitive_person_level_output_is_rejected_by_phase1_policy(
        self,
    ):
        class UnsafeProvider(IVisionProvider):
            async def extract_async(self, image):
                return VisionProviderResult(
                    image_id=image.image_id,
                    observations=(
                        VisionObservation(
                            VisionEvidenceKind.TOPIC,
                            "person religion",
                            0.9,
                        ),
                    ),
                    caption=None,
                    provider="llama_cpp",
                    provider_version="b123",
                    model_id="candidate",
                    model_version="v1",
                    config_version="phase5-v1",
                    confidence_semantics=(
                        "model_self_reported_uncalibrated"
                    ),
                )

        with self.assertRaisesRegex(
            VisionValidationError,
            "unsupported claim",
        ):
            await EvidenceExtractionService(
                UnsafeProvider()
            ).extract_async(_image())

    async def test_wrong_provider_result_type_is_rejected(
        self,
    ):
        class WrongTypeProvider(IVisionProvider):
            async def extract_async(self, image):
                return {"image_id": image.image_id}

        with self.assertRaisesRegex(
            VisionValidationError,
            "invalid result type",
        ):
            await EvidenceExtractionService(
                WrongTypeProvider()
            ).extract_async(_image())

    async def test_sensitive_caption_is_rejected_by_phase1_policy(
        self,
    ):
        class UnsafeCaptionProvider(IVisionProvider):
            async def extract_async(self, image):
                return VisionProviderResult(
                    image_id=image.image_id,
                    observations=(),
                    caption=VisionCaption(
                        "The person religion is visible.",
                        0.9,
                    ),
                    provider="llama_cpp",
                    provider_version="b123",
                    model_id="candidate",
                    model_version="v1",
                    config_version="phase5-v1",
                    confidence_semantics=(
                        "model_self_reported_uncalibrated"
                    ),
                )

        with self.assertRaisesRegex(
            VisionValidationError,
            "caption contains unsupported claim",
        ):
            await EvidenceExtractionService(
                UnsafeCaptionProvider()
            ).extract_async(_image())

    async def test_concurrent_requests_do_not_share_state(
        self,
    ):
        service = EvidenceExtractionService(FakeVisionProvider())
        results = await asyncio.gather(
            service.extract_async(_image("img-a")),
            service.extract_async(_image("img-b")),
        )
        self.assertEqual(
            {result.image_id for result in results},
            {"img-a", "img-b"},
        )
        self.assertTrue(
            all(
                item.image_id == results[0].image_id
                for item in results[0].evidence
            )
        )
        self.assertTrue(
            all(
                item.image_id == results[1].image_id
                for item in results[1].evidence
            )
        )


if __name__ == "__main__":
    unittest.main()
