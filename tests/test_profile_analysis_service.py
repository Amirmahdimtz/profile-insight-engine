from types import SimpleNamespace
import unittest

from src.core.profile_analysis.contracts import (
    Evidence,
    EvidenceType,
    InsightType,
    ProfileAnalysisRequest,
    ProfileAnalysisStatus,
    ProfileInsight,
)
from src.core.profile_analysis.image_contracts import CanonicalImage, ProcessedImageBatch, RawImageInput
from src.core.profile_analysis.insight_contracts import InsightGenerationResult
from src.core.services.profile_analysis.profile_analysis_service import (
    ProfileAnalysisAlreadyExistsError,
    ProfileAnalysisExecutionError,
    ProfileAnalysisService,
    ProfileAnalysisTransitionError,
)
from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import (
    ProfileAnalysisAlreadyExistsError as RepositoryAlreadyExistsError,
)


class _FakeImageProcessingService:
    def __init__(self):
        self.released = False

    async def process_async(self, request, images):
        return ProcessedImageBatch(
            analysis_id=request.analysis_id,
            images=tuple(
                CanonicalImage(
                    image_id=image.image_id,
                    content_hash=("a" if index == 0 else "b") * 64,
                    storage_reference=f"scope/{image.image_id}.png",
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
                for index, image in enumerate(images)
            ),
            duplicates=(),
            storage_scope_reference="scope",
        )

    async def release_async(self, batch):
        self.released = True


class _FakeOcrService:
    async def extract_async(self, image):
        evidence = Evidence(
            id=f"ocr-{image.image_id}",
            image_id=image.image_id,
            type=EvidenceType.OCR_TEXT,
            label="ocr_text",
            value="Private visible text",
            confidence=0.8,
            source="ocr",
            metadata={"raw_text": "Private visible text"},
        )
        return SimpleNamespace(evidence=(evidence,))


class _FakeVisionService:
    def __init__(self, fail=False):
        self.fail = fail

    async def extract_async(self, image):
        if self.fail:
            raise RuntimeError("provider failed")
        evidence = Evidence(
            id=f"topic-{image.image_id}",
            image_id=image.image_id,
            type=EvidenceType.TOPIC,
            label="football",
            value="football",
            confidence=0.9,
            source="vision",
        )
        return SimpleNamespace(evidence=(evidence,))


class _FakeAggregationService:
    def aggregate(self, batch, evidence):
        return ("aggregate",)


class _FakeInsightService:
    def generate(self, aggregates, evidence):
        topic = tuple(item for item in evidence if item.type is EvidenceType.TOPIC)
        insight = ProfileInsight(
            key="visible_interest_1",
            type=InsightType.VISIBLE_INTEREST,
            label="Recurring topic-related content: football",
            explanation="Supported by two evidence signals across two images.",
            confidence=0.85,
            evidence_count=2,
            image_coverage=1.0,
            supporting_evidence_ids=tuple(item.id for item in topic),
            supporting_image_ids=tuple(item.image_id for item in topic),
        )
        return InsightGenerationResult(
            policy_version="phase8-candidate-v1",
            insights=(insight,),
            summary="Supported insights: Recurring topic-related content: football",
        )


class _FakeRepository:
    def __init__(self, duplicate=False):
        self.duplicate = duplicate
        self.entity = None
        self.transitions = []

    async def create_pending_async(self, analysis_id, image_ids):
        if self.duplicate:
            raise RepositoryAlreadyExistsError("already exists")
        self.entity = SimpleNamespace(
            id=analysis_id,
            status=ProfileAnalysisStatus.PENDING.value,
            image_ids=list(image_ids),
            result_payload=None,
        )
        return self.entity

    async def transition_async(self, analysis_id, *, expected_status, new_status, result_payload=None):
        if self.entity is None or self.entity.status != expected_status.value:
            raise AssertionError("unexpected repository transition")
        self.transitions.append((expected_status, new_status, result_payload))
        self.entity.status = new_status.value
        self.entity.result_payload = result_payload
        return self.entity

    async def get_by_id_async(self, analysis_id):
        if self.entity is None or self.entity.id != analysis_id:
            return None
        return self.entity


class ProfileAnalysisServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_workflow_persists_atomic_completed_snapshot(self):
        image_service = _FakeImageProcessingService()
        repository = _FakeRepository()
        service = ProfileAnalysisService(
            image_service,
            _FakeOcrService(),
            _FakeVisionService(),
            _FakeAggregationService(),
            _FakeInsightService(),
            repository,
        )
        result = await service.create_async(self._request(), self._images())
        self.assertTrue(image_service.released)
        self.assertEqual(
            [(item[0], item[1]) for item in repository.transitions],
            [
                (ProfileAnalysisStatus.PENDING, ProfileAnalysisStatus.IN_PROGRESS),
                (ProfileAnalysisStatus.IN_PROGRESS, ProfileAnalysisStatus.COMPLETED),
            ],
        )
        persisted = repository.transitions[-1][2]
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted["schema_version"], "phase9-analysis-result-v1")
        self.assertIsNone(result.result.images[0].evidence[0].value)
        self.assertNotIn("raw_text", result.result.images[0].evidence[0].metadata)
        restored = await service.get_result_async("analysis-1")
        self.assertEqual(restored, result)

    async def test_execution_failure_marks_analysis_failed_without_partial_result(self):
        repository = _FakeRepository()
        image_service = _FakeImageProcessingService()
        service = ProfileAnalysisService(
            image_service,
            _FakeOcrService(),
            _FakeVisionService(fail=True),
            _FakeAggregationService(),
            _FakeInsightService(),
            repository,
        )
        with self.assertRaises(ProfileAnalysisExecutionError):
            await service.create_async(self._request(), self._images())
        self.assertTrue(image_service.released)
        self.assertEqual(repository.entity.status, ProfileAnalysisStatus.FAILED.value)
        self.assertIsNone(repository.entity.result_payload)

    async def test_duplicate_analysis_id_is_rejected_before_pipeline(self):
        service = ProfileAnalysisService(
            _FakeImageProcessingService(),
            _FakeOcrService(),
            _FakeVisionService(),
            _FakeAggregationService(),
            _FakeInsightService(),
            _FakeRepository(duplicate=True),
        )
        with self.assertRaises(ProfileAnalysisAlreadyExistsError):
            await service.create_async(self._request(), self._images())

    async def test_invalid_lifecycle_transition_is_rejected_by_service(self):
        service = ProfileAnalysisService(
            _FakeImageProcessingService(),
            _FakeOcrService(),
            _FakeVisionService(),
            _FakeAggregationService(),
            _FakeInsightService(),
            _FakeRepository(),
        )
        with self.assertRaises(ProfileAnalysisTransitionError):
            await service._transition_async(
                "analysis-1",
                ProfileAnalysisStatus.COMPLETED,
                ProfileAnalysisStatus.FAILED,
            )

    @staticmethod
    def _request():
        return ProfileAnalysisRequest(
            analysis_id="analysis-1",
            image_ids=("img-1", "img-2"),
        )

    @staticmethod
    def _images():
        return (
            RawImageInput("img-1", "1.png", "image/png", b"x"),
            RawImageInput("img-2", "2.png", "image/png", b"y"),
        )


if __name__ == "__main__":
    unittest.main()
