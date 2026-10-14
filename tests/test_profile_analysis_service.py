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
from src.core.profile_analysis.self_report_contracts import (
    SelfReportedClaim,
    SelfReportedClaimCategory,
    SelfReportedSourceType,
)
from src.core.profile_analysis.personality_contracts import (
    BigFiveTrait,
    PersonalityProfile,
    PersonalityQuestionnaire,
    PersonalityQuestionnaireItemResponse,
    PersonalityTraitScore,
)
from src.core.services.profile_analysis.profile_analysis_service import (
    ProfileAnalysisAlreadyExistsError,
    ProfileAnalysisExecutionError,
    ProfileAnalysisInputError,
    ProfileAnalysisService,
    ProfileAnalysisTransitionError,
)
from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import (
    ProfileAnalysisAlreadyExistsError as RepositoryAlreadyExistsError,
)


class _FakeConfigReader:
    def __init__(self, vision_enabled=True):
        self.vision_enabled = vision_enabled

    def get(self, key):
        if key != "vision.enabled":
            raise AssertionError(f"unexpected config key: {key}")
        return self.vision_enabled

    def get_positive_int(self, key):
        values = {
            "profile_analysis.max_self_reported_claims": 5,
            "profile_analysis.max_self_reported_claim_chars": 64,
        }
        if key not in values:
            raise AssertionError(f"unexpected config key: {key}")
        return values[key]


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
        self.call_count = 0

    async def extract_async(self, image):
        self.call_count += 1
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


class _FakeNoInsightService:
    def generate(self, aggregates, evidence):
        return InsightGenerationResult(
            policy_version="phase8-candidate-v1",
            insights=(),
            summary="No supported insights met the configured evidence policy.",
        )


class _FakePersonalityAnalysisService:
    def __init__(self, profile=None, error=None):
        self.profile = profile
        self.error = error
        self.received = None

    def analyze(self, questionnaire):
        self.received = questionnaire
        if self.error is not None:
            raise self.error
        return self.profile


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
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(True),
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
        self.assertEqual(persisted["schema_version"], "profile-analysis-result-v3")
        self.assertIsNone(result.result.images[0].evidence[0].value)
        self.assertNotIn("raw_text", result.result.images[0].evidence[0].metadata)
        restored = await service.get_result_async("analysis-1")
        self.assertEqual(restored, result)

    async def test_self_reported_claims_are_separate_persisted_and_restored(self):
        repository = _FakeRepository()
        service = ProfileAnalysisService(
            _FakeImageProcessingService(),
            _FakeOcrService(),
            _FakeVisionService(),
            _FakeAggregationService(),
            _FakeInsightService(),
            repository,
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(True),
        )
        claim = SelfReportedClaim(
            claim_id="claim-1",
            category=SelfReportedClaimCategory.RELIGION,
            value="Muslim",
            source_id="questionnaire-q1",
            source_type=SelfReportedSourceType.QUESTIONNAIRE,
            consent_confirmed=True,
        )

        result = await service.create_async(
            self._request(),
            self._images(),
            (claim,),
        )

        self.assertEqual(result.self_reported_claims, (claim,))
        persisted = repository.entity.result_payload
        self.assertEqual(
            persisted["self_reported_claims"][0]["basis"],
            "explicit_self_report",
        )
        self.assertNotIn("self_reported_claims", persisted["result"])
        restored = await service.get_result_async("analysis-1")
        self.assertEqual(restored.self_reported_claims, (claim,))

    async def test_self_reported_claim_limits_fail_before_repository_write(self):
        repository = _FakeRepository()
        service = ProfileAnalysisService(
            _FakeImageProcessingService(),
            _FakeOcrService(),
            _FakeVisionService(),
            _FakeAggregationService(),
            _FakeInsightService(),
            repository,
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(True),
        )
        oversized = SelfReportedClaim(
            claim_id="claim-1",
            category=SelfReportedClaimCategory.PERSONALITY,
            value="x" * 65,
            source_id="bio-1",
            source_type=SelfReportedSourceType.BIO,
            consent_confirmed=True,
        )

        with self.assertRaisesRegex(
            ProfileAnalysisInputError,
            "exceeds configured text limit",
        ):
            await service.create_async(
                self._request(),
                self._images(),
                (oversized,),
            )

        self.assertIsNone(repository.entity)

    async def test_personality_profile_is_separate_persisted_and_restored(self):
        repository = _FakeRepository()
        profile = PersonalityProfile(
            instrument_id="authorized-form",
            instrument_version="1",
            model="big_five_self_report_scoring_v1",
            source="explicit_self_report_questionnaire",
            traits=tuple(
                PersonalityTraitScore(
                    trait=trait,
                    normalized_score=0.5,
                    scored_mean=3.0,
                    item_count=1,
                    supporting_item_ids=(f"{trait.value}-1",),
                    explanation=f"Supported self-report score for {trait.value}.",
                )
                for trait in BigFiveTrait
            ),
            summary=(
                "Self-reported Big Five scores: "
                "openness=0.500000 | conscientiousness=0.500000 | "
                "extraversion=0.500000 | agreeableness=0.500000 | "
                "neuroticism=0.500000"
            ),
            disclaimer="Self-report score; not image-derived or diagnostic.",
        )
        personality_service = _FakePersonalityAnalysisService(profile=profile)
        service = ProfileAnalysisService(
            _FakeImageProcessingService(),
            _FakeOcrService(),
            _FakeVisionService(),
            _FakeAggregationService(),
            _FakeInsightService(),
            repository,
            personality_service,
            _FakeConfigReader(True),
        )
        questionnaire = PersonalityQuestionnaire(
            instrument_id="authorized-form",
            instrument_version="1",
            scale_min=1,
            scale_max=5,
            responses=tuple(
                PersonalityQuestionnaireItemResponse(
                    item_id=f"{trait.value}-1",
                    trait=trait,
                    response_value=3,
                )
                for trait in BigFiveTrait
            ),
            consent_confirmed=True,
        )

        result = await service.create_async(
            self._request(),
            self._images(),
            (),
            questionnaire,
        )

        self.assertIs(personality_service.received, questionnaire)
        self.assertEqual(result.personality_profile, profile)
        persisted = repository.entity.result_payload
        self.assertEqual(persisted["schema_version"], "profile-analysis-result-v3")
        self.assertEqual(
            persisted["personality_profile"]["source"],
            "explicit_self_report_questionnaire",
        )
        self.assertNotIn("responses", persisted["personality_profile"])
        restored = await service.get_result_async("analysis-1")
        self.assertEqual(restored.personality_profile, profile)

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
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(True),
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
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(True),
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
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(True),
        )
        with self.assertRaises(ProfileAnalysisTransitionError):
            await service._transition_async(
                "analysis-1",
                ProfileAnalysisStatus.COMPLETED,
                ProfileAnalysisStatus.FAILED,
            )

    async def test_disabled_vision_skips_provider_and_completes_with_ocr_only(self):
        repository = _FakeRepository()
        image_service = _FakeImageProcessingService()
        vision_service = _FakeVisionService(fail=True)
        service = ProfileAnalysisService(
            image_service,
            _FakeOcrService(),
            vision_service,
            _FakeAggregationService(),
            _FakeNoInsightService(),
            repository,
            _FakePersonalityAnalysisService(),
            _FakeConfigReader(False),
        )

        result = await service.create_async(
            self._request(),
            self._images(),
        )

        self.assertEqual(vision_service.call_count, 0)
        self.assertEqual(
            repository.entity.status,
            ProfileAnalysisStatus.COMPLETED.value,
        )
        self.assertTrue(image_service.released)
        self.assertEqual(result.result.insights, ())
        self.assertTrue(
            all(
                evidence.type is EvidenceType.OCR_TEXT
                for image in result.result.images
                for evidence in image.evidence
            )
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
