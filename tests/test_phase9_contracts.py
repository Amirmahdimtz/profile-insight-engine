import unittest

from pydantic import ValidationError

from src.application.profile_analysis.dtos.profile_analysis_dto import CreateProfileAnalysisDto
from src.core.profile_analysis.contracts import (
    Evidence,
    EvidenceType,
    ImageAnalysisResult,
    InsightType,
    ProfileAnalysisResult,
    ProfileAnalysisStatus,
    ProfileInsight,
)
from src.core.profile_analysis.self_report_contracts import (
    SelfReportedClaim,
    SelfReportedClaimCategory,
    SelfReportedSourceType,
)
from src.core.profile_analysis.lifecycle_contracts import (
    CompletedProfileAnalysis,
    ProfileAnalysisLifecycle,
    ProfileAnalysisLifecycleError,
)
from src.core.services.profile_analysis.profile_analysis_service import (
    ProfileAnalysisExecutionError,
    ProfileAnalysisService,
)


class Phase9ContractTests(unittest.TestCase):
    def test_create_dto_rejects_unknown_fields_and_duplicate_image_ids(self):
        with self.assertRaises(ValidationError):
            CreateProfileAnalysisDto.model_validate(
                {
                    "analysis_id": "analysis-1",
                    "image_ids": ["img-1", "img-1"],
                    "unexpected": True,
                }
            )

    def test_lifecycle_uses_existing_status_vocabulary(self):
        lifecycle = ProfileAnalysisLifecycle(
            analysis_id="analysis-1",
            status=ProfileAnalysisStatus.PENDING,
            image_ids=("img-1",),
        )
        self.assertEqual(lifecycle.status, ProfileAnalysisStatus.PENDING)
        with self.assertRaises(ProfileAnalysisLifecycleError):
            ProfileAnalysisLifecycle(
                analysis_id="analysis-1",
                status="queued",  # type: ignore[arg-type]
                image_ids=("img-1",),
            )

    def test_persistence_redacts_ocr_text_but_preserves_support_graph(self):
        ocr = Evidence(
            id="ocr-1",
            image_id="img-1",
            type=EvidenceType.OCR_TEXT,
            label="ocr_text",
            value="Sensitive OCR content",
            confidence=0.9,
            source="ocr",
            metadata={"raw_text": "Sensitive OCR content", "language_hint": "en"},
        )
        insight = ProfileInsight(
            key="textual_reference_1",
            type=InsightType.TEXTUAL_REFERENCE,
            label="Recurring textual reference: observable reference",
            explanation="Supported by one observable evidence signal.",
            confidence=0.9,
            evidence_count=1,
            image_coverage=1.0,
            supporting_evidence_ids=("ocr-1",),
            supporting_image_ids=("img-1",),
        )
        completed = CompletedProfileAnalysis(
            result=ProfileAnalysisResult(
                analysis_id="analysis-1",
                status=ProfileAnalysisStatus.COMPLETED,
                images=(
                    ImageAnalysisResult(
                        analysis_id="analysis-1",
                        image_id="img-1",
                        evidence=(ocr,),
                    ),
                ),
                insights=(insight,),
            ),
            insight_policy_version="phase8-candidate-v1",
            summary="Supported insights: Recurring textual reference: observable reference",
        )
        sanitized = ProfileAnalysisService._sanitize_for_persistence(completed)
        evidence = sanitized.result.images[0].evidence[0]
        self.assertIsNone(evidence.value)
        self.assertNotIn("raw_text", evidence.metadata)
        self.assertEqual(evidence.id, "ocr-1")
        self.assertEqual(sanitized.result.insights[0].supporting_evidence_ids, ("ocr-1",))

    def test_persisted_snapshot_round_trip_is_deterministic(self):
        completed = self._completed_fixture()
        payload1 = ProfileAnalysisService._serialize_completed(completed)
        payload2 = ProfileAnalysisService._serialize_completed(completed)
        self.assertEqual(payload1, payload2)
        restored = ProfileAnalysisService._deserialize_completed(payload1)
        self.assertEqual(restored, completed)

    def test_persisted_snapshot_round_trip_keeps_explicit_self_report_separate(self):
        completed = self._completed_fixture()
        claim = SelfReportedClaim(
            claim_id="claim-1",
            category=SelfReportedClaimCategory.POLITICAL_ORIENTATION,
            value="self-described political view",
            source_id="questionnaire-q4",
            source_type=SelfReportedSourceType.QUESTIONNAIRE,
            consent_confirmed=True,
        )
        completed = CompletedProfileAnalysis(
            result=completed.result,
            insight_policy_version=completed.insight_policy_version,
            summary=completed.summary,
            self_reported_claims=(claim,),
        )

        payload = ProfileAnalysisService._serialize_completed(completed)
        self.assertEqual(payload["schema_version"], "profile-analysis-result-v2")
        self.assertEqual(
            payload["self_reported_claims"][0]["basis"],
            "explicit_self_report",
        )
        restored = ProfileAnalysisService._deserialize_completed(payload)
        self.assertEqual(restored, completed)

    def test_v1_persisted_snapshot_remains_readable(self):
        completed = self._completed_fixture()
        v2 = ProfileAnalysisService._serialize_completed(completed)
        v1 = {
            "schema_version": "phase9-analysis-result-v1",
            "result": v2["result"],
            "insight_policy_version": v2["insight_policy_version"],
            "summary": v2["summary"],
        }
        restored = ProfileAnalysisService._deserialize_completed(v1)
        self.assertEqual(restored.result, completed.result)
        self.assertEqual(restored.self_reported_claims, ())

    def test_persisted_snapshot_revalidates_sensitive_inference_boundary(self):
        completed = self._completed_fixture()
        payload = ProfileAnalysisService._serialize_completed(completed)
        payload["result"]["insights"][0]["label"] = "The person religion is X"
        with self.assertRaises(ProfileAnalysisExecutionError):
            ProfileAnalysisService._deserialize_completed(payload)

    def test_persisted_snapshot_rejects_summary_outside_insight_graph(self):
        completed = self._completed_fixture()
        payload = ProfileAnalysisService._serialize_completed(completed)
        payload["summary"] = "This summary adds an unsupported profile claim."
        with self.assertRaises(ProfileAnalysisExecutionError):
            ProfileAnalysisService._deserialize_completed(payload)

    @staticmethod
    def _completed_fixture() -> CompletedProfileAnalysis:
        evidence = Evidence(
            id="ev-1",
            image_id="img-1",
            type=EvidenceType.TOPIC,
            label="football",
            value="football",
            confidence=0.8,
            source="vision",
        )
        insight = ProfileInsight(
            key="visible_interest_1",
            type=InsightType.VISIBLE_INTEREST,
            label="Recurring topic-related content: football",
            explanation="Supported by one evidence signal across one image.",
            confidence=0.8,
            evidence_count=1,
            image_coverage=1.0,
            supporting_evidence_ids=("ev-1",),
            supporting_image_ids=("img-1",),
        )
        return CompletedProfileAnalysis(
            result=ProfileAnalysisResult(
                analysis_id="analysis-1",
                status=ProfileAnalysisStatus.COMPLETED,
                images=(
                    ImageAnalysisResult(
                        analysis_id="analysis-1",
                        image_id="img-1",
                        evidence=(evidence,),
                    ),
                ),
                insights=(insight,),
            ),
            insight_policy_version="phase8-candidate-v1",
            summary="Supported insights: Recurring topic-related content: football",
        )


if __name__ == "__main__":
    unittest.main()
