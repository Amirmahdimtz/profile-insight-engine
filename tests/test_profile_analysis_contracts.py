import math
import unittest

from src.core.profile_analysis.contracts import (
    ContractValidationError,
    Evidence,
    EvidenceType,
    ImageAnalysisResult,
    InsightType,
    ProfileAnalysisRequest,
    ProfileAnalysisResult,
    ProfileAnalysisStatus,
    ProfileInsight,
    Theme,
    ThemeType,
)


class ProfileAnalysisContractTests(unittest.TestCase):
    def _evidence(self, *, evidence_id="ev_1", image_id="img_1", confidence=0.9):
        return Evidence(
            id=evidence_id,
            image_id=image_id,
            type=EvidenceType.OBJECT,
            label="football_object",
            value="football",
            confidence=confidence,
            source="normalized_visual_evidence",
        )

    def _minimum_result(self):
        evidence = self._evidence()
        image = ImageAnalysisResult(
            analysis_id="analysis_1",
            image_id="img_1",
            evidence=(evidence,),
        )
        theme = Theme(
            key="football_content",
            type=ThemeType.RECURRING_CONTENT,
            label="Football-related content",
            confidence=0.9,
            evidence_count=1,
            image_coverage=1.0,
            supporting_evidence_ids=("ev_1",),
            supporting_image_ids=("img_1",),
        )
        insight = ProfileInsight(
            key="football_interest",
            type=InsightType.VISIBLE_INTEREST,
            label="Football-related content is visible",
            explanation="Supported by observable football-related evidence in the image",
            confidence=0.9,
            evidence_count=1,
            image_coverage=1.0,
            supporting_evidence_ids=("ev_1",),
            supporting_image_ids=("img_1",),
        )
        return ProfileAnalysisResult(
            analysis_id="analysis_1",
            status=ProfileAnalysisStatus.COMPLETED,
            images=(image,),
            themes=(theme,),
            insights=(insight,),
        )

    def test_valid_minimum_contract(self):
        request = ProfileAnalysisRequest(analysis_id="analysis_1", image_ids=("img_1",))
        result = self._minimum_result()

        self.assertEqual(request.image_ids, ("img_1",))
        self.assertEqual(result.insights[0].supporting_evidence_ids, ("ev_1",))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisRequest(analysis_id="analysis_1", image_ids=())

    def test_configured_image_count_limit_is_enforced_without_hardcoded_maximum(self):
        request = ProfileAnalysisRequest(
            analysis_id="analysis_1",
            image_ids=("img_1", "img_2", "img_3"),
        )
        with self.assertRaises(ContractValidationError):
            request.validate_image_count(max_images=2)

    def test_duplicate_image_id_is_rejected_in_request(self):
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisRequest(
                analysis_id="analysis_1",
                image_ids=("img_1", "img_1"),
            )

    def test_duplicate_image_id_is_rejected_in_result(self):
        image_1 = ImageAnalysisResult(analysis_id="analysis_1", image_id="img_1")
        image_2 = ImageAnalysisResult(analysis_id="analysis_1", image_id="img_1")
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult(
                analysis_id="analysis_1",
                status=ProfileAnalysisStatus.PENDING,
                images=(image_1, image_2),
            )

    def test_confidence_outside_range_is_rejected(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ContractValidationError):
                self._evidence(confidence=value)

    def test_nan_and_infinity_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ContractValidationError):
                self._evidence(confidence=value)

        with self.assertRaises(ContractValidationError):
            Evidence(
                id="ev_1",
                image_id="img_1",
                type=EvidenceType.OTHER_OBSERVABLE,
                label="numeric_observation",
                value=math.nan,
                confidence=0.5,
                source="normalized_source",
            )

    def test_unknown_fields_are_rejected_at_strict_boundaries(self):
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisRequest.from_dict(
                {
                    "analysis_id": "analysis_1",
                    "image_ids": ["img_1"],
                    "unexpected": True,
                }
            )

        payload = self._minimum_result().to_dict()
        payload["images"][0]["evidence"][0]["unexpected"] = True
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult.from_dict(payload)

    def test_invalid_evidence_reference_is_rejected(self):
        result = self._minimum_result().to_dict()
        result["insights"][0]["supporting_evidence_ids"] = ["ev_missing"]
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult.from_dict(result)

    def test_unknown_supporting_image_id_is_rejected(self):
        result = self._minimum_result().to_dict()
        result["insights"][0]["supporting_image_ids"] = ["img_missing"]
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult.from_dict(result)

    def test_invalid_evidence_count_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            ProfileInsight(
                key="football_interest",
                type=InsightType.VISIBLE_INTEREST,
                label="Football-related content is visible",
                explanation="Supported by observable evidence",
                confidence=0.8,
                evidence_count=2,
                image_coverage=1.0,
                supporting_evidence_ids=("ev_1",),
                supporting_image_ids=("img_1",),
            )

    def test_invalid_image_coverage_is_rejected_by_result_graph(self):
        result = self._minimum_result().to_dict()
        result["insights"][0]["image_coverage"] = 0.5
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult.from_dict(result)

    def test_inconsistent_result_graph_is_rejected(self):
        result = self._minimum_result().to_dict()
        result["images"][0]["analysis_id"] = "different_analysis"
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult.from_dict(result)

        result = self._minimum_result().to_dict()
        result["images"][0]["evidence"][0]["image_id"] = "img_other"
        with self.assertRaises(ContractValidationError):
            ProfileAnalysisResult.from_dict(result)

    def test_deterministic_serialization(self):
        result = self._minimum_result()
        first = result.to_json()
        second = result.to_json()
        reparsed = ProfileAnalysisResult.from_json(first)

        self.assertEqual(first, second)
        self.assertEqual(first, reparsed.to_json())
        self.assertNotIn('": ', first)
        self.assertNotIn('", ', first)

    def test_metadata_serialization_is_key_order_deterministic(self):
        first = Evidence(
            id="ev_1",
            image_id="img_1",
            type=EvidenceType.OTHER_OBSERVABLE,
            label="observable_metadata",
            value="value",
            confidence=0.5,
            source="normalized_source",
            metadata={"z": 1, "a": 2},
        )
        second = Evidence(
            id="ev_1",
            image_id="img_1",
            type=EvidenceType.OTHER_OBSERVABLE,
            label="observable_metadata",
            value="value",
            confidence=0.5,
            source="normalized_source",
            metadata={"a": 2, "z": 1},
        )

        image_1 = ImageAnalysisResult("analysis_1", "img_1", (first,))
        image_2 = ImageAnalysisResult("analysis_1", "img_1", (second,))
        result_1 = ProfileAnalysisResult(
            "analysis_1", ProfileAnalysisStatus.COMPLETED, (image_1,)
        )
        result_2 = ProfileAnalysisResult(
            "analysis_1", ProfileAnalysisStatus.COMPLETED, (image_2,)
        )
        self.assertEqual(result_1.to_json(), result_2.to_json())

    def test_religious_content_observation_is_allowed(self):
        evidence = Evidence(
            id="ev_religious_content",
            image_id="img_1",
            type=EvidenceType.RELIGIOUS_CONTENT,
            label="religious-themed content detected",
            value="religious_symbol",
            confidence=0.8,
            source="normalized_visual_evidence",
        )
        self.assertEqual(evidence.type, EvidenceType.RELIGIOUS_CONTENT)

    def test_unsupported_sensitive_inferences_are_rejected(self):
        unsupported = (
            ("religion", "Religion"),
            ("political_orientation", "Political orientation"),
            ("ethnicity", "Ethnicity"),
            ("mental_health", "Mental health"),
            ("sexual_orientation", "Sexual orientation"),
            ("intelligence", "Intelligence"),
            ("honesty", "Honesty"),
            ("family_relationship", "Family relationship"),
            ("personality_trait", "Personality trait"),
        )
        for key, label in unsupported:
            with self.subTest(key=key), self.assertRaises(ContractValidationError):
                ProfileInsight(
                    key=key,
                    type=InsightType.CONTENT_PATTERN,
                    label=label,
                    explanation="Unsupported claim",
                    confidence=0.9,
                    evidence_count=1,
                    image_coverage=1.0,
                    supporting_evidence_ids=("ev_1",),
                    supporting_image_ids=("img_1",),
                )

        with self.assertRaises(ContractValidationError):
            ProfileInsight(
                key="unsupported_claim",
                type=InsightType.CONTENT_PATTERN,
                label="Person religion = example",
                explanation="Unsupported personal sensitive-trait claim",
                confidence=0.9,
                evidence_count=1,
                image_coverage=1.0,
                supporting_evidence_ids=("ev_1",),
                supporting_image_ids=("img_1",),
            )


    def test_persian_sensitive_inferences_are_rejected_without_blocking_observable_content(self):
        blocked_labels = (
            "مذهب شخص = اسلام",
            "گرایش سیاسی فرد = نمونه",
            "سلامت روان کاربر = نمونه",
            "هوش صاحب پروفایل = بالا",
            "نسبت خانوادگی فرد = نمونه",
        )
        for label in blocked_labels:
            with self.subTest(label=label), self.assertRaises(ContractValidationError):
                Evidence(
                    id="ev_1",
                    image_id="img_1",
                    type=EvidenceType.OTHER_OBSERVABLE,
                    label=label,
                    value=None,
                    confidence=0.8,
                    source="normalized_source",
                )

        allowed = Evidence(
            id="ev_religious_content_fa",
            image_id="img_1",
            type=EvidenceType.RELIGIOUS_CONTENT,
            label="محتوای مذهبی قابل مشاهده",
            value="نماد مذهبی",
            confidence=0.8,
            source="normalized_visual_evidence",
        )
        self.assertEqual(allowed.type, EvidenceType.RELIGIOUS_CONTENT)


if __name__ == "__main__":
    unittest.main()
