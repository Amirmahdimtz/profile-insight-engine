import unittest

from src.core.profile_analysis.contracts import InsightType, ProfileInsight
from src.core.profile_analysis.insight_contracts import (
    InsightGenerationError,
    InsightGenerationResult,
)


class InsightContractsTests(unittest.TestCase):
    def _insight(self, key="content_pattern_0123456789abcdef"):
        return ProfileInsight(
            key=key,
            type=InsightType.CONTENT_PATTERN,
            label="Recurring object: camera",
            explanation="Supported by observable aggregate facts",
            confidence=0.7,
            evidence_count=2,
            image_coverage=0.5,
            supporting_evidence_ids=("ev-1", "ev-2"),
            supporting_image_ids=("img-1", "img-2"),
        )

    def test_result_serializes_deterministically(self):
        result = InsightGenerationResult(
            policy_version="phase8-candidate-v1",
            insights=(self._insight(),),
            summary="Supported insights: Recurring object: camera",
        )
        self.assertEqual(result.to_json(), result.to_json())
        self.assertIn('"policy_version":"phase8-candidate-v1"', result.to_json())

    def test_duplicate_insight_keys_are_rejected(self):
        insight = self._insight()
        with self.assertRaises(InsightGenerationError):
            InsightGenerationResult(
                policy_version="phase8-candidate-v1",
                insights=(insight, insight),
                summary="Supported insights: Recurring object: camera",
            )

    def test_empty_summary_is_rejected(self):
        with self.assertRaises(InsightGenerationError):
            InsightGenerationResult(
                policy_version="phase8-candidate-v1",
                insights=(),
                summary="",
            )


if __name__ == "__main__":
    unittest.main()
