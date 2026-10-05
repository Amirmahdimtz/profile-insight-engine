import unittest

from src.core.profile_analysis.aggregation_contracts import (
    AggregatedTheme,
    AggregationKind,
    AggregationValidationError,
    aggregation_key,
)


class AggregationContractTests(unittest.TestCase):
    def _aggregate(self, **overrides):
        values = {
            "kind": AggregationKind.TOPIC,
            "key": "football",
            "label": "Football",
            "evidence_count": 2,
            "unique_image_count": 2,
            "image_universe_count": 4,
            "image_coverage": 0.5,
            "average_confidence": 0.75,
            "max_confidence": 0.9,
            "source_diversity": 2,
            "cross_image_consistency": 1.0,
            "sources": ("vision_a", "vision_b"),
            "supporting_image_ids": ("img_b", "img_a"),
            "supporting_signal_ids": ("ev_b", "ev_a"),
        }
        values.update(overrides)
        return AggregatedTheme(**values)

    def test_key_normalization_is_exact_casefold_not_fuzzy(self):
        self.assertEqual(aggregation_key("  FootBall  "), "football")
        self.assertNotEqual(aggregation_key("football"), aggregation_key("foot ball"))

    def test_contract_sorts_traceability_fields_and_serializes_deterministically(self):
        aggregate = self._aggregate()
        self.assertEqual(aggregate.supporting_image_ids, ("img_a", "img_b"))
        self.assertEqual(aggregate.supporting_signal_ids, ("ev_a", "ev_b"))
        self.assertEqual(aggregate.to_json(), aggregate.to_json())

    def test_coverage_formula_is_enforced(self):
        with self.assertRaises(AggregationValidationError):
            self._aggregate(image_coverage=0.75)

    def test_confidence_is_all_present_or_all_missing(self):
        with self.assertRaises(AggregationValidationError):
            self._aggregate(average_confidence=None, max_confidence=0.9)
        semantic = self._aggregate(
            kind=AggregationKind.SEMANTIC_THEME,
            average_confidence=None,
            max_confidence=None,
        )
        self.assertIsNone(semantic.average_confidence)
        self.assertIsNone(semantic.max_confidence)


if __name__ == "__main__":
    unittest.main()
