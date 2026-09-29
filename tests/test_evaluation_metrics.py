import math
import unittest

from evaluation.contracts import EvaluationValidationError, MetricResult, SystemMetrics
from evaluation.metrics import (
    average_precision,
    character_error_rate,
    expected_calibration_error,
    macro_f1,
    mean_average_precision,
    precision_recall_f1,
    recall_at_k,
    unsupported_claim_rate,
    word_error_rate,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_cer_exact_insertion_deletion_substitution(self):
        self.assertEqual(character_error_rate("abc", "abc"), 0.0)
        self.assertEqual(character_error_rate("abc", "abxc"), 1 / 3)
        self.assertEqual(character_error_rate("abc", "ac"), 1 / 3)
        self.assertEqual(character_error_rate("abc", "axc"), 1 / 3)

    def test_cer_empty_reference_semantics(self):
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "x"), 1.0)

    def test_cer_preserves_persian_unicode_without_hidden_normalization(self):
        self.assertEqual(character_error_rate("سلام", "سلام"), 0.0)
        self.assertEqual(character_error_rate("ی", "ي"), 1.0)

    def test_wer_exact_insertion_deletion_substitution(self):
        self.assertEqual(word_error_rate("a b", "a b"), 0.0)
        self.assertEqual(word_error_rate("a b", "a x b"), 0.5)
        self.assertEqual(word_error_rate("a b", "a"), 0.5)
        self.assertEqual(word_error_rate("a b", "a x"), 0.5)

    def test_wer_empty_reference_semantics(self):
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "word"), 1.0)

    def test_cer_and_wer_reject_non_string_input(self):
        with self.assertRaises(EvaluationValidationError):
            character_error_rate("a", 1)
        with self.assertRaises(EvaluationValidationError):
            word_error_rate([], "a")

    def test_precision_recall_f1_known_case(self):
        result = precision_recall_f1({"a", "b"}, {"b", "c"})
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1, 0.5)

    def test_precision_recall_f1_empty_semantics(self):
        self.assertEqual(precision_recall_f1(set(), set()).f1, 1.0)
        self.assertEqual(precision_recall_f1({"a"}, set()).f1, 0.0)
        self.assertEqual(precision_recall_f1(set(), {"a"}).f1, 0.0)

    def test_duplicate_classification_predictions_are_set_semantics(self):
        result = precision_recall_f1(["a"], ["a", "a"])
        self.assertEqual(result.f1, 1.0)

    def test_macro_f1_known_case_and_zero_support_class(self):
        value = macro_f1(
            [{"a"}, {"b"}],
            [{"a"}, set()],
            classes={"a", "b", "zero_support"},
        )
        self.assertEqual(value, 0.5)

    def test_macro_f1_penalizes_predicted_only_classes(self):
        value = macro_f1(
            [{"a"}],
            [{"a", "b"}],
            classes={"a", "b"},
        )
        self.assertEqual(value, 0.5)

    def test_macro_f1_all_zero_support_semantics(self):
        self.assertEqual(macro_f1([set()], [set()], classes={"a"}), 1.0)
        self.assertEqual(macro_f1([set()], [{"a"}], classes={"a"}), 0.0)

    def test_macro_f1_rejects_misaligned_or_unknown_classes(self):
        with self.assertRaises(EvaluationValidationError):
            macro_f1([{"a"}], [], classes={"a"})
        with self.assertRaises(EvaluationValidationError):
            macro_f1([{"a"}], [{"b"}], classes={"a"})

    def test_recall_at_k_and_average_precision(self):
        relevant = {"a", "c"}
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at_k(relevant, ranked, 2), 0.5)
        self.assertAlmostEqual(average_precision(relevant, ranked), (1.0 + 2 / 3) / 2)

    def test_retrieval_invalid_k_duplicates_and_no_relevant(self):
        with self.assertRaises(EvaluationValidationError):
            recall_at_k({"a"}, ["a"], 0)
        with self.assertRaises(EvaluationValidationError):
            recall_at_k({"a"}, ["a", "a"], 1)
        self.assertEqual(recall_at_k(set(), ["a"], 1), 0.0)
        self.assertEqual(average_precision(set(), ["a"]), 0.0)

    def test_map_is_deterministic_and_validates_query_graph(self):
        relevant = {"q2": {"b"}, "q1": {"a"}}
        ranked = {"q1": ["a", "x"], "q2": ["x", "b"]}
        self.assertEqual(mean_average_precision(relevant, ranked), 0.75)
        with self.assertRaises(EvaluationValidationError):
            mean_average_precision({"q1": {"a"}}, {"q2": ["a"]})
        self.assertEqual(mean_average_precision({}, {}), 0.0)

    def test_calibration_error_known_case(self):
        value = expected_calibration_error([(0.8, True), (0.2, False)], bins=10)
        self.assertAlmostEqual(value, 0.2)

    def test_calibration_rejects_invalid_input(self):
        for confidence in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(confidence=confidence), self.assertRaises(EvaluationValidationError):
                expected_calibration_error([(confidence, True)], bins=10)
        with self.assertRaises(EvaluationValidationError):
            expected_calibration_error([(0.5, True)], bins=0)
        with self.assertRaises(EvaluationValidationError):
            expected_calibration_error([(0.5, 1)], bins=10)

    def test_unsupported_claim_rate(self):
        self.assertEqual(unsupported_claim_rate([]), 0.0)
        self.assertEqual(
            unsupported_claim_rate(
                [
                    ("visible_sports_content", "Visible sports content"),
                    ("person_religion", "Person religion = x"),
                    ("observable_note", "مذهب شخص = نمونه"),
                ]
            ),
            2 / 3,
        )

    def test_metric_result_rejects_nan_infinity_and_bounded_range_violation(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(EvaluationValidationError):
                MetricResult("metric", value, 1)
        with self.assertRaises(EvaluationValidationError):
            MetricResult("bounded", 1.1, 1, minimum=0.0, maximum=1.0)

    def test_system_metrics_json_contract_allows_omitted_optional_vram(self):
        metrics = SystemMetrics.from_dict(
            {
                "p50_latency_ms": 10.0,
                "p95_latency_ms": 20.0,
                "ram_mb": 512.0,
            }
        )
        self.assertIsNone(metrics.vram_mb)

    def test_system_metric_representation_validates_order_and_finiteness(self):
        metrics = SystemMetrics(10.0, 20.0, 512.0, 128.0)
        self.assertEqual(metrics.p95_latency_ms, 20.0)
        with self.assertRaises(EvaluationValidationError):
            SystemMetrics(20.0, 10.0, 512.0)
        with self.assertRaises(EvaluationValidationError):
            SystemMetrics(10.0, 20.0, math.nan)


if __name__ == "__main__":
    unittest.main()
