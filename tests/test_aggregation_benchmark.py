import unittest

from evaluation.aggregation_benchmark import (
    REPORT_SCHEMA_VERSION,
    _percentile,
    run_benchmark,
)


class AggregationBenchmarkTests(unittest.TestCase):
    def test_percentile_is_deterministic(self):
        self.assertEqual(_percentile([30.0, 10.0, 20.0], 0.5), 20.0)
        self.assertEqual(_percentile([], 0.95), 0.0)

    def test_small_benchmark_reports_reproducibility_and_resource_fields(self):
        report = run_benchmark(
            image_counts=(1, 3),
            evidence_per_image=4,
            iterations=2,
            semantic_labels=("semantic_alpha",),
        )
        self.assertEqual(report.schema_version, REPORT_SCHEMA_VERSION)
        self.assertEqual(report.complexity_time, "O(S log S + I)")
        self.assertEqual(report.complexity_auxiliary_space, "O(S + I)")
        self.assertEqual([item.image_count for item in report.results], [1, 3])
        for item in report.results:
            self.assertTrue(item.deterministic_rerun)
            self.assertTrue(item.reversed_input_deterministic)
            self.assertGreater(item.total_input_signal_count, 0)
            self.assertGreater(item.aggregate_count, 0)
            self.assertGreaterEqual(item.p50_latency_ms, 0.0)
            self.assertGreaterEqual(item.p95_latency_ms, item.p50_latency_ms)
            self.assertGreater(item.traced_peak_memory_mb, 0.0)
            self.assertEqual(len(item.result_hash), 64)


if __name__ == "__main__":
    unittest.main()
