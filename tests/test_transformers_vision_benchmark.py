import unittest

from evaluation.transformers_vision_benchmark import (
    METRIC_SCOPE,
    REPORT_SCHEMA_VERSION,
    _report_exit_code,
    _sample_iteration_count,
)


class TransformersVisionBenchmarkContractTests(unittest.TestCase):
    def test_report_contract_is_phase10_scoped(self):
        self.assertEqual(
            REPORT_SCHEMA_VERSION,
            "phase10-transformers-vision-benchmark-v1",
        )
        self.assertIn("phase10", METRIC_SCOPE)

    def test_determinism_repeats_only_configured_prefix(self):
        self.assertEqual(
            _sample_iteration_count(
                sample_index=0,
                iterations=2,
                determinism_sample_count=3,
            ),
            2,
        )
        self.assertEqual(
            _sample_iteration_count(
                sample_index=3,
                iterations=2,
                determinism_sample_count=3,
            ),
            1,
        )

    def test_cli_fails_closed_without_valid_candidate(self):
        self.assertEqual(
            _report_exit_code(
                {
                    "candidate": {
                        "benchmark_valid": False
                    }
                }
            ),
            2,
        )
        self.assertEqual(
            _report_exit_code(
                {
                    "candidate": {
                        "benchmark_valid": True
                    }
                }
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
