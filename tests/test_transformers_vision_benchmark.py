import inspect
import unittest
from pathlib import Path

from evaluation.benchmark import load_manifest
from evaluation.transformers_vision_benchmark import (
    METRIC_SCOPE,
    REPORT_SCHEMA_VERSION,
    _manifest_identity,
    _report_exit_code,
    _sample_iteration_count,
    run_transformers_vision_benchmark_async,
)


class TransformersVisionBenchmarkContractTests(unittest.TestCase):
    def test_report_metadata_uses_real_manifest_dataset_version_contract(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "fixtures"
            / "sanity_manifest.json"
        )
        self.assertEqual(
            _manifest_identity(manifest),
            ("phase2_sanity_fixture", "1.0.0"),
        )
        self.assertFalse(hasattr(manifest, "version"))

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

    def test_runtime_benchmark_fails_fast_when_first_sample_inference_fails(self):
        source = inspect.getsource(
            run_transformers_vision_benchmark_async
        )
        self.assertIn("if sample_index == 0:", source)
        self.assertIn("preflight_failed = True", source)
        self.assertIn('"rejected_preflight"', source)

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
