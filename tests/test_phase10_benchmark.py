import argparse
import json
import pathlib
import tempfile
import unittest

from evaluation.phase10_benchmark import REPORT_SCHEMA_VERSION, build_report
from evaluation.phase10_end_to_end_benchmark import (
    METRIC_SCOPE as E2E_METRIC_SCOPE,
    REPORT_SCHEMA_VERSION as E2E_SCHEMA_VERSION,
    _DEFAULT_WORKLOAD_SIZES,
)


class Phase10BenchmarkTests(unittest.TestCase):
    def test_final_report_keeps_missing_metrics_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hardware_path = pathlib.Path(temp_dir) / "hardware.json"
            hardware_path.write_text(
                json.dumps({"schema_version": "phase10-hardware-profile-v1"}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                ocr_report=None,
                vision_report=None,
                embedding_report=None,
                insight_report=None,
                api_persistence_report=None,
                e2e_report=None,
                hardware_report=hardware_path,
            )
            report = build_report(args)
        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertFalse(report["acceptance_ready"])
        self.assertEqual(report["metrics"]["quality"]["end_to_end_accuracy"]["status"], "unavailable")
        self.assertEqual(report["production_selection"]["vision"], "unresolved_requires_phase10_evidence_review")

    def test_e2e_contract_covers_required_workload_sizes_and_has_explicit_scope(self):
        self.assertEqual(_DEFAULT_WORKLOAD_SIZES, (1, 5, 20, 50, 100))
        self.assertEqual(E2E_SCHEMA_VERSION, "phase10-end-to-end-benchmark-v1")
        self.assertIn("without_semantic_theme_injection", E2E_METRIC_SCOPE)

    def test_final_matrix_covers_required_failure_scenarios(self):
        args = argparse.Namespace(
            ocr_report=None,
            vision_report=None,
            embedding_report=None,
            insight_report=None,
            api_persistence_report=None,
            e2e_report=None,
            hardware_report=None,
        )
        report = build_report(args)
        failures = set(report["benchmark_matrix"]["failure_scenarios"])
        self.assertTrue(
            {
                "gpu_unavailable",
                "ocr_provider_unavailable_or_timeout",
                "vision_provider_timeout",
                "vision_provider_crash",
                "embedding_provider_failure",
                "partial_pipeline_failure",
                "database_runtime_failure",
                "concurrent_requests",
                "restart_recovery",
                "request_isolation",
            }.issubset(failures)
        )


if __name__ == "__main__":
    unittest.main()
