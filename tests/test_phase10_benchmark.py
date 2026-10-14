import argparse
import json
import pathlib
import subprocess
import sys
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

    def test_e2e_cli_writes_report_then_fails_when_no_measured_iteration_succeeds(self):
        source = pathlib.Path("evaluation/phase10_end_to_end_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("successful_iterations == 0", source)
        self.assertIn("raise SystemExit(1)", source)
        self.assertLess(source.index("args.output.write_text"), source.index("raise SystemExit(1)"))

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

    def test_degraded_profile_acceptance_requires_rejection_and_clean_e2e(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)

            def write(name, payload):
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            ocr = write("ocr.json", {"schema_version": "1.0.0", "candidates": [{"candidate": "tessdata_best", "deterministic_rerun": True}]})
            vision = write("vision.json", {"schema_version": "phase10-siglip-zero-shot-vision-benchmark-v1", "candidate": {"selection_protocol_safe": True, "benchmark_valid": True, "failure_rate": 0.0, "f1": 0.0}})
            insight = write("insight.json", {"schema_version": "phase8-insight-benchmark-v1"})
            api = write("api.json", {"schema_version": "phase9-api-persistence-benchmark-v1"})
            hardware = write("hardware.json", {"schema_version": "phase10-hardware-profile-v1"})
            e2e = write("e2e.json", {"schema_version": "phase10-end-to-end-benchmark-v1", "deployment_profile": {"vision_enabled": False, "semantic_theme_active": False}, "performance": {"failure_rate": 0.0}, "quality": {"unsupported_claim_rate_post_validation": 0.0, "summary_structural_factuality": True}, "concurrency": {"failed_requests": 0, "request_isolation": True}})
            args = argparse.Namespace(ocr_report=ocr, vision_report=vision, embedding_report=None, insight_report=insight, api_persistence_report=api, e2e_report=e2e, hardware_report=hardware, deployment_profile="degraded_ocr_only")
            report = build_report(args)

        self.assertTrue(report["acceptance_ready"])
        self.assertEqual(report["missing_evidence"], [])
        self.assertEqual(report["input_evidence"]["embedding"]["status"], "not_required")
        self.assertEqual(report["production_selection"]["vision"], "disabled_rejected_by_holdout_evidence")
        self.assertTrue(report["selection_evidence"]["vision_rejection_evidence_valid"])
        self.assertTrue(report["selection_evidence"]["e2e_degraded_profile_valid"])



if __name__ == "__main__":
    unittest.main()
