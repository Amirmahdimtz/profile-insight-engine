import inspect
import unittest
from pathlib import Path

from evaluation.benchmark import load_manifest
from evaluation.siglip_zero_shot_vision_benchmark import (
    METRIC_SCOPE,
    REPORT_SCHEMA_VERSION,
    _report_exit_code,
    _taxonomy_fingerprint,
    _taxonomy_from_manifest,
    run_siglip_zero_shot_vision_benchmark_async,
)
from src.core.profile_analysis.vision_contracts import (
    VisionEvidenceKind,
)


class SiglipZeroShotVisionBenchmarkContractTests(
    unittest.TestCase
):
    def test_benchmark_module_is_parseable(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "siglip_zero_shot_vision_benchmark.py"
        )
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

    def test_report_contract_is_phase10_scoped(self):
        self.assertEqual(
            REPORT_SCHEMA_VERSION,
            "phase10-siglip-zero-shot-vision-benchmark-v1",
        )
        self.assertIn("phase10", METRIC_SCOPE)

    def test_taxonomy_uses_global_class_universe_not_sample_mapping(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "fixtures"
            / "sanity_manifest.json"
        )
        taxonomy = _taxonomy_from_manifest(manifest)
        self.assertEqual(
            taxonomy[VisionEvidenceKind.OBJECT],
            ("bicycle", "cup"),
        )
        self.assertEqual(
            taxonomy[VisionEvidenceKind.SCENE],
            ("park",),
        )
        self.assertEqual(
            taxonomy[VisionEvidenceKind.ACTIVITY],
            (),
        )
        self.assertEqual(
            _taxonomy_fingerprint(taxonomy),
            _taxonomy_fingerprint(taxonomy),
        )

    def test_runtime_requires_zero_failures_for_valid_benchmark(self):
        source = inspect.getsource(
            run_siglip_zero_shot_vision_benchmark_async
        )
        self.assertIn(
            "successful_samples == len(eval_samples)",
            source,
        )
        self.assertIn(
            '"rejected_runtime_failures"',
            source,
        )
        self.assertIn(
            "global_visual_class_universe_from_versioned_manifest",
            source,
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
