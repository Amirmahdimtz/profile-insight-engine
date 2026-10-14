import inspect
import unittest
from pathlib import Path

from evaluation.benchmark import load_manifest
from evaluation.siglip_zero_shot_vision_benchmark import (
    METRIC_SCOPE,
    REPORT_SCHEMA_VERSION,
    _report_exit_code,
    _taxonomy_coverage,
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

    def test_taxonomy_uses_train_split_only_and_excludes_eval_answers(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "fixtures"
            / "sanity_manifest.json"
        )
        taxonomy = _taxonomy_from_manifest(manifest)
        self.assertEqual(
            taxonomy[VisionEvidenceKind.OBJECT],
            ("cup",),
        )
        self.assertEqual(
            taxonomy[VisionEvidenceKind.SCENE],
            (),
        )
        self.assertNotIn(
            "bicycle",
            taxonomy[VisionEvidenceKind.OBJECT],
        )
        self.assertNotIn(
            "park",
            taxonomy[VisionEvidenceKind.SCENE],
        )
        self.assertEqual(
            taxonomy[VisionEvidenceKind.ACTIVITY],
            (),
        )
        self.assertEqual(
            _taxonomy_fingerprint(taxonomy),
            _taxonomy_fingerprint(taxonomy),
        )

    def test_taxonomy_coverage_penalizes_eval_labels_absent_from_train(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "fixtures"
            / "sanity_manifest.json"
        )
        taxonomy = _taxonomy_from_manifest(manifest)
        eval_samples = tuple(
            sample
            for sample in manifest.samples
            if sample.split.value == "eval"
        )
        coverage = _taxonomy_coverage(
            eval_samples=eval_samples,
            taxonomy=taxonomy,
        )
        self.assertEqual(
            coverage["object"]["eval_expected_count"],
            1,
        )
        self.assertEqual(
            coverage["object"]["eval_expected_covered_count"],
            0,
        )
        self.assertEqual(
            coverage["object"]["eval_expected_coverage"],
            0.0,
        )
        self.assertEqual(
            coverage["scene"]["eval_expected_coverage"],
            0.0,
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
            "train_visual_class_universe_from_versioned_manifest",
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
