import argparse
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from evaluation.common import DatasetSlice, DatasetSplit, EvaluationLabelType
from evaluation.metrics import unsupported_claim_rate
from evaluation.vision_benchmark import (
    _benchmark_label,
    _build_parser,
    _parse_candidate,
    _deterministic_rerun_status,
    _provider_claims,
    _report_exit_code,
    _runtime_command,
    _sanitized_failure_reason,
    _wait_for_health,
    audit_visual_label_coverage,
    run_vision_benchmark_async,
)
from src.core.profile_analysis.vision_contracts import (
    VisionEvidenceKind,
    VisionProviderError,
    VisionObservation,
    VisionProviderResult,
)


class VisionBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_candidate_parser_requires_name_and_hf_spec(self):
        candidate = _parse_candidate(
            "qwen3b=ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
        )
        self.assertEqual(candidate.name, "qwen3b")
        self.assertIn("Qwen2.5-VL-3B", candidate.hf_model)
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_candidate("broken")

    def test_cli_accepts_benchmark_inference_overrides(self):
        args = _build_parser().parse_args(
            [
                "--manifest",
                "manifest.json",
                "--dataset-root",
                ".",
                "--candidate",
                "qwen=repo:model",
                "--iterations",
                "1",
                "--request-timeout-seconds",
                "300",
                "--max-tokens",
                "256",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(args.request_timeout_seconds, 300)
        self.assertEqual(args.max_tokens, 256)

    def test_cli_accepts_cached_diagnostic_skip_acquisition(self):
        args = _build_parser().parse_args(
            [
                "--manifest",
                "manifest.json",
                "--dataset-root",
                ".",
                "--candidate",
                "qwen=repo:model",
                "--iterations",
                "1",
                "--skip-model-acquisition",
                "--output",
                "report.json",
            ]
        )
        self.assertTrue(args.skip_model_acquisition)

    def test_benchmark_label_normalization_is_case_and_whitespace_stable(
        self,
    ):
        self.assertEqual(_benchmark_label("  Dog  "), "dog")
        self.assertEqual(_benchmark_label("DOG"), "dog")

    def test_runtime_command_separates_acquisition_from_measured_load(
        self,
    ):
        candidate = _parse_candidate(
            "qwen3b=ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
        )
        acquisition = _runtime_command(
            "llama-server",
            candidate,
            1234,
            offline=False,
        )
        measured = _runtime_command(
            "llama-server",
            candidate,
            1234,
            offline=True,
        )
        self.assertNotIn("--offline", acquisition)
        self.assertIn("--offline", measured)
        self.assertEqual(
            acquisition[:3],
            [
                "llama-server",
                "-hf",
                candidate.hf_model,
            ],
        )

    def test_health_poll_retries_transient_connection_reset(
        self,
    ):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b'{"status":"ok"}'
        process = mock.MagicMock()
        process.poll.return_value = None

        with (
            mock.patch(
                "evaluation.vision_benchmark.urllib.request.urlopen",
                side_effect=(
                    ConnectionResetError(10054, "connection reset"),
                    response,
                ),
            ),
            mock.patch(
                "evaluation.vision_benchmark.time.sleep",
            ),
        ):
            elapsed_ms = _wait_for_health(
                "http://127.0.0.1:1234",
                1,
                process,
            )

        self.assertGreaterEqual(elapsed_ms, 0.0)
        self.assertEqual(process.poll.call_count, 2)

    def test_failure_reason_is_sanitized_and_fatal_report_is_nonzero(
        self,
    ):
        self.assertEqual(
            _sanitized_failure_reason(
                VisionProviderError(
                    "llama.cpp vision request failed"
                )
            ),
            "VisionProviderError: llama.cpp vision request failed",
        )
        self.assertEqual(
            _sanitized_failure_reason(
                RuntimeError("raw provider content must not leak")
            ),
            "RuntimeError",
        )
        self.assertEqual(
            _report_exit_code(
                {
                    "candidates": [
                        {"benchmark_valid": False},
                    ]
                }
            ),
            2,
        )
        self.assertEqual(
            _report_exit_code(
                {
                    "candidates": [
                        {"benchmark_valid": True},
                    ]
                }
            ),
            0,
        )

    def test_single_iteration_does_not_claim_deterministic_rerun(
        self,
    ):
        self.assertIsNone(
            _deterministic_rerun_status(1, True)
        )
        self.assertTrue(
            _deterministic_rerun_status(2, True)
        )
        self.assertFalse(
            _deterministic_rerun_status(2, False)
        )

    async def test_fingerprint_mismatch_fails_before_runtime_start(
        self,
    ):
        manifest = SimpleNamespace(
            samples=(),
            fingerprint=lambda: "actual",
            validate_references=lambda root: "content",
        )
        with self.assertRaisesRegex(
            ValueError,
            "manifest fingerprint",
        ):
            await run_vision_benchmark_async(
                manifest=manifest,
                dataset_root=Path("."),
                candidates=(
                    _parse_candidate("qwen=repo:model"),
                ),
                iterations=1,
                expected_manifest_fingerprint="expected",
                expected_dataset_content_fingerprint=None,
                runtime_executable=None,
                startup_timeout_seconds=None,
            )

    def test_pre_policy_claims_preserve_unsupported_output_for_metric(
        self,
    ):
        result = VisionProviderResult(
            image_id="img-1",
            observations=(
                VisionObservation(
                    VisionEvidenceKind.TOPIC,
                    "person religion",
                    0.9,
                ),
            ),
            caption=None,
            provider="llama_cpp",
            provider_version="b123",
            model_id="candidate",
            model_version="v1",
            config_version="phase5-v1",
            confidence_semantics=(
                "model_self_reported_uncalibrated"
            ),
        )
        claims = _provider_claims(result)
        self.assertEqual(
            unsupported_claim_rate(claims),
            1.0,
        )

    def test_visual_coverage_reports_missing_label_types(
        self,
    ):
        sample = SimpleNamespace(
            split=DatasetSplit.EVAL,
            slices=(DatasetSlice.OUTDOOR,),
            ground_truth=SimpleNamespace(
                labels=(
                    SimpleNamespace(
                        type=EvaluationLabelType.OBJECT
                    ),
                    SimpleNamespace(
                        type=EvaluationLabelType.SCENE
                    ),
                )
            ),
        )
        manifest = SimpleNamespace(samples=(sample,))
        coverage = audit_visual_label_coverage(manifest)
        self.assertEqual(
            coverage["label_count_by_type"]["object"],
            1,
        )
        self.assertEqual(
            coverage["label_count_by_type"]["scene"],
            1,
        )
        self.assertEqual(
            coverage["missing_required_label_types"],
            ["activity", "topic"],
        )
        self.assertEqual(
            coverage["label_count_by_slice"]["outdoor"][
                "object"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
