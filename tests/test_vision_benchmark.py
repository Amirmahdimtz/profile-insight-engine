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
    _rejected_candidate_report,
    _report_exit_code,
    _runtime_command,
    _sample_iteration_count,
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
                "--determinism-sample-count",
                "3",
                "--request-timeout-seconds",
                "300",
                "--max-tokens",
                "256",
                "--max-observations-per-kind",
                "1",
                "--max-label-chars",
                "32",
                "--max-caption-chars",
                "80",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(args.determinism_sample_count, 3)
        self.assertEqual(args.request_timeout_seconds, 300)
        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.max_observations_per_kind, 1)
        self.assertEqual(args.max_label_chars, 32)
        self.assertEqual(args.max_caption_chars, 80)

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

    def test_cli_accepts_existing_runtime_for_fast_diagnostic(self):
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
                "--existing-runtime-base-url",
                "http://127.0.0.1:51749",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(
            args.existing_runtime_base_url,
            "http://127.0.0.1:51749",
        )

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
            context_size=4096,
            parallel=1,
        )
        measured = _runtime_command(
            "llama-server",
            candidate,
            1234,
            offline=True,
            context_size=4096,
            parallel=1,
        )
        self.assertNotIn("--offline", acquisition)
        self.assertIn("--offline", measured)
        for command in (acquisition, measured):
            self.assertIn("--ctx-size", command)
            self.assertEqual(
                command[command.index("--ctx-size") + 1],
                "4096",
            )
            self.assertIn("--parallel", command)
            self.assertEqual(
                command[command.index("--parallel") + 1],
                "1",
            )
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

    def test_inference_preflight_rejection_fields_are_traceable(self):
        source = (
            pathlib.Path(__file__).resolve().parents[1]
            / "evaluation"
            / "vision_benchmark.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"rejection_stage": "inference_preflight"',
            source,
        )
        self.assertIn(
            '"runtime_context_size": (',
            source,
        )
        self.assertIn(
            '"runtime_parallel": runtime_parallel',
            source,
        )

    def test_runtime_failure_rejection_report_preserves_evidence(self):
        candidate = _parse_candidate(
            "gemma=ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
        )
        report = _rejected_candidate_report(
            candidate=candidate,
            stage="model_acquisition",
            reason="RuntimeError: model acquisition failed",
            sample_count=18,
            request_timeout_seconds=300,
            max_tokens=768,
            max_observations_per_kind=3,
            max_label_chars=64,
            max_caption_chars=160,
            runtime_context_size=4096,
            runtime_parallel=1,
            model_acquisition_time_ms=1234.5,
        )
        self.assertEqual(
            report["candidate_status"],
            "rejected_preflight",
        )
        self.assertEqual(
            report["rejection_stage"],
            "model_acquisition",
        )
        self.assertFalse(report["benchmark_valid"])
        self.assertEqual(report["runtime_context_size"], 4096)
        self.assertEqual(report["runtime_parallel"], 1)
        self.assertEqual(report["evaluated_sample_count"], 0)
        self.assertIsNone(report["precision"])
        self.assertEqual(
            report["failure_reasons"],
            {"RuntimeError: model acquisition failed": 1},
        )
        self.assertEqual(
            _report_exit_code(
                {
                    "candidates": [
                        report,
                        {
                            "candidate_status": "evaluated",
                            "benchmark_valid": True,
                        },
                    ]
                }
            ),
            0,
        )

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

        self.assertEqual(
            _report_exit_code(
                {
                    "candidates": [
                        {
                            "benchmark_valid": False,
                            "candidate_status": "rejected_preflight",
                        },
                        {
                            "benchmark_valid": True,
                            "candidate_status": "evaluated",
                        },
                    ]
                }
            ),
            0,
        )
        self.assertEqual(
            _report_exit_code(
                {
                    "candidates": [
                        {
                            "benchmark_valid": False,
                            "candidate_status": "rejected_preflight",
                        },
                    ]
                }
            ),
            2,
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

    def test_determinism_reruns_are_limited_to_fixed_subset(self):
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
                sample_index=2,
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
        self.assertEqual(
            _sample_iteration_count(
                sample_index=0,
                iterations=1,
                determinism_sample_count=3,
            ),
            1,
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
