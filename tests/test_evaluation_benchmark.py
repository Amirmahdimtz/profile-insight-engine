import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark import load_manifest, load_predictions, main, run_benchmark
from evaluation.contracts import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkReportKind,
    EvaluationDatasetManifest,
    EvaluationValidationError,
    PredictionSet,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "evaluation" / "fixtures" / "sanity_manifest.json"
PREDICTIONS_PATH = REPOSITORY_ROOT / "evaluation" / "fixtures" / "sanity_predictions.json"
BASELINE_PATH = REPOSITORY_ROOT / "evaluation" / "baselines" / "infrastructure_sanity_baseline.json"


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST_PATH)
        self.predictions = load_predictions(PREDICTIONS_PATH)

    def test_same_input_and_config_produce_identical_report(self):
        config = BenchmarkConfig(calibration_bins=10)
        first = run_benchmark(
            self.manifest,
            self.predictions,
            config=config,
            report_kind=BenchmarkReportKind.INFRASTRUCTURE_SANITY_BASELINE,
        )
        second = run_benchmark(
            self.manifest,
            self.predictions,
            config=config,
            report_kind=BenchmarkReportKind.INFRASTRUCTURE_SANITY_BASELINE,
        )
        self.assertEqual(first.to_json(), second.to_json())

    def test_registered_infrastructure_sanity_baseline_matches_runner(self):
        report = run_benchmark(
            self.manifest,
            self.predictions,
            config=BenchmarkConfig(calibration_bins=10),
            report_kind=BenchmarkReportKind.INFRASTRUCTURE_SANITY_BASELINE,
        )
        self.assertEqual(BASELINE_PATH.read_text(encoding="utf-8").strip(), report.to_json())

    def test_metric_report_has_stable_sorted_names(self):
        report = run_benchmark(self.manifest, self.predictions)
        names = [metric.name for metric in report.metrics]
        self.assertEqual(names, sorted(names))
        self.assertEqual(
            names,
            [
                "confidence_ece",
                "ocr_cer",
                "ocr_wer",
                "unsupported_claim_rate",
                "visual_f1",
                "visual_precision",
                "visual_recall",
            ],
        )

    def test_report_contains_reproducibility_metadata(self):
        report = run_benchmark(self.manifest, self.predictions)
        self.assertEqual(report.reproducibility.dataset_version, "1.0.0")
        self.assertEqual(
            report.reproducibility.manifest_fingerprint,
            self.manifest.fingerprint(),
        )
        self.assertEqual(report.reproducibility.config.calibration_bins, 10)

    def test_dataset_id_and_version_must_match_predictions(self):
        payload = json.loads(self.predictions.to_json())
        payload["dataset_id"] = "other_dataset"
        with self.assertRaisesRegex(EvaluationValidationError, "dataset_id"):
            run_benchmark(self.manifest, PredictionSet.from_dict(payload))

        payload = json.loads(self.predictions.to_json())
        payload["dataset_version"] = "9.9.9"
        with self.assertRaisesRegex(EvaluationValidationError, "dataset_version"):
            run_benchmark(self.manifest, PredictionSet.from_dict(payload))

    def test_predictions_must_exactly_cover_eval_split(self):
        payload = json.loads(self.predictions.to_json())
        payload["predictions"] = payload["predictions"][:-1]
        with self.assertRaisesRegex(EvaluationValidationError, "missing eval predictions"):
            run_benchmark(self.manifest, PredictionSet.from_dict(payload))

        payload = json.loads(self.predictions.to_json())
        payload["predictions"].append(
            {"sample_id": "sample_train_001", "labels": [], "claims": []}
        )
        with self.assertRaisesRegex(EvaluationValidationError, "unexpected predictions"):
            run_benchmark(self.manifest, PredictionSet.from_dict(payload))

    def test_malformed_duplicate_predicted_label_is_rejected(self):
        payload = json.loads(self.predictions.to_json())
        duplicate = dict(payload["predictions"][0]["labels"][0])
        payload["predictions"][0]["labels"].append(duplicate)
        with self.assertRaisesRegex(EvaluationValidationError, "duplicates"):
            PredictionSet.from_dict(payload)

    def test_missing_ocr_prediction_for_labeled_sample_is_rejected(self):
        payload = json.loads(self.predictions.to_json())
        del payload["predictions"][0]["ocr_text"]
        predictions = PredictionSet.from_dict(payload)
        with self.assertRaisesRegex(EvaluationValidationError, "no OCR prediction"):
            run_benchmark(self.manifest, predictions)

    def test_unknown_prediction_field_is_rejected(self):
        payload = json.loads(self.predictions.to_json())
        payload["predictions"][0]["unexpected"] = True
        with self.assertRaisesRegex(EvaluationValidationError, "unknown fields"):
            PredictionSet.from_dict(payload)

    def test_cli_writes_same_deterministic_sanity_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            exit_code = main(
                [
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--predictions",
                    str(PREDICTIONS_PATH),
                    "--report-kind",
                    "infrastructure_sanity_baseline",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                output.read_text(encoding="utf-8").strip(),
                BASELINE_PATH.read_text(encoding="utf-8").strip(),
            )

    def test_report_contract_roundtrip_and_consistency_validation(self):
        report = run_benchmark(self.manifest, self.predictions)
        reparsed = BenchmarkReport.from_json(report.to_json())
        self.assertEqual(report.to_json(), reparsed.to_json())

        payload = json.loads(report.to_json())
        payload["reproducibility"]["evaluator_version"] = "different-evaluator"
        with self.assertRaisesRegex(EvaluationValidationError, "must match"):
            BenchmarkReport.from_dict(payload)

        payload = json.loads(report.to_json())
        payload["metrics"][0]["unexpected"] = True
        with self.assertRaisesRegex(EvaluationValidationError, "unknown fields"):
            BenchmarkReport.from_dict(payload)

    def test_manifest_and_prediction_serialization_are_order_stable(self):
        manifest_payload = json.loads(self.manifest.to_json())
        manifest_payload["samples"].reverse()
        reparsed_manifest = EvaluationDatasetManifest.from_dict(manifest_payload)
        self.assertEqual(self.manifest.to_json(), reparsed_manifest.to_json())

        prediction_payload = json.loads(self.predictions.to_json())
        prediction_payload["predictions"].reverse()
        reparsed_predictions = PredictionSet.from_dict(prediction_payload)
        self.assertEqual(self.predictions.to_json(), reparsed_predictions.to_json())


if __name__ == "__main__":
    unittest.main()
