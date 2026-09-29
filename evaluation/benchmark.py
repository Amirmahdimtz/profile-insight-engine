from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.contracts import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkReportKind,
    DatasetSplit,
    EvaluationDatasetManifest,
    EvaluationLabelType,
    EvaluationValidationError,
    MetricResult,
    PredictionSet,
    ReproducibilityMetadata,
    SystemMetrics,
)
from evaluation.metrics import (
    character_error_rate,
    expected_calibration_error,
    precision_recall_f1,
    unsupported_claim_rate,
    word_error_rate,
)


def load_manifest(path: str | Path) -> EvaluationDatasetManifest:
    return EvaluationDatasetManifest.from_json(Path(path).read_text(encoding="utf-8"))


def load_predictions(path: str | Path) -> PredictionSet:
    return PredictionSet.from_json(Path(path).read_text(encoding="utf-8"))


def load_system_metrics(path: str | Path) -> SystemMetrics:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError) as exc:
        raise EvaluationValidationError("system metrics contain invalid JSON") from exc
    return SystemMetrics.from_dict(payload)


def run_benchmark(
    manifest: EvaluationDatasetManifest,
    predictions: PredictionSet,
    *,
    config: BenchmarkConfig | None = None,
    report_kind: BenchmarkReportKind = BenchmarkReportKind.BENCHMARK,
    dataset_content_fingerprint: str | None = None,
    system_metrics: SystemMetrics | None = None,
) -> BenchmarkReport:
    if not isinstance(manifest, EvaluationDatasetManifest):
        raise EvaluationValidationError("manifest must be an EvaluationDatasetManifest")
    if not isinstance(predictions, PredictionSet):
        raise EvaluationValidationError("predictions must be a PredictionSet")
    config = config or BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise EvaluationValidationError("config must be a BenchmarkConfig")
    if not isinstance(report_kind, BenchmarkReportKind):
        raise EvaluationValidationError("report_kind must be a BenchmarkReportKind")
    if system_metrics is not None and not isinstance(system_metrics, SystemMetrics):
        raise EvaluationValidationError("system_metrics must be SystemMetrics or null")

    if predictions.dataset_id != manifest.dataset_id:
        raise EvaluationValidationError("prediction dataset_id does not match manifest")
    if predictions.dataset_version != manifest.dataset_version:
        raise EvaluationValidationError("prediction dataset_version does not match manifest")

    eval_samples = tuple(
        sample for sample in manifest.samples if sample.split is DatasetSplit.EVAL
    )
    expected_sample_ids = {sample.sample_id for sample in eval_samples}
    actual_sample_ids = {prediction.sample_id for prediction in predictions.predictions}
    if actual_sample_ids != expected_sample_ids:
        missing = sorted(expected_sample_ids - actual_sample_ids)
        unexpected = sorted(actual_sample_ids - expected_sample_ids)
        details: list[str] = []
        if missing:
            details.append(f"missing eval predictions: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected predictions: {', '.join(unexpected)}")
        raise EvaluationValidationError("; ".join(details))

    prediction_by_sample = {
        prediction.sample_id: prediction for prediction in predictions.predictions
    }

    expected_label_instances: set[tuple[str, str, str, str]] = set()
    predicted_label_instances: set[tuple[str, str, str, str]] = set()
    calibration_observations: list[tuple[float, bool]] = []
    cer_values: list[float] = []
    wer_values: list[float] = []
    claims: list[tuple[str, str]] = []

    for sample in eval_samples:
        prediction = prediction_by_sample[sample.sample_id]
        expected_identities = {
            label.identity
            for label in sample.ground_truth.labels
            if label.type is not EvaluationLabelType.OCR_TEXT
        }
        predicted_identities = {
            label.identity
            for label in prediction.labels
            if label.type is not EvaluationLabelType.OCR_TEXT
        }

        expected_label_instances.update(
            (sample.sample_id, label_type, label, value)
            for label_type, label, value in expected_identities
        )
        predicted_label_instances.update(
            (sample.sample_id, label_type, label, value)
            for label_type, label, value in predicted_identities
        )

        for predicted_label in prediction.labels:
            if predicted_label.type is EvaluationLabelType.OCR_TEXT:
                continue
            calibration_observations.append(
                (
                    predicted_label.confidence,
                    predicted_label.identity in expected_identities,
                )
            )

        if sample.ground_truth.ocr_text is not None:
            if prediction.ocr_text is None:
                raise EvaluationValidationError(
                    f"sample '{sample.sample_id}' has OCR ground truth but no OCR prediction"
                )
            cer_values.append(
                character_error_rate(sample.ground_truth.ocr_text, prediction.ocr_text)
            )
            wer_values.append(
                word_error_rate(sample.ground_truth.ocr_text, prediction.ocr_text)
            )

        claims.extend((claim.key, claim.label) for claim in prediction.claims)

    classification = precision_recall_f1(
        expected_label_instances, predicted_label_instances
    )
    metrics = [
        MetricResult(
            name="visual_precision",
            value=classification.precision,
            sample_count=len(eval_samples),
            minimum=0.0,
            maximum=1.0,
        ),
        MetricResult(
            name="visual_recall",
            value=classification.recall,
            sample_count=len(eval_samples),
            minimum=0.0,
            maximum=1.0,
        ),
        MetricResult(
            name="visual_f1",
            value=classification.f1,
            sample_count=len(eval_samples),
            minimum=0.0,
            maximum=1.0,
        ),
        MetricResult(
            name="confidence_ece",
            value=expected_calibration_error(
                calibration_observations, bins=config.calibration_bins
            ),
            sample_count=len(calibration_observations),
            minimum=0.0,
            maximum=1.0,
        ),
        MetricResult(
            name="unsupported_claim_rate",
            value=unsupported_claim_rate(claims),
            sample_count=len(claims),
            minimum=0.0,
            maximum=1.0,
        ),
    ]
    if cer_values:
        metrics.extend(
            [
                MetricResult(
                    name="ocr_cer",
                    value=sum(cer_values) / len(cer_values),
                    sample_count=len(cer_values),
                    minimum=0.0,
                ),
                MetricResult(
                    name="ocr_wer",
                    value=sum(wer_values) / len(wer_values),
                    sample_count=len(wer_values),
                    minimum=0.0,
                ),
            ]
        )

    reproducibility = ReproducibilityMetadata(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        manifest_fingerprint=manifest.fingerprint(),
        dataset_content_fingerprint=dataset_content_fingerprint,
        evaluator_version=config.evaluator_version,
        config=config,
    )
    return BenchmarkReport(
        schema_version=BENCHMARK_REPORT_SCHEMA_VERSION,
        report_kind=report_kind,
        reproducibility=reproducibility,
        metrics=tuple(metrics),
        system_metrics=system_metrics,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Phase 2 structured-data evaluation."
    )
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest JSON")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--output", help="Optional report output path; stdout is used otherwise")
    parser.add_argument(
        "--dataset-root",
        help="Optional dataset root for structural file-reference existence checks",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of deterministic ECE bins (recorded in report metadata)",
    )
    parser.add_argument(
        "--report-kind",
        choices=[member.value for member in BenchmarkReportKind],
        default=BenchmarkReportKind.BENCHMARK.value,
    )
    parser.add_argument(
        "--system-metrics",
        help="Optional JSON file with measured p50/p95 latency and RAM/VRAM values",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    dataset_content_fingerprint = None
    if args.dataset_root:
        dataset_content_fingerprint = manifest.validate_references(args.dataset_root)
    predictions = load_predictions(args.predictions)
    system_metrics = (
        None if args.system_metrics is None else load_system_metrics(args.system_metrics)
    )
    report = run_benchmark(
        manifest,
        predictions,
        config=BenchmarkConfig(calibration_bins=args.calibration_bins),
        report_kind=BenchmarkReportKind(args.report_kind),
        dataset_content_fingerprint=dataset_content_fingerprint,
        system_metrics=system_metrics,
    )
    serialized = report.to_json() + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
