from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.benchmark import load_manifest
from evaluation.common import DatasetSplit, EvaluationLabelType
from evaluation.metrics import precision_recall_f1, unsupported_claim_rate
from evaluation.vision_benchmark import (
    _benchmark_label,
    _mime_from_path,
    _percentile,
    _provider_claims,
    _sanitized_failure_reason,
    audit_visual_label_coverage,
)
from src.core.profile_analysis.contracts import EvidenceType, ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import RawImageInput
from src.core.services.profile_analysis.evidence_extraction_service import (
    EvidenceExtractionService,
)
from src.core.services.profile_analysis.image_processing_service import (
    ImageProcessingService,
)
from src.infrastructure.providers.image.local_image_storage import (
    LocalImageStorage,
)
from src.infrastructure.providers.vision.transformers_vision_provider import (
    TransformersVisionProvider,
    TransformersVisionSettings,
)
from src.infrastructure.utils.config_reader import ConfigReader


REPORT_SCHEMA_VERSION = "phase10-transformers-vision-benchmark-v1"
METRIC_SCOPE = "phase10_transformers_vision_candidate_on_target_hardware"
_VISUAL_LABEL_TYPES = (
    EvaluationLabelType.SCENE,
    EvaluationLabelType.OBJECT,
    EvaluationLabelType.ACTIVITY,
    EvaluationLabelType.TOPIC,
)


def _deterministic_status(
    iterations: int,
    deterministic: bool,
) -> bool | None:
    return deterministic if iterations >= 2 else None


def _sample_iteration_count(
    *,
    sample_index: int,
    iterations: int,
    determinism_sample_count: int,
) -> int:
    if iterations < 2:
        return 1
    return iterations if sample_index < determinism_sample_count else 1


def _report_exit_code(report: Mapping[str, Any]) -> int:
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping):
        return 2
    return 0 if candidate.get("benchmark_valid") is True else 2


def _manifest_identity(manifest: Any) -> tuple[str, str]:
    return manifest.dataset_id, manifest.dataset_version


async def run_transformers_vision_benchmark_async(
    *,
    manifest: Any,
    dataset_root: Path,
    iterations: int,
    determinism_sample_count: int,
    expected_manifest_fingerprint: str | None,
    expected_dataset_content_fingerprint: str | None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if (
        isinstance(determinism_sample_count, bool)
        or not isinstance(determinism_sample_count, int)
        or determinism_sample_count <= 0
    ):
        raise ValueError(
            "determinism_sample_count must be a positive integer"
        )

    manifest_fingerprint = manifest.fingerprint()
    if (
        expected_manifest_fingerprint is not None
        and manifest_fingerprint != expected_manifest_fingerprint
    ):
        raise ValueError(
            "manifest fingerprint does not match the expected Phase 2 dataset"
        )
    dataset_content_fingerprint = manifest.validate_references(dataset_root)
    if (
        expected_dataset_content_fingerprint is not None
        and dataset_content_fingerprint
        != expected_dataset_content_fingerprint
    ):
        raise ValueError(
            "dataset content fingerprint does not match "
            "the expected Phase 2 dataset"
        )

    eval_samples = tuple(
        sample
        for sample in manifest.samples
        if sample.split is DatasetSplit.EVAL
    )
    if not eval_samples:
        raise ValueError("evaluation split must contain at least one sample")

    config = ConfigReader()
    settings = TransformersVisionSettings.from_config(config)
    provider = TransformersVisionProvider(config, settings)
    service = EvidenceExtractionService(provider)
    image_service = ImageProcessingService(
        config,
        LocalImageStorage(),
    )

    expected_by_type: dict[str, set[tuple[str, str]]] = {
        item.value: set() for item in _VISUAL_LABEL_TYPES
    }
    predicted_by_type: dict[str, set[tuple[str, str]]] = {
        item.value: set() for item in _VISUAL_LABEL_TYPES
    }
    pre_policy_claims: list[tuple[str, str]] = []
    post_validation_claims: list[tuple[str, str]] = []
    latencies_ms: list[float] = []
    failed_samples = 0
    failure_reasons: dict[str, int] = {}
    deterministic = True
    provider_version: str | None = None
    model_version: str | None = None

    effective_determinism_sample_count = (
        min(determinism_sample_count, len(eval_samples))
        if iterations >= 2
        else 0
    )

    for sample in eval_samples:
        for label in sample.ground_truth.labels:
            if label.type in _VISUAL_LABEL_TYPES:
                expected_value = (
                    label.value.strip()
                    if isinstance(label.value, str) and label.value.strip()
                    else label.label
                )
                expected_by_type[label.type.value].add(
                    (
                        sample.sample_id,
                        _benchmark_label(expected_value),
                    )
                )

    attempted_samples = 0
    successful_samples = 0
    preflight_failed = False

    for sample_index, sample in enumerate(eval_samples):
        source_path = dataset_root / sample.image.relative_path
        raw = RawImageInput(
            image_id=sample.image.image_id,
            filename=source_path.name,
            declared_mime_type=_mime_from_path(source_path),
            content=source_path.read_bytes(),
        )
        request = ProfileAnalysisRequest(
            analysis_id=(
                f"phase10-transformers-vision-{sample.sample_id}"
            ),
            image_ids=(sample.image.image_id,),
        )

        attempted_samples += 1
        batch = await image_service.process_async(request, (raw,))
        try:
            signatures: list[
                tuple[tuple[str, str, float], ...]
            ] = []
            first_result = None
            run_count = _sample_iteration_count(
                sample_index=sample_index,
                iterations=iterations,
                determinism_sample_count=(
                    effective_determinism_sample_count
                ),
            )
            try:
                for iteration_index in range(run_count):
                    started = time.perf_counter()
                    provider_result = await provider.extract_async(
                        batch.images[0]
                    )
                    if iteration_index == 0:
                        provider_version = provider_result.provider_version
                        model_version = provider_result.model_version
                        pre_policy_claims.extend(
                            _provider_claims(provider_result)
                        )
                    result = service.normalize_result(
                        batch.images[0],
                        provider_result,
                    )
                    if iteration_index == 0:
                        latencies_ms.append(
                            (time.perf_counter() - started) * 1000.0
                        )
                        first_result = result
                    signatures.append(
                        tuple(
                            (
                                item.type.value,
                                str(item.value),
                                item.confidence,
                            )
                            for item in result.evidence
                        )
                    )
                if any(
                    signature != signatures[0]
                    for signature in signatures[1:]
                ):
                    deterministic = False
            except Exception as exc:
                failed_samples += 1
                reason = _sanitized_failure_reason(exc)
                failure_reasons[reason] = (
                    failure_reasons.get(reason, 0) + 1
                )
                if sample_index == 0:
                    preflight_failed = True
                    break
                continue

            successful_samples += 1
            if first_result is not None:
                for evidence in first_result.evidence:
                    if evidence.type.value in predicted_by_type:
                        predicted_by_type[evidence.type.value].add(
                            (
                                sample.sample_id,
                                _benchmark_label(str(evidence.value)),
                            )
                        )
                        post_validation_claims.append(
                            (
                                evidence.type.value,
                                str(evidence.value),
                            )
                        )
                    elif evidence.type is EvidenceType.OTHER_OBSERVABLE:
                        post_validation_claims.append(
                            ("caption", str(evidence.value))
                        )
        finally:
            await image_service.release_async(batch)

    per_label_type: dict[str, Any] = {}
    aggregate_expected: set[tuple[str, str, str]] = set()
    aggregate_predicted: set[tuple[str, str, str]] = set()
    evaluated_label_types: list[str] = []

    for label_type in _VISUAL_LABEL_TYPES:
        expected = expected_by_type[label_type.value]
        predicted = predicted_by_type[label_type.value]
        if preflight_failed:
            per_label_type[label_type.value] = {
                "evaluated": False,
                "reason": "candidate rejected during inference preflight",
                "precision": None,
                "recall": None,
                "f1": None,
                "expected_count": len(expected),
                "predicted_count": 0,
            }
            continue
        if not expected:
            per_label_type[label_type.value] = {
                "evaluated": False,
                "reason": "no ground-truth labels for this visual type",
                "precision": None,
                "recall": None,
                "f1": None,
                "expected_count": 0,
                "predicted_count": len(predicted),
            }
            continue
        metric = precision_recall_f1(expected, predicted)
        evaluated_label_types.append(label_type.value)
        per_label_type[label_type.value] = {
            "evaluated": True,
            "reason": None,
            "precision": metric.precision,
            "recall": metric.recall,
            "f1": metric.f1,
            "expected_count": len(expected),
            "predicted_count": len(predicted),
        }
        aggregate_expected.update(
            (label_type.value, *item) for item in expected
        )
        aggregate_predicted.update(
            (label_type.value, *item) for item in predicted
        )

    overall = (
        precision_recall_f1(
            aggregate_expected,
            aggregate_predicted,
        )
        if aggregate_expected and not preflight_failed
        else None
    )

    candidate_report = {
        "candidate": "smolvlm2_256m_transformers_single_tile",
        "candidate_status": (
            "rejected_preflight"
            if preflight_failed
            else "evaluated"
        ),
        "provider": "transformers",
        "provider_version": provider_version,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "model_version": model_version,
        "config_version": settings.config_version,
        "device": settings.device,
        "local_files_only": settings.local_files_only,
        "image_longest_edge": settings.image_longest_edge,
        "do_image_splitting": settings.do_image_splitting,
        "structured_backend": settings.structured_backend,
        "max_tokens": settings.max_tokens,
        "max_observations_per_kind": settings.max_observations_per_kind,
        "max_label_chars": settings.max_label_chars,
        "max_caption_chars": settings.max_caption_chars,
        "sample_count": len(eval_samples),
        "evaluated_sample_count": successful_samples,
        "attempted_sample_count": attempted_samples,
        "failed_sample_count": failed_samples,
        "failure_rate": (
            failed_samples / attempted_samples
            if attempted_samples
            else None
        ),
        "benchmark_valid": (
            successful_samples > 0
            and not preflight_failed
        ),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "evaluated_label_types": evaluated_label_types,
        "precision": None if overall is None else overall.precision,
        "recall": None if overall is None else overall.recall,
        "f1": None if overall is None else overall.f1,
        "unsupported_claim_rate_pre_policy": unsupported_claim_rate(
            pre_policy_claims
        ),
        "unsupported_claim_rate_post_validation": unsupported_claim_rate(
            post_validation_claims
        ),
        "p50_latency_ms": (
            statistics.median(latencies_ms)
            if latencies_ms
            else None
        ),
        "p95_latency_ms": (
            _percentile(latencies_ms, 0.95)
            if latencies_ms
            else None
        ),
        "determinism_sample_count": (
            0
            if preflight_failed
            else effective_determinism_sample_count
        ),
        "deterministic_rerun": (
            None
            if preflight_failed
            else _deterministic_status(
                iterations,
                deterministic,
            )
        ),
        "per_label_type": per_label_type,
    }

    dataset_id, dataset_version = _manifest_identity(manifest)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metric_scope": METRIC_SCOPE,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "manifest_fingerprint": manifest_fingerprint,
        "dataset_content_fingerprint": dataset_content_fingerprint,
        "coverage": audit_visual_label_coverage(manifest),
        "iterations": iterations,
        "candidate": candidate_report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--determinism-sample-count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--expected-manifest-fingerprint",
    )
    parser.add_argument(
        "--expected-dataset-content-fingerprint",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = asyncio.run(
        run_transformers_vision_benchmark_async(
            manifest=manifest,
            dataset_root=args.dataset_root,
            iterations=args.iterations,
            determinism_sample_count=(
                args.determinism_sample_count
            ),
            expected_manifest_fingerprint=(
                args.expected_manifest_fingerprint
            ),
            expected_dataset_content_fingerprint=(
                args.expected_dataset_content_fingerprint
            ),
        )
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return _report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
