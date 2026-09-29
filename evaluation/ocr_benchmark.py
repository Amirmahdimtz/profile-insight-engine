from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from evaluation.benchmark import load_manifest
from evaluation.common import DatasetSlice, DatasetSplit
from evaluation.image_preprocessing_benchmark import _peak_rss_mb
from evaluation.metrics import character_error_rate, word_error_rate
from src.core.profile_analysis.contracts import ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import RawImageInput
from src.core.services.profile_analysis.image_processing_service import ImageProcessingService
from src.core.services.profile_analysis.ocr_evidence_service import (
    OcrEvidenceService,
    normalize_ocr_text,
)
from src.infrastructure.providers.image.local_image_storage import LocalImageStorage
from src.infrastructure.providers.ocr.tesseract_ocr_provider import (
    TesseractOcrProvider,
    TesseractOcrSettings,
)
from src.infrastructure.utils.config_reader import ConfigReader


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    tessdata_dir: str | None


def _parse_candidate(value: str) -> CandidateSpec:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("candidate must be NAME=TESDATA_DIR or NAME=default")
    name, separator, location = value.partition("=")
    name = name.strip()
    location = location.strip()
    if not separator or not name or not location:
        raise argparse.ArgumentTypeError("candidate must be NAME=TESDATA_DIR or NAME=default")
    return CandidateSpec(name=name, tessdata_dir=None if location.lower() == "default" else location)


def _mime_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    try:
        return mapping[suffix]
    except KeyError as exc:
        raise ValueError(f"unsupported Phase 3 evaluation image extension: {suffix or '<none>'}") from exc


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.999999)))
    return ordered[index]


def _cpu_seconds() -> float | None:
    if sys.platform.startswith("win"):
        return None
    values = os.times()
    return values.user + values.system + values.children_user + values.children_system


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _sample_signature(result: Any) -> tuple[object, ...]:
    return (
        result.raw_text,
        result.normalized_text,
        tuple(
            (
                block.order,
                block.raw_text,
                block.normalized_text,
                block.confidence,
                block.script_hint,
                block.language_hint,
                None if block.region is None else (
                    block.region.x,
                    block.region.y,
                    block.region.width,
                    block.region.height,
                ),
            )
            for block in result.blocks
        ),
        tuple(item.to_dict() for item in result.evidence),
    )


async def run_ocr_benchmark_async(
    manifest: Any,
    dataset_root: Path,
    candidates: tuple[CandidateSpec, ...],
    iterations: int,
    expected_manifest_fingerprint: str | None,
    expected_dataset_content_fingerprint: str | None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not candidates or len({item.name for item in candidates}) != len(candidates):
        raise ValueError("candidates must contain unique names")

    manifest_fingerprint = manifest.fingerprint()
    if expected_manifest_fingerprint is not None and manifest_fingerprint != expected_manifest_fingerprint:
        raise ValueError("manifest fingerprint does not match the expected Phase 2 dataset")
    dataset_content_fingerprint = manifest.validate_references(dataset_root)
    if (
        expected_dataset_content_fingerprint is not None
        and dataset_content_fingerprint != expected_dataset_content_fingerprint
    ):
        raise ValueError("dataset content fingerprint does not match the expected Phase 2 dataset")

    eval_samples = tuple(sample for sample in manifest.samples if sample.split is DatasetSplit.EVAL)
    ocr_samples = tuple(sample for sample in eval_samples if sample.ground_truth.ocr_text is not None)
    if not ocr_samples:
        raise ValueError("evaluation dataset contains no OCR ground truth")

    coverage_by_slice = {
        item.value: sum(1 for sample in ocr_samples if item in sample.slices)
        for item in DatasetSlice
    }

    base_config = ConfigReader()
    image_service = ImageProcessingService(base_config, LocalImageStorage())
    base_settings = TesseractOcrSettings.from_config(base_config)
    candidate_reports: list[dict[str, Any]] = []

    for candidate in candidates:
        settings = TesseractOcrSettings(
            executable=base_settings.executable,
            languages=base_settings.languages,
            page_segmentation_mode=base_settings.page_segmentation_mode,
            timeout_seconds=base_settings.timeout_seconds,
            model_id=candidate.name,
            config_version=base_settings.config_version,
            tessdata_dir=candidate.tessdata_dir,
        )
        provider = TesseractOcrProvider(base_config, settings)
        service = OcrEvidenceService(provider)
        latencies_ms: list[float] = []
        raw_cer: list[float] = []
        raw_wer: list[float] = []
        normalized_cer: list[float] = []
        normalized_wer: list[float] = []
        per_slice_raw_cer: dict[str, list[float]] = {item.value: [] for item in DatasetSlice}
        per_slice_raw_wer: dict[str, list[float]] = {item.value: [] for item in DatasetSlice}
        deterministic = True
        provider_version: str | None = None
        peak_ram_mb = _peak_rss_mb()
        wall_started = time.perf_counter()
        cpu_started = _cpu_seconds()

        for sample in ocr_samples:
            source_path = dataset_root / sample.image.relative_path
            raw = RawImageInput(
                image_id=sample.image.image_id,
                filename=source_path.name,
                declared_mime_type=_mime_from_path(source_path),
                content=source_path.read_bytes(),
            )
            request = ProfileAnalysisRequest(
                analysis_id=f"phase4-{candidate.name}-{sample.sample_id}",
                image_ids=(sample.image.image_id,),
            )
            batch = await image_service.process_async(request, (raw,))
            try:
                signatures: list[tuple[object, ...]] = []
                first_result = None
                for _ in range(iterations):
                    started = time.perf_counter()
                    result = await service.extract_async(batch.images[0])
                    latencies_ms.append((time.perf_counter() - started) * 1000.0)
                    peak_ram_mb = max(peak_ram_mb, _peak_rss_mb())
                    signatures.append(_sample_signature(result))
                    first_result = first_result or result
                    provider_version = provider_version or result.provider_version
                    if result.provider_version != provider_version:
                        deterministic = False
                if any(signature != signatures[0] for signature in signatures[1:]):
                    deterministic = False
                if first_result is None:
                    raise RuntimeError("OCR benchmark produced no result for an OCR-annotated sample")
                reference_raw = sample.ground_truth.ocr_text
                reference_normalized = normalize_ocr_text(reference_raw)
                sample_raw_cer = character_error_rate(reference_raw, first_result.raw_text)
                sample_raw_wer = word_error_rate(reference_raw, first_result.raw_text)
                sample_normalized_cer = character_error_rate(
                    reference_normalized, first_result.normalized_text
                )
                sample_normalized_wer = word_error_rate(
                    reference_normalized, first_result.normalized_text
                )
                raw_cer.append(sample_raw_cer)
                raw_wer.append(sample_raw_wer)
                normalized_cer.append(sample_normalized_cer)
                normalized_wer.append(sample_normalized_wer)
                for slice_value in sample.slices:
                    per_slice_raw_cer[slice_value.value].append(sample_raw_cer)
                    per_slice_raw_wer[slice_value.value].append(sample_raw_wer)
            finally:
                await image_service.release_async(batch)

        wall_seconds = time.perf_counter() - wall_started
        cpu_ended = _cpu_seconds()
        cpu_usage_percent = None
        if cpu_started is not None and cpu_ended is not None and wall_seconds > 0:
            cpu_usage_percent = max(0.0, (cpu_ended - cpu_started) / wall_seconds * 100.0)

        slice_metrics = {
            key: {
                "sample_count": len(per_slice_raw_cer[key]),
                "raw_cer": _mean(per_slice_raw_cer[key]),
                "raw_wer": _mean(per_slice_raw_wer[key]),
            }
            for key in sorted(per_slice_raw_cer)
            if per_slice_raw_cer[key]
        }
        candidate_reports.append(
            {
                "candidate": candidate.name,
                "provider": "tesseract",
                "provider_version": provider_version,
                "model_id": candidate.name,
                "config_version": settings.config_version,
                "languages": list(settings.languages),
                "tessdata_dir": candidate.tessdata_dir,
                "sample_count": len(ocr_samples),
                "raw_cer": _mean(raw_cer),
                "raw_wer": _mean(raw_wer),
                "normalized_cer": _mean(normalized_cer),
                "normalized_wer": _mean(normalized_wer),
                "p50_latency_ms": statistics.median(latencies_ms),
                "p95_latency_ms": _percentile(latencies_ms, 0.95),
                "process_peak_ram_mb": peak_ram_mb,
                "cpu_usage_percent": cpu_usage_percent,
                "gpu_required": False,
                "gpu_usage": "not_applicable_cpu_provider",
                "deterministic_rerun": deterministic,
                "slice_metrics": slice_metrics,
            }
        )

    return {
        "schema_version": "1.0.0",
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "manifest_fingerprint": manifest_fingerprint,
        "dataset_content_fingerprint": dataset_content_fingerprint,
        "eval_sample_count": len(eval_samples),
        "ocr_ground_truth_sample_count": len(ocr_samples),
        "ocr_ground_truth_coverage_by_slice": coverage_by_slice,
        "iterations": iterations,
        "candidates": candidate_reports,
        "resource_measurement_notes": {
            "cpu_usage_percent": "null on Windows because stdlib process accounting cannot include Tesseract child CPU reliably",
            "process_peak_ram_mb": "Python process peak RSS; Tesseract child peak memory is not claimed by this metric",
            "gpu_usage": "Tesseract provider is CPU-only in this Phase 4 implementation",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Phase 4 OCR candidates on authorized data")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--expected-manifest-fingerprint")
    parser.add_argument("--expected-dataset-content-fingerprint")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = asyncio.run(
        run_ocr_benchmark_async(
            manifest=manifest,
            dataset_root=Path(args.dataset_root),
            candidates=tuple(args.candidate),
            iterations=args.iterations,
            expected_manifest_fingerprint=args.expected_manifest_fingerprint,
            expected_dataset_content_fingerprint=args.expected_dataset_content_fingerprint,
        )
    )
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(
        f"WROTE {output} manifest_fingerprint={report['manifest_fingerprint']} "
        f"dataset_content_fingerprint={report['dataset_content_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
