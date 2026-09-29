from __future__ import annotations

import argparse
import asyncio
import io
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps, __version__ as pillow_version

from evaluation.contracts import EvaluationDatasetManifest
from src.core.profile_analysis.contracts import ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import RawImageInput
from src.core.services.profile_analysis.image_processing_service import (
    ImageProcessingService,
    calculate_resized_dimensions,
)
from src.infrastructure.providers.image.local_image_storage import LocalImageStorage
from src.infrastructure.utils.config_reader import ConfigReader


_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@dataclass(frozen=True)
class BatchBenchmarkResult:
    image_count: int
    iterations: int
    source_sample_count: int
    replayed_sample_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    decode_images_per_second: float
    isolated_resize_cost_ms: float
    resized_image_count: int
    process_peak_ram_mb: float
    deterministic_hashes: bool


@dataclass(frozen=True)
class ImagePreprocessingBenchmarkReport:
    schema_version: str
    dataset_id: str
    dataset_version: str
    dataset_sample_count: int
    manifest_fingerprint: str
    dataset_content_fingerprint: str
    pillow_version: str
    batch_results: tuple[BatchBenchmarkResult, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_manifest(path: Path) -> EvaluationDatasetManifest:
    return EvaluationDatasetManifest.from_json(path.read_text(encoding="utf-8"))


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated positive integers") from exc
    if not sizes or any(size <= 0 for size in sizes) or len(sizes) != len(set(sizes)):
        raise argparse.ArgumentTypeError("batch sizes must be unique positive integers")
    return sizes


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _workload_sample_indices(dataset_size: int, image_count: int) -> tuple[int, ...]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    return tuple(index % dataset_size for index in range(image_count))


def _benchmark_image_id(position: int) -> str:
    if position < 0:
        raise ValueError("position must not be negative")
    return f"phase3-benchmark-image-{position + 1:04d}"


def _windows_peak_working_set_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    ):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "GetProcessMemoryInfo failed")
    return float(counters.PeakWorkingSetSize) / (1024 * 1024)


def _peak_rss_mb() -> float:
    if sys.platform == "win32":
        return _windows_peak_working_set_mb()

    try:
        import resource
    except ImportError as exc:
        raise RuntimeError("RAM peak measurement is unsupported on this platform") from exc
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _declared_mime(content: bytes, image_id: str) -> str:
    with Image.open(io.BytesIO(content)) as image:
        mime = _FORMAT_TO_MIME.get((image.format or "").upper())
    if mime is None:
        raise ValueError(f"benchmark image '{image_id}' uses an unsupported format")
    return mime


def _measure_decode_throughput(contents: Sequence[bytes]) -> float:
    started = time.perf_counter()
    for content in contents:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("decode benchmark clock did not advance")
    return len(contents) / elapsed


def _measure_resize_cost(contents: Sequence[bytes], max_dimension: int) -> tuple[float, int]:
    elapsed = 0.0
    resized_count = 0
    for content in contents:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            target = calculate_resized_dimensions(
                normalized.width, normalized.height, max_dimension
            )
            if target == normalized.size:
                continue
            started = time.perf_counter()
            normalized.resize(
                target,
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            elapsed += time.perf_counter() - started
            resized_count += 1
    return elapsed * 1000.0, resized_count


def _stable_signature(batch) -> tuple[object, ...]:
    return (
        tuple(
            (
                image.image_id,
                image.content_hash,
                image.width,
                image.height,
                image.mime_type,
                image.was_resized,
                image.exif_orientation_applied,
            )
            for image in batch.images
        ),
        tuple(
            (item.duplicate_image_id, item.canonical_image_id, item.content_hash)
            for item in batch.duplicates
        ),
    )


async def run_benchmark_async(
    manifest: EvaluationDatasetManifest,
    dataset_root: Path,
    batch_sizes: tuple[int, ...],
    iterations: int,
) -> ImagePreprocessingBenchmarkReport:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not batch_sizes or any(size <= 0 for size in batch_sizes) or len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError("batch_sizes must contain unique positive integers")
    content_fingerprint = manifest.validate_references(dataset_root)

    config_reader = ConfigReader()
    service = ImageProcessingService(config_reader, LocalImageStorage())
    max_dimension = config_reader.get_positive_int("image.max_dimension")
    results: list[BatchBenchmarkResult] = []

    for image_count in batch_sizes:
        sample_indices = _workload_sample_indices(len(manifest.samples), image_count)
        selected = tuple(manifest.samples[index] for index in sample_indices)
        contents = tuple(
            (dataset_root / sample.image.relative_path).read_bytes() for sample in selected
        )
        inputs = tuple(
            RawImageInput(
                image_id=_benchmark_image_id(position),
                filename=Path(sample.image.relative_path).name,
                declared_mime_type=_declared_mime(content, sample.image.image_id),
                content=content,
            )
            for position, (sample, content) in enumerate(
                zip(selected, contents, strict=True)
            )
        )
        request = ProfileAnalysisRequest(
            analysis_id=f"phase3-benchmark-{image_count}",
            image_ids=tuple(_benchmark_image_id(position) for position in range(image_count)),
        )

        latencies: list[float] = []
        signatures: list[tuple[object, ...]] = []
        peak_ram_mb = _peak_rss_mb()
        for _ in range(iterations):
            started = time.perf_counter()
            batch = await service.process_async(request, inputs)
            try:
                latencies.append((time.perf_counter() - started) * 1000.0)
                signatures.append(_stable_signature(batch))
                peak_ram_mb = max(peak_ram_mb, _peak_rss_mb())
            finally:
                await service.release_async(batch)

        decode_throughput = _measure_decode_throughput(contents)
        resize_cost_ms, resized_count = _measure_resize_cost(contents, max_dimension)
        results.append(
            BatchBenchmarkResult(
                image_count=image_count,
                iterations=iterations,
                source_sample_count=len(set(sample_indices)),
                replayed_sample_count=image_count - len(set(sample_indices)),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=_percentile(latencies, 0.95),
                decode_images_per_second=decode_throughput,
                isolated_resize_cost_ms=resize_cost_ms,
                resized_image_count=resized_count,
                process_peak_ram_mb=peak_ram_mb,
                deterministic_hashes=all(signature == signatures[0] for signature in signatures[1:]),
            )
        )

    return ImagePreprocessingBenchmarkReport(
        schema_version="1.1.0",
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_sample_count=len(manifest.samples),
        manifest_fingerprint=manifest.fingerprint(),
        dataset_content_fingerprint=content_fingerprint,
        pillow_version=pillow_version,
        batch_results=tuple(results),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Phase 3 image ingestion/preprocessing")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=(1, 5, 20, 50))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = _load_manifest(Path(args.manifest))
    report = asyncio.run(
        run_benchmark_async(
            manifest,
            Path(args.dataset_root),
            args.batch_sizes,
            args.iterations,
        )
    )
    Path(args.output).write_text(report.to_json() + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output} manifest_fingerprint={report.manifest_fingerprint} "
        f"dataset_content_fingerprint={report.dataset_content_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
