from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from src.core.profile_analysis.contracts import Evidence, EvidenceType
from src.core.profile_analysis.embedding_contracts import (
    SemanticThemeResult,
    ThemeSimilarityEvidence,
)
from src.core.profile_analysis.image_contracts import CanonicalImage, ProcessedImageBatch
from src.core.services.profile_analysis.evidence_aggregation_service import (
    EvidenceAggregationService,
)


REPORT_SCHEMA_VERSION = "phase7-aggregation-benchmark-v1"


@dataclass(frozen=True)
class AggregationScaleResult:
    image_count: int
    standard_evidence_count: int
    semantic_signal_count: int
    total_input_signal_count: int
    aggregate_count: int
    iterations: int
    p50_latency_ms: float
    p95_latency_ms: float
    traced_peak_memory_mb: float
    process_peak_ram_mb: float | None
    result_hash: str
    deterministic_rerun: bool
    reversed_input_deterministic: bool


@dataclass(frozen=True)
class AggregationBenchmarkReport:
    schema_version: str
    evidence_per_image: int
    semantic_labels: tuple[str, ...]
    complexity_time: str
    complexity_auxiliary_space: str
    results: tuple[AggregationScaleResult, ...]

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _process_peak_ram_mb() -> float | None:
    if sys.platform == "win32":
        try:
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
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return None
            return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except (OSError, AttributeError, ValueError):
            return None

    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0
    except (ImportError, OSError, ValueError):
        return None


def _parse_image_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "image counts must be comma-separated positive integers"
        ) from exc
    if not counts or any(count <= 0 for count in counts) or len(counts) != len(set(counts)):
        raise argparse.ArgumentTypeError("image counts must be unique positive integers")
    return counts


def _image(image_id: str, position: int) -> CanonicalImage:
    digest = hashlib.sha256(f"phase7-image-{position}".encode("ascii")).hexdigest()
    return CanonicalImage(
        image_id=image_id,
        content_hash=digest,
        storage_reference=f"benchmark://{image_id}",
        mime_type="image/png",
        width=224,
        height=224,
        size_bytes=1024,
        original_mime_type="image/png",
        original_width=224,
        original_height=224,
        original_size_bytes=1024,
        was_resized=False,
        exif_orientation_applied=False,
    )


def _workload(
    image_count: int,
    evidence_per_image: int,
    semantic_labels: tuple[str, ...],
) -> tuple[ProcessedImageBatch, tuple[Evidence, ...], SemanticThemeResult]:
    images = tuple(_image(f"img-{index:04d}", index) for index in range(image_count))
    evidence: list[Evidence] = []
    evidence_types = (
        EvidenceType.OBJECT,
        EvidenceType.SCENE,
        EvidenceType.ACTIVITY,
        EvidenceType.TOPIC,
    )
    for image_index, image in enumerate(images):
        for item_index in range(evidence_per_image):
            evidence_type = evidence_types[item_index % len(evidence_types)]
            concept_index = item_index % max(1, evidence_per_image // 4)
            label = f"{evidence_type.value}_concept_{concept_index:03d}"
            confidence = 0.5 + (((image_index + item_index) % 50) / 100.0)
            evidence.append(
                Evidence(
                    id=f"ev-{image_index:04d}-{item_index:04d}",
                    image_id=image.image_id,
                    type=evidence_type,
                    label=label,
                    value=label,
                    confidence=confidence,
                    source=f"benchmark_{evidence_type.value}",
                    metadata={"ordinal": item_index},
                )
            )

    semantic = tuple(
        ThemeSimilarityEvidence(
            image_id=image.image_id,
            theme_label=label,
            similarity=0.8,
            threshold=0.5,
        )
        for image in images
        for label in semantic_labels
    )
    semantic_result = SemanticThemeResult(
        theme_similarities=semantic,
        retrieval_results=(),
        near_duplicate_groups=(),
        provider="phase7-benchmark-semantic",
        provider_version="1",
        model_id="synthetic-contract-input",
        model_version="1",
        config_version="phase7-benchmark-v1",
        embedding_dimension=2,
    )
    batch = ProcessedImageBatch(
        analysis_id=f"phase7-benchmark-{image_count}",
        images=images,
        duplicates=(),
        storage_scope_reference=f"benchmark://scope/{image_count}",
    )
    return batch, tuple(evidence), semantic_result


def _result_hash(result) -> str:
    payload = [item.to_dict() for item in result]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_benchmark(
    *,
    image_counts: tuple[int, ...],
    evidence_per_image: int,
    iterations: int,
    semantic_labels: tuple[str, ...] = ("semantic_alpha", "semantic_beta"),
) -> AggregationBenchmarkReport:
    if not image_counts or any(value <= 0 for value in image_counts):
        raise ValueError("image_counts must contain positive integers")
    if evidence_per_image <= 0:
        raise ValueError("evidence_per_image must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not semantic_labels or any(not value or value != value.strip() for value in semantic_labels):
        raise ValueError("semantic_labels must contain non-empty trimmed strings")

    service = EvidenceAggregationService()
    scale_results: list[AggregationScaleResult] = []
    for image_count in image_counts:
        batch, evidence, semantic_result = _workload(
            image_count,
            evidence_per_image,
            semantic_labels,
        )
        latencies: list[float] = []
        hashes: list[str] = []
        traced_peaks: list[int] = []
        last_result = ()
        for _ in range(iterations):
            tracemalloc.start()
            started = time.perf_counter()
            last_result = service.aggregate(batch, evidence, semantic_result)
            latencies.append((time.perf_counter() - started) * 1000.0)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            traced_peaks.append(peak)
            hashes.append(_result_hash(last_result))

        reversed_result = service.aggregate(
            batch,
            tuple(reversed(evidence)),
            SemanticThemeResult(
                theme_similarities=tuple(reversed(semantic_result.theme_similarities)),
                retrieval_results=semantic_result.retrieval_results,
                near_duplicate_groups=semantic_result.near_duplicate_groups,
                provider=semantic_result.provider,
                provider_version=semantic_result.provider_version,
                model_id=semantic_result.model_id,
                model_version=semantic_result.model_version,
                config_version=semantic_result.config_version,
                embedding_dimension=semantic_result.embedding_dimension,
            ),
        )
        stable_hash = hashes[0]
        scale_results.append(
            AggregationScaleResult(
                image_count=image_count,
                standard_evidence_count=len(evidence),
                semantic_signal_count=len(semantic_result.theme_similarities),
                total_input_signal_count=len(evidence) + len(semantic_result.theme_similarities),
                aggregate_count=len(last_result),
                iterations=iterations,
                p50_latency_ms=_percentile(latencies, 0.50),
                p95_latency_ms=_percentile(latencies, 0.95),
                traced_peak_memory_mb=max(traced_peaks) / (1024.0 * 1024.0),
                process_peak_ram_mb=_process_peak_ram_mb(),
                result_hash=stable_hash,
                deterministic_rerun=len(set(hashes)) == 1,
                reversed_input_deterministic=_result_hash(reversed_result) == stable_hash,
            )
        )

    return AggregationBenchmarkReport(
        schema_version=REPORT_SCHEMA_VERSION,
        evidence_per_image=evidence_per_image,
        semantic_labels=semantic_labels,
        complexity_time="O(S log S + I)",
        complexity_auxiliary_space="O(S + I)",
        results=tuple(scale_results),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark deterministic Phase 7 aggregation")
    parser.add_argument(
        "--image-counts",
        type=_parse_image_counts,
        default=(1, 5, 20, 50),
        help="comma-separated image counts (default: 1,5,20,50)",
    )
    parser.add_argument("--evidence-per-image", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_benchmark(
        image_counts=args.image_counts,
        evidence_per_image=args.evidence_per_image,
        iterations=args.iterations,
    )
    payload = report.to_json()
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
