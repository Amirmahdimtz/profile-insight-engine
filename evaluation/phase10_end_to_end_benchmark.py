from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from evaluation.benchmark import load_manifest
from evaluation.common import DatasetSplit
from evaluation.metrics import unsupported_claim_rate
from src.core.profile_analysis.contracts import ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import RawImageInput
from src.core.services.profile_analysis.profile_analysis_service import ProfileAnalysisService
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve
from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import ProfileAnalysisRepository
from src.infrastructure.utils.config_reader import ConfigReader


REPORT_SCHEMA_VERSION = "phase10-end-to-end-benchmark-v1"
METRIC_SCOPE = "phase10_production_http_workflow_core_path_without_semantic_theme_injection"
_DEFAULT_WORKLOAD_SIZES = (1, 5, 20, 50, 100)


def _mime_from_path(path: Path) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    try:
        return mapping[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation image extension: {path.suffix or '<none>'}") from exc


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _peak_process_ram_mb() -> float | None:
    if sys.platform.startswith("win"):
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
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return None
            return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except (OSError, AttributeError, ValueError):
            return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return float(value) / divisor
    except (ImportError, OSError, ValueError):
        return None


def _result_signature(completed: Any) -> str:
    payload = {
        "result": json.loads(completed.result.to_json()),
        "insight_policy_version": completed.insight_policy_version,
        "summary": completed.summary,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _summary_is_structurally_factual(completed: Any) -> bool:
    labels = [item.label for item in completed.result.insights]
    expected = (
        "Supported insights: " + " | ".join(labels)
        if labels
        else "No supported insights met the configured evidence policy."
    )
    return completed.summary == expected


def _claims_from_completed(completed: Any) -> tuple[tuple[str, str], ...]:
    claims: list[tuple[str, str]] = []
    for image in completed.result.images:
        for evidence in image.evidence:
            claims.append((evidence.type.value, evidence.label))
    for insight in completed.result.insights:
        claims.append((insight.type.value, insight.label))
    return tuple(claims)


def _build_raw_images(samples: Sequence[Any], dataset_root: Path, workload_size: int) -> tuple[RawImageInput, ...]:
    if not samples:
        raise ValueError("evaluation manifest has no eval samples")
    raw_images: list[RawImageInput] = []
    for index in range(workload_size):
        sample = samples[index % len(samples)]
        path = (dataset_root / sample.image.relative_path).resolve()
        image_id = f"phase10-{workload_size:03d}-{index:03d}-{sample.image.image_id}"
        raw_images.append(
            RawImageInput(
                image_id=image_id,
                filename=path.name,
                declared_mime_type=_mime_from_path(path),
                content=path.read_bytes(),
            )
        )
    return tuple(raw_images)


async def _delete_if_present(repository: ProfileAnalysisRepository, analysis_id: str) -> None:
    await repository.delete_async(analysis_id)


async def _run_workload(
    service: ProfileAnalysisService,
    repository: ProfileAnalysisRepository,
    raw_images: tuple[RawImageInput, ...],
    workload_size: int,
    iterations: int,
) -> dict[str, Any]:
    analysis_id = f"phase10-e2e-{workload_size:03d}"
    request = ProfileAnalysisRequest(
        analysis_id=analysis_id,
        image_ids=tuple(image.image_id for image in raw_images),
    )
    latencies: list[float] = []
    signatures: list[str] = []
    unsupported_rates: list[float] = []
    summary_factuality: list[bool] = []
    failures: list[str] = []
    for _ in range(iterations):
        await _delete_if_present(repository, analysis_id)
        started = time.perf_counter()
        try:
            completed = await service.create_async(request, raw_images)
        except Exception as exc:
            latencies.append((time.perf_counter() - started) * 1000.0)
            failures.append(f"{type(exc).__name__}: {str(exc)}")
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)
        signatures.append(_result_signature(completed))
        unsupported_rates.append(unsupported_claim_rate(_claims_from_completed(completed)))
        summary_factuality.append(_summary_is_structurally_factual(completed))
    await _delete_if_present(repository, analysis_id)
    successful = len(signatures)
    return {
        "workload_size": workload_size,
        "iterations": iterations,
        "successful_iterations": successful,
        "failed_iterations": len(failures),
        "failure_reasons": sorted(set(failures)),
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "throughput_images_per_second": (
            (workload_size * successful) / (sum(latencies) / 1000.0)
            if successful and sum(latencies) > 0.0
            else None
        ),
        "unsupported_claim_rate_post_validation": (
            sum(unsupported_rates) / len(unsupported_rates) if unsupported_rates else None
        ),
        "summary_structural_factuality": (
            all(summary_factuality) if summary_factuality else None
        ),
        "deterministic_rerun": (
            len(set(signatures)) == 1 if len(signatures) >= 2 else None
        ),
        "latency_samples_ms": latencies,
    }


async def _run_concurrency_probe(
    service: ProfileAnalysisService,
    repository: ProfileAnalysisRepository,
    sample: Any,
    dataset_root: Path,
    concurrency: int,
) -> dict[str, Any]:
    path = (dataset_root / sample.image.relative_path).resolve()

    async def run_one(index: int) -> tuple[str, str]:
        analysis_id = f"phase10-concurrency-{index:03d}"
        image_id = f"phase10-concurrency-image-{index:03d}"
        await _delete_if_present(repository, analysis_id)
        request = ProfileAnalysisRequest(analysis_id=analysis_id, image_ids=(image_id,))
        image = RawImageInput(
            image_id=image_id,
            filename=path.name,
            declared_mime_type=_mime_from_path(path),
            content=path.read_bytes(),
        )
        completed = await service.create_async(request, (image,))
        return analysis_id, completed.result.analysis_id

    started = time.perf_counter()
    results = await asyncio.gather(*(run_one(index) for index in range(concurrency)), return_exceptions=True)
    wall_ms = (time.perf_counter() - started) * 1000.0
    failures = [f"{type(item).__name__}: {str(item)}" for item in results if isinstance(item, Exception)]
    pairs = [item for item in results if not isinstance(item, Exception)]
    isolation_ok = all(expected == actual for expected, actual in pairs) and len({actual for _, actual in pairs}) == len(pairs)
    for index in range(concurrency):
        await _delete_if_present(repository, f"phase10-concurrency-{index:03d}")
    return {
        "concurrency": concurrency,
        "successful_requests": len(pairs),
        "failed_requests": len(failures),
        "failure_reasons": sorted(set(failures)),
        "wall_ms": wall_ms,
        "request_isolation": isolation_ok if pairs else None,
    }


async def run_async(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    manifest_fingerprint = manifest.fingerprint()
    if args.expected_manifest_fingerprint and manifest_fingerprint != args.expected_manifest_fingerprint:
        raise ValueError("manifest fingerprint does not match expected Phase 2 dataset")
    content_fingerprint = manifest.validate_references(args.dataset_root)
    if (
        args.expected_dataset_content_fingerprint
        and content_fingerprint != args.expected_dataset_content_fingerprint
    ):
        raise ValueError("dataset content fingerprint does not match expected Phase 2 dataset")

    eval_samples = tuple(sample for sample in manifest.samples if sample.split is DatasetSplit.EVAL)
    if not eval_samples:
        raise ValueError("evaluation manifest has no eval samples")
    config = ConfigReader()
    max_images = config.get_positive_int("profile_analysis.max_images")
    bootstrap_di()
    service = resolve(ProfileAnalysisService)
    repository = resolve(ProfileAnalysisRepository)

    workloads: list[dict[str, Any]] = []
    for workload_size in args.workload_size:
        if workload_size > max_images:
            workloads.append(
                {
                    "workload_size": workload_size,
                    "iterations": args.iterations,
                    "status": "expected_rejection_by_config",
                    "configured_max_images": max_images,
                    "reason": "workload exceeds profile_analysis.max_images",
                }
            )
            continue
        raw_images = _build_raw_images(eval_samples, args.dataset_root, workload_size)
        report = await _run_workload(
            service,
            repository,
            raw_images,
            workload_size,
            args.iterations,
        )
        report["status"] = "measured"
        workloads.append(report)

    concurrency = await _run_concurrency_probe(
        service,
        repository,
        eval_samples[0],
        args.dataset_root,
        args.concurrency,
    )
    measured = [item for item in workloads if item.get("status") == "measured"]
    total_iterations = sum(int(item["iterations"]) for item in measured)
    failed_iterations = sum(int(item["failed_iterations"]) for item in measured)
    all_latencies = [
        float(latency)
        for item in measured
        for latency in item.get("latency_samples_ms", [])
    ]
    successful_images = sum(
        int(item.get("successful_iterations", 0)) * int(item["workload_size"])
        for item in measured
    )
    total_execution_latency_ms = sum(all_latencies)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metric_scope": METRIC_SCOPE,
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "manifest_fingerprint": manifest_fingerprint,
        "dataset_content_fingerprint": content_fingerprint,
        "workload_sizes": list(args.workload_size),
        "configured_max_images": max_images,
        "iterations": args.iterations,
        "workloads": workloads,
        "concurrency": concurrency,
        "performance": {
            "total_end_to_end_p50_latency_ms": statistics.median(all_latencies) if all_latencies else None,
            "total_end_to_end_p95_latency_ms": _percentile(all_latencies, 0.95),
            "throughput_images_per_second": (
                successful_images / (total_execution_latency_ms / 1000.0)
                if successful_images and total_execution_latency_ms > 0.0
                else None
            ),
            "model_provider_warmup_ms": None,
            "peak_process_ram_mb": _peak_process_ram_mb(),
            "peak_vram_mb": None,
            "failure_rate": (
                failed_iterations / total_iterations if total_iterations else None
            ),
        },
        "quality": {
            "end_to_end_accuracy": "unavailable",
            "insight_precision": "unavailable",
            "confidence_calibration": "unavailable",
            "unsupported_claim_rate_post_validation": (
                max(
                    (
                        item["unsupported_claim_rate_post_validation"]
                        for item in measured
                        if item.get("unsupported_claim_rate_post_validation") is not None
                    ),
                    default=None,
                )
            ),
            "summary_structural_factuality": (
                all(
                    item.get("summary_structural_factuality") is True
                    for item in measured
                    if item.get("successful_iterations", 0) > 0
                )
                if any(item.get("successful_iterations", 0) > 0 for item in measured)
                else None
            ),
        },
        "limitations": [
            "The production HTTP workflow intentionally does not inject SemanticThemeService because no production candidate-theme source exists in the Phase 9 contract.",
            "End-to-end accuracy and insight precision are unavailable without authorized end-to-end/insight-level ground truth.",
            "VRAM and provider-specific warmup are supplied by provider benchmark reports rather than inferred from this service-process runner.",
            "Workloads larger than configured max_images are expected input-limit rejections rather than inference workloads.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workload-size", type=int, action="append", default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--expected-manifest-fingerprint")
    parser.add_argument("--expected-dataset-content-fingerprint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.workload_size = tuple(args.workload_size or _DEFAULT_WORKLOAD_SIZES)
    if any(value <= 0 for value in args.workload_size):
        parser.error("workload sizes must be positive")
    if args.iterations <= 0 or args.concurrency <= 0:
        parser.error("iterations and concurrency must be positive")
    report = asyncio.run(run_async(args))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
