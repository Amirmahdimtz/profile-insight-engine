from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import httpx

from evaluation.benchmark import load_manifest
from evaluation.common import DatasetSplit
from evaluation.metrics import unsupported_claim_rate
from src.application.web import WebService
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve
from src.infrastructure.utils.config_reader import ConfigReader


REPORT_SCHEMA_VERSION = "phase10-benchmark-v1"
METRIC_SCOPE = "in_process_asgi_full_deployed_pipeline_real_providers_postgresql"
DEFAULT_WORKLOADS = (1, 5, 20, 50, 100)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.999999)),
    )
    return ordered[index]


def _peak_process_ram_mb() -> float | None:
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
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
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return None
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
        except (OSError, AttributeError, ValueError):
            return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return float(value) / divisor
    except (ImportError, OSError, ValueError):
        return None


def _total_ram_mb() -> float | None:
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullTotalPhys / (1024.0 * 1024.0)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024.0 * 1024.0)
    except (AttributeError, OSError, ValueError):
        return None


def _gpu_inventory() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    result: list[dict[str, object]] = []
    for row in completed.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        try:
            memory_mb: float | None = float(parts[2])
        except ValueError:
            memory_mb = None
        result.append(
            {"name": parts[0], "driver_version": parts[1], "memory_total_mb": memory_mb}
        )
    return result


def hardware_profile() -> dict[str, object]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "ram_total_mb": _total_ram_mb(),
        "nvidia_gpus": _gpu_inventory(),
    }


def _mime(path: Path) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    try:
        return mapping[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation image extension: {path.suffix}") from exc


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def oversized_dimension_png(max_pixels: int) -> bytes:
    width = max(2, int(max_pixels**0.5) + 1)
    height = width
    while width * height <= max_pixels:
        width += 1
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")


def _load_optional_report(path: Path | None, expected_schema: str | None) -> dict[str, object]:
    if path is None:
        return {"status": "unavailable"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"component report must be an object: {path}")
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected component report schema: {path}")
    return {"status": "available", "path": str(path), "report": dict(payload)}


def _quality_summary(component_evidence: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "end_to_end_accuracy": {
            "status": "unavailable",
            "reason": "the current authorized manifest has no profile-level expected final result contract",
        },
        "ocr_quality": component_evidence["ocr"],
        "vision_quality": component_evidence["vision"],
        "embedding_quality": component_evidence["embedding"],
        "insight_precision": {
            "status": "unavailable",
            "reason": "Phase 8 evidence is synthetic contract evidence, not product insight-level ground truth",
        },
        "confidence_calibration": {
            "status": "unavailable",
            "reason": "no authorized final-insight correctness labels are available",
        },
    }


def _files_for_workload(
    eval_samples: Sequence[Any], dataset_root: Path, count: int, analysis_id: str
) -> tuple[list[str], list[tuple[str, tuple[str, bytes, str]]]]:
    image_ids: list[str] = []
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for index in range(count):
        sample = eval_samples[index % len(eval_samples)]
        path = dataset_root / sample.image.relative_path
        image_id = f"{analysis_id}-img-{index + 1:03d}"
        image_ids.append(image_id)
        files.append(("images", (path.name, path.read_bytes(), _mime(path))))
    return image_ids, files


async def _post_analysis(
    client: httpx.AsyncClient,
    image_ids: Sequence[str],
    files: Sequence[tuple[str, tuple[str, bytes, str]]],
    analysis_id: str,
) -> tuple[httpx.Response, float]:
    started = time.perf_counter()
    response = await client.post(
        "/api/v1/profile_analysis/",
        data={"request_json": json.dumps({"analysis_id": analysis_id, "image_ids": list(image_ids)})},
        files=list(files),
    )
    return response, (time.perf_counter() - started) * 1000.0


def _claims_and_summary(response: httpx.Response) -> tuple[list[tuple[str, str]], bool | None]:
    if response.status_code != 201:
        return [], None
    payload = response.json()
    insights = payload.get("insights", [])
    if not isinstance(insights, list):
        return [], None
    claims = [
        (item["key"], item["label"])
        for item in insights
        if isinstance(item, Mapping) and isinstance(item.get("key"), str) and isinstance(item.get("label"), str)
    ]
    expected = (
        "No supported insights met the configured evidence policy."
        if not insights
        else "Supported insights: " + " | ".join(item["label"] for item in insights)
    )
    return claims, payload.get("summary") == expected


async def _input_scenarios(
    client: httpx.AsyncClient,
    sample_path: Path,
    max_size_bytes: int,
    max_pixels: int,
) -> list[dict[str, object]]:
    valid = sample_path.read_bytes()
    mime = _mime(sample_path)
    cases = [
        ("blank", b"", "image/png", 422),
        ("corrupt", b"not-an-image", "image/png", 422),
        ("huge_dimensions", oversized_dimension_png(max_pixels), "image/png", 422),
        ("oversized_upload", b"x" * (max_size_bytes + 1), "image/png", 413),
    ]
    results: list[dict[str, object]] = []
    for name, content, declared_mime, expected in cases:
        analysis_id = f"phase10-{name}-{uuid4().hex}"
        response, latency = await _post_analysis(
            client,
            (f"{analysis_id}-img-1",),
            (("images", (f"{name}.png", content, declared_mime)),),
            analysis_id,
        )
        results.append(
            {
                "scenario": name,
                "expected_status": expected,
                "actual_status": response.status_code,
                "passed": response.status_code == expected,
                "latency_ms": latency,
            }
        )

    mixed_id = f"phase10-mixed-{uuid4().hex}"
    mixed, mixed_latency = await _post_analysis(
        client,
        (f"{mixed_id}-1", f"{mixed_id}-2"),
        (
            ("images", (sample_path.name, valid, mime)),
            ("images", ("corrupt.png", b"invalid", "image/png")),
        ),
        mixed_id,
    )
    results.append(
        {
            "scenario": "mixed_valid_invalid",
            "expected_status": 422,
            "actual_status": mixed.status_code,
            "passed": mixed.status_code == 422,
            "latency_ms": mixed_latency,
        }
    )

    dup_id = f"phase10-duplicate-{uuid4().hex}"
    duplicate, dup_latency = await _post_analysis(
        client,
        (f"{dup_id}-1", f"{dup_id}-2"),
        (
            ("images", (sample_path.name, valid, mime)),
            ("images", (sample_path.name, valid, mime)),
        ),
        dup_id,
    )
    results.append(
        {
            "scenario": "duplicate_images",
            "expected_status": 201,
            "actual_status": duplicate.status_code,
            "passed": duplicate.status_code == 201,
            "latency_ms": dup_latency,
        }
    )
    return results


async def _concurrency(
    client: httpx.AsyncClient,
    eval_samples: Sequence[Any],
    dataset_root: Path,
    concurrency: int,
) -> dict[str, object]:
    async def one(index: int) -> tuple[int, str, float, bool]:
        analysis_id = f"phase10-concurrent-{index}-{uuid4().hex}"
        image_ids, files = _files_for_workload(eval_samples, dataset_root, 1, analysis_id)
        response, latency = await _post_analysis(client, image_ids, files, analysis_id)
        isolated = False
        if response.status_code == 201:
            payload = response.json()
            support_ids = {
                image_id
                for insight in payload.get("insights", [])
                for image_id in insight.get("supporting_image_ids", [])
            }
            isolated = payload.get("analysis_id") == analysis_id and support_ids.issubset(set(image_ids))
        return response.status_code, analysis_id, latency, isolated

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(index) for index in range(concurrency)))
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "request_count": concurrency,
        "success_count": sum(1 for status, _, _, _ in rows if status == 201),
        "failure_count": sum(1 for status, _, _, _ in rows if status != 201),
        "wall_ms": wall_ms,
        "p50_latency_ms": _percentile([row[2] for row in rows], 0.50),
        "p95_latency_ms": _percentile([row[2] for row in rows], 0.95),
        "request_isolation": all(row[3] for row in rows),
    }


async def run_benchmark_async(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("env_type", "").strip().lower() == "development":
        raise ValueError("Phase 10 final benchmark requires production appsettings")
    manifest = load_manifest(args.manifest)
    dataset_root = Path(args.dataset_root).resolve()
    manifest_fingerprint = manifest.fingerprint()
    content_fingerprint = manifest.validate_references(dataset_root)
    if args.expected_manifest_fingerprint and args.expected_manifest_fingerprint != manifest_fingerprint:
        raise ValueError("manifest fingerprint mismatch")
    if args.expected_dataset_content_fingerprint and args.expected_dataset_content_fingerprint != content_fingerprint:
        raise ValueError("dataset content fingerprint mismatch")
    eval_samples = tuple(sample for sample in manifest.samples if sample.split is DatasetSplit.EVAL)
    if not eval_samples:
        raise ValueError("Phase 10 benchmark requires eval samples")

    config = ConfigReader()
    max_images = config.get_positive_int("profile_analysis.max_images")
    max_size_bytes = config.get_positive_int("profile_analysis.max_image_size_mb") * 1024 * 1024
    max_pixels = config.get_positive_int("profile_analysis.max_image_pixels")
    component_evidence = {
        "ocr": _load_optional_report(args.ocr_report, None),
        "vision": _load_optional_report(args.vision_report, None),
        "embedding": _load_optional_report(args.embedding_report, "phase6-embedding-benchmark-v2"),
        "insight": _load_optional_report(args.insight_report, "phase8-insight-benchmark-v1"),
    }

    bootstrap_di()
    app = resolve(WebService).create_app()
    transport = httpx.ASGITransport(app=app)
    workload_results: list[dict[str, object]] = []
    final_claims: list[tuple[str, str]] = []
    factuality_values: list[bool] = []
    successful_latencies: list[float] = []
    successful_images = 0
    attempts = 0
    failures = 0

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://phase10.local") as client:
            warmup_id = f"phase10-warmup-{uuid4().hex}"
            warmup_ids, warmup_files = _files_for_workload(eval_samples, dataset_root, 1, warmup_id)
            warmup_response, warmup_latency = await _post_analysis(
                client, warmup_ids, warmup_files, warmup_id
            )
            warmup_success = warmup_response.status_code == 201

            previous_failed = False
            for count in DEFAULT_WORKLOADS:
                if count > max_images:
                    analysis_id = f"phase10-limit-{count}-{uuid4().hex}"
                    ids, files = _files_for_workload(eval_samples, dataset_root, count, analysis_id)
                    response, latency = await _post_analysis(client, ids, files, analysis_id)
                    workload_results.append(
                        {
                            "image_count": count,
                            "measured_full_pipeline": False,
                            "status": "expected_config_limit_rejection",
                            "expected_http_status": 422,
                            "actual_http_status": response.status_code,
                            "passed": response.status_code == 422,
                            "latency_ms": latency,
                            "source_sample_reuse": count > len(eval_samples),
                        }
                    )
                    continue
                if previous_failed and not args.continue_after_workload_failure:
                    workload_results.append(
                        {
                            "image_count": count,
                            "measured_full_pipeline": False,
                            "status": "blocked_by_prior_workload_failure",
                            "passed": False,
                            "source_sample_reuse": count > len(eval_samples),
                        }
                    )
                    continue

                latencies: list[float] = []
                statuses: list[int] = []
                for iteration in range(args.iterations):
                    analysis_id = f"phase10-{count}-{iteration}-{uuid4().hex}"
                    ids, files = _files_for_workload(eval_samples, dataset_root, count, analysis_id)
                    response, latency = await _post_analysis(client, ids, files, analysis_id)
                    attempts += 1
                    statuses.append(response.status_code)
                    if response.status_code == 201:
                        latencies.append(latency)
                        successful_latencies.append(latency)
                        successful_images += count
                        claims, factual = _claims_and_summary(response)
                        final_claims.extend(claims)
                        if factual is not None:
                            factuality_values.append(factual)
                    else:
                        failures += 1
                passed = all(status == 201 for status in statuses)
                previous_failed = previous_failed or not passed
                workload_results.append(
                    {
                        "image_count": count,
                        "measured_full_pipeline": True,
                        "status": "measured",
                        "iterations": args.iterations,
                        "http_statuses": statuses,
                        "passed": passed,
                        "p50_latency_ms": _percentile(latencies, 0.50),
                        "p95_latency_ms": _percentile(latencies, 0.95),
                        "throughput_images_per_second": (
                            count / ((_percentile(latencies, 0.50) or 0) / 1000.0)
                            if latencies and (_percentile(latencies, 0.50) or 0) > 0
                            else None
                        ),
                        "source_sample_reuse": count > len(eval_samples),
                    }
                )

            scenarios = await _input_scenarios(
                client,
                dataset_root / eval_samples[0].image.relative_path,
                max_size_bytes,
                max_pixels,
            )
            concurrency_result = await _concurrency(
                client, eval_samples, dataset_root, args.concurrency
            )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metric_scope": METRIC_SCOPE,
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "manifest_fingerprint": manifest_fingerprint,
            "dataset_content_fingerprint": content_fingerprint,
            "eval_sample_count": len(eval_samples),
        },
        "hardware": hardware_profile(),
        "configuration": {
            "max_images": max_images,
            "ocr_provider": config.get_non_empty_string("ocr.provider"),
            "ocr_model_id": config.get_non_empty_string("ocr.model_id"),
            "ocr_config_version": config.get_non_empty_string("ocr.config_version"),
            "vision_provider": config.get_non_empty_string("vision.provider"),
            "vision_model_id": config.get_non_empty_string("vision.model_id"),
            "vision_model_version": config.get_non_empty_string("vision.model_version"),
            "vision_config_version": config.get_non_empty_string("vision.config_version"),
            "embedding_provider": config.get_non_empty_string("embedding.provider"),
            "embedding_model_id": config.get_non_empty_string("embedding.model_id"),
            "embedding_model_revision": config.get_non_empty_string("embedding.model_revision"),
            "embedding_config_version": config.get_non_empty_string("embedding.config_version"),
            "insight_policy_version": config.get_non_empty_string("insights.policy_version"),
        },
        "pipeline_scope": {
            "included": [
                "http_dto_validation",
                "image_preprocessing",
                "ocr",
                "vision",
                "cross_image_aggregation",
                "insight_generation",
                "postgresql_persistence",
                "response_mapping",
            ],
            "not_in_current_http_workflow": ["embedding_semantic_theme"],
        },
        "warmup": {"latency_ms": warmup_latency, "successful": warmup_success},
        "workloads": workload_results,
        "input_scenarios": scenarios,
        "concurrency": concurrency_result,
        "performance": {
            "end_to_end_p50_latency_ms": _percentile(successful_latencies, 0.50),
            "end_to_end_p95_latency_ms": _percentile(successful_latencies, 0.95),
            "throughput_images_per_second_sequential": (
                successful_images / (sum(successful_latencies) / 1000.0)
                if successful_latencies and sum(successful_latencies) > 0
                else None
            ),
            "failure_rate": failures / attempts if attempts else None,
            "full_pipeline_attempts": attempts,
            "full_pipeline_failures": failures,
            "peak_python_process_ram_mb": _peak_process_ram_mb(),
            "vision_runtime_ram_mb": None,
            "vision_runtime_vram_mb": None,
            "vision_runtime_resource_scope": "use the Phase 5 candidate report for llama.cpp runtime RAM/VRAM",
            "per_stage_latency": {
                "status": "unavailable",
                "reason": "production services do not expose stage timing telemetry; phase-specific benchmarks remain the non-invasive stage evidence",
            },
        },
        "quality": {
            **_quality_summary(component_evidence),
            "final_pipeline_unsupported_claim_rate": {
                "status": "available" if final_claims else "unavailable",
                "value": unsupported_claim_rate(final_claims) if final_claims else None,
                "scope": "policy_validator_on_emitted_final_insight_labels",
            },
            "summary_factuality": {
                "status": "available" if factuality_values else "unavailable",
                "value": (
                    sum(1 for value in factuality_values if value) / len(factuality_values)
                    if factuality_values
                    else None
                ),
                "scope": "structural_exact_match_against_emitted_insight_labels",
            },
        },
        "component_evidence": component_evidence,
        "failure_scenario_evidence": {
            "provider_timeout_crash_malformed": "covered by existing provider/service regression tests; no synthetic production metric is fabricated",
            "gpu_unavailable": "vision runtime is allowed to run CPU/other backend; target hardware behavior requires local benchmark evidence",
            "database_runtime_failure": "sanitized by Phase 10 service hardening and covered by focused tests",
            "restart_recovery": "pending/in_progress rows are failed and stale temp scopes are cleaned during single-process startup",
        },
        "model_selection": {
            "status": "unresolved",
            "reason": "provider/model/config promotion requires returned target-hardware Phase 10 evidence",
        },
    }
    return report


def report_is_complete(report: Mapping[str, Any]) -> bool:
    warmup = report.get("warmup")
    workloads = report.get("workloads")
    scenarios = report.get("input_scenarios")
    concurrency = report.get("concurrency")
    if not isinstance(warmup, Mapping) or warmup.get("successful") is not True:
        return False
    if not isinstance(workloads, list):
        return False
    counts = {
        item.get("image_count")
        for item in workloads
        if isinstance(item, Mapping)
    }
    if counts != set(DEFAULT_WORKLOADS):
        return False
    max_images = report["configuration"]["max_images"]
    for item in workloads:
        if item["image_count"] <= max_images:
            if item.get("measured_full_pipeline") is not True or item.get("passed") is not True:
                return False
        elif item.get("status") != "expected_config_limit_rejection" or item.get("passed") is not True:
            return False
    if not isinstance(scenarios, list) or not scenarios or not all(
        isinstance(item, Mapping) and item.get("passed") is True for item in scenarios
    ):
        return False
    return (
        isinstance(concurrency, Mapping)
        and concurrency.get("failure_count") == 0
        and concurrency.get("request_isolation") is True
    )


def _parse_workloads(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workloads must be comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("workloads must contain unique positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 10 end-to-end benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workloads", type=_parse_workloads, default=DEFAULT_WORKLOADS)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--expected-manifest-fingerprint")
    parser.add_argument("--expected-dataset-content-fingerprint")
    parser.add_argument("--ocr-report", type=Path)
    parser.add_argument("--vision-report", type=Path)
    parser.add_argument("--embedding-report", type=Path)
    parser.add_argument("--insight-report", type=Path)
    parser.add_argument("--continue-after-workload-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.iterations <= 0 or args.concurrency <= 0:
        raise SystemExit("--iterations and --concurrency must be positive")
    if tuple(args.workloads) != DEFAULT_WORKLOADS:
        raise SystemExit("Phase 10 final benchmark workloads must be exactly 1,5,20,50,100")
    report = asyncio.run(run_benchmark_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report_is_complete(report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
