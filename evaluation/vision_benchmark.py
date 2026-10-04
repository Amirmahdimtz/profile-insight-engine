from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.benchmark import load_manifest
from evaluation.common import (
    DatasetSlice,
    DatasetSplit,
    EvaluationLabelType,
)
from evaluation.metrics import (
    precision_recall_f1,
    unsupported_claim_rate,
)
from src.core.profile_analysis.contracts import (
    EvidenceType,
    ProfileAnalysisRequest,
)
from src.core.profile_analysis.image_contracts import RawImageInput
from src.core.profile_analysis.vision_contracts import (
    VisionProviderError,
    VisionValidationError,
)
from src.core.services.profile_analysis.evidence_extraction_service import (
    EvidenceExtractionService,
    normalize_visual_text,
)
from src.core.services.profile_analysis.image_processing_service import (
    ImageProcessingService,
)
from src.infrastructure.providers.image.local_image_storage import (
    LocalImageStorage,
)
from src.infrastructure.providers.vision.llama_cpp_vision_provider import (
    LlamaCppVisionProvider,
    LlamaCppVisionSettings,
)
from src.infrastructure.utils.config_reader import ConfigReader


_VISUAL_LABEL_TYPES = (
    EvaluationLabelType.OBJECT,
    EvaluationLabelType.SCENE,
    EvaluationLabelType.ACTIVITY,
    EvaluationLabelType.TOPIC,
)


def _sanitized_failure_reason(exc: Exception) -> str:
    if isinstance(
        exc,
        (VisionProviderError, VisionValidationError),
    ):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


def _deterministic_rerun_status(
    iterations: int,
    deterministic: bool,
) -> bool | None:
    if iterations < 2:
        return None
    return deterministic


def _sample_iteration_count(
    *,
    sample_index: int,
    iterations: int,
    determinism_sample_count: int,
) -> int:
    if iterations < 2:
        return 1
    if sample_index < determinism_sample_count:
        return iterations
    return 1


def _report_exit_code(report: Mapping[str, Any]) -> int:
    candidates = report.get("candidates")
    if not isinstance(candidates, Sequence):
        return 2
    valid_count = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return 2
        if candidate.get("benchmark_valid") is True:
            valid_count += 1
            continue
        if candidate.get("candidate_status") != "rejected_preflight":
            return 2
    return 0 if valid_count > 0 else 2


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    hf_model: str


def _rejected_candidate_report(
    *,
    candidate: CandidateSpec,
    stage: str,
    reason: str,
    sample_count: int,
    request_timeout_seconds: int,
    max_tokens: int,
    max_observations_per_kind: int,
    max_label_chars: int,
    max_caption_chars: int,
    runtime_context_size: int,
    runtime_parallel: int,
    model_acquisition_time_ms: float | None = None,
    model_load_time_ms: float | None = None,
    runtime_peak_ram_mb: float | None = None,
    vram_peak_mb: float | None = None,
) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "candidate_status": "rejected_preflight",
        "rejection_stage": stage,
        "hf_model": candidate.hf_model,
        "provider": "llama_cpp",
        "provider_version": None,
        "model_path": None,
        "model_sha256": None,
        "request_timeout_seconds": request_timeout_seconds,
        "max_tokens": max_tokens,
        "max_observations_per_kind": max_observations_per_kind,
        "max_label_chars": max_label_chars,
        "max_caption_chars": max_caption_chars,
        "runtime_context_size": runtime_context_size,
        "runtime_parallel": runtime_parallel,
        "sample_count": sample_count,
        "evaluated_sample_count": 0,
        "failed_sample_count": 1,
        "benchmark_valid": False,
        "failure_reasons": {reason: 1},
        "preflight_sample_id": None,
        "preflight_latency_ms": None,
        "evaluated_label_types": [],
        "precision": None,
        "recall": None,
        "f1": None,
        "unsupported_claim_rate_pre_policy": None,
        "unsupported_claim_rate_post_validation": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "model_acquisition_time_ms": model_acquisition_time_ms,
        "model_acquisition_skipped": False,
        "existing_runtime_reused": False,
        "model_load_time_ms": model_load_time_ms,
        "runtime_peak_ram_mb": runtime_peak_ram_mb,
        "vram_peak_mb": vram_peak_mb,
        "determinism_sample_count": 0,
        "deterministic_rerun": None,
        "per_label_type": {},
    }


def _parse_candidate(value: str) -> CandidateSpec:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(
            "candidate must be NAME=HF_MODEL_SPEC"
        )
    name, separator, hf_model = value.partition("=")
    name = name.strip()
    hf_model = hf_model.strip()
    if not separator or not name or not hf_model:
        raise argparse.ArgumentTypeError(
            "candidate must be NAME=HF_MODEL_SPEC"
        )
    return CandidateSpec(
        name=name,
        hf_model=hf_model,
    )


def audit_visual_label_coverage(
    manifest: Any,
) -> dict[str, Any]:
    eval_samples = tuple(
        sample
        for sample in manifest.samples
        if sample.split is DatasetSplit.EVAL
    )
    by_type = {
        label_type.value: sum(
            1
            for sample in eval_samples
            for label in sample.ground_truth.labels
            if label.type is label_type
        )
        for label_type in _VISUAL_LABEL_TYPES
    }
    by_slice = {
        item.value: {
            label_type.value: sum(
                1
                for sample in eval_samples
                if item in sample.slices
                for label in sample.ground_truth.labels
                if label.type is label_type
            )
            for label_type in _VISUAL_LABEL_TYPES
        }
        for item in DatasetSlice
    }
    missing = [
        key
        for key, count in by_type.items()
        if count == 0
    ]
    return {
        "eval_sample_count": len(eval_samples),
        "label_count_by_type": by_type,
        "label_count_by_slice": by_slice,
        "missing_required_label_types": missing,
    }


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
        raise ValueError(
            "unsupported Phase 3 evaluation image extension: "
            f"{path.suffix or '<none>'}"
        ) from exc


def _percentile(
    values: Sequence[float],
    quantile: float,
) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(
                (len(ordered) - 1)
                * quantile
                + 0.999999
            ),
        ),
    )
    return ordered[index]


def _free_port() -> int:
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _runtime_command(
    runtime_executable: str,
    candidate: CandidateSpec,
    port: int,
    *,
    offline: bool,
    context_size: int,
    parallel: int,
) -> list[str]:
    command = [
        runtime_executable,
        "-hf",
        candidate.hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--parallel",
        str(parallel),
        "--log-disable",
    ]
    if offline:
        command.append("--offline")
    return command


def _start_runtime(
    command: Sequence[str],
) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "llama.cpp runtime executable was not found"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "llama.cpp runtime could not be started"
        ) from exc


def _stop_runtime(
    process: subprocess.Popen[bytes],
) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_health(
    base_url: str,
    timeout_seconds: int,
    process: subprocess.Popen[bytes],
) -> float:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "llama.cpp runtime exited before becoming healthy"
            )
        request = urllib.request.Request(
            f"{base_url}/health",
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=2,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
                if (
                    response.status == 200
                    and isinstance(payload, Mapping)
                    and payload.get("status") == "ok"
                ):
                    return (
                        time.perf_counter() - started
                    ) * 1000.0
        except (
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            UnicodeError,
        ):
            pass
        time.sleep(0.25)
    raise RuntimeError(
        "llama.cpp runtime did not become healthy "
        "before startup timeout"
    )


def _runtime_props(
    base_url: str,
    timeout_seconds: int,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/props",
        method="GET",
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "llama.cpp /props response is invalid"
        )
    return payload


def _nvidia_vram_mb(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None
    if completed.returncode != 0:
        return None
    total = 0.0
    found = False
    for line in completed.stdout.splitlines():
        parts = [
            part.strip()
            for part in line.split(",")
        ]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            used = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += used
            found = True
    return total if found else None


def _process_rss_mb(pid: int) -> float | None:
    if pid <= 0:
        return None
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            process_vm_read = 0x0010

            class ProcessMemoryCounters(
                ctypes.Structure
            ):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    (
                        "PeakWorkingSetSize",
                        ctypes.c_size_t,
                    ),
                    (
                        "WorkingSetSize",
                        ctypes.c_size_t,
                    ),
                    (
                        "QuotaPeakPagedPoolUsage",
                        ctypes.c_size_t,
                    ),
                    (
                        "QuotaPagedPoolUsage",
                        ctypes.c_size_t,
                    ),
                    (
                        "QuotaPeakNonPagedPoolUsage",
                        ctypes.c_size_t,
                    ),
                    (
                        "QuotaNonPagedPoolUsage",
                        ctypes.c_size_t,
                    ),
                    (
                        "PagefileUsage",
                        ctypes.c_size_t,
                    ),
                    (
                        "PeakPagefileUsage",
                        ctypes.c_size_t,
                    ),
                ]

            kernel32 = ctypes.WinDLL(
                "kernel32",
                use_last_error=True,
            )
            psapi = ctypes.WinDLL(
                "psapi",
                use_last_error=True,
            )
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = (
                wintypes.HANDLE
            )
            kernel32.CloseHandle.argtypes = [
                wintypes.HANDLE
            ]
            kernel32.CloseHandle.restype = (
                wintypes.BOOL
            )
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(
                    ProcessMemoryCounters
                ),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = (
                wintypes.BOOL
            )
            handle = kernel32.OpenProcess(
                process_query_limited_information
                | process_vm_read,
                False,
                pid,
            )
            if not handle:
                return None
            try:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(
                    counters
                )
                if not psapi.GetProcessMemoryInfo(
                    handle,
                    ctypes.byref(counters),
                    counters.cb,
                ):
                    return None
                return (
                    float(counters.WorkingSetSize)
                    / (1024.0 * 1024.0)
                )
            finally:
                kernel32.CloseHandle(handle)
        except (
            OSError,
            AttributeError,
            ValueError,
        ):
            return None
    if sys.platform.startswith("linux"):
        try:
            lines = Path(
                f"/proc/{pid}/status"
            ).read_text(
                encoding="utf-8"
            ).splitlines()
            for line in lines:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
        except (
            OSError,
            ValueError,
            IndexError,
        ):
            return None
    return None


def _file_sha256_if_present(
    value: object,
) -> str | None:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        return None
    path = Path(value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


class _RuntimeResourceMonitor:
    def __init__(
        self,
        pid: int,
        interval_seconds: float = 0.25,
    ):
        self._pid = pid
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.peak_ram_mb: float | None = None
        self.peak_vram_mb: float | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(
            timeout=max(
                2.0,
                self._interval_seconds * 4,
            )
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            ram = _process_rss_mb(self._pid)
            if ram is not None:
                self.peak_ram_mb = (
                    ram
                    if self.peak_ram_mb is None
                    else max(
                        self.peak_ram_mb,
                        ram,
                    )
                )
            vram = _nvidia_vram_mb(self._pid)
            if vram is not None:
                self.peak_vram_mb = (
                    vram
                    if self.peak_vram_mb is None
                    else max(
                        self.peak_vram_mb,
                        vram,
                    )
                )
            self._stop.wait(
                self._interval_seconds
            )


def _provider_claims(
    provider_result: object,
) -> tuple[tuple[str, str], ...]:
    observations = getattr(
        provider_result,
        "observations",
        (),
    )
    claims = [
        (item.kind.value, item.label)
        for item in observations
    ]
    caption = getattr(
        provider_result,
        "caption",
        None,
    )
    if caption is not None:
        claims.append(("caption", caption.text))
    return tuple(claims)


def _benchmark_label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            "visual benchmark labels must be non-empty strings"
        )
    normalized = normalize_visual_text(value)
    if not normalized:
        raise ValueError(
            "visual benchmark labels must remain non-empty "
            "after normalization"
        )
    return normalized.casefold()


async def run_vision_benchmark_async(
    manifest: Any,
    dataset_root: Path,
    candidates: tuple[CandidateSpec, ...],
    iterations: int,
    expected_manifest_fingerprint: str | None,
    expected_dataset_content_fingerprint: str | None,
    runtime_executable: str | None,
    startup_timeout_seconds: int | None,
    model_acquisition_timeout_seconds: int | None = None,
    request_timeout_seconds: int | None = None,
    max_tokens: int | None = None,
    max_observations_per_kind: int | None = None,
    max_label_chars: int | None = None,
    max_caption_chars: int | None = None,
    determinism_sample_count: int = 3,
    skip_model_acquisition: bool = False,
    existing_runtime_base_url: str | None = None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError(
            "iterations must be positive"
        )
    if (
        isinstance(determinism_sample_count, bool)
        or not isinstance(determinism_sample_count, int)
        or determinism_sample_count <= 0
    ):
        raise ValueError(
            "determinism_sample_count must be a positive integer"
        )
    if (
        not candidates
        or len(
            {item.name for item in candidates}
        )
        != len(candidates)
    ):
        raise ValueError(
            "candidates must contain unique names"
        )
    if runtime_executable is not None and (
        not isinstance(runtime_executable, str)
        or not runtime_executable.strip()
    ):
        raise ValueError(
            "runtime_executable must be a non-empty "
            "string when provided"
        )
    if startup_timeout_seconds is not None and (
        isinstance(
            startup_timeout_seconds,
            bool,
        )
        or not isinstance(
            startup_timeout_seconds,
            int,
        )
        or startup_timeout_seconds <= 0
    ):
        raise ValueError(
            "startup_timeout_seconds must be a "
            "positive integer when provided"
        )
    if not isinstance(skip_model_acquisition, bool):
        raise ValueError("skip_model_acquisition must be a boolean")
    if existing_runtime_base_url is not None:
        if (
            not isinstance(existing_runtime_base_url, str)
            or not existing_runtime_base_url.strip()
        ):
            raise ValueError(
                "existing_runtime_base_url must be a non-empty "
                "string when provided"
            )
        existing_runtime_base_url = (
            existing_runtime_base_url.strip().rstrip("/")
        )
        if not (
            existing_runtime_base_url.startswith(
                "http://127.0.0.1:"
            )
            or existing_runtime_base_url.startswith(
                "http://localhost:"
            )
        ):
            raise ValueError(
                "existing_runtime_base_url must reference a "
                "local HTTP llama.cpp runtime"
            )
        if len(candidates) != 1:
            raise ValueError(
                "existing_runtime_base_url requires exactly "
                "one candidate"
            )
    if model_acquisition_timeout_seconds is not None and (
        isinstance(
            model_acquisition_timeout_seconds,
            bool,
        )
        or not isinstance(
            model_acquisition_timeout_seconds,
            int,
        )
        or model_acquisition_timeout_seconds <= 0
    ):
        raise ValueError(
            "model_acquisition_timeout_seconds must be a "
            "positive integer when provided"
        )
    for field_name, value in (
        ("request_timeout_seconds", request_timeout_seconds),
        ("max_tokens", max_tokens),
        ("max_observations_per_kind", max_observations_per_kind),
        ("max_label_chars", max_label_chars),
        ("max_caption_chars", max_caption_chars),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be a positive integer "
                "when provided"
            )

    manifest_fingerprint = manifest.fingerprint()
    if (
        expected_manifest_fingerprint is not None
        and manifest_fingerprint
        != expected_manifest_fingerprint
    ):
        raise ValueError(
            "manifest fingerprint does not match "
            "the expected Phase 2 dataset"
        )
    dataset_content_fingerprint = (
        manifest.validate_references(
            dataset_root
        )
    )
    if (
        expected_dataset_content_fingerprint
        is not None
        and dataset_content_fingerprint
        != expected_dataset_content_fingerprint
    ):
        raise ValueError(
            "dataset content fingerprint does not "
            "match the expected Phase 2 dataset"
        )

    coverage = audit_visual_label_coverage(
        manifest
    )
    eval_samples = tuple(
        sample
        for sample in manifest.samples
        if sample.split is DatasetSplit.EVAL
    )
    config = ConfigReader()
    runtime_executable = (
        runtime_executable
        or config.get_non_empty_string(
            "vision.runtime_executable"
        )
    )
    startup_timeout_seconds = (
        startup_timeout_seconds
        or config.get_positive_int(
            "vision.startup_timeout_seconds"
        )
    )
    model_acquisition_timeout_seconds = (
        model_acquisition_timeout_seconds
        or config.get_positive_int(
            "vision.model_acquisition_timeout_seconds"
        )
    )
    runtime_context_size = config.get_positive_int(
        "vision.runtime_context_size"
    )
    runtime_parallel = config.get_positive_int(
        "vision.runtime_parallel"
    )
    image_service = ImageProcessingService(
        config,
        LocalImageStorage(),
    )
    base_settings = (
        LlamaCppVisionSettings.from_config(config)
    )
    candidate_reports: list[dict[str, Any]] = []
    effective_determinism_sample_count = (
        min(determinism_sample_count, len(eval_samples))
        if iterations >= 2
        else 0
    )

    for candidate in candidates:
        model_acquisition_time_ms: float | None = None
        using_existing_runtime = (
            existing_runtime_base_url is not None
        )
        effective_request_timeout_seconds = (
            request_timeout_seconds
            or base_settings.request_timeout_seconds
        )
        effective_max_tokens = (
            max_tokens
            or base_settings.max_tokens
        )
        effective_max_observations_per_kind = (
            max_observations_per_kind
            or base_settings.max_observations_per_kind
        )
        effective_max_label_chars = (
            max_label_chars
            or base_settings.max_label_chars
        )
        effective_max_caption_chars = (
            max_caption_chars
            or base_settings.max_caption_chars
        )
        if not skip_model_acquisition and not using_existing_runtime:
            acquisition_port = _free_port()
            acquisition_base_url = (
                f"http://127.0.0.1:{acquisition_port}"
            )
            acquisition_process = _start_runtime(
                _runtime_command(
                    runtime_executable,
                    candidate,
                    acquisition_port,
                    offline=False,
                    context_size=runtime_context_size,
                    parallel=runtime_parallel,
                )
            )
            acquisition_started = time.perf_counter()
            try:
                try:
                    model_acquisition_time_ms = (
                        await asyncio.to_thread(
                            _wait_for_health,
                            acquisition_base_url,
                            model_acquisition_timeout_seconds,
                            acquisition_process,
                        )
                    )
                except RuntimeError:
                    model_acquisition_time_ms = (
                        time.perf_counter()
                        - acquisition_started
                    ) * 1000.0
                    candidate_reports.append(
                        _rejected_candidate_report(
                            candidate=candidate,
                            stage="model_acquisition",
                            reason=(
                                "RuntimeError: model acquisition failed"
                            ),
                            sample_count=len(eval_samples),
                            request_timeout_seconds=(
                                effective_request_timeout_seconds
                            ),
                            max_tokens=effective_max_tokens,
                            max_observations_per_kind=(
                                effective_max_observations_per_kind
                            ),
                            max_label_chars=(
                                effective_max_label_chars
                            ),
                            max_caption_chars=(
                                effective_max_caption_chars
                            ),
                            runtime_context_size=(
                                runtime_context_size
                            ),
                            runtime_parallel=runtime_parallel,
                            model_acquisition_time_ms=(
                                model_acquisition_time_ms
                            ),
                        )
                    )
                    continue
            finally:
                _stop_runtime(acquisition_process)

        process: subprocess.Popen[bytes] | None = None
        resource_monitor: _RuntimeResourceMonitor | None = None
        model_load_time_ms: float | None = None
        if using_existing_runtime:
            base_url = existing_runtime_base_url
        else:
            port = _free_port()
            base_url = f"http://127.0.0.1:{port}"
            process = _start_runtime(
                _runtime_command(
                    runtime_executable,
                    candidate,
                    port,
                    offline=True,
                    context_size=runtime_context_size,
                    parallel=runtime_parallel,
                )
            )
            resource_monitor = _RuntimeResourceMonitor(
                process.pid
            )
            resource_monitor.start()
            load_started = time.perf_counter()
            try:
                model_load_time_ms = await asyncio.to_thread(
                    _wait_for_health,
                    base_url,
                    startup_timeout_seconds,
                    process,
                )
            except RuntimeError:
                model_load_time_ms = (
                    time.perf_counter()
                    - load_started
                ) * 1000.0
                resource_monitor.stop()
                _stop_runtime(process)
                candidate_reports.append(
                    _rejected_candidate_report(
                        candidate=candidate,
                        stage="offline_model_load",
                        reason=(
                            "RuntimeError: offline model load failed"
                        ),
                        sample_count=len(eval_samples),
                        request_timeout_seconds=(
                            effective_request_timeout_seconds
                        ),
                        max_tokens=effective_max_tokens,
                        max_observations_per_kind=(
                            effective_max_observations_per_kind
                        ),
                        max_label_chars=(
                            effective_max_label_chars
                        ),
                        max_caption_chars=(
                            effective_max_caption_chars
                        ),
                        runtime_context_size=runtime_context_size,
                        runtime_parallel=runtime_parallel,
                        model_acquisition_time_ms=(
                            model_acquisition_time_ms
                        ),
                        model_load_time_ms=(
                            model_load_time_ms
                        ),
                        runtime_peak_ram_mb=(
                            resource_monitor.peak_ram_mb
                        ),
                        vram_peak_mb=(
                            resource_monitor.peak_vram_mb
                        ),
                    )
                )
                continue
        try:
            props = await asyncio.to_thread(
                _runtime_props,
                base_url,
                request_timeout_seconds
                or base_settings.request_timeout_seconds,
            )
            settings = LlamaCppVisionSettings(
                base_url=base_url,
                request_timeout_seconds=(
                    effective_request_timeout_seconds
                ),
                model_id=candidate.hf_model,
                model_version=candidate.name,
                config_version=(
                    base_settings.config_version
                ),
                max_tokens=effective_max_tokens,
                max_observations_per_kind=(
                    effective_max_observations_per_kind
                ),
                max_label_chars=effective_max_label_chars,
                max_caption_chars=effective_max_caption_chars,
                temperature=base_settings.temperature,
                top_p=base_settings.top_p,
                seed=base_settings.seed,
                confidence_semantics=(
                    base_settings.confidence_semantics
                ),
            )
            provider = LlamaCppVisionProvider(
                config,
                settings,
            )
            service = EvidenceExtractionService(
                provider
            )
            preflight_sample_id: str | None = None
            preflight_latency_ms: float | None = None
            run_preflight = (
                existing_runtime_base_url is None
                and len(candidates) > 1
                and bool(eval_samples)
            )
            if run_preflight:
                preflight_sample = eval_samples[0]
                preflight_sample_id = preflight_sample.sample_id
                preflight_source_path = (
                    dataset_root
                    / preflight_sample.image.relative_path
                )
                preflight_raw = RawImageInput(
                    image_id=preflight_sample.image.image_id,
                    filename=preflight_source_path.name,
                    declared_mime_type=(
                        _mime_from_path(preflight_source_path)
                    ),
                    content=preflight_source_path.read_bytes(),
                )
                preflight_request = ProfileAnalysisRequest(
                    analysis_id=(
                        f"phase5-{candidate.name}-preflight-"
                        f"{preflight_sample.sample_id}"
                    ),
                    image_ids=(
                        preflight_sample.image.image_id,
                    ),
                )
                preflight_batch = await image_service.process_async(
                    preflight_request,
                    (preflight_raw,),
                )
                preflight_started = time.perf_counter()
                try:
                    try:
                        preflight_provider_result = (
                            await provider.extract_async(
                                preflight_batch.images[0]
                            )
                        )
                        service.normalize_result(
                            preflight_batch.images[0],
                            preflight_provider_result,
                        )
                    except Exception as exc:
                        preflight_latency_ms = (
                            time.perf_counter()
                            - preflight_started
                        ) * 1000.0
                        reason = _sanitized_failure_reason(exc)
                        model_sha256 = await asyncio.to_thread(
                            _file_sha256_if_present,
                            props.get("model_path"),
                        )
                        candidate_reports.append(
                            {
                                "candidate": candidate.name,
                                "candidate_status": "rejected_preflight",
                                "hf_model": candidate.hf_model,
                                "provider": "llama_cpp",
                                "provider_version": props.get(
                                    "build_info"
                                ),
                                "model_path": props.get("model_path"),
                                "model_sha256": model_sha256,
                                "request_timeout_seconds": (
                                    settings.request_timeout_seconds
                                ),
                                "max_tokens": settings.max_tokens,
                                "max_observations_per_kind": (
                                    settings.max_observations_per_kind
                                ),
                                "max_label_chars": (
                                    settings.max_label_chars
                                ),
                                "max_caption_chars": (
                                    settings.max_caption_chars
                                ),
                                "sample_count": len(eval_samples),
                                "evaluated_sample_count": 0,
                                "failed_sample_count": 1,
                                "benchmark_valid": False,
                                "failure_reasons": {reason: 1},
                                "preflight_sample_id": (
                                    preflight_sample_id
                                ),
                                "preflight_latency_ms": (
                                    preflight_latency_ms
                                ),
                                "evaluated_label_types": [],
                                "precision": None,
                                "recall": None,
                                "f1": None,
                                "unsupported_claim_rate_pre_policy": (
                                    None
                                ),
                                "unsupported_claim_rate_post_validation": (
                                    None
                                ),
                                "p50_latency_ms": None,
                                "p95_latency_ms": None,
                                "model_acquisition_time_ms": (
                                    model_acquisition_time_ms
                                ),
                                "model_acquisition_skipped": (
                                    skip_model_acquisition
                                    or using_existing_runtime
                                ),
                                "existing_runtime_reused": (
                                    using_existing_runtime
                                ),
                                "model_load_time_ms": (
                                    model_load_time_ms
                                ),
                                "runtime_peak_ram_mb": (
                                    None
                                    if resource_monitor is None
                                    else resource_monitor.peak_ram_mb
                                ),
                                "vram_peak_mb": (
                                    None
                                    if resource_monitor is None
                                    else resource_monitor.peak_vram_mb
                                ),
                                "determinism_sample_count": 0,
                                "deterministic_rerun": None,
                                "per_label_type": {},
                            }
                        )
                        continue
                    preflight_latency_ms = (
                        time.perf_counter()
                        - preflight_started
                    ) * 1000.0
                finally:
                    await image_service.release_async(
                        preflight_batch
                    )

            latencies_ms: list[float] = []
            expected_by_type: dict[
                str,
                set[tuple[str, str]],
            ] = {
                item.value: set()
                for item in _VISUAL_LABEL_TYPES
            }
            predicted_by_type: dict[
                str,
                set[tuple[str, str]],
            ] = {
                item.value: set()
                for item in _VISUAL_LABEL_TYPES
            }
            pre_policy_claims: list[
                tuple[str, str]
            ] = []
            post_validation_claims: list[
                tuple[str, str]
            ] = []
            deterministic = True
            failed_samples = 0
            failure_reasons: dict[str, int] = {}

            for sample_index, sample in enumerate(eval_samples):
                source_path = (
                    dataset_root
                    / sample.image.relative_path
                )
                raw = RawImageInput(
                    image_id=sample.image.image_id,
                    filename=source_path.name,
                    declared_mime_type=(
                        _mime_from_path(source_path)
                    ),
                    content=source_path.read_bytes(),
                )
                request = ProfileAnalysisRequest(
                    analysis_id=(
                        f"phase5-{candidate.name}-"
                        f"{sample.sample_id}"
                    ),
                    image_ids=(
                        sample.image.image_id,
                    ),
                )
                for label in (
                    sample.ground_truth.labels
                ):
                    if (
                        label.type
                        in _VISUAL_LABEL_TYPES
                    ):
                        expected_value = (
                            label.value.strip()
                            if isinstance(
                                label.value,
                                str,
                            )
                            and label.value.strip()
                            else label.label
                        )
                        expected_by_type[
                            label.type.value
                        ].add(
                            (
                                sample.sample_id,
                                _benchmark_label(
                                    expected_value
                                ),
                            )
                        )

                batch = await image_service.process_async(
                    request,
                    (raw,),
                )
                try:
                    signatures: list[
                        tuple[
                            tuple[str, str, float],
                            ...,
                        ]
                    ] = []
                    first_result = None
                    try:
                        sample_iterations = _sample_iteration_count(
                            sample_index=sample_index,
                            iterations=iterations,
                            determinism_sample_count=(
                                effective_determinism_sample_count
                            ),
                        )
                        for iteration_index in range(
                            sample_iterations
                        ):
                            started = (
                                time.perf_counter()
                            )
                            provider_result = (
                                await provider.extract_async(
                                    batch.images[0]
                                )
                            )
                            if iteration_index == 0:
                                pre_policy_claims.extend(
                                    _provider_claims(
                                        provider_result
                                    )
                                )
                            result = service.normalize_result(
                                batch.images[0],
                                provider_result,
                            )
                            if iteration_index == 0:
                                latencies_ms.append(
                                    (
                                        time.perf_counter()
                                        - started
                                    )
                                    * 1000.0
                                )
                            signature = tuple(
                                (
                                    item.type.value,
                                    str(item.value),
                                    item.confidence,
                                )
                                for item in result.evidence
                            )
                            signatures.append(
                                signature
                            )
                            first_result = (
                                first_result
                                or result
                            )
                        if any(
                            signature
                            != signatures[0]
                            for signature
                            in signatures[1:]
                        ):
                            deterministic = False
                    except Exception as exc:
                        failed_samples += 1
                        reason = _sanitized_failure_reason(exc)
                        failure_reasons[reason] = (
                            failure_reasons.get(reason, 0) + 1
                        )
                        continue

                    if first_result is not None:
                        for evidence in (
                            first_result.evidence
                        ):
                            if (
                                evidence.type.value
                                in predicted_by_type
                            ):
                                predicted_by_type[
                                    evidence.type.value
                                ].add(
                                    (
                                        sample.sample_id,
                                        _benchmark_label(
                                            str(
                                                evidence.value
                                            )
                                        ),
                                    )
                                )
                                post_validation_claims.append(
                                    (
                                        evidence.type.value,
                                        str(
                                            evidence.value
                                        ),
                                    )
                                )
                            elif (
                                evidence.type
                                is EvidenceType.OTHER_OBSERVABLE
                            ):
                                post_validation_claims.append(
                                    (
                                        "caption",
                                        str(
                                            evidence.value
                                        ),
                                    )
                                )
                finally:
                    await image_service.release_async(
                        batch
                    )

            per_label_type: dict[
                str,
                Any,
            ] = {}
            aggregate_expected: set[
                tuple[str, str, str]
            ] = set()
            aggregate_predicted: set[
                tuple[str, str, str]
            ] = set()
            evaluated_label_types: list[str] = []
            for label_type in _VISUAL_LABEL_TYPES:
                expected = expected_by_type[
                    label_type.value
                ]
                predicted = predicted_by_type[
                    label_type.value
                ]
                if not expected:
                    per_label_type[
                        label_type.value
                    ] = {
                        "evaluated": False,
                        "reason": (
                            "no ground-truth labels "
                            "for this visual type"
                        ),
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "expected_count": 0,
                        "predicted_count": len(
                            predicted
                        ),
                    }
                    continue
                metric = precision_recall_f1(
                    expected,
                    predicted,
                )
                evaluated_label_types.append(
                    label_type.value
                )
                per_label_type[
                    label_type.value
                ] = {
                    "evaluated": True,
                    "reason": None,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1": metric.f1,
                    "expected_count": len(
                        expected
                    ),
                    "predicted_count": len(
                        predicted
                    ),
                }
                aggregate_expected.update(
                    (
                        label_type.value,
                        *item,
                    )
                    for item in expected
                )
                aggregate_predicted.update(
                    (
                        label_type.value,
                        *item,
                    )
                    for item in predicted
                )
            overall = (
                precision_recall_f1(
                    aggregate_expected,
                    aggregate_predicted,
                )
                if aggregate_expected
                else None
            )
            model_sha256 = (
                await asyncio.to_thread(
                    _file_sha256_if_present,
                    props.get("model_path"),
                )
            )

            candidate_reports.append(
                {
                    "candidate": candidate.name,
                    "candidate_status": "evaluated",
                    "hf_model": candidate.hf_model,
                    "provider": "llama_cpp",
                    "provider_version": (
                        props.get("build_info")
                    ),
                    "model_path": (
                        props.get("model_path")
                    ),
                    "model_sha256": model_sha256,
                    "request_timeout_seconds": (
                        settings.request_timeout_seconds
                    ),
                    "max_tokens": settings.max_tokens,
                    "max_observations_per_kind": (
                        settings.max_observations_per_kind
                    ),
                    "max_label_chars": settings.max_label_chars,
                    "max_caption_chars": settings.max_caption_chars,
                    "runtime_context_size": runtime_context_size,
                    "runtime_parallel": runtime_parallel,
                    "sample_count": len(
                        eval_samples
                    ),
                    "evaluated_sample_count": len(
                        eval_samples
                    ),
                    "preflight_sample_id": (
                        preflight_sample_id
                    ),
                    "preflight_latency_ms": (
                        preflight_latency_ms
                    ),
                    "failed_sample_count": (
                        failed_samples
                    ),
                    "benchmark_valid": (
                        failed_samples < len(eval_samples)
                    ),
                    "failure_reasons": dict(
                        sorted(failure_reasons.items())
                    ),
                    "evaluated_label_types": (
                        evaluated_label_types
                    ),
                    "precision": (
                        None
                        if overall is None
                        else overall.precision
                    ),
                    "recall": (
                        None
                        if overall is None
                        else overall.recall
                    ),
                    "f1": (
                        None
                        if overall is None
                        else overall.f1
                    ),
                    "unsupported_claim_rate_pre_policy": (
                        unsupported_claim_rate(
                            pre_policy_claims
                        )
                    ),
                    "unsupported_claim_rate_post_validation": (
                        unsupported_claim_rate(
                            post_validation_claims
                        )
                    ),
                    "p50_latency_ms": (
                        statistics.median(
                            latencies_ms
                        )
                        if latencies_ms
                        else None
                    ),
                    "p95_latency_ms": (
                        _percentile(
                            latencies_ms,
                            0.95,
                        )
                        if latencies_ms
                        else None
                    ),
                    "model_acquisition_time_ms": (
                        model_acquisition_time_ms
                    ),
                    "model_acquisition_skipped": (
                        skip_model_acquisition
                        or using_existing_runtime
                    ),
                    "existing_runtime_reused": (
                        using_existing_runtime
                    ),
                    "model_load_time_ms": (
                        model_load_time_ms
                    ),
                    "runtime_peak_ram_mb": (
                        None
                        if resource_monitor is None
                        else resource_monitor.peak_ram_mb
                    ),
                    "vram_peak_mb": (
                        None
                        if resource_monitor is None
                        else resource_monitor.peak_vram_mb
                    ),
                    "determinism_sample_count": (
                        effective_determinism_sample_count
                    ),
                    "deterministic_rerun": (
                        _deterministic_rerun_status(
                            iterations,
                            deterministic,
                        )
                    ),
                    "per_label_type": (
                        per_label_type
                    ),
                }
            )
        finally:
            if resource_monitor is not None:
                resource_monitor.stop()
            if process is not None:
                _stop_runtime(process)

    return {
        "schema_version": "1.0.0",
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "manifest_fingerprint": (
            manifest_fingerprint
        ),
        "dataset_content_fingerprint": (
            dataset_content_fingerprint
        ),
        "visual_label_coverage": coverage,
        "iterations": iterations,
        "candidates": candidate_reports,
        "measurement_notes": {
            "confidence": (
                "provider confidence is model-self-reported "
                "and uncalibrated; no production threshold "
                "is selected in Phase 5"
            ),
            "unsupported_claim_rate_pre_policy": (
                "measured on strictly parsed structured "
                "provider claims before Core policy rejection; "
                "raw model response text is never retained"
            ),
            "unsupported_claim_rate_post_validation": (
                "measured on evidence that survived the "
                "Phase 1 observable-claim policy; "
                "provider-policy rejections are counted "
                "in failed_sample_count"
            ),
            "model_acquisition_time_ms": (
                "network-enabled llama.cpp -hf warm-up "
                "used to populate the configured cache before "
                "measured startup; includes download and initial "
                "load work when the candidate is not cached"
            ),
            "model_load_time_ms": (
                "measured on a second llama.cpp startup with "
                "--offline after acquisition succeeds, so network "
                "download time is excluded"
            ),
            "runtime_peak_ram_mb": (
                "llama.cpp process working-set/RSS "
                "sampled every 250 ms when supported; "
                "null otherwise"
            ),
            "vram_peak_mb": (
                "NVIDIA process VRAM sampled from "
                "nvidia-smi every 250 ms when available; "
                "null otherwise"
            ),
            "model_sha256": (
                "SHA-256 of the local GGUF model_path "
                "exposed by llama.cpp /props when that "
                "path is readable"
            ),
            "failure_reasons": (
                "sanitized exception categories/messages only; "
                "raw model output and image content are not retained"
            ),
            "deterministic_rerun": (
                "null when iterations < 2 because no rerun was "
                "performed"
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Phase 5 local vision candidates"
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--determinism-sample-count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--expected-manifest-fingerprint"
    )
    parser.add_argument(
        "--expected-dataset-content-fingerprint"
    )
    parser.add_argument(
        "--runtime-executable"
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=int,
    )
    parser.add_argument(
        "--model-acquisition-timeout-seconds",
        type=int,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
    )
    parser.add_argument(
        "--skip-model-acquisition",
        action="store_true",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
    )
    parser.add_argument(
        "--max-observations-per-kind",
        type=int,
    )
    parser.add_argument(
        "--max-label-chars",
        type=int,
    )
    parser.add_argument(
        "--max-caption-chars",
        type=int,
    )
    parser.add_argument(
        "--existing-runtime-base-url",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = asyncio.run(
        run_vision_benchmark_async(
            manifest=manifest,
            dataset_root=Path(
                args.dataset_root
            ),
            candidates=tuple(
                args.candidate
            ),
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
            runtime_executable=(
                args.runtime_executable
            ),
            startup_timeout_seconds=(
                args.startup_timeout_seconds
            ),
            model_acquisition_timeout_seconds=(
                args.model_acquisition_timeout_seconds
            ),
            request_timeout_seconds=(
                args.request_timeout_seconds
            ),
            max_tokens=args.max_tokens,
            max_observations_per_kind=(
                args.max_observations_per_kind
            ),
            max_label_chars=args.max_label_chars,
            max_caption_chars=args.max_caption_chars,
            skip_model_acquisition=(
                args.skip_model_acquisition
            ),
            existing_runtime_base_url=(
                args.existing_runtime_base_url
            ),
        )
    )
    output_path = Path(args.output)
    output_path.write_text(
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
    return _report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
