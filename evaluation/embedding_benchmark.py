from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluation.dataset import EvaluationDatasetManifest
from evaluation.metrics import macro_f1, mean_average_precision, recall_at_k
from src.core.profile_analysis.contracts import ProfileAnalysisRequest
from src.core.profile_analysis.image_contracts import RawImageInput
from src.core.services.profile_analysis.image_processing_service import ImageProcessingService
from src.core.services.profile_analysis.semantic_theme_service import SemanticThemeService
from src.infrastructure.providers.embedding.transformers_embedding_provider import TransformersEmbeddingProvider
from src.infrastructure.providers.image.local_image_storage import LocalImageStorage
from src.infrastructure.utils.config_reader import ConfigReader


REPORT_SCHEMA_VERSION = "phase6-embedding-benchmark-v2"


def _peak_ram_mb() -> float | None:
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
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = kernel32.GetCurrentProcess()
            if not psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                return None
            return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except (OSError, AttributeError, ValueError):
            return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return value / divisor
    except (ImportError, OSError, ValueError):
        return None


def _vram_mb() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    total = 0.0
    matched = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, memory = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid == os.getpid():
            total += memory
            matched = True
    return total if matched else None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _load_manifest(path: Path) -> EvaluationDatasetManifest:
    return EvaluationDatasetManifest.from_json(path.read_text(encoding="utf-8"))


def _raw_image(sample, root: Path) -> RawImageInput:
    path = root / sample.image.relative_path
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"unsupported benchmark image type: {path.name}")
    return RawImageInput(
        image_id=sample.image.image_id,
        filename=path.name,
        declared_mime_type=mime,
        content=path.read_bytes(),
    )


def _calculate_semantic_metrics(eval_samples, result, labels: tuple[str, ...], recall_k: int) -> dict[str, object]:
    expected_sets: list[set[str]] = []
    predicted_sets: list[set[str]] = []
    matched_by_image: dict[str, set[str]] = {
        sample.image.image_id: set() for sample in eval_samples
    }
    for item in result.theme_similarities:
        if item.matched:
            matched_by_image[item.image_id].add(item.theme_label)

    relevant_by_label: dict[str, set[str]] = {label: set() for label in labels}
    for sample in eval_samples:
        expected = {label.label for label in sample.ground_truth.labels}
        expected_sets.append(expected)
        predicted_sets.append(matched_by_image[sample.image.image_id])
        for label in expected:
            relevant_by_label[label].add(sample.image.image_id)

    ranked_by_label = {
        item.query_label: item.ranked_image_ids for item in result.retrieval_results
    }
    map_value = mean_average_precision(relevant_by_label, ranked_by_label)
    recall_values = [
        recall_at_k(relevant_by_label[label], ranked_by_label[label], recall_k)
        for label in labels
        if relevant_by_label[label]
    ]
    cross_language_f1: dict[str, float | None] = {}
    for slice_name in ("persian_heavy", "english_heavy", "mixed_persian_english"):
        indices = [
            index
            for index, sample in enumerate(eval_samples)
            if any(item.value == slice_name for item in sample.slices)
        ]
        has_ground_truth = any(expected_sets[index] for index in indices)
        cross_language_f1[slice_name] = (
            macro_f1(
                [expected_sets[index] for index in indices],
                [predicted_sets[index] for index in indices],
                classes=labels,
            )
            if indices and has_ground_truth
            else None
        )

    expected_pairs = {
        (sample.image.image_id, label.label)
        for sample in eval_samples
        for label in sample.ground_truth.labels
    }
    similarity_scores = [item.similarity for item in result.theme_similarities]
    expected_similarity_scores = [
        item.similarity
        for item in result.theme_similarities
        if (item.image_id, item.theme_label) in expected_pairs
    ]
    similarity_diagnostics = {
        "matched_count": sum(1 for item in result.theme_similarities if item.matched),
        "score_min": min(similarity_scores) if similarity_scores else None,
        "score_max": max(similarity_scores) if similarity_scores else None,
        "expected_score_min": (
            min(expected_similarity_scores) if expected_similarity_scores else None
        ),
        "expected_score_max": (
            max(expected_similarity_scores) if expected_similarity_scores else None
        ),
    }

    return {
        "theme_macro_f1": macro_f1(expected_sets, predicted_sets, classes=labels),
        "mean_recall_at_k": statistics.fmean(recall_values) if recall_values else 0.0,
        "map": map_value,
        "cross_language_theme_macro_f1": cross_language_f1,
        "theme_similarity_diagnostics": similarity_diagnostics,
    }


async def _run(args: argparse.Namespace) -> dict[str, object]:
    manifest = _load_manifest(Path(args.manifest))
    dataset_root = Path(args.dataset_root).resolve()
    content_fingerprint = manifest.validate_references(dataset_root)
    eval_samples = tuple(sample for sample in manifest.samples if sample.split.value == "eval")
    if not eval_samples:
        raise ValueError("embedding benchmark requires eval samples")

    labels = tuple(
        sorted(
            {label.label for sample in eval_samples for label in sample.ground_truth.labels},
            key=lambda value: (value.casefold(), value),
        )
    )
    if not labels:
        raise ValueError("embedding benchmark requires observable ground-truth labels")

    config = ConfigReader(args.config)
    provider = TransformersEmbeddingProvider(
        config,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    image_processing = ImageProcessingService(config, LocalImageStorage())
    semantic_service = SemanticThemeService(provider, config)
    request = ProfileAnalysisRequest(
        analysis_id="phase6-embedding-benchmark",
        image_ids=tuple(sample.image.image_id for sample in eval_samples),
    )
    batch = await image_processing.process_async(
        request,
        tuple(_raw_image(sample, dataset_root) for sample in eval_samples),
    )
    try:
        warmup_started = time.perf_counter()
        await semantic_service.analyze_async((batch.images[0],), labels)
        warmup_latency_ms = (time.perf_counter() - warmup_started) * 1000.0

        latencies: list[float] = []
        results = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            results.append(await semantic_service.analyze_async(batch.images, labels))
            latencies.append((time.perf_counter() - started) * 1000.0)
        result = results[-1]
        deterministic_rerun = None
        if len(results) > 1:
            deterministic_rerun = all(current == results[0] for current in results[1:])

        semantic_metrics = _calculate_semantic_metrics(
            eval_samples, result, labels, args.recall_k
        )

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "manifest_fingerprint": manifest.fingerprint(),
            "dataset_content_fingerprint": content_fingerprint,
            "model_id": result.model_id,
            "model_version": result.model_version,
            "provider": result.provider,
            "provider_version": result.provider_version,
            "config_version": result.config_version,
            "embedding_dimension": result.embedding_dimension,
            "eval_sample_count": len(eval_samples),
            "theme_label_count": len(labels),
            "theme_macro_f1": semantic_metrics["theme_macro_f1"],
            "mean_recall_at_k": semantic_metrics["mean_recall_at_k"],
            "recall_k": args.recall_k,
            "map": semantic_metrics["map"],
            "cross_language_theme_macro_f1": semantic_metrics["cross_language_theme_macro_f1"],
            "theme_similarity_diagnostics": semantic_metrics["theme_similarity_diagnostics"],
            "iterations": args.iterations,
            "warmup_latency_ms": warmup_latency_ms,
            "latency_scope": "post_warmup_full_eval_batch",
            "p50_batch_latency_ms": _percentile(latencies, 0.50),
            "p95_batch_latency_ms": _percentile(latencies, 0.95),
            "peak_process_ram_mb": _peak_ram_mb(),
            "process_vram_mb": _vram_mb(),
            "deterministic_rerun": deterministic_rerun,
            "near_duplicate_group_count": len(result.near_duplicate_groups),
            "candidate_thresholds": {
                "theme_similarity_threshold": config.get("embedding.theme_similarity_threshold"),
                "near_duplicate_similarity_threshold": config.get(
                    "embedding.near_duplicate_similarity_threshold"
                ),
            },
        }
    finally:
        await image_processing.release_async(batch)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a Phase 6 embedding candidate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="src/host/res/appsettings.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.iterations <= 0 or args.recall_k <= 0:
        raise SystemExit("--iterations and --recall-k must be positive")
    report = asyncio.run(_run(args))
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
