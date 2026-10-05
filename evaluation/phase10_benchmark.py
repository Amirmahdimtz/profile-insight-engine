from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "phase10-final-benchmark-v1"
METRIC_SCOPE = "phase10_final_acceptance_evidence"
_EXPECTED_SCHEMAS = {
    "e2e": "phase10-end-to-end-benchmark-v1",
    "hardware": "phase10-hardware-profile-v1",
}


def _load_optional(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} report is missing or invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} report must be a JSON object")
    return payload


def _metric(value: object, reason: str) -> dict[str, object]:
    if value is None or value == "unavailable":
        return {"status": "unavailable", "value": None, "reason": reason}
    return {"status": "measured", "value": value, "reason": None}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        "ocr": _load_optional(args.ocr_report, "OCR"),
        "vision": _load_optional(args.vision_report, "Vision"),
        "embedding": _load_optional(args.embedding_report, "Embedding"),
        "insight": _load_optional(args.insight_report, "Insight"),
        "api_persistence": _load_optional(args.api_persistence_report, "API/persistence"),
        "e2e": _load_optional(args.e2e_report, "end-to-end"),
        "hardware": _load_optional(args.hardware_report, "hardware"),
    }
    for key, schema in _EXPECTED_SCHEMAS.items():
        report = reports[key]
        if report is not None and report.get("schema_version") != schema:
            raise ValueError(f"unexpected {key} report schema_version")

    e2e = reports["e2e"] or {}
    quality = e2e.get("quality") if isinstance(e2e.get("quality"), dict) else {}
    performance = e2e.get("performance") if isinstance(e2e.get("performance"), dict) else {}
    required_inputs = ("ocr", "vision", "embedding", "insight", "api_persistence", "e2e", "hardware")
    missing = [name for name in required_inputs if reports[name] is None]

    mandatory_metrics = {
        "quality": {
            "end_to_end_accuracy": _metric(
                quality.get("end_to_end_accuracy"),
                "authorized end-to-end ground truth is not available in the supplied evidence",
            ),
            "insight_precision": _metric(
                quality.get("insight_precision"),
                "authorized insight-level correctness annotations are not available in the supplied evidence",
            ),
            "unsupported_claim_rate": _metric(
                quality.get("unsupported_claim_rate_post_validation"),
                "no measured end-to-end output was supplied",
            ),
            "summary_factuality": _metric(
                quality.get("summary_structural_factuality"),
                "no measured end-to-end output was supplied",
            ),
            "confidence_calibration": _metric(
                quality.get("confidence_calibration"),
                "ground-truth correctness outcomes are insufficient for calibration",
            ),
        },
        "performance": {
            "p50_latency_ms": _metric(
                performance.get("total_end_to_end_p50_latency_ms"),
                "no final end-to-end aggregate latency was supplied",
            ),
            "p95_latency_ms": _metric(
                performance.get("total_end_to_end_p95_latency_ms"),
                "no final end-to-end aggregate latency was supplied",
            ),
            "throughput_images_per_second": _metric(
                performance.get("throughput_images_per_second"),
                "no final end-to-end aggregate throughput was supplied",
            ),
            "model_provider_warmup_ms": _metric(
                performance.get("model_provider_warmup_ms"),
                "provider-specific warmup must come from real provider benchmark evidence",
            ),
            "peak_ram_mb": _metric(
                performance.get("peak_process_ram_mb"),
                "no measured service-process RAM evidence was supplied",
            ),
            "peak_vram_mb": _metric(
                performance.get("peak_vram_mb"),
                "VRAM is unavailable when the target runtime has no measurable GPU process evidence",
            ),
            "failure_rate": _metric(
                performance.get("failure_rate"),
                "no measured end-to-end failure evidence was supplied",
            ),
        },
    }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metric_scope": METRIC_SCOPE,
        "input_evidence": {
            name: {
                "status": "available" if report is not None else "missing",
                "schema_version": report.get("schema_version") if report else None,
            }
            for name, report in reports.items()
        },
        "benchmark_matrix": {
            "workload_sizes": [1, 5, 20, 50, 100],
            "input_scenarios": [
                "normal_images",
                "corrupt_images",
                "huge_images",
                "blank_images",
                "duplicate_images",
                "mixed_valid_invalid_batches",
            ],
            "failure_scenarios": [
                "gpu_unavailable",
                "ocr_provider_unavailable_or_timeout",
                "vision_provider_timeout",
                "vision_provider_crash",
                "embedding_provider_failure",
                "partial_pipeline_failure",
                "database_runtime_failure",
                "concurrent_requests",
                "restart_recovery",
                "request_isolation",
            ],
        },
        "metrics": mandatory_metrics,
        "production_selection": {
            "ocr": "unresolved_requires_phase10_evidence_review",
            "vision": "unresolved_requires_phase10_evidence_review",
            "embedding": "unresolved_requires_phase10_evidence_review",
            "thresholds": "unresolved_requires_phase10_evidence_review",
        },
        "missing_evidence": missing,
        "acceptance_ready": False,
        "notes": [
            "This report never promotes a model/provider/config automatically.",
            "Production selection requires explicit evidence review and a versioned config change after target-hardware benchmark results are accepted.",
            "Metrics without valid annotations or measurements remain unavailable rather than estimated.",
            "Acceptance is fail-closed in this implementation commit; a later evidence-review commit must explicitly finalize production selection and closure.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-report", type=Path)
    parser.add_argument("--vision-report", type=Path)
    parser.add_argument("--embedding-report", type=Path)
    parser.add_argument("--insight-report", type=Path)
    parser.add_argument("--api-persistence-report", type=Path)
    parser.add_argument("--e2e-report", type=Path)
    parser.add_argument("--hardware-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
