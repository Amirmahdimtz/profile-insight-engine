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


def _degraded_vision_rejection_is_valid(report: dict[str, Any] | None) -> bool:
    if report is None or report.get("schema_version") != "phase10-siglip-zero-shot-vision-benchmark-v1":
        return False
    candidate = report.get("candidate")
    return bool(
        isinstance(candidate, dict)
        and candidate.get("selection_protocol_safe") is True
        and candidate.get("benchmark_valid") is True
        and candidate.get("failure_rate") == 0.0
        and candidate.get("f1") == 0.0
    )


def _ocr_best_evidence_is_valid(report: dict[str, Any] | None) -> bool:
    if report is None:
        return False
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("candidate") == "tessdata_best"
        and item.get("deterministic_rerun") is True
        for item in candidates
    )


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
    deployment_profile = getattr(args, "deployment_profile", "strict_full")
    if deployment_profile not in {"strict_full", "degraded_ocr_only"}:
        raise ValueError("unsupported deployment profile")
    required_inputs = (
        ("ocr", "vision", "insight", "api_persistence", "e2e", "hardware")
        if deployment_profile == "degraded_ocr_only"
        else ("ocr", "vision", "embedding", "insight", "api_persistence", "e2e", "hardware")
    )
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

    vision_rejection_valid = _degraded_vision_rejection_is_valid(reports["vision"])
    ocr_selection_valid = _ocr_best_evidence_is_valid(reports["ocr"])
    e2e_profile = e2e.get("deployment_profile") if isinstance(e2e.get("deployment_profile"), dict) else {}
    concurrency = e2e.get("concurrency") if isinstance(e2e.get("concurrency"), dict) else {}
    e2e_profile_valid = (
        deployment_profile == "degraded_ocr_only"
        and e2e_profile.get("vision_enabled") is False
        and performance.get("failure_rate") == 0.0
        and quality.get("summary_structural_factuality") is True
        and quality.get("unsupported_claim_rate_post_validation") == 0.0
        and concurrency.get("failed_requests") == 0
        and concurrency.get("request_isolation") is True
    )
    degraded_acceptance_ready = (
        deployment_profile == "degraded_ocr_only"
        and not missing
        and vision_rejection_valid
        and ocr_selection_valid
        and e2e_profile_valid
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "deployment_profile": deployment_profile,
        "metric_scope": METRIC_SCOPE,
        "input_evidence": {
            name: {
                "status": (
                    "not_required"
                    if deployment_profile == "degraded_ocr_only" and name == "embedding" and report is None
                    else ("available" if report is not None else "missing")
                ),
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
        "production_selection": (
            {
                "ocr": "tesseract_tessdata_best_fas_eng",
                "vision": "disabled_rejected_by_holdout_evidence",
                "embedding": "not_in_active_http_workflow",
                "thresholds": "phase8_versioned_policy_no_posthoc_tuning",
            }
            if deployment_profile == "degraded_ocr_only"
            else {
                "ocr": "unresolved_requires_phase10_evidence_review",
                "vision": "unresolved_requires_phase10_evidence_review",
                "embedding": "unresolved_requires_phase10_evidence_review",
                "thresholds": "unresolved_requires_phase10_evidence_review",
            }
        ),
        "selection_evidence": {
            "ocr_best_evidence_valid": ocr_selection_valid,
            "vision_rejection_evidence_valid": vision_rejection_valid,
            "e2e_degraded_profile_valid": e2e_profile_valid,
        },
        "missing_evidence": missing,
        "acceptance_ready": degraded_acceptance_ready,
        "notes": [
            "This report never promotes a failed Vision candidate as production-capable.",
            "The degraded_ocr_only profile is an explicit scope decision backed by rejected target-host Vision evidence and versioned vision.enabled=false configuration.",
            "Metrics without valid annotations or measurements remain unavailable rather than estimated.",
            "Acceptance is fail-closed unless required reports, OCR selection evidence, Vision rejection evidence and the exact degraded E2E profile all validate.",
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
    parser.add_argument("--deployment-profile", choices=("strict_full", "degraded_ocr_only"), default="strict_full")
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
