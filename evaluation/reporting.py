from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from evaluation.common import (
    BENCHMARK_REPORT_SCHEMA_VERSION, EVALUATOR_VERSION, BenchmarkReportKind,
    EvaluationValidationError, _expect_keys, _expect_mapping, _parse_enum,
    _require_finite, _require_id, _require_non_negative_finite, _require_non_negative_int,
    _require_positive_int, _require_sequence, _require_text, _require_version,
    _to_primitive, deterministic_json,
)

@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    sample_count: int
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "metric.name"))
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise EvaluationValidationError("metric.value must be a finite number")
        value = float(self.value)
        if not math.isfinite(value):
            raise EvaluationValidationError("metric.value must be a finite number")
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self, "sample_count", _require_non_negative_int(self.sample_count, "metric.sample_count")
        )
        minimum = self.minimum
        maximum = self.maximum
        if minimum is not None:
            minimum = _require_finite(minimum, "metric.minimum")
            object.__setattr__(self, "minimum", minimum)
        if maximum is not None:
            maximum = _require_finite(maximum, "metric.maximum")
            object.__setattr__(self, "maximum", maximum)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise EvaluationValidationError("metric.minimum must be <= metric.maximum")
        if minimum is not None and value < minimum:
            raise EvaluationValidationError("metric.value is below metric.minimum")
        if maximum is not None and value > maximum:
            raise EvaluationValidationError("metric.value is above metric.maximum")

    @classmethod
    def from_dict(cls, payload: Any) -> "MetricResult":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"name", "value", "sample_count", "minimum", "maximum"},
        )
        return cls(
            name=payload["name"],
            value=payload["value"],
            sample_count=payload["sample_count"],
            minimum=payload["minimum"],
            maximum=payload["maximum"],
        )


@dataclass(frozen=True)
class SystemMetrics:
    p50_latency_ms: float
    p95_latency_ms: float
    ram_mb: float
    vram_mb: float | None = None

    def __post_init__(self) -> None:
        p50 = _require_non_negative_finite(self.p50_latency_ms, "system.p50_latency_ms")
        p95 = _require_non_negative_finite(self.p95_latency_ms, "system.p95_latency_ms")
        ram = _require_non_negative_finite(self.ram_mb, "system.ram_mb")
        if p95 < p50:
            raise EvaluationValidationError("system.p95_latency_ms must be >= p50_latency_ms")
        object.__setattr__(self, "p50_latency_ms", p50)
        object.__setattr__(self, "p95_latency_ms", p95)
        object.__setattr__(self, "ram_mb", ram)
        if self.vram_mb is not None:
            object.__setattr__(
                self,
                "vram_mb",
                _require_non_negative_finite(self.vram_mb, "system.vram_mb"),
            )

    @classmethod
    def from_dict(cls, payload: Any) -> "SystemMetrics":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"p50_latency_ms", "p95_latency_ms", "ram_mb", "vram_mb"},
        )
        return cls(
            p50_latency_ms=payload["p50_latency_ms"],
            p95_latency_ms=payload["p95_latency_ms"],
            ram_mb=payload["ram_mb"],
            vram_mb=payload["vram_mb"],
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    evaluator_version: str = EVALUATOR_VERSION
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluator_version",
            _require_text(self.evaluator_version, "config.evaluator_version"),
        )
        object.__setattr__(
            self,
            "calibration_bins",
            _require_positive_int(self.calibration_bins, "config.calibration_bins"),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> "BenchmarkConfig":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"evaluator_version", "calibration_bins"})
        return cls(
            evaluator_version=payload["evaluator_version"],
            calibration_bins=payload["calibration_bins"],
        )


@dataclass(frozen=True)
class ReproducibilityMetadata:
    dataset_id: str
    dataset_version: str
    manifest_fingerprint: str
    evaluator_version: str
    config: BenchmarkConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _require_id(self.dataset_id, "repro.dataset_id"))
        object.__setattr__(
            self,
            "dataset_version",
            _require_version(self.dataset_version, "repro.dataset_version"),
        )
        fingerprint = _require_text(self.manifest_fingerprint, "repro.manifest_fingerprint")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise EvaluationValidationError(
                "repro.manifest_fingerprint must be a lowercase SHA-256 hex digest"
            )
        object.__setattr__(self, "manifest_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "evaluator_version",
            _require_text(self.evaluator_version, "repro.evaluator_version"),
        )
        if not isinstance(self.config, BenchmarkConfig):
            raise EvaluationValidationError("repro.config must be a BenchmarkConfig")
        if self.evaluator_version != self.config.evaluator_version:
            raise EvaluationValidationError(
                "repro.evaluator_version must match repro.config.evaluator_version"
            )

    @classmethod
    def from_dict(cls, payload: Any) -> "ReproducibilityMetadata":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {
                "dataset_id",
                "dataset_version",
                "manifest_fingerprint",
                "evaluator_version",
                "config",
            },
        )
        return cls(
            dataset_id=payload["dataset_id"],
            dataset_version=payload["dataset_version"],
            manifest_fingerprint=payload["manifest_fingerprint"],
            evaluator_version=payload["evaluator_version"],
            config=BenchmarkConfig.from_dict(payload["config"]),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: str
    report_kind: BenchmarkReportKind
    reproducibility: ReproducibilityMetadata
    metrics: tuple[MetricResult, ...]
    system_metrics: SystemMetrics | None = None

    def __post_init__(self) -> None:
        schema_version = _require_version(self.schema_version, "report.schema_version")
        if schema_version != BENCHMARK_REPORT_SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"report.schema_version must be {BENCHMARK_REPORT_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.report_kind, BenchmarkReportKind):
            raise EvaluationValidationError("report.report_kind must be a BenchmarkReportKind")
        if not isinstance(self.reproducibility, ReproducibilityMetadata):
            raise EvaluationValidationError(
                "report.reproducibility must be ReproducibilityMetadata"
            )
        metrics = _require_sequence(self.metrics, "report.metrics")
        if not all(isinstance(item, MetricResult) for item in metrics):
            raise EvaluationValidationError("report.metrics must contain MetricResult values")
        names = [item.name for item in metrics]
        if len(names) != len(set(names)):
            raise EvaluationValidationError("report metric names must be unique")
        object.__setattr__(self, "metrics", tuple(sorted(metrics, key=lambda item: item.name)))
        if self.system_metrics is not None and not isinstance(self.system_metrics, SystemMetrics):
            raise EvaluationValidationError("report.system_metrics must be SystemMetrics or null")

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    def to_json(self) -> str:
        return deterministic_json(self)

    @classmethod
    def from_dict(cls, payload: Any) -> "BenchmarkReport":
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"schema_version", "report_kind", "reproducibility", "metrics", "system_metrics"},
        )
        metrics = _require_sequence(payload["metrics"], "report.metrics")
        system_metrics = payload["system_metrics"]
        return cls(
            schema_version=payload["schema_version"],
            report_kind=_parse_enum(
                BenchmarkReportKind, payload["report_kind"], "report.report_kind"
            ),
            reproducibility=ReproducibilityMetadata.from_dict(payload["reproducibility"]),
            metrics=tuple(MetricResult.from_dict(item) for item in metrics),
            system_metrics=(
                None
                if system_metrics is None
                else SystemMetrics.from_dict(system_metrics)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "BenchmarkReport":
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise EvaluationValidationError("report contains invalid JSON") from exc
        return cls.from_dict(parsed)
