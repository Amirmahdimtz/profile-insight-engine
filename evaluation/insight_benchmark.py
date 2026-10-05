from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from evaluation.metrics import unsupported_claim_rate
from src.core.profile_analysis.aggregation_contracts import AggregatedTheme, AggregationKind
from src.core.profile_analysis.contracts import Evidence, EvidenceType, InsightType
from src.core.services.profile_analysis.insight_generation_service import InsightGenerationService
from src.infrastructure.utils.config_reader import ConfigReader


REPORT_SCHEMA_VERSION = "phase8-insight-benchmark-v1"
METRIC_SCOPE = "synthetic_contract_scenarios_not_product_quality"


@dataclass(frozen=True)
class InsightBenchmarkReport:
    schema_version: str
    metric_scope: str
    policy_version: str
    aggregate_count: int
    expected_supported_count: int
    generated_insight_count: int
    insight_precision: float
    insight_recall: float
    false_unsupported_insight_rate: float
    summary_factuality: float
    confidence_calibration: float | None
    human_review_agreement: float | None
    deterministic_rerun: bool
    reversed_input_deterministic: bool
    result_hash: str
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _evidence(
    evidence_id: str,
    image_id: str,
    evidence_type: EvidenceType,
    label: str,
    confidence: float,
    source: str,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        image_id=image_id,
        type=evidence_type,
        label=label,
        value=label,
        confidence=confidence,
        source=source,
    )


def _aggregate(
    *,
    kind: AggregationKind,
    key: str,
    label: str,
    signal_ids: tuple[str, ...],
    image_ids: tuple[str, ...],
    image_universe_count: int,
    average_confidence: float | None,
    max_confidence: float | None,
    sources: tuple[str, ...],
    consistency: float = 1.0,
) -> AggregatedTheme:
    return AggregatedTheme(
        kind=kind,
        key=key,
        label=label,
        evidence_count=len(signal_ids),
        unique_image_count=len(image_ids),
        image_universe_count=image_universe_count,
        image_coverage=len(image_ids) / image_universe_count,
        average_confidence=average_confidence,
        max_confidence=max_confidence,
        source_diversity=len(sources),
        cross_image_consistency=consistency,
        sources=sources,
        supporting_image_ids=image_ids,
        supporting_signal_ids=signal_ids,
    )


def _workload() -> tuple[tuple[AggregatedTheme, ...], tuple[Evidence, ...]]:
    evidence = (
        _evidence("topic-1", "img-1", EvidenceType.TOPIC, "football", 0.8, "vision"),
        _evidence("topic-2", "img-2", EvidenceType.TOPIC, "football", 0.7, "vision"),
        _evidence("topic-3", "img-3", EvidenceType.TOPIC, "football", 0.9, "ocr"),
        _evidence("brand-1", "img-1", EvidenceType.BRAND, "example_brand", 0.99, "vision"),
        _evidence(
            "religious-1",
            "img-2",
            EvidenceType.RELIGIOUS_CONTENT,
            "religious-themed content",
            0.75,
            "vision",
        ),
        _evidence(
            "religious-2",
            "img-4",
            EvidenceType.RELIGIOUS_CONTENT,
            "religious-themed content",
            0.8,
            "ocr",
        ),
    )
    aggregates = (
        _aggregate(
            kind=AggregationKind.TOPIC,
            key="football",
            label="football",
            signal_ids=("topic-1", "topic-2", "topic-3"),
            image_ids=("img-1", "img-2", "img-3"),
            image_universe_count=4,
            average_confidence=0.8,
            max_confidence=0.9,
            sources=("ocr", "vision"),
            consistency=1.0,
        ),
        _aggregate(
            kind=AggregationKind.BRAND,
            key="example_brand",
            label="example_brand",
            signal_ids=("brand-1",),
            image_ids=("img-1",),
            image_universe_count=4,
            average_confidence=0.99,
            max_confidence=0.99,
            sources=("vision",),
            consistency=1.0,
        ),
        _aggregate(
            kind=AggregationKind.RELIGIOUS_CONTENT,
            key="religious-themed content",
            label="religious-themed content",
            signal_ids=("religious-1", "religious-2"),
            image_ids=("img-2", "img-4"),
            image_universe_count=4,
            average_confidence=0.775,
            max_confidence=0.8,
            sources=("ocr", "vision"),
            consistency=1.0,
        ),
        _aggregate(
            kind=AggregationKind.SEMANTIC_THEME,
            key="travel",
            label="travel",
            signal_ids=("semantic-1", "semantic-2"),
            image_ids=("img-1", "img-4"),
            image_universe_count=4,
            average_confidence=None,
            max_confidence=None,
            sources=("embedding",),
            consistency=1.0,
        ),
    )
    return aggregates, evidence


def _hash_result(result) -> str:
    return hashlib.sha256(result.to_json().encode("utf-8")).hexdigest()


def run_benchmark(config_reader: ConfigReader) -> InsightBenchmarkReport:
    service = InsightGenerationService(config_reader)
    aggregates, evidence = _workload()
    first = service.generate(aggregates, evidence)
    second = service.generate(aggregates, evidence)
    reversed_result = service.generate(tuple(reversed(aggregates)), tuple(reversed(evidence)))

    expected = {
        (InsightType.VISIBLE_INTEREST.value, "Recurring topic-related content: football"),
        (
            InsightType.RELIGIOUS_CONTENT.value,
            "Recurring religious-themed content: religious-themed content",
        ),
    }
    actual = {(item.type.value, item.label) for item in first.insights}
    true_positive = len(actual & expected)
    insight_precision = true_positive / len(actual) if actual else (1.0 if not expected else 0.0)
    insight_recall = true_positive / len(expected) if expected else 1.0

    false_unsupported_rate = unsupported_claim_rate(
        [(item.key, item.label) for item in first.insights]
    )

    expected_summary = (
        "No supported insights met the configured evidence policy."
        if not first.insights
        else "Supported insights: " + " | ".join(item.label for item in first.insights)
    )
    summary_factuality = 1.0 if first.summary == expected_summary else 0.0
    result_hash = _hash_result(first)

    return InsightBenchmarkReport(
        schema_version=REPORT_SCHEMA_VERSION,
        metric_scope=METRIC_SCOPE,
        policy_version=first.policy_version,
        aggregate_count=len(aggregates),
        expected_supported_count=len(expected),
        generated_insight_count=len(first.insights),
        insight_precision=insight_precision,
        insight_recall=insight_recall,
        false_unsupported_insight_rate=false_unsupported_rate,
        summary_factuality=summary_factuality,
        confidence_calibration=None,
        human_review_agreement=None,
        deterministic_rerun=first == second,
        reversed_input_deterministic=first == reversed_result,
        result_hash=result_hash,
        limitations=(
            "No authorized Phase 8 insight-level gold annotations are committed, so product insight precision is not measured by this synthetic contract benchmark.",
            "Confidence calibration requires labeled correctness outcomes and is unavailable until a real annotated insight dataset is supplied.",
            "Human review agreement requires independent human annotations and is unavailable in the repository-only environment.",
            "Summary factuality here is a structural check that the deterministic summary contains exactly the accepted insight labels, not a substitute for human factuality review on real data.",
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Phase 8 Insight Engine contract evaluation."
    )
    parser.add_argument(
        "--config",
        help="Optional appsettings YAML path; normal ConfigReader environment selection is used when omitted.",
    )
    parser.add_argument("--output", help="Optional report output path; stdout is used otherwise")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_reader = ConfigReader(args.config) if args.config else ConfigReader()
    report = run_benchmark(config_reader)
    serialized = report.to_json() + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
