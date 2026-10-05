from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from src.core.profile_analysis.aggregation_contracts import AggregatedTheme, AggregationKind
from src.core.profile_analysis.contracts import (
    ContractValidationError,
    Evidence,
    InsightType,
    ProfileInsight,
    validate_observable_claim,
)
from src.core.profile_analysis.insight_contracts import (
    InsightGenerationError,
    InsightGenerationResult,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


@dataclass(frozen=True)
class _InsightRule:
    insight_type: InsightType
    label_prefix: str


@dataclass(frozen=True)
class _InsightPolicy:
    policy_version: str
    minimum_evidence_count: int
    minimum_unique_image_count: int
    minimum_image_coverage: float
    minimum_source_diversity: int
    minimum_cross_image_consistency: float
    minimum_average_confidence: float
    source_diversity_saturation: int

    @classmethod
    def from_config(cls, config_reader: ConfigReader) -> "_InsightPolicy":
        minimum_source_diversity = config_reader.get_positive_int(
            "insights.minimum_source_diversity"
        )
        source_diversity_saturation = config_reader.get_positive_int(
            "insights.source_diversity_saturation"
        )
        if source_diversity_saturation < minimum_source_diversity:
            raise InsightGenerationError(
                "insights.source_diversity_saturation must be >= minimum_source_diversity"
            )
        return cls(
            policy_version=config_reader.get_non_empty_string("insights.policy_version"),
            minimum_evidence_count=config_reader.get_positive_int(
                "insights.minimum_evidence_count"
            ),
            minimum_unique_image_count=config_reader.get_positive_int(
                "insights.minimum_unique_image_count"
            ),
            minimum_image_coverage=_read_probability(
                config_reader, "insights.minimum_image_coverage"
            ),
            minimum_source_diversity=minimum_source_diversity,
            minimum_cross_image_consistency=_read_probability(
                config_reader, "insights.minimum_cross_image_consistency"
            ),
            minimum_average_confidence=_read_probability(
                config_reader, "insights.minimum_average_confidence"
            ),
            source_diversity_saturation=source_diversity_saturation,
        )


_RULES: dict[AggregationKind, _InsightRule | None] = {
    AggregationKind.OBJECT: _InsightRule(InsightType.CONTENT_PATTERN, "Recurring object"),
    AggregationKind.SCENE: _InsightRule(InsightType.ENVIRONMENT, "Recurring scene"),
    AggregationKind.OCR_TEXT: _InsightRule(
        InsightType.TEXTUAL_REFERENCE, "Recurring textual reference"
    ),
    AggregationKind.ACTIVITY: _InsightRule(
        InsightType.ACTIVITY, "Recurring visible activity"
    ),
    AggregationKind.ENVIRONMENT: _InsightRule(
        InsightType.ENVIRONMENT, "Recurring environment"
    ),
    AggregationKind.TOPIC: _InsightRule(
        InsightType.VISIBLE_INTEREST, "Recurring topic-related content"
    ),
    AggregationKind.BRAND: _InsightRule(
        InsightType.BRAND_REFERENCE, "Recurring brand reference"
    ),
    AggregationKind.TEAM: _InsightRule(
        InsightType.TEAM_REFERENCE, "Recurring team reference"
    ),
    AggregationKind.RELIGIOUS_CONTENT: _InsightRule(
        InsightType.RELIGIOUS_CONTENT, "Recurring religious-themed content"
    ),
    AggregationKind.SOCIAL_CONTEXT: _InsightRule(
        InsightType.SOCIAL_CONTEXT, "Recurring visible social context"
    ),
    AggregationKind.OTHER_OBSERVABLE: _InsightRule(
        InsightType.CONTENT_PATTERN, "Recurring observable content"
    ),
    # Phase 7 intentionally exposes semantic similarity without probability confidence.
    # ProfileInsight requires bounded confidence and real Evidence IDs, so semantic-only
    # aggregates cannot independently become ProfileInsight values in Phase 8.
    AggregationKind.SEMANTIC_THEME: None,
}


def _read_probability(config_reader: ConfigReader, key: str) -> float:
    value = config_reader.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InsightGenerationError(f"configuration key '{key}' must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise InsightGenerationError(f"configuration key '{key}' must be in [0, 1]")
    return normalized


def _insight_key(aggregate: AggregatedTheme, insight_type: InsightType) -> str:
    digest = hashlib.sha256(
        f"{aggregate.kind.value}\0{aggregate.key}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{insight_type.value}_{digest}"


def _summary(insights: Sequence[ProfileInsight]) -> str:
    if not insights:
        return "No supported insights met the configured evidence policy."
    return "Supported insights: " + " | ".join(item.label for item in insights)


@inject
class InsightGenerationService:
    """Generate deterministic, evidence-backed ProfileInsight values from Phase 7 aggregates."""

    def __init__(self, config_reader: ConfigReader):
        self._policy = _InsightPolicy.from_config(config_reader)

    def generate(
        self,
        aggregates: Sequence[AggregatedTheme],
        evidence: Sequence[Evidence],
    ) -> InsightGenerationResult:
        aggregates = self._validate_aggregates(aggregates)
        evidence_by_id = self._evidence_registry(evidence)

        candidates: list[tuple[tuple[object, ...], ProfileInsight]] = []
        for aggregate in aggregates:
            rule = _RULES[aggregate.kind]
            if rule is None or not self._is_eligible(aggregate):
                continue
            if not self._is_observable(aggregate):
                continue

            supporting_evidence = self._resolve_supporting_evidence(
                aggregate, evidence_by_id
            )
            supporting_image_ids = tuple(
                sorted({item.image_id for item in supporting_evidence})
            )
            if len(supporting_image_ids) != aggregate.unique_image_count:
                raise InsightGenerationError(
                    "aggregate supporting signal references do not preserve unique-image support"
                )

            confidence = self._support_score(aggregate)
            label = f"{rule.label_prefix}: {aggregate.label}"
            explanation = self._explanation(aggregate, supporting_image_ids)
            try:
                insight = ProfileInsight(
                    key=_insight_key(aggregate, rule.insight_type),
                    type=rule.insight_type,
                    label=label,
                    explanation=explanation,
                    confidence=confidence,
                    evidence_count=aggregate.evidence_count,
                    image_coverage=aggregate.image_coverage,
                    supporting_evidence_ids=aggregate.supporting_signal_ids,
                    supporting_image_ids=supporting_image_ids,
                )
            except ContractValidationError as exc:
                raise InsightGenerationError(
                    f"aggregate '{aggregate.kind.value}:{aggregate.key}' cannot form a valid ProfileInsight"
                ) from exc

            ranking = (
                -confidence,
                -aggregate.image_coverage,
                -aggregate.unique_image_count,
                -aggregate.evidence_count,
                -aggregate.source_diversity,
                -aggregate.cross_image_consistency,
                insight.type.value,
                aggregate.key,
                insight.label,
            )
            candidates.append((ranking, insight))

        candidates.sort(key=lambda item: item[0])
        insights = tuple(item[1] for item in candidates)
        return InsightGenerationResult(
            policy_version=self._policy.policy_version,
            insights=insights,
            summary=_summary(insights),
        )

    @staticmethod
    def _validate_aggregates(
        aggregates: Sequence[AggregatedTheme],
    ) -> tuple[AggregatedTheme, ...]:
        if isinstance(aggregates, (str, bytes, bytearray)) or not isinstance(
            aggregates, Sequence
        ):
            raise InsightGenerationError(
                "aggregates must be a sequence of AggregatedTheme values"
            )
        items = tuple(aggregates)
        if not all(isinstance(item, AggregatedTheme) for item in items):
            raise InsightGenerationError(
                "aggregates must contain only AggregatedTheme values"
            )

        by_identity: dict[tuple[AggregationKind, str], AggregatedTheme] = {}
        for item in items:
            identity = (item.kind, item.key)
            existing = by_identity.get(identity)
            if existing is not None and existing != item:
                raise InsightGenerationError(
                    f"conflicting aggregate identity '{item.kind.value}:{item.key}'"
                )
            by_identity.setdefault(identity, item)
        return tuple(
            by_identity[identity]
            for identity in sorted(by_identity, key=lambda value: (value[0].value, value[1]))
        )

    @staticmethod
    def _evidence_registry(evidence: Sequence[Evidence]) -> dict[str, Evidence]:
        if isinstance(evidence, (str, bytes, bytearray)) or not isinstance(
            evidence, Sequence
        ):
            raise InsightGenerationError("evidence must be a sequence of Evidence values")
        if not all(isinstance(item, Evidence) for item in evidence):
            raise InsightGenerationError("evidence must contain only Evidence values")

        by_id: dict[str, Evidence] = {}
        for item in evidence:
            existing = by_id.get(item.id)
            if existing is not None and existing != item:
                raise InsightGenerationError(
                    f"conflicting evidence id '{item.id}' in insight support registry"
                )
            by_id.setdefault(item.id, item)
        return by_id

    def _is_eligible(self, aggregate: AggregatedTheme) -> bool:
        if aggregate.evidence_count < self._policy.minimum_evidence_count:
            return False
        if aggregate.unique_image_count < self._policy.minimum_unique_image_count:
            return False
        if aggregate.image_coverage < self._policy.minimum_image_coverage:
            return False
        if aggregate.source_diversity < self._policy.minimum_source_diversity:
            return False
        if aggregate.cross_image_consistency < self._policy.minimum_cross_image_consistency:
            return False
        if aggregate.average_confidence is None:
            return False
        if aggregate.average_confidence < self._policy.minimum_average_confidence:
            return False
        return True

    @staticmethod
    def _is_observable(aggregate: AggregatedTheme) -> bool:
        try:
            validate_observable_claim(aggregate.key, aggregate.label)
        except ContractValidationError:
            return False
        return True

    @staticmethod
    def _resolve_supporting_evidence(
        aggregate: AggregatedTheme,
        evidence_by_id: dict[str, Evidence],
    ) -> tuple[Evidence, ...]:
        resolved: list[Evidence] = []
        for signal_id in aggregate.supporting_signal_ids:
            item = evidence_by_id.get(signal_id)
            if item is None:
                raise InsightGenerationError(
                    f"aggregate '{aggregate.kind.value}:{aggregate.key}' references unknown evidence id '{signal_id}'"
                )
            if item.type.value != aggregate.kind.value:
                raise InsightGenerationError(
                    f"aggregate '{aggregate.kind.value}:{aggregate.key}' references evidence '{signal_id}' with incompatible type '{item.type.value}'"
                )
            resolved.append(item)
        return tuple(resolved)

    def _support_score(self, aggregate: AggregatedTheme) -> float:
        if aggregate.average_confidence is None:
            raise InsightGenerationError("support score requires aggregate average confidence")
        source_score = min(
            1.0,
            aggregate.source_diversity / self._policy.source_diversity_saturation,
        )
        score = math.fsum(
            (
                aggregate.average_confidence,
                aggregate.image_coverage,
                aggregate.cross_image_consistency,
                source_score,
            )
        ) / 4.0
        return min(1.0, max(0.0, score))

    @staticmethod
    def _explanation(
        aggregate: AggregatedTheme,
        supporting_image_ids: Sequence[str],
    ) -> str:
        image_ids = ", ".join(supporting_image_ids)
        return (
            f"Supported by {aggregate.evidence_count} evidence signals across "
            f"{aggregate.unique_image_count}/{aggregate.image_universe_count} images "
            f"(coverage={aggregate.image_coverage:.6f}); "
            f"source_diversity={aggregate.source_diversity}; "
            f"cross_image_consistency={aggregate.cross_image_consistency:.6f}; "
            f"average_evidence_confidence={aggregate.average_confidence:.6f}; "
            f"max_evidence_confidence={aggregate.max_confidence:.6f}; "
            f"supporting_image_ids=[{image_ids}]."
        )
