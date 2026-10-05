from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.core.profile_analysis.aggregation_contracts import (
    AggregatedTheme,
    AggregationKind,
    AggregationValidationError,
    aggregation_key,
    normalize_aggregation_text,
)
from src.core.profile_analysis.contracts import Evidence, EvidenceType
from src.core.profile_analysis.embedding_contracts import SemanticThemeResult, ThemeSimilarityEvidence
from src.core.profile_analysis.image_contracts import ProcessedImageBatch
from src.infrastructure.di.inject import inject


@dataclass(frozen=True)
class _SupportSignal:
    signal_id: str
    original_image_id: str
    canonical_image_id: str
    kind: AggregationKind
    key: str
    label: str
    source: str
    confidence: float | None


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AggregationValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _json_primitive(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_primitive(item) for item in value]
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(
        _json_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _evidence_concept_label(evidence: Evidence) -> str:
    if evidence.type is EvidenceType.OCR_TEXT:
        if not isinstance(evidence.value, str):
            raise AggregationValidationError("OCR evidence value must be normalized text")
        return normalize_aggregation_text(evidence.value)
    if evidence.label == "factual_caption":
        if not isinstance(evidence.value, str):
            raise AggregationValidationError("factual caption evidence value must be text")
        return normalize_aggregation_text(evidence.value)
    return normalize_aggregation_text(evidence.label)


def _evidence_payload(evidence: Evidence) -> dict[str, Any]:
    return {
        "type": evidence.type.value,
        "label": evidence.label,
        "value": evidence.value,
        "confidence": evidence.confidence,
        "source": evidence.source,
        "metadata": evidence.metadata,
    }


def _semantic_payload(item: ThemeSimilarityEvidence, result: SemanticThemeResult) -> dict[str, Any]:
    return {
        "kind": AggregationKind.SEMANTIC_THEME.value,
        "theme_label": item.theme_label,
        "similarity": item.similarity,
        "threshold": item.threshold,
        "provider": result.provider,
        "provider_version": result.provider_version,
        "model_id": result.model_id,
        "model_version": result.model_version,
        "config_version": result.config_version,
    }


def _semantic_signal_id(item: ThemeSimilarityEvidence, result: SemanticThemeResult) -> str:
    payload = {
        "image_id": item.image_id,
        **_semantic_payload(item, result),
    }
    digest = hashlib.sha256()
    digest.update(b"phase7-semantic-support-v1\0")
    digest.update(_stable_json(payload).encode("utf-8"))
    return f"semantic-{digest.hexdigest()[:24]}"


@inject
class EvidenceAggregationService:
    """Aggregate normalized per-image support without generating profile Insights."""

    def __init__(self):
        pass

    def aggregate(
        self,
        image_batch: ProcessedImageBatch,
        evidence: Sequence[Evidence],
        semantic_result: SemanticThemeResult | None = None,
    ) -> tuple[AggregatedTheme, ...]:
        if not isinstance(image_batch, ProcessedImageBatch):
            raise AggregationValidationError("image_batch must be a ProcessedImageBatch")
        if isinstance(evidence, (str, bytes, bytearray)) or not isinstance(evidence, Sequence):
            raise AggregationValidationError("evidence must be a sequence of Evidence values")
        evidence_items = tuple(evidence)
        if not all(isinstance(item, Evidence) for item in evidence_items):
            raise AggregationValidationError("evidence must contain only Evidence values")
        if semantic_result is not None and not isinstance(semantic_result, SemanticThemeResult):
            raise AggregationValidationError("semantic_result must be SemanticThemeResult or null")

        image_ids = {image.image_id for image in image_batch.images}
        canonical_by_image_id = {image.image_id: image.image_id for image in image_batch.images}
        for duplicate in image_batch.duplicates:
            canonical_by_image_id[duplicate.duplicate_image_id] = duplicate.canonical_image_id

        unique_evidence = self._deduplicate_evidence_ids(evidence_items)
        supports: list[_SupportSignal] = []
        for item in unique_evidence:
            if item.image_id not in image_ids:
                raise AggregationValidationError(
                    f"evidence '{item.id}' references image outside the aggregation batch"
                )
            source = _require_text(item.source, "evidence.source")
            label = _evidence_concept_label(item)
            supports.append(
                _SupportSignal(
                    signal_id=item.id,
                    original_image_id=item.image_id,
                    canonical_image_id=canonical_by_image_id[item.image_id],
                    kind=AggregationKind(item.type.value),
                    key=aggregation_key(label),
                    label=label,
                    source=source,
                    confidence=item.confidence,
                )
            )

        if semantic_result is not None:
            supports.extend(
                self._semantic_supports(
                    semantic_result,
                    image_ids=image_ids,
                    canonical_by_image_id=canonical_by_image_id,
                )
            )

        deduplicated_supports = self._deduplicate_supports(supports)
        return self._aggregate_supports(
            deduplicated_supports,
            image_universe_count=len(image_batch.images),
        )

    @staticmethod
    def _deduplicate_evidence_ids(evidence: Sequence[Evidence]) -> tuple[Evidence, ...]:
        by_id: dict[str, tuple[str, Evidence]] = {}
        for item in evidence:
            payload_signature = _stable_json(_evidence_payload(item))
            existing = by_id.get(item.id)
            if existing is None:
                by_id[item.id] = (payload_signature, item)
                continue
            existing_signature, existing_item = existing
            if existing_item.image_id != item.image_id or existing_signature != payload_signature:
                raise AggregationValidationError(
                    f"duplicate evidence id '{item.id}' has conflicting payload"
                )
        return tuple(by_id[evidence_id][1] for evidence_id in sorted(by_id))

    @staticmethod
    def _semantic_supports(
        semantic_result: SemanticThemeResult,
        *,
        image_ids: set[str],
        canonical_by_image_id: Mapping[str, str],
    ) -> tuple[_SupportSignal, ...]:
        source = _require_text(semantic_result.provider, "semantic_result.provider")
        by_identity: dict[tuple[str, str], tuple[str, ThemeSimilarityEvidence]] = {}
        for item in semantic_result.theme_similarities:
            if item.image_id not in image_ids:
                raise AggregationValidationError(
                    "semantic theme evidence references image outside the aggregation batch"
                )
            label = normalize_aggregation_text(item.theme_label)
            identity = (item.image_id, aggregation_key(label))
            signature = _stable_json(_semantic_payload(item, semantic_result))
            existing = by_identity.get(identity)
            if existing is not None and existing[0] != signature:
                raise AggregationValidationError(
                    "duplicate semantic theme identity has conflicting payload"
                )
            by_identity.setdefault(identity, (signature, item))

        supports: list[_SupportSignal] = []
        for identity in sorted(by_identity):
            item = by_identity[identity][1]
            if not item.matched:
                continue
            label = normalize_aggregation_text(item.theme_label)
            supports.append(
                _SupportSignal(
                    signal_id=_semantic_signal_id(item, semantic_result),
                    original_image_id=item.image_id,
                    canonical_image_id=canonical_by_image_id[item.image_id],
                    kind=AggregationKind.SEMANTIC_THEME,
                    key=aggregation_key(label),
                    label=label,
                    source=source,
                    confidence=None,
                )
            )
        return tuple(supports)

    @staticmethod
    def _deduplicate_supports(supports: Sequence[_SupportSignal]) -> tuple[_SupportSignal, ...]:
        """Prevent exact duplicate uploads from contributing repeated cross-image support.

        Repeated signals within one image remain distinct. For a given canonical-content
        identity, aggregation kind, normalized key, and standardized source, exactly one
        original image supplies support. The Phase 3 canonical image is preferred; when
        it has no such support, the lexically smallest duplicate image is used.
        """

        candidates: dict[tuple[str, str, str, str], dict[str, list[_SupportSignal]]] = {}
        for support in supports:
            identity = (
                support.canonical_image_id,
                support.kind.value,
                support.key,
                support.source,
            )
            candidates.setdefault(identity, {}).setdefault(
                support.original_image_id, []
            ).append(support)

        retained: list[_SupportSignal] = []
        for identity in sorted(candidates):
            canonical_image_id = identity[0]
            by_original_image = candidates[identity]
            representative = (
                canonical_image_id
                if canonical_image_id in by_original_image
                else min(by_original_image)
            )
            retained.extend(by_original_image[representative])

        return tuple(
            sorted(
                retained,
                key=lambda item: (
                    item.kind.value,
                    item.key,
                    item.canonical_image_id,
                    item.source.casefold(),
                    item.source,
                    item.signal_id,
                ),
            )
        )

    @staticmethod
    def _aggregate_supports(
        supports: Sequence[_SupportSignal],
        *,
        image_universe_count: int,
    ) -> tuple[AggregatedTheme, ...]:
        grouped: dict[tuple[AggregationKind, str], list[_SupportSignal]] = {}
        for support in supports:
            grouped.setdefault((support.kind, support.key), []).append(support)

        aggregates: list[AggregatedTheme] = []
        for group_key in sorted(grouped, key=lambda item: (item[0].value, item[1])):
            items = grouped[group_key]
            per_image_count: dict[str, int] = {}
            for item in items:
                per_image_count[item.canonical_image_id] = (
                    per_image_count.get(item.canonical_image_id, 0) + 1
                )
            image_counts = tuple(per_image_count.values())
            consistency = min(image_counts) / max(image_counts)
            confidence_values = sorted(
                item.confidence for item in items if item.confidence is not None
            )
            average_confidence = (
                math.fsum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            )
            max_confidence = max(confidence_values) if confidence_values else None
            labels = sorted(
                {item.label for item in items},
                key=lambda value: (value.casefold(), value),
            )
            sources = tuple(
                sorted(
                    {item.source for item in items},
                    key=lambda value: (value.casefold(), value),
                )
            )
            supporting_image_ids = tuple(sorted(per_image_count))
            supporting_signal_ids = tuple(sorted(item.signal_id for item in items))
            aggregates.append(
                AggregatedTheme(
                    kind=group_key[0],
                    key=group_key[1],
                    label=labels[0],
                    evidence_count=len(items),
                    unique_image_count=len(supporting_image_ids),
                    image_universe_count=image_universe_count,
                    image_coverage=len(supporting_image_ids) / image_universe_count,
                    average_confidence=average_confidence,
                    max_confidence=max_confidence,
                    source_diversity=len(sources),
                    cross_image_consistency=consistency,
                    sources=sources,
                    supporting_image_ids=supporting_image_ids,
                    supporting_signal_ids=supporting_signal_ids,
                )
            )
        return tuple(aggregates)
