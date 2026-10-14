from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from src.core.profile_analysis.contracts import ContractValidationError


_PERSONALITY_BASIS = "explicit_self_report_questionnaire"
_PERSONALITY_MODEL = "big_five_self_report_scoring_v1"
_PERSONALITY_SOURCE = "explicit_self_report_questionnaire"


class BigFiveTrait(str, Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractValidationError(
            f"{field_name} must be a non-empty trimmed string"
        )
    return value


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True)
class PersonalityQuestionnaireItemResponse:
    item_id: str
    trait: BigFiveTrait
    response_value: float
    reverse_scored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "item_id",
            _require_text(self.item_id, "personality_questionnaire.item_id"),
        )
        if not isinstance(self.trait, BigFiveTrait):
            raise ContractValidationError(
                "personality_questionnaire.trait must be a BigFiveTrait"
            )
        object.__setattr__(
            self,
            "response_value",
            _require_finite_number(
                self.response_value,
                "personality_questionnaire.response_value",
            ),
        )
        if not isinstance(self.reverse_scored, bool):
            raise ContractValidationError(
                "personality_questionnaire.reverse_scored must be boolean"
            )


@dataclass(frozen=True)
class PersonalityQuestionnaire:
    instrument_id: str
    instrument_version: str
    scale_min: float
    scale_max: float
    responses: tuple[PersonalityQuestionnaireItemResponse, ...]
    consent_confirmed: bool
    basis: str = _PERSONALITY_BASIS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _require_text(
                self.instrument_id,
                "personality_questionnaire.instrument_id",
            ),
        )
        object.__setattr__(
            self,
            "instrument_version",
            _require_text(
                self.instrument_version,
                "personality_questionnaire.instrument_version",
            ),
        )
        scale_min = _require_finite_number(
            self.scale_min,
            "personality_questionnaire.scale_min",
        )
        scale_max = _require_finite_number(
            self.scale_max,
            "personality_questionnaire.scale_max",
        )
        if scale_min >= scale_max:
            raise ContractValidationError(
                "personality_questionnaire.scale_min must be less than scale_max"
            )
        object.__setattr__(self, "scale_min", scale_min)
        object.__setattr__(self, "scale_max", scale_max)

        if (
            isinstance(self.responses, (str, bytes, bytearray))
            or not isinstance(self.responses, Sequence)
        ):
            raise ContractValidationError(
                "personality_questionnaire.responses must be a sequence"
            )
        responses = tuple(self.responses)
        if not responses:
            raise ContractValidationError(
                "personality_questionnaire.responses cannot be empty"
            )
        if not all(
            isinstance(item, PersonalityQuestionnaireItemResponse)
            for item in responses
        ):
            raise ContractValidationError(
                "personality_questionnaire.responses must contain item responses"
            )
        item_ids = [item.item_id for item in responses]
        if len(item_ids) != len(set(item_ids)):
            raise ContractValidationError(
                "personality_questionnaire.item_id values must be unique"
            )
        for item in responses:
            if not scale_min <= item.response_value <= scale_max:
                raise ContractValidationError(
                    f"personality questionnaire response '{item.item_id}' "
                    "is outside the declared scale"
                )
        present_traits = {item.trait for item in responses}
        missing_traits = [
            trait.value for trait in BigFiveTrait if trait not in present_traits
        ]
        if missing_traits:
            raise ContractValidationError(
                "personality questionnaire requires at least one item for every "
                f"Big Five trait; missing: {', '.join(missing_traits)}"
            )
        if self.consent_confirmed is not True:
            raise ContractValidationError(
                "personality_questionnaire.consent_confirmed must be true"
            )
        if self.basis != _PERSONALITY_BASIS:
            raise ContractValidationError(
                "personality_questionnaire.basis must be "
                "'explicit_self_report_questionnaire'"
            )
        object.__setattr__(
            self,
            "responses",
            tuple(sorted(responses, key=lambda item: item.item_id)),
        )


@dataclass(frozen=True)
class PersonalityTraitScore:
    trait: BigFiveTrait
    normalized_score: float
    scored_mean: float
    item_count: int
    supporting_item_ids: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(self.trait, BigFiveTrait):
            raise ContractValidationError("personality trait must be a BigFiveTrait")
        normalized = _require_finite_number(
            self.normalized_score,
            "personality_trait.normalized_score",
        )
        if not 0.0 <= normalized <= 1.0:
            raise ContractValidationError(
                "personality_trait.normalized_score must be within [0, 1]"
            )
        object.__setattr__(self, "normalized_score", normalized)
        object.__setattr__(
            self,
            "scored_mean",
            _require_finite_number(
                self.scored_mean,
                "personality_trait.scored_mean",
            ),
        )
        if isinstance(self.item_count, bool) or not isinstance(self.item_count, int):
            raise ContractValidationError(
                "personality_trait.item_count must be an integer"
            )
        if self.item_count <= 0:
            raise ContractValidationError(
                "personality_trait.item_count must be positive"
            )
        item_ids = tuple(
            _require_text(item, "personality_trait.supporting_item_ids")
            for item in self.supporting_item_ids
        )
        if len(item_ids) != self.item_count or len(item_ids) != len(set(item_ids)):
            raise ContractValidationError(
                "personality_trait supporting_item_ids must be unique and "
                "match item_count"
            )
        object.__setattr__(
            self,
            "supporting_item_ids",
            tuple(sorted(item_ids)),
        )
        object.__setattr__(
            self,
            "explanation",
            _require_text(self.explanation, "personality_trait.explanation"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trait": self.trait.value,
            "normalized_score": self.normalized_score,
            "scored_mean": self.scored_mean,
            "item_count": self.item_count,
            "supporting_item_ids": list(self.supporting_item_ids),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PersonalityTraitScore":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("PersonalityTraitScore must be an object")
        required = {
            "trait",
            "normalized_score",
            "scored_mean",
            "item_count",
            "supporting_item_ids",
            "explanation",
        }
        if set(payload) != required:
            raise ContractValidationError(
                "PersonalityTraitScore has an invalid field set"
            )
        try:
            trait = BigFiveTrait(payload["trait"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                "PersonalityTraitScore contains an unsupported trait"
            ) from exc
        item_ids = payload["supporting_item_ids"]
        if (
            isinstance(item_ids, (str, bytes, bytearray))
            or not isinstance(item_ids, Sequence)
        ):
            raise ContractValidationError(
                "PersonalityTraitScore supporting_item_ids must be a sequence"
            )
        return cls(
            trait=trait,
            normalized_score=payload["normalized_score"],
            scored_mean=payload["scored_mean"],
            item_count=payload["item_count"],
            supporting_item_ids=tuple(item_ids),
            explanation=payload["explanation"],
        )


@dataclass(frozen=True)
class PersonalityProfile:
    instrument_id: str
    instrument_version: str
    model: str
    source: str
    traits: tuple[PersonalityTraitScore, ...]
    summary: str
    disclaimer: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _require_text(self.instrument_id, "personality_profile.instrument_id"),
        )
        object.__setattr__(
            self,
            "instrument_version",
            _require_text(
                self.instrument_version,
                "personality_profile.instrument_version",
            ),
        )
        if self.model != _PERSONALITY_MODEL:
            raise ContractValidationError(
                f"personality_profile.model must be '{_PERSONALITY_MODEL}'"
            )
        if self.source != _PERSONALITY_SOURCE:
            raise ContractValidationError(
                f"personality_profile.source must be '{_PERSONALITY_SOURCE}'"
            )
        traits = tuple(self.traits)
        if len(traits) != len(BigFiveTrait) or not all(
            isinstance(item, PersonalityTraitScore) for item in traits
        ):
            raise ContractValidationError(
                "personality_profile must contain exactly five trait scores"
            )
        trait_values = [item.trait for item in traits]
        if set(trait_values) != set(BigFiveTrait):
            raise ContractValidationError(
                "personality_profile must contain each Big Five trait exactly once"
            )
        object.__setattr__(
            self,
            "traits",
            tuple(
                sorted(
                    traits,
                    key=lambda item: list(BigFiveTrait).index(item.trait),
                )
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _require_text(self.summary, "personality_profile.summary"),
        )
        object.__setattr__(
            self,
            "disclaimer",
            _require_text(self.disclaimer, "personality_profile.disclaimer"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "instrument_version": self.instrument_version,
            "model": self.model,
            "source": self.source,
            "traits": [item.to_dict() for item in self.traits],
            "summary": self.summary,
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PersonalityProfile":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("PersonalityProfile must be an object")
        required = {
            "instrument_id",
            "instrument_version",
            "model",
            "source",
            "traits",
            "summary",
            "disclaimer",
        }
        if set(payload) != required:
            raise ContractValidationError(
                "PersonalityProfile has an invalid field set"
            )
        traits = payload["traits"]
        if (
            isinstance(traits, (str, bytes, bytearray))
            or not isinstance(traits, Sequence)
        ):
            raise ContractValidationError(
                "PersonalityProfile traits must be a sequence"
            )
        return cls(
            instrument_id=payload["instrument_id"],
            instrument_version=payload["instrument_version"],
            model=payload["model"],
            source=payload["source"],
            traits=tuple(PersonalityTraitScore.from_dict(item) for item in traits),
            summary=payload["summary"],
            disclaimer=payload["disclaimer"],
        )
