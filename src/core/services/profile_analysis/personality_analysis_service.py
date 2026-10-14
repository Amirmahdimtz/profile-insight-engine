from __future__ import annotations

from collections import defaultdict

from src.core.profile_analysis.personality_contracts import (
    BigFiveTrait,
    PersonalityProfile,
    PersonalityQuestionnaire,
    PersonalityTraitScore,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class PersonalityAnalysisValidationError(ValueError):
    """Raised when a supplied self-report questionnaire cannot be scored safely."""


@inject
class PersonalityAnalysisService:
    def __init__(self, config_reader: ConfigReader):
        self._max_items = config_reader.get_positive_int(
            "profile_analysis.max_personality_questionnaire_items"
        )
        self._max_text_chars = config_reader.get_positive_int(
            "profile_analysis.max_self_reported_claim_chars"
        )

    def analyze(
        self,
        questionnaire: PersonalityQuestionnaire | None,
    ) -> PersonalityProfile | None:
        if questionnaire is None:
            return None
        if not isinstance(questionnaire, PersonalityQuestionnaire):
            raise PersonalityAnalysisValidationError(
                "personality_questionnaire must be a PersonalityQuestionnaire"
            )
        if len(questionnaire.responses) > self._max_items:
            raise PersonalityAnalysisValidationError(
                "personality questionnaire item count exceeds configured limit"
            )
        for field_name, field_value in (
            ("instrument_id", questionnaire.instrument_id),
            ("instrument_version", questionnaire.instrument_version),
        ):
            if len(field_value) > self._max_text_chars:
                raise PersonalityAnalysisValidationError(
                    f"personality questionnaire {field_name} exceeds configured text limit"
                )
        for item in questionnaire.responses:
            if len(item.item_id) > self._max_text_chars:
                raise PersonalityAnalysisValidationError(
                    "personality questionnaire item_id exceeds configured text limit"
                )

        grouped: dict[BigFiveTrait, list[tuple[str, float]]] = defaultdict(list)
        scale_span = questionnaire.scale_max - questionnaire.scale_min

        for item in questionnaire.responses:
            adjusted = item.response_value
            if item.reverse_scored:
                adjusted = (
                    questionnaire.scale_min
                    + questionnaire.scale_max
                    - item.response_value
                )
            grouped[item.trait].append((item.item_id, adjusted))

        trait_scores: list[PersonalityTraitScore] = []
        for trait in BigFiveTrait:
            items = grouped[trait]
            scored_mean = sum(value for _, value in items) / len(items)
            normalized = (
                scored_mean - questionnaire.scale_min
            ) / scale_span
            item_ids = tuple(sorted(item_id for item_id, _ in items))
            trait_scores.append(
                PersonalityTraitScore(
                    trait=trait,
                    normalized_score=round(normalized, 6),
                    scored_mean=round(scored_mean, 6),
                    item_count=len(items),
                    supporting_item_ids=item_ids,
                    explanation=(
                        f"Normalized mean of {len(items)} explicit self-report "
                        f"questionnaire item(s) mapped by instrument "
                        f"'{questionnaire.instrument_id}' to {trait.value}; "
                        "reverse-scored items were transformed before averaging."
                    ),
                )
            )

        summary = "Self-reported Big Five scores: " + " | ".join(
            f"{item.trait.value}={item.normalized_score:.6f}"
            for item in trait_scores
        )
        return PersonalityProfile(
            instrument_id=questionnaire.instrument_id,
            instrument_version=questionnaire.instrument_version,
            model="big_five_self_report_scoring_v1",
            source="explicit_self_report_questionnaire",
            traits=tuple(trait_scores),
            summary=summary,
            disclaimer=(
                "Scores describe explicit self-report questionnaire responses "
                "using the caller-supplied instrument mapping. They are not "
                "derived from images and are not a clinical diagnosis or an "
                "independent validation of the instrument's psychometric quality."
            ),
        )
