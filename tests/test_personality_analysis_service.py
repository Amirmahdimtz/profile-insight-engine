import unittest

from src.core.profile_analysis.contracts import ContractValidationError
from src.core.profile_analysis.personality_contracts import (
    BigFiveTrait,
    PersonalityProfile,
    PersonalityQuestionnaire,
    PersonalityQuestionnaireItemResponse,
)
from src.core.services.profile_analysis.personality_analysis_service import (
    PersonalityAnalysisService,
    PersonalityAnalysisValidationError,
)


class _Config:
    def __init__(self, max_items=50):
        self.max_items = max_items

    def get_positive_int(self, key):
        values = {
            "profile_analysis.max_personality_questionnaire_items": self.max_items,
            "profile_analysis.max_self_reported_claim_chars": 64,
        }
        if key not in values:
            raise AssertionError(f"unexpected config key: {key}")
        return values[key]


def _questionnaire():
    return PersonalityQuestionnaire(
        instrument_id="authorized-big-five-form",
        instrument_version="1.0",
        scale_min=1,
        scale_max=5,
        responses=(
            PersonalityQuestionnaireItemResponse(
                "o-1", BigFiveTrait.OPENNESS, 5
            ),
            PersonalityQuestionnaireItemResponse(
                "o-2", BigFiveTrait.OPENNESS, 1, reverse_scored=True
            ),
            PersonalityQuestionnaireItemResponse(
                "c-1", BigFiveTrait.CONSCIENTIOUSNESS, 3
            ),
            PersonalityQuestionnaireItemResponse(
                "e-1", BigFiveTrait.EXTRAVERSION, 2
            ),
            PersonalityQuestionnaireItemResponse(
                "a-1", BigFiveTrait.AGREEABLENESS, 4
            ),
            PersonalityQuestionnaireItemResponse(
                "n-1", BigFiveTrait.NEUROTICISM, 1
            ),
        ),
        consent_confirmed=True,
    )


class PersonalityContractTests(unittest.TestCase):
    def test_questionnaire_requires_consent_complete_traits_and_valid_scale(self):
        with self.assertRaisesRegex(
            ContractValidationError,
            "consent_confirmed must be true",
        ):
            PersonalityQuestionnaire(
                instrument_id="instrument",
                instrument_version="1",
                scale_min=1,
                scale_max=5,
                responses=_questionnaire().responses,
                consent_confirmed=False,
            )

        with self.assertRaisesRegex(
            ContractValidationError,
            "missing: neuroticism",
        ):
            PersonalityQuestionnaire(
                instrument_id="instrument",
                instrument_version="1",
                scale_min=1,
                scale_max=5,
                responses=tuple(
                    item for item in _questionnaire().responses
                    if item.trait is not BigFiveTrait.NEUROTICISM
                ),
                consent_confirmed=True,
            )

        with self.assertRaisesRegex(
            ContractValidationError,
            "outside the declared scale",
        ):
            responses = list(_questionnaire().responses)
            responses[0] = PersonalityQuestionnaireItemResponse(
                "o-1", BigFiveTrait.OPENNESS, 6
            )
            PersonalityQuestionnaire(
                instrument_id="instrument",
                instrument_version="1",
                scale_min=1,
                scale_max=5,
                responses=tuple(responses),
                consent_confirmed=True,
            )

    def test_profile_round_trip_is_deterministic(self):
        profile = PersonalityAnalysisService(_Config()).analyze(
            _questionnaire()
        )
        self.assertIsNotNone(profile)
        payload = profile.to_dict()
        self.assertEqual(PersonalityProfile.from_dict(payload), profile)


class PersonalityAnalysisServiceTests(unittest.TestCase):
    def test_scoring_is_deterministic_and_reverse_scoring_is_applied(self):
        service = PersonalityAnalysisService(_Config())
        first = service.analyze(_questionnaire())
        second = service.analyze(_questionnaire())

        self.assertEqual(first, second)
        scores = {item.trait: item for item in first.traits}
        self.assertEqual(scores[BigFiveTrait.OPENNESS].normalized_score, 1.0)
        self.assertEqual(scores[BigFiveTrait.OPENNESS].raw_mean, 5.0)
        self.assertEqual(
            scores[BigFiveTrait.CONSCIENTIOUSNESS].normalized_score,
            0.5,
        )
        self.assertEqual(
            scores[BigFiveTrait.EXTRAVERSION].normalized_score,
            0.25,
        )
        self.assertEqual(
            scores[BigFiveTrait.AGREEABLENESS].normalized_score,
            0.75,
        )
        self.assertEqual(
            scores[BigFiveTrait.NEUROTICISM].normalized_score,
            0.0,
        )
        self.assertEqual(
            scores[BigFiveTrait.OPENNESS].supporting_item_ids,
            ("o-1", "o-2"),
        )
        self.assertIn("not a clinical diagnosis", first.disclaimer)

    def test_none_questionnaire_returns_none(self):
        self.assertIsNone(PersonalityAnalysisService(_Config()).analyze(None))

    def test_configured_text_limit_fails_closed(self):
        service = PersonalityAnalysisService(_Config())
        questionnaire = _questionnaire()
        oversized = PersonalityQuestionnaire(
            instrument_id="x" * 65,
            instrument_version=questionnaire.instrument_version,
            scale_min=questionnaire.scale_min,
            scale_max=questionnaire.scale_max,
            responses=questionnaire.responses,
            consent_confirmed=True,
        )
        with self.assertRaisesRegex(
            PersonalityAnalysisValidationError,
            "instrument_id exceeds configured text limit",
        ):
            service.analyze(oversized)

    def test_configured_item_limit_fails_closed(self):
        service = PersonalityAnalysisService(_Config(max_items=5))
        with self.assertRaisesRegex(
            PersonalityAnalysisValidationError,
            "item count exceeds configured limit",
        ):
            service.analyze(_questionnaire())


if __name__ == "__main__":
    unittest.main()
