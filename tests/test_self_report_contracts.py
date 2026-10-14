import unittest

from src.core.profile_analysis.contracts import ContractValidationError
from src.core.profile_analysis.self_report_contracts import (
    SelfReportedClaim,
    SelfReportedClaimCategory,
    SelfReportedSourceType,
)


class SelfReportedClaimContractTests(unittest.TestCase):
    def test_explicit_self_report_with_consent_is_valid_and_deterministic(self):
        claim = SelfReportedClaim(
            claim_id="claim-1",
            category=SelfReportedClaimCategory.RELIGION,
            value="Muslim",
            source_id="questionnaire-q1",
            source_type=SelfReportedSourceType.QUESTIONNAIRE,
            consent_confirmed=True,
        )
        payload = claim.to_dict()
        self.assertEqual(payload["basis"], "explicit_self_report")
        self.assertEqual(SelfReportedClaim.from_dict(payload), claim)

    def test_claim_without_consent_is_rejected(self):
        with self.assertRaisesRegex(
            ContractValidationError,
            "consent_confirmed must be true",
        ):
            SelfReportedClaim(
                claim_id="claim-1",
                category=SelfReportedClaimCategory.MENTAL_HEALTH,
                value="self-described condition",
                source_id="questionnaire-q2",
                source_type=SelfReportedSourceType.QUESTIONNAIRE,
                consent_confirmed=False,
            )

    def test_non_self_report_basis_is_rejected(self):
        with self.assertRaisesRegex(
            ContractValidationError,
            "basis must be 'explicit_self_report'",
        ):
            SelfReportedClaim(
                claim_id="claim-1",
                category=SelfReportedClaimCategory.PERSONALITY,
                value="introverted",
                source_id="bio-1",
                source_type=SelfReportedSourceType.BIO,
                consent_confirmed=True,
                basis="image_inference",
            )

    def test_image_source_type_is_unrepresentable(self):
        with self.assertRaises(ValueError):
            SelfReportedSourceType("image")

    def test_unknown_or_extra_fields_are_rejected(self):
        payload = {
            "claim_id": "claim-1",
            "category": "personality",
            "value": "introverted",
            "source_id": "bio-1",
            "source_type": "bio",
            "consent_confirmed": True,
            "basis": "explicit_self_report",
            "raw_text": "private original bio",
        }
        with self.assertRaisesRegex(
            ContractValidationError,
            "invalid field set",
        ):
            SelfReportedClaim.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
