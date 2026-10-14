from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from src.core.profile_analysis.contracts import ContractValidationError


_SELF_REPORT_BASIS = "explicit_self_report"


class SelfReportedClaimCategory(str, Enum):
    RELIGION = "religion"
    POLITICAL_ORIENTATION = "political_orientation"
    ETHNICITY = "ethnicity"
    MENTAL_HEALTH = "mental_health"
    SEXUAL_ORIENTATION = "sexual_orientation"
    PERSONALITY = "personality"
    INTELLIGENCE = "intelligence"
    OTHER_SELF_DESCRIPTION = "other_self_description"


class SelfReportedSourceType(str, Enum):
    QUESTIONNAIRE = "questionnaire"
    BIO = "bio"
    POST = "post"
    DIRECT_STATEMENT = "direct_statement"
    OTHER = "other"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractValidationError(
            f"{field_name} must be a non-empty trimmed string"
        )
    return value


@dataclass(frozen=True)
class SelfReportedClaim:
    """Explicit user-provided profile claim; never derived from image/OCR/Vision evidence."""

    claim_id: str
    category: SelfReportedClaimCategory
    value: str
    source_id: str
    source_type: SelfReportedSourceType
    consent_confirmed: bool
    basis: str = _SELF_REPORT_BASIS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claim_id",
            _require_text(self.claim_id, "self_reported_claim.claim_id"),
        )
        if not isinstance(self.category, SelfReportedClaimCategory):
            raise ContractValidationError(
                "self_reported_claim.category must be a SelfReportedClaimCategory"
            )
        object.__setattr__(
            self,
            "value",
            _require_text(self.value, "self_reported_claim.value"),
        )
        object.__setattr__(
            self,
            "source_id",
            _require_text(self.source_id, "self_reported_claim.source_id"),
        )
        if not isinstance(self.source_type, SelfReportedSourceType):
            raise ContractValidationError(
                "self_reported_claim.source_type must be a SelfReportedSourceType"
            )
        if self.consent_confirmed is not True:
            raise ContractValidationError(
                "self_reported_claim.consent_confirmed must be true"
            )
        if self.basis != _SELF_REPORT_BASIS:
            raise ContractValidationError(
                "self_reported_claim.basis must be 'explicit_self_report'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "category": self.category.value,
            "value": self.value,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "consent_confirmed": self.consent_confirmed,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SelfReportedClaim":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("SelfReportedClaim must be an object")
        required = {
            "claim_id",
            "category",
            "value",
            "source_id",
            "source_type",
            "consent_confirmed",
            "basis",
        }
        if set(payload) != required:
            raise ContractValidationError(
                "SelfReportedClaim has an invalid field set"
            )
        try:
            category = SelfReportedClaimCategory(payload["category"])
            source_type = SelfReportedSourceType(payload["source_type"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                "SelfReportedClaim contains an unsupported enum value"
            ) from exc
        return cls(
            claim_id=payload["claim_id"],
            category=category,
            value=payload["value"],
            source_id=payload["source_id"],
            source_type=source_type,
            consent_confirmed=payload["consent_confirmed"],
            basis=payload["basis"],
        )
