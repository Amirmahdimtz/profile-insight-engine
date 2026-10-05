from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.core.profile_analysis.contracts import (
    ContractValidationError,
    ProfileAnalysisResult,
    ProfileAnalysisStatus,
)
from src.core.profile_analysis.insight_contracts import InsightGenerationResult


class ProfileAnalysisLifecycleError(ValueError):
    """Raised when Phase 9 lifecycle data or a transition is invalid."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProfileAnalysisLifecycleError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_image_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProfileAnalysisLifecycleError("image_ids must be a sequence")
    image_ids = tuple(_require_text(item, "image_ids") for item in value)
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ProfileAnalysisLifecycleError("image_ids must contain unique values")
    return image_ids


def _expected_summary(result: ProfileAnalysisResult) -> str:
    if not result.insights:
        return "No supported insights met the configured evidence policy."
    return "Supported insights: " + " | ".join(item.label for item in result.insights)


@dataclass(frozen=True)
class ProfileAnalysisLifecycle:
    analysis_id: str
    status: ProfileAnalysisStatus
    image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _require_text(self.analysis_id, "analysis_id"))
        if not isinstance(self.status, ProfileAnalysisStatus):
            raise ProfileAnalysisLifecycleError("status must be a ProfileAnalysisStatus")
        object.__setattr__(self, "image_ids", _require_image_ids(self.image_ids))


@dataclass(frozen=True)
class CompletedProfileAnalysis:
    result: ProfileAnalysisResult
    insight_policy_version: str
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, ProfileAnalysisResult):
            raise ProfileAnalysisLifecycleError("result must be a ProfileAnalysisResult")
        if self.result.status is not ProfileAnalysisStatus.COMPLETED:
            raise ProfileAnalysisLifecycleError("completed result must have completed status")
        object.__setattr__(
            self,
            "insight_policy_version",
            _require_text(self.insight_policy_version, "insight_policy_version"),
        )
        summary = _require_text(self.summary, "summary")
        if summary != _expected_summary(self.result):
            raise ProfileAnalysisLifecycleError(
                "summary must be derived exactly from the completed result insights"
            )
        object.__setattr__(self, "summary", summary)

    @classmethod
    def from_insight_result(
        cls,
        result: ProfileAnalysisResult,
        insight_result: InsightGenerationResult,
    ) -> "CompletedProfileAnalysis":
        if not isinstance(insight_result, InsightGenerationResult):
            raise ContractValidationError("insight_result must be an InsightGenerationResult")
        return cls(
            result=result,
            insight_policy_version=insight_result.policy_version,
            summary=insight_result.summary,
        )
