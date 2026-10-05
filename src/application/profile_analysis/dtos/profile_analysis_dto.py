from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.profile_analysis.contracts import ProfileAnalysisRequest, ProfileAnalysisStatus
from src.core.profile_analysis.lifecycle_contracts import CompletedProfileAnalysis, ProfileAnalysisLifecycle


class CreateProfileAnalysisDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str = Field(min_length=1)
    image_ids: list[str] = Field(min_length=1)

    @field_validator("analysis_id")
    @classmethod
    def validate_analysis_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("analysis_id must be trimmed")
        return value

    @field_validator("image_ids")
    @classmethod
    def validate_image_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            if not value or value != value.strip():
                raise ValueError("image_ids must contain non-empty trimmed strings")
            normalized.append(value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("image_ids must contain unique values")
        return normalized

    def to_core(self) -> ProfileAnalysisRequest:
        return ProfileAnalysisRequest(
            analysis_id=self.analysis_id,
            image_ids=tuple(self.image_ids),
        )


class EvidenceDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    image_id: str
    type: str
    label: str
    value: Any
    confidence: float
    source: str
    metadata: dict[str, Any]


class ImageAnalysisResultDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    image_id: str
    evidence: list[EvidenceDto]


class ThemeDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    type: str
    label: str
    confidence: float
    evidence_count: int
    image_coverage: float
    supporting_evidence_ids: list[str]
    supporting_image_ids: list[str]


class ProfileInsightDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    type: str
    label: str
    explanation: str
    confidence: float
    evidence_count: int
    image_coverage: float
    supporting_evidence_ids: list[str]
    supporting_image_ids: list[str]


class ProfileAnalysisStatusDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    status: ProfileAnalysisStatus
    image_ids: list[str]

    @classmethod
    def from_core(cls, value: ProfileAnalysisLifecycle) -> "ProfileAnalysisStatusDto":
        return cls(
            analysis_id=value.analysis_id,
            status=value.status,
            image_ids=list(value.image_ids),
        )


class ProfileAnalysisResultDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    status: ProfileAnalysisStatus
    images: list[ImageAnalysisResultDto]
    themes: list[ThemeDto]
    insights: list[ProfileInsightDto]
    insight_policy_version: str
    summary: str

    @classmethod
    def from_core(cls, value: CompletedProfileAnalysis) -> "ProfileAnalysisResultDto":
        payload = value.result.to_dict()
        payload["insight_policy_version"] = value.insight_policy_version
        payload["summary"] = value.summary
        return cls.model_validate(payload)
