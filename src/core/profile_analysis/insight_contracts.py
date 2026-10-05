from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from src.core.profile_analysis.contracts import ProfileInsight


class InsightGenerationError(ValueError):
    """Raised when Phase 8 insight-generation input or output is invalid."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InsightGenerationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_insights(value: object) -> tuple[ProfileInsight, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise InsightGenerationError("insights must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, ProfileInsight) for item in result):
        raise InsightGenerationError("insights must contain ProfileInsight values")
    keys = [item.key for item in result]
    if len(keys) != len(set(keys)):
        raise InsightGenerationError("insights must have unique keys")
    return result


@dataclass(frozen=True)
class InsightGenerationResult:
    """Phase 8 output kept separate from Phase 9 API/persistence contracts."""

    policy_version: str
    insights: tuple[ProfileInsight, ...]
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_version",
            _require_text(self.policy_version, "insight_result.policy_version"),
        )
        object.__setattr__(self, "insights", _require_insights(self.insights))
        object.__setattr__(
            self,
            "summary",
            _require_text(self.summary, "insight_result.summary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "insights": [item.to_dict() for item in self.insights],
            "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
