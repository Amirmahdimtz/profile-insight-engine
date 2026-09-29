from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ContractValidationError(ValueError):
    """Raised when a profile-analysis contract violates Phase 1 invariants."""


class ProfileAnalysisStatus(str, Enum):
    """Stable vocabulary only; lifecycle transitions are intentionally out of scope."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceType(str, Enum):
    """Observable evidence categories allowed by the product analysis boundary."""

    OBJECT = "object"
    SCENE = "scene"
    OCR_TEXT = "ocr_text"
    ACTIVITY = "activity"
    ENVIRONMENT = "environment"
    TOPIC = "topic"
    BRAND = "brand"
    TEAM = "team"
    RELIGIOUS_CONTENT = "religious_content"
    SOCIAL_CONTEXT = "social_context"
    OTHER_OBSERVABLE = "other_observable"


class ThemeType(str, Enum):
    """Cross-image observable theme categories; no sensitive-trait categories exist."""

    RECURRING_CONTENT = "recurring_content"
    VISIBLE_INTEREST = "visible_interest"
    ACTIVITY = "activity"
    ENVIRONMENT = "environment"
    BRAND_TEAM = "brand_team"
    RELIGIOUS_CONTENT = "religious_content"
    SOCIAL_CONTEXT = "social_context"
    TEXTUAL_REFERENCE = "textual_reference"
    OTHER_OBSERVABLE = "other_observable"


class InsightType(str, Enum):
    """Evidence-backed insight categories allowed by the Phase 1 contract."""

    RECURRING_THEME = "recurring_theme"
    VISIBLE_INTEREST = "visible_interest"
    ACTIVITY = "activity"
    ENVIRONMENT = "environment"
    CONTENT_PATTERN = "content_pattern"
    SOCIAL_CONTEXT = "social_context"
    BRAND_REFERENCE = "brand_reference"
    TEAM_REFERENCE = "team_reference"
    TEXTUAL_REFERENCE = "textual_reference"
    RELIGIOUS_CONTENT = "religious_content"


_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# Exact structured claim keys that would encode unsupported personal inferences.
_SENSITIVE_TRAIT_KEYS = {
    "religion",
    "person_religion",
    "political_orientation",
    "person_political_orientation",
    "ethnicity",
    "person_ethnicity",
    "mental_health",
    "person_mental_health",
    "sexual_orientation",
    "person_sexual_orientation",
    "intelligence",
    "person_intelligence",
    "honesty",
    "person_honesty",
    "family_relationship",
    "person_family_relationship",
    "personality_trait",
    "personality_traits",
    "inner_personality",
}

_SENSITIVE_TRAIT_PHRASES = (
    "religion",
    "political orientation",
    "political ideology",
    "ethnicity",
    "mental health",
    "sexual orientation",
    "intelligence",
    "honesty",
    "family relationship",
    "personality trait",
    "inner personality",
    "مذهب",
    "دین",
    "گرایش سیاسی",
    "ایدئولوژی سیاسی",
    "قومیت",
    "سلامت روان",
    "گرایش جنسی",
    "هوش",
    "صداقت",
    "رابطه خانوادگی",
    "نسبت خانوادگی",
    "ویژگی شخصیتی",
    "صفات شخصیتی",
    "شخصیت درونی",
)
_SUBJECT_TERMS = (
    "person",
    "user",
    "subject",
    "individual",
    "owner",
    "profile owner",
    "شخص",
    "فرد",
    "کاربر",
    "سوژه",
    "صاحب پروفایل",
    "صاحب حساب",
)


def _require_non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ContractValidationError(f"{field_name} must be non-empty and trimmed")
    return value


def _require_key(value: Any, field_name: str) -> str:
    value = _require_non_empty_text(value, field_name)
    if not _KEY_PATTERN.fullmatch(value):
        raise ContractValidationError(
            f"{field_name} must use lower_snake_case and start with a letter"
        )
    return value


def _require_probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ContractValidationError(f"{field_name} must be a finite number in [0, 1]")
    return normalized


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _require_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractValidationError(f"{field_name} must be a sequence")
    return tuple(value)


def _require_unique_ids(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = tuple(_require_non_empty_text(value, field_name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ContractValidationError(f"{field_name} must contain unique values")
    return normalized


def _normalize_policy_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = normalized.replace("ي", "ی").replace("ك", "ک")
    normalized = normalized.replace("\u200c", " ").replace("_", " ")
    return " ".join(normalized.split())


def _contains_policy_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def validate_observable_claim(key: str | None, label: str) -> None:
    """Reject unsupported personal sensitive-trait claims in English or Persian."""

    normalized_label = _normalize_policy_text(label)
    normalized_key = key.lower() if key is not None else None

    if normalized_key in _SENSITIVE_TRAIT_KEYS:
        raise ContractValidationError(
            f"unsupported sensitive inference key: {normalized_key}"
        )

    normalized_traits = tuple(_normalize_policy_text(value) for value in _SENSITIVE_TRAIT_PHRASES)
    normalized_subjects = tuple(_normalize_policy_text(value) for value in _SUBJECT_TERMS)

    if normalized_label in normalized_traits:
        raise ContractValidationError(
            f"unsupported sensitive inference label: {label}"
        )

    if any(_contains_policy_phrase(normalized_label, trait) for trait in normalized_traits) and any(
        _contains_policy_phrase(normalized_label, subject) for subject in normalized_subjects
    ):
        raise ContractValidationError(
            f"unsupported sensitive inference label: {label}"
        )

    if normalized_key is not None:
        normalized_key_phrase = _normalize_policy_text(normalized_key)
        if any(
            _contains_policy_phrase(normalized_key_phrase, trait) for trait in normalized_traits
        ) and any(
            _contains_policy_phrase(normalized_key_phrase, subject) for subject in normalized_subjects
        ):
            raise ContractValidationError(
                f"unsupported sensitive inference key: {normalized_key}"
            )


# Backward-compatible private alias used internally by the verified Phase 1 contract.
_validate_observable_claim = validate_observable_claim


def _normalize_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{field_name} must not contain NaN or Infinity")
        return value
    if isinstance(value, _JSON_SCALAR_TYPES):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{field_name} object keys must be strings")
            normalized[key] = _normalize_json_value(item, field_name)
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normalize_json_value(item, field_name) for item in value)
    raise ContractValidationError(f"{field_name} must be JSON-compatible")


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _to_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    return value


def _to_json(value: Any) -> str:
    return json.dumps(
        _to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _expect_mapping(payload: Any, model_name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"{model_name} must be an object")
    return payload


def _expect_keys(
    payload: Mapping[str, Any],
    model_name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    unknown = set(payload) - required - optional
    if unknown:
        raise ContractValidationError(
            f"{model_name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    missing = required - set(payload)
    if missing:
        raise ContractValidationError(
            f"{model_name} is missing required fields: {', '.join(sorted(missing))}"
        )


def _parse_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


@dataclass(frozen=True)
class ProfileAnalysisRequest:
    """Identity-only Phase 1 request contract; raw image transport is intentionally absent."""

    analysis_id: str
    image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _require_non_empty_text(self.analysis_id, "analysis_id"))
        image_ids = _require_sequence(self.image_ids, "image_ids")
        image_ids = _require_unique_ids(image_ids, "image_ids")
        if not image_ids:
            raise ContractValidationError("image_ids must contain at least one image")
        object.__setattr__(self, "image_ids", image_ids)

    def validate_image_count(self, max_images: int) -> None:
        """Validate a caller-supplied configured limit without freezing a Phase 3 tuning value."""

        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0:
            raise ContractValidationError("max_images must be a positive integer")
        if len(self.image_ids) > max_images:
            raise ContractValidationError(
                f"image count {len(self.image_ids)} exceeds configured max_images {max_images}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    def to_json(self) -> str:
        return _to_json(self)

    @classmethod
    def from_dict(cls, payload: Any) -> ProfileAnalysisRequest:
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"analysis_id", "image_ids"})
        return cls(
            analysis_id=payload["analysis_id"],
            image_ids=_require_sequence(payload["image_ids"], "image_ids"),
        )

    @classmethod
    def from_json(cls, payload: str) -> ProfileAnalysisRequest:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("invalid JSON") from exc
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class Evidence:
    """Atomic observable support unit. Raw provider responses are not represented here."""

    id: str
    image_id: str
    type: EvidenceType
    label: str
    value: Any
    confidence: float
    source: str
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty_text(self.id, "evidence.id"))
        object.__setattr__(self, "image_id", _require_non_empty_text(self.image_id, "evidence.image_id"))
        if not isinstance(self.type, EvidenceType):
            raise ContractValidationError("evidence.type must be an EvidenceType")
        label = _require_non_empty_text(self.label, "evidence.label")
        _validate_observable_claim(None, label)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "value", _normalize_json_value(self.value, "evidence.value"))
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "evidence.confidence"))
        object.__setattr__(self, "source", _require_non_empty_text(self.source, "evidence.source"))
        if not isinstance(self.metadata, Mapping):
            raise ContractValidationError("evidence.metadata must be an object")
        object.__setattr__(self, "metadata", _normalize_json_value(self.metadata, "evidence.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    @classmethod
    def from_dict(cls, payload: Any) -> Evidence:
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"id", "image_id", "type", "label", "value", "confidence", "source"},
            {"metadata"},
        )
        return cls(
            id=payload["id"],
            image_id=payload["image_id"],
            type=_parse_enum(EvidenceType, payload["type"], "evidence.type"),
            label=payload["label"],
            value=payload["value"],
            confidence=payload["confidence"],
            source=payload["source"],
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Per-image evidence container tied to one analysis identity."""

    analysis_id: str
    image_id: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _require_non_empty_text(self.analysis_id, "image.analysis_id"))
        object.__setattr__(self, "image_id", _require_non_empty_text(self.image_id, "image.image_id"))
        evidence = _require_sequence(self.evidence, "image.evidence")
        for item in evidence:
            if not isinstance(item, Evidence):
                raise ContractValidationError("image.evidence must contain Evidence values")
            if item.image_id != self.image_id:
                raise ContractValidationError(
                    f"evidence '{item.id}' references image '{item.image_id}' instead of '{self.image_id}'"
                )
        evidence_ids = [item.id for item in evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ContractValidationError("image.evidence ids must be unique")
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    @classmethod
    def from_dict(cls, payload: Any) -> ImageAnalysisResult:
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(payload, cls.__name__, {"analysis_id", "image_id", "evidence"})
        evidence_payload = _require_sequence(payload["evidence"], "image.evidence")
        return cls(
            analysis_id=payload["analysis_id"],
            image_id=payload["image_id"],
            evidence=tuple(Evidence.from_dict(item) for item in evidence_payload),
        )


@dataclass(frozen=True)
class Theme:
    """Observable recurring theme backed by explicit evidence and image references."""

    key: str
    type: ThemeType
    label: str
    confidence: float
    evidence_count: int
    image_coverage: float
    supporting_evidence_ids: tuple[str, ...]
    supporting_image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        key = _require_key(self.key, "theme.key")
        if not isinstance(self.type, ThemeType):
            raise ContractValidationError("theme.type must be a ThemeType")
        label = _require_non_empty_text(self.label, "theme.label")
        _validate_observable_claim(key, label)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "theme.confidence"))
        object.__setattr__(self, "evidence_count", _require_positive_int(self.evidence_count, "theme.evidence_count"))
        object.__setattr__(self, "image_coverage", _require_probability(self.image_coverage, "theme.image_coverage"))
        evidence_ids = _require_unique_ids(
            _require_sequence(self.supporting_evidence_ids, "theme.supporting_evidence_ids"),
            "theme.supporting_evidence_ids",
        )
        image_ids = _require_unique_ids(
            _require_sequence(self.supporting_image_ids, "theme.supporting_image_ids"),
            "theme.supporting_image_ids",
        )
        if not evidence_ids:
            raise ContractValidationError("theme.supporting_evidence_ids must not be empty")
        if not image_ids:
            raise ContractValidationError("theme.supporting_image_ids must not be empty")
        if self.evidence_count != len(evidence_ids):
            raise ContractValidationError(
                "theme.evidence_count must equal the number of supporting_evidence_ids"
            )
        object.__setattr__(self, "supporting_evidence_ids", evidence_ids)
        object.__setattr__(self, "supporting_image_ids", image_ids)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    @classmethod
    def from_dict(cls, payload: Any) -> Theme:
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {
                "key",
                "type",
                "label",
                "confidence",
                "evidence_count",
                "image_coverage",
                "supporting_evidence_ids",
                "supporting_image_ids",
            },
        )
        return cls(
            key=payload["key"],
            type=_parse_enum(ThemeType, payload["type"], "theme.type"),
            label=payload["label"],
            confidence=payload["confidence"],
            evidence_count=payload["evidence_count"],
            image_coverage=payload["image_coverage"],
            supporting_evidence_ids=_require_sequence(
                payload["supporting_evidence_ids"], "theme.supporting_evidence_ids"
            ),
            supporting_image_ids=_require_sequence(
                payload["supporting_image_ids"], "theme.supporting_image_ids"
            ),
        )


@dataclass(frozen=True)
class ProfileInsight:
    """Evidence-backed profile-content insight; unsupported personal traits are unrepresentable."""

    key: str
    type: InsightType
    label: str
    explanation: str
    confidence: float
    evidence_count: int
    image_coverage: float
    supporting_evidence_ids: tuple[str, ...]
    supporting_image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        key = _require_key(self.key, "insight.key")
        if not isinstance(self.type, InsightType):
            raise ContractValidationError("insight.type must be an InsightType")
        label = _require_non_empty_text(self.label, "insight.label")
        explanation = _require_non_empty_text(self.explanation, "insight.explanation")
        _validate_observable_claim(key, label)
        _validate_observable_claim(key, explanation)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "confidence", _require_probability(self.confidence, "insight.confidence"))
        object.__setattr__(self, "evidence_count", _require_positive_int(self.evidence_count, "insight.evidence_count"))
        object.__setattr__(self, "image_coverage", _require_probability(self.image_coverage, "insight.image_coverage"))
        evidence_ids = _require_unique_ids(
            _require_sequence(self.supporting_evidence_ids, "insight.supporting_evidence_ids"),
            "insight.supporting_evidence_ids",
        )
        image_ids = _require_unique_ids(
            _require_sequence(self.supporting_image_ids, "insight.supporting_image_ids"),
            "insight.supporting_image_ids",
        )
        if not evidence_ids:
            raise ContractValidationError("insight.supporting_evidence_ids must not be empty")
        if not image_ids:
            raise ContractValidationError("insight.supporting_image_ids must not be empty")
        if self.evidence_count != len(evidence_ids):
            raise ContractValidationError(
                "insight.evidence_count must equal the number of supporting_evidence_ids"
            )
        object.__setattr__(self, "supporting_evidence_ids", evidence_ids)
        object.__setattr__(self, "supporting_image_ids", image_ids)

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    @classmethod
    def from_dict(cls, payload: Any) -> ProfileInsight:
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {
                "key",
                "type",
                "label",
                "explanation",
                "confidence",
                "evidence_count",
                "image_coverage",
                "supporting_evidence_ids",
                "supporting_image_ids",
            },
        )
        return cls(
            key=payload["key"],
            type=_parse_enum(InsightType, payload["type"], "insight.type"),
            label=payload["label"],
            explanation=payload["explanation"],
            confidence=payload["confidence"],
            evidence_count=payload["evidence_count"],
            image_coverage=payload["image_coverage"],
            supporting_evidence_ids=_require_sequence(
                payload["supporting_evidence_ids"], "insight.supporting_evidence_ids"
            ),
            supporting_image_ids=_require_sequence(
                payload["supporting_image_ids"], "insight.supporting_image_ids"
            ),
        )


@dataclass(frozen=True)
class ProfileAnalysisResult:
    """Validated analysis graph linking images, evidence, themes, and insights."""

    analysis_id: str
    status: ProfileAnalysisStatus
    images: tuple[ImageAnalysisResult, ...]
    themes: tuple[Theme, ...] = ()
    insights: tuple[ProfileInsight, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _require_non_empty_text(self.analysis_id, "result.analysis_id"))
        if not isinstance(self.status, ProfileAnalysisStatus):
            raise ContractValidationError("result.status must be a ProfileAnalysisStatus")

        images = _require_sequence(self.images, "result.images")
        themes = _require_sequence(self.themes, "result.themes")
        insights = _require_sequence(self.insights, "result.insights")

        if not images:
            raise ContractValidationError("result.images must contain at least one image")
        if not all(isinstance(item, ImageAnalysisResult) for item in images):
            raise ContractValidationError("result.images must contain ImageAnalysisResult values")
        if not all(isinstance(item, Theme) for item in themes):
            raise ContractValidationError("result.themes must contain Theme values")
        if not all(isinstance(item, ProfileInsight) for item in insights):
            raise ContractValidationError("result.insights must contain ProfileInsight values")

        object.__setattr__(self, "images", images)
        object.__setattr__(self, "themes", themes)
        object.__setattr__(self, "insights", insights)
        self._validate_graph()

    def _validate_graph(self) -> None:
        image_by_id: dict[str, ImageAnalysisResult] = {}
        evidence_by_id: dict[str, Evidence] = {}

        for image in self.images:
            if image.analysis_id != self.analysis_id:
                raise ContractValidationError(
                    f"image '{image.image_id}' belongs to analysis '{image.analysis_id}', expected '{self.analysis_id}'"
                )
            if image.image_id in image_by_id:
                raise ContractValidationError(f"duplicate image id: {image.image_id}")
            image_by_id[image.image_id] = image
            for evidence in image.evidence:
                if evidence.id in evidence_by_id:
                    raise ContractValidationError(f"duplicate evidence id: {evidence.id}")
                evidence_by_id[evidence.id] = evidence

        theme_keys: set[str] = set()
        for theme in self.themes:
            if theme.key in theme_keys:
                raise ContractValidationError(f"duplicate theme key: {theme.key}")
            theme_keys.add(theme.key)
            self._validate_support_graph(
                owner=f"theme '{theme.key}'",
                evidence_count=theme.evidence_count,
                image_coverage=theme.image_coverage,
                supporting_evidence_ids=theme.supporting_evidence_ids,
                supporting_image_ids=theme.supporting_image_ids,
                image_by_id=image_by_id,
                evidence_by_id=evidence_by_id,
            )

        insight_keys: set[str] = set()
        for insight in self.insights:
            if insight.key in insight_keys:
                raise ContractValidationError(f"duplicate insight key: {insight.key}")
            insight_keys.add(insight.key)
            self._validate_support_graph(
                owner=f"insight '{insight.key}'",
                evidence_count=insight.evidence_count,
                image_coverage=insight.image_coverage,
                supporting_evidence_ids=insight.supporting_evidence_ids,
                supporting_image_ids=insight.supporting_image_ids,
                image_by_id=image_by_id,
                evidence_by_id=evidence_by_id,
            )

    def _validate_support_graph(
        self,
        *,
        owner: str,
        evidence_count: int,
        image_coverage: float,
        supporting_evidence_ids: tuple[str, ...],
        supporting_image_ids: tuple[str, ...],
        image_by_id: Mapping[str, ImageAnalysisResult],
        evidence_by_id: Mapping[str, Evidence],
    ) -> None:
        unknown_evidence = [
            evidence_id
            for evidence_id in supporting_evidence_ids
            if evidence_id not in evidence_by_id
        ]
        if unknown_evidence:
            raise ContractValidationError(
                f"{owner} references unknown evidence ids: {', '.join(sorted(unknown_evidence))}"
            )

        unknown_images = [image_id for image_id in supporting_image_ids if image_id not in image_by_id]
        if unknown_images:
            raise ContractValidationError(
                f"{owner} references unknown image ids: {', '.join(sorted(unknown_images))}"
            )

        if evidence_count != len(supporting_evidence_ids):
            raise ContractValidationError(
                f"{owner} evidence_count does not match supporting_evidence_ids"
            )

        evidence_image_ids = {
            evidence_by_id[evidence_id].image_id for evidence_id in supporting_evidence_ids
        }
        if set(supporting_image_ids) != evidence_image_ids:
            raise ContractValidationError(
                f"{owner} supporting_image_ids must exactly match images referenced by supporting evidence"
            )

        expected_coverage = len(evidence_image_ids) / len(image_by_id)
        if not math.isclose(image_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            raise ContractValidationError(
                f"{owner} image_coverage must equal supporting image count / total image count"
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_primitive(self)

    def to_json(self) -> str:
        return _to_json(self)

    @classmethod
    def from_dict(cls, payload: Any) -> ProfileAnalysisResult:
        payload = _expect_mapping(payload, cls.__name__)
        _expect_keys(
            payload,
            cls.__name__,
            {"analysis_id", "status", "images", "themes", "insights"},
        )
        images = _require_sequence(payload["images"], "result.images")
        themes = _require_sequence(payload["themes"], "result.themes")
        insights = _require_sequence(payload["insights"], "result.insights")
        return cls(
            analysis_id=payload["analysis_id"],
            status=_parse_enum(ProfileAnalysisStatus, payload["status"], "result.status"),
            images=tuple(ImageAnalysisResult.from_dict(item) for item in images),
            themes=tuple(Theme.from_dict(item) for item in themes),
            insights=tuple(ProfileInsight.from_dict(item) for item in insights),
        )

    @classmethod
    def from_json(cls, payload: str) -> ProfileAnalysisResult:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("invalid JSON") from exc
        return cls.from_dict(parsed)
