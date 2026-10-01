from __future__ import annotations

import asyncio
import base64
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import (
    VisionCaption,
    VisionEvidenceKind,
    VisionObservation,
    VisionProviderError,
    VisionProviderResult,
    VisionValidationError,
)
from src.core.profile_analysis.vision_provider import IVisionProvider
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


def _observation_schema(
    *,
    max_items: int,
    max_label_chars: int,
) -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": max_items,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "confidence"],
            "properties": {
                "label": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_label_chars,
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    }


def _output_schema(
    *,
    max_observations_per_kind: int,
    max_label_chars: int,
    max_caption_chars: int,
) -> dict[str, Any]:
    observation_schema = _observation_schema(
        max_items=max_observations_per_kind,
        max_label_chars=max_label_chars,
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["scenes", "objects", "activities", "topics", "caption"],
        "properties": {
            "scenes": observation_schema,
            "objects": observation_schema,
            "activities": observation_schema,
            "topics": observation_schema,
            "caption": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "confidence"],
                        "properties": {
                            "text": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": max_caption_chars,
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                    },
                    {"type": "null"},
                ]
            },
        },
    }

_SYSTEM_PROMPT = (
    "Extract only directly observable visual evidence from the supplied image. "
    "Do not infer religion, political orientation, ethnicity, mental health, sexual orientation, "
    "intelligence, honesty, family relationships, personality, motives, identity, or other hidden traits. "
    "Use short factual labels. The caption must be one short factual sentence and must not add details "
    "that are not directly visible. Confidence is your own uncalibrated self-reported certainty in [0,1]; "
    "it is not a calibrated probability and will be benchmarked before any threshold is frozen."
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VisionValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_int(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VisionValidationError(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisionValidationError(
            f"{field_name} must be a finite number in [{minimum}, {maximum}]"
        )
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise VisionValidationError(
            f"{field_name} must be a finite number in [{minimum}, {maximum}]"
        )
    return result


@dataclass(frozen=True)
class LlamaCppVisionSettings:
    base_url: str
    request_timeout_seconds: int
    model_id: str
    model_version: str
    config_version: str
    max_tokens: int
    max_observations_per_kind: int
    max_label_chars: int
    max_caption_chars: int
    temperature: float
    top_p: float
    seed: int
    confidence_semantics: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_url",
            _require_text(self.base_url, "vision.base_url").rstrip("/"),
        )
        object.__setattr__(
            self,
            "request_timeout_seconds",
            _require_int(
                self.request_timeout_seconds,
                "vision.request_timeout_seconds",
                minimum=1,
            ),
        )
        for field_name in (
            "model_id",
            "model_version",
            "config_version",
            "confidence_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), f"vision.{field_name}"),
            )
        for field_name in (
            "max_tokens",
            "max_observations_per_kind",
            "max_label_chars",
            "max_caption_chars",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_int(
                    getattr(self, field_name),
                    f"vision.{field_name}",
                    minimum=1,
                ),
            )
        object.__setattr__(
            self,
            "temperature",
            _require_float(
                self.temperature,
                "vision.temperature",
                minimum=0.0,
                maximum=2.0,
            ),
        )
        object.__setattr__(
            self,
            "top_p",
            _require_float(
                self.top_p,
                "vision.top_p",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise VisionValidationError("vision.seed must be a non-negative integer")

    @classmethod
    def from_config(cls, config_reader: ConfigReader) -> "LlamaCppVisionSettings":
        provider = config_reader.get_non_empty_string("vision.provider").lower()
        if provider != "llama_cpp":
            raise VisionValidationError(
                "vision.provider must be 'llama_cpp' for LlamaCppVisionProvider"
            )
        return cls(
            base_url=config_reader.get_non_empty_string("vision.base_url"),
            request_timeout_seconds=config_reader.get_positive_int(
                "vision.request_timeout_seconds"
            ),
            model_id=config_reader.get_non_empty_string("vision.model_id"),
            model_version=config_reader.get_non_empty_string("vision.model_version"),
            config_version=config_reader.get_non_empty_string("vision.config_version"),
            max_tokens=config_reader.get_positive_int("vision.max_tokens"),
            max_observations_per_kind=config_reader.get_positive_int(
                "vision.max_observations_per_kind"
            ),
            max_label_chars=config_reader.get_positive_int(
                "vision.max_label_chars"
            ),
            max_caption_chars=config_reader.get_positive_int(
                "vision.max_caption_chars"
            ),
            temperature=config_reader.get("vision.temperature"),
            top_p=config_reader.get("vision.top_p"),
            seed=config_reader.get("vision.seed"),
            confidence_semantics=config_reader.get_non_empty_string(
                "vision.confidence_semantics"
            ),
        )


@inject
class LlamaCppVisionProvider(IVisionProvider):
    def __init__(
        self,
        config_reader: ConfigReader,
        settings: LlamaCppVisionSettings | None = None,
    ):
        self._settings = settings or LlamaCppVisionSettings.from_config(config_reader)
        self._provider_version: str | None = None

    async def extract_async(self, image: CanonicalImage) -> VisionProviderResult:
        if not isinstance(image, CanonicalImage):
            raise VisionValidationError("image must be a CanonicalImage")
        path = Path(image.storage_reference)
        if not path.is_file():
            raise VisionProviderError("canonical image storage reference does not exist")
        try:
            image_bytes = await asyncio.to_thread(path.read_bytes)
            if not image_bytes:
                raise VisionProviderError(
                    "canonical image storage reference is empty"
                )
            provider_version = await asyncio.to_thread(self._get_provider_version)
            payload = await asyncio.to_thread(
                self._request_completion,
                image_bytes,
            )
            structured = self._parse_completion_payload(payload)
            observations, caption = self._parse_structured_output(
                structured,
                max_observations_per_kind=(
                    self._settings.max_observations_per_kind
                ),
                max_label_chars=self._settings.max_label_chars,
                max_caption_chars=self._settings.max_caption_chars,
            )
        except VisionProviderError:
            raise
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            VisionValidationError,
            ValueError,
            TypeError,
        ) as exc:
            raise VisionProviderError(
                "vision provider returned malformed output"
            ) from exc

        return VisionProviderResult(
            image_id=image.image_id,
            observations=observations,
            caption=caption,
            provider="llama_cpp",
            provider_version=provider_version,
            model_id=self._settings.model_id,
            model_version=self._settings.model_version,
            config_version=self._settings.config_version,
            confidence_semantics=self._settings.confidence_semantics,
        )

    def _urlopen_json(
        self,
        request: urllib.request.Request,
    ) -> Mapping[str, Any]:
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._settings.request_timeout_seconds,
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise VisionProviderError("llama.cpp vision request failed") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise VisionProviderError(
                    "llama.cpp vision request timed out"
                ) from exc
            raise VisionProviderError(
                "llama.cpp vision runtime is unavailable"
            ) from exc
        except TimeoutError as exc:
            raise VisionProviderError(
                "llama.cpp vision request timed out"
            ) from exc
        try:
            parsed = json.loads(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisionProviderError(
                "llama.cpp returned invalid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise VisionProviderError(
                "llama.cpp returned invalid response shape"
            )
        return parsed

    def _get_provider_version(self) -> str:
        if self._provider_version is not None:
            return self._provider_version
        request = urllib.request.Request(
            f"{self._settings.base_url}/props",
            headers={"Accept": "application/json"},
            method="GET",
        )
        payload = self._urlopen_json(request)
        build_info = payload.get("build_info")
        modalities = payload.get("modalities")
        if not isinstance(build_info, str) or not build_info.strip():
            raise VisionProviderError(
                "llama.cpp /props response is missing build_info"
            )
        if (
            not isinstance(modalities, Mapping)
            or modalities.get("vision") is not True
        ):
            raise VisionProviderError(
                "llama.cpp runtime does not report vision capability"
            )
        self._provider_version = build_info.strip()
        return self._provider_version

    def _request_completion(
        self,
        image_bytes: bytes,
    ) -> Mapping[str, Any]:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self._settings.model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Return scene, object, activity, topic observations "
                                "and one optional short factual caption using exactly "
                                "the requested JSON schema. Return no more than "
                                f"{self._settings.max_observations_per_kind} items "
                                "per observation category."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    f"{encoded_image}"
                                )
                            },
                        },
                    ],
                },
            ],
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "seed": self._settings.seed,
            "max_tokens": self._settings.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "schema": _output_schema(
                    max_observations_per_kind=(
                        self._settings.max_observations_per_kind
                    ),
                    max_label_chars=self._settings.max_label_chars,
                    max_caption_chars=self._settings.max_caption_chars,
                ),
            },
        }
        request = urllib.request.Request(
            f"{self._settings.base_url}/v1/chat/completions",
            data=json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        return self._urlopen_json(request)

    @staticmethod
    def _parse_completion_payload(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        choices = payload.get("choices")
        if (
            isinstance(choices, (str, bytes, bytearray))
            or not isinstance(choices, Sequence)
            or len(choices) != 1
        ):
            raise VisionProviderError(
                "llama.cpp completion must contain exactly one choice"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise VisionProviderError(
                "llama.cpp completion choice is invalid"
            )
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise VisionProviderError(
                "llama.cpp completion message is invalid"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise VisionProviderError(
                "llama.cpp completion content is empty"
            )
        try:
            structured = json.loads(content)
        except json.JSONDecodeError as exc:
            if finish_reason == "length":
                raise VisionProviderError(
                    "llama.cpp structured content was truncated "
                    "at the token limit"
                ) from exc
            raise VisionProviderError(
                "llama.cpp structured content is not valid JSON"
            ) from exc
        if not isinstance(structured, Mapping):
            raise VisionProviderError(
                "llama.cpp structured content must be an object"
            )
        return structured

    @classmethod
    def _parse_structured_output(
        cls,
        payload: Mapping[str, Any],
        *,
        max_observations_per_kind: int | None = None,
        max_label_chars: int | None = None,
        max_caption_chars: int | None = None,
    ) -> tuple[tuple[VisionObservation, ...], VisionCaption | None]:
        required = {
            "scenes",
            "objects",
            "activities",
            "topics",
            "caption",
        }
        if set(payload) != required:
            raise VisionProviderError(
                "vision structured output has invalid fields"
            )

        observations: list[VisionObservation] = []
        category_map = (
            ("scenes", VisionEvidenceKind.SCENE),
            ("objects", VisionEvidenceKind.OBJECT),
            ("activities", VisionEvidenceKind.ACTIVITY),
            ("topics", VisionEvidenceKind.TOPIC),
        )
        for field_name, kind in category_map:
            values = payload[field_name]
            if (
                isinstance(values, (str, bytes, bytearray))
                or not isinstance(values, Sequence)
            ):
                raise VisionProviderError(
                    f"vision structured field '{field_name}' must be an array"
                )
            if (
                max_observations_per_kind is not None
                and len(values) > max_observations_per_kind
            ):
                raise VisionProviderError(
                    f"vision structured field '{field_name}' "
                    "exceeds configured item limit"
                )
            for item in values:
                if (
                    not isinstance(item, Mapping)
                    or set(item) != {"label", "confidence"}
                ):
                    raise VisionProviderError(
                        "vision observation has invalid fields"
                    )
                label = item["label"]
                if (
                    max_label_chars is not None
                    and isinstance(label, str)
                    and len(label) > max_label_chars
                ):
                    raise VisionProviderError(
                        "vision observation label exceeds configured "
                        "length limit"
                    )
                observations.append(
                    VisionObservation(
                        kind=kind,
                        label=label,
                        confidence=item["confidence"],
                    )
                )

        caption_payload = payload["caption"]
        caption: VisionCaption | None
        if caption_payload is None:
            caption = None
        elif (
            isinstance(caption_payload, Mapping)
            and set(caption_payload) == {"text", "confidence"}
        ):
            caption_text = caption_payload["text"]
            if (
                max_caption_chars is not None
                and isinstance(caption_text, str)
                and len(caption_text) > max_caption_chars
            ):
                raise VisionProviderError(
                    "vision caption exceeds configured length limit"
                )
            caption = VisionCaption(
                text=caption_text,
                confidence=caption_payload["confidence"],
            )
        else:
            raise VisionProviderError(
                "vision caption has invalid fields"
            )

        identities = [
            (item.kind.value, item.label.casefold())
            for item in observations
        ]
        if len(identities) != len(set(identities)):
            raise VisionProviderError(
                "vision structured output contains duplicate observations"
            )
        return tuple(observations), caption
