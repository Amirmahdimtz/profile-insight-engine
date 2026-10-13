from __future__ import annotations

import asyncio
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import (
    VisionProviderError,
    VisionProviderResult,
    VisionValidationError,
)
from src.core.profile_analysis.vision_provider import IVisionProvider
from src.infrastructure.providers.vision.llama_cpp_vision_provider import (
    LlamaCppVisionProvider,
    _SYSTEM_PROMPT,
    _output_schema,
)
from src.infrastructure.utils.config_reader import ConfigReader


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VisionValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _require_int(value: object, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VisionValidationError(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise VisionValidationError(f"{field_name} must be boolean")
    return value


def _build_prefix_allowed_tokens_fn(
    *,
    guide_type: type[Any],
    index: Any,
    prompt_token_count: int,
    output_vocab_size: int,
):
    if (
        isinstance(prompt_token_count, bool)
        or not isinstance(prompt_token_count, int)
        or prompt_token_count < 0
    ):
        raise VisionProviderError(
            "transformers vision prompt token count is invalid"
        )
    if (
        isinstance(output_vocab_size, bool)
        or not isinstance(output_vocab_size, int)
        or output_vocab_size <= 0
    ):
        raise VisionProviderError(
            "transformers vision output vocabulary size is invalid"
        )

    def allowed_tokens(batch_id: int, input_ids: Any) -> list[int]:
        if batch_id != 0:
            raise VisionProviderError(
                "transformers vision constrained generation only supports batch size one"
            )
        try:
            token_ids = input_ids.tolist()
            if (
                token_ids
                and isinstance(token_ids[0], list)
            ):
                if len(token_ids) != 1:
                    raise VisionProviderError(
                        "transformers vision constrained generation received an unexpected batch"
                    )
                token_ids = token_ids[0]
            generated_ids = token_ids[prompt_token_count:]
            guide = guide_type(index)
            for token_id in generated_ids:
                guide.advance(
                    token_id=int(token_id),
                    return_tokens=False,
                )
            allowed = [
                int(token_id)
                for token_id in guide.get_tokens()
                if 0 <= int(token_id) < output_vocab_size
            ]
        except VisionProviderError:
            raise
        except Exception as exc:
            raise VisionProviderError(
                "transformers vision constraint state update failed"
            ) from exc
        if not allowed:
            raise VisionProviderError(
                "transformers vision constraint produced no valid model tokens"
            )
        return allowed

    return allowed_tokens


@dataclass(frozen=True)
class TransformersVisionSettings:
    model_id: str
    model_revision: str
    config_version: str
    device: str
    local_files_only: bool
    image_longest_edge: int
    do_image_splitting: bool
    max_tokens: int
    max_observations_per_kind: int
    max_label_chars: int
    max_caption_chars: int
    confidence_semantics: str
    structured_backend: str

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "model_revision",
            "config_version",
            "device",
            "confidence_semantics",
            "structured_backend",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(
                    getattr(self, field_name),
                    f"vision.transformers_candidate.{field_name}",
                ),
            )
        object.__setattr__(
            self,
            "local_files_only",
            _require_bool(
                self.local_files_only,
                "vision.transformers_candidate.local_files_only",
            ),
        )
        object.__setattr__(
            self,
            "do_image_splitting",
            _require_bool(
                self.do_image_splitting,
                "vision.transformers_candidate.do_image_splitting",
            ),
        )
        for field_name in (
            "image_longest_edge",
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
                    f"vision.transformers_candidate.{field_name}",
                ),
            )

    @classmethod
    def from_config(
        cls,
        config_reader: ConfigReader,
    ) -> "TransformersVisionSettings":
        prefix = "vision.transformers_candidate"
        return cls(
            model_id=config_reader.get_non_empty_string(f"{prefix}.model_id"),
            model_revision=config_reader.get_non_empty_string(
                f"{prefix}.model_revision"
            ),
            config_version=config_reader.get_non_empty_string(
                f"{prefix}.config_version"
            ),
            device=config_reader.get_non_empty_string(f"{prefix}.device"),
            local_files_only=config_reader.get(f"{prefix}.local_files_only"),
            image_longest_edge=config_reader.get_positive_int(
                f"{prefix}.image_longest_edge"
            ),
            do_image_splitting=config_reader.get(
                f"{prefix}.do_image_splitting"
            ),
            max_tokens=config_reader.get_positive_int(f"{prefix}.max_tokens"),
            max_observations_per_kind=config_reader.get_positive_int(
                f"{prefix}.max_observations_per_kind"
            ),
            max_label_chars=config_reader.get_positive_int(
                f"{prefix}.max_label_chars"
            ),
            max_caption_chars=config_reader.get_positive_int(
                f"{prefix}.max_caption_chars"
            ),
            confidence_semantics=config_reader.get_non_empty_string(
                f"{prefix}.confidence_semantics"
            ),
            structured_backend=config_reader.get_non_empty_string(
                f"{prefix}.structured_backend"
            ),
        )


class TransformersVisionProvider(IVisionProvider):
    """Phase 10 local candidate adapter; not DI-bound until benchmark evidence selects it."""

    def __init__(
        self,
        config_reader: ConfigReader,
        settings: TransformersVisionSettings | None = None,
    ):
        self._settings = settings or TransformersVisionSettings.from_config(
            config_reader
        )
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._constraint_index: Any | None = None
        self._guide_type: type[Any] | None = None
        self._output_vocab_size: int | None = None
        self._provider_version: str | None = None
        self._resolved_model_version: str | None = None

    async def extract_async(self, image: CanonicalImage) -> VisionProviderResult:
        if not isinstance(image, CanonicalImage):
            raise VisionValidationError("image must be a CanonicalImage")
        path = Path(image.storage_reference)
        if not path.is_file():
            raise VisionProviderError(
                "canonical image storage reference does not exist"
            )

        await self._ensure_loaded_async()
        async with self._inference_lock:
            try:
                structured = await asyncio.to_thread(
                    self._generate_structured_sync,
                    path,
                )
                observations, caption = (
                    LlamaCppVisionProvider._parse_structured_output(
                        structured,
                        max_observations_per_kind=(
                            self._settings.max_observations_per_kind
                        ),
                        max_label_chars=self._settings.max_label_chars,
                        max_caption_chars=self._settings.max_caption_chars,
                    )
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
                    "transformers vision returned malformed structured output"
                ) from exc

        if (
            self._provider_version is None
            or self._resolved_model_version is None
        ):
            raise VisionProviderError(
                "transformers vision runtime metadata is unavailable"
            )

        return VisionProviderResult(
            image_id=image.image_id,
            observations=observations,
            caption=caption,
            provider="transformers",
            provider_version=self._provider_version,
            model_id=self._settings.model_id,
            model_version=self._resolved_model_version,
            config_version=self._settings.config_version,
            confidence_semantics=self._settings.confidence_semantics,
        )

    async def _ensure_loaded_async(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                await asyncio.to_thread(self._load_sync)
            except VisionProviderError:
                raise
            except Exception as exc:
                raise VisionProviderError(
                    "transformers vision model initialization failed"
                ) from exc

    def _load_sync(self) -> None:
        try:
            import outlines
            import torch
            import transformers
            import torchvision  # noqa: F401
            import num2words  # noqa: F401
            from huggingface_hub import hf_hub_download
            from outlines.backends import get_json_schema_logits_processor
            from outlines_core import Guide
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise VisionProviderError(
                "transformers vision runtime dependencies are not installed"
            ) from exc

        try:
            if self._settings.local_files_only:
                config_path = Path(
                    hf_hub_download(
                        repo_id=self._settings.model_id,
                        filename="config.json",
                        revision=self._settings.model_revision,
                        local_files_only=True,
                    )
                )
                model_source: str | Path = config_path.parent
                load_revision: str | None = None
            else:
                model_source = self._settings.model_id
                load_revision = self._settings.model_revision

            processor = AutoProcessor.from_pretrained(
                model_source,
                revision=load_revision,
                local_files_only=self._settings.local_files_only,
            )
            model = AutoModelForMultimodalLM.from_pretrained(
                model_source,
                revision=load_revision,
                local_files_only=self._settings.local_files_only,
                dtype=torch.float32,
            )
            config_path = Path(
                hf_hub_download(
                    repo_id=self._settings.model_id,
                    filename="config.json",
                    revision=self._settings.model_revision,
                    local_files_only=True,
                )
            )
            snapshot_path = config_path.parent
            model.to(self._settings.device)
            model.eval()

            output_embeddings = model.get_output_embeddings()
            output_weight = getattr(
                output_embeddings,
                "weight",
                None,
            )
            output_shape = getattr(
                output_weight,
                "shape",
                (),
            )
            if not output_shape:
                raise VisionProviderError(
                    "transformers vision output vocabulary is unavailable"
                )
            output_vocab_size = int(output_shape[0])
            if output_vocab_size <= 0:
                raise VisionProviderError(
                    "transformers vision output vocabulary is invalid"
                )

            outlines_model = outlines.from_transformers(
                model,
                processor.tokenizer,
            )
            schema_text = json.dumps(
                _output_schema(
                    max_observations_per_kind=(
                        self._settings.max_observations_per_kind
                    ),
                    max_label_chars=self._settings.max_label_chars,
                    max_caption_chars=self._settings.max_caption_chars,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            logits_processor = get_json_schema_logits_processor(
                self._settings.structured_backend,
                outlines_model,
                schema_text,
            )
            constraint_index = getattr(
                logits_processor,
                "index",
                None,
            )
            if constraint_index is None:
                raise VisionProviderError(
                    "transformers vision structured backend did not expose a constraint index"
                )
        except VisionProviderError:
            raise
        except Exception as exc:
            raise VisionProviderError(
                "transformers vision model could not be loaded"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model
        self._constraint_index = constraint_index
        self._guide_type = Guide
        self._output_vocab_size = output_vocab_size
        self._provider_version = (
            f"transformers-{transformers.__version__};"
            f"outlines-{importlib.metadata.version('outlines')}"
        )
        self._resolved_model_version = snapshot_path.name

    def _generate_structured_sync(
        self,
        image_path: Path,
    ) -> Mapping[str, Any]:
        if (
            self._torch is None
            or self._processor is None
            or self._model is None
            or self._constraint_index is None
            or self._guide_type is None
            or self._output_vocab_size is None
        ):
            raise VisionProviderError(
                "transformers vision model is not initialized"
            )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract scene, object, activity and topic observations "
                            "from the image. Include one optional factual caption. "
                            "Use short directly observable facts only. "
                            "Return the required structured response."
                        ),
                    },
                    {
                        "type": "image",
                        "path": str(image_path),
                    },
                ],
            },
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=self._processor_kwargs(),
            )
            inputs = inputs.to(self._settings.device)
            pixel_values = inputs.get("pixel_values")
            pixel_shape = getattr(pixel_values, "shape", ())
            if (
                not self._settings.do_image_splitting
                and (
                    len(pixel_shape) < 2
                    or int(pixel_shape[1]) != 1
                )
            ):
                raise VisionProviderError(
                    "transformers vision processor violated single-tile configuration"
                )
            input_token_count = int(inputs["input_ids"].shape[-1])
            prefix_allowed_tokens_fn = (
                _build_prefix_allowed_tokens_fn(
                    guide_type=self._guide_type,
                    index=self._constraint_index,
                    prompt_token_count=input_token_count,
                    output_vocab_size=self._output_vocab_size,
                )
            )
            try:
                with self._torch.inference_mode():
                    output_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self._settings.max_tokens,
                        prefix_allowed_tokens_fn=(
                            prefix_allowed_tokens_fn
                        ),
                    )
            except VisionProviderError:
                raise
            except Exception as exc:
                raise VisionProviderError(
                    "transformers vision constrained generation failed "
                    f"({type(exc).__name__})"
                ) from exc
            generated_ids = output_ids[0][input_token_count:]
            generated_count = int(generated_ids.shape[-1])
            generated_text = self._processor.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            if not generated_text:
                raise VisionProviderError(
                    "transformers vision returned empty structured output"
                )
            try:
                parsed = json.loads(generated_text)
            except json.JSONDecodeError as exc:
                if generated_count >= self._settings.max_tokens:
                    raise VisionProviderError(
                        "transformers vision structured output was truncated "
                        "at the token limit"
                    ) from exc
                raise VisionProviderError(
                    "transformers vision returned invalid structured JSON"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise VisionProviderError(
                    "transformers vision structured output must be an object"
                )
            return parsed
        except VisionProviderError:
            raise
        except Exception as exc:
            raise VisionProviderError(
                "transformers vision inference failed"
            ) from exc

    def _processor_kwargs(self) -> dict[str, object]:
        longest_edge = self._settings.image_longest_edge
        return {
            "images_kwargs": {
                "do_image_splitting": self._settings.do_image_splitting,
                "size": {"longest_edge": longest_edge},
                "max_image_size": {"longest_edge": longest_edge},
            }
        }
