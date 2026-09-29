import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.vision_contracts import VisionProviderError
from src.infrastructure.providers.vision.llama_cpp_vision_provider import (
    LlamaCppVisionProvider,
    LlamaCppVisionSettings,
)


def _settings() -> LlamaCppVisionSettings:
    return LlamaCppVisionSettings(
        base_url="http://127.0.0.1:8080",
        request_timeout_seconds=10,
        model_id="candidate",
        model_version="v1",
        config_version="phase5-v1",
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        confidence_semantics="model_self_reported_uncalibrated",
    )


class LlamaCppVisionProviderParsingTests(unittest.TestCase):
    def test_valid_structured_output_parses(self):
        observations, caption = (
            LlamaCppVisionProvider._parse_structured_output(
                {
                    "scenes": [
                        {"label": "park", "confidence": 0.8}
                    ],
                    "objects": [
                        {"label": "dog", "confidence": 0.9}
                    ],
                    "activities": [
                        {"label": "walking", "confidence": 0.7}
                    ],
                    "topics": [
                        {"label": "pets", "confidence": 0.6}
                    ],
                    "caption": {
                        "text": "A dog is walking in a park.",
                        "confidence": 0.75,
                    },
                }
            )
        )
        self.assertEqual(len(observations), 4)
        self.assertEqual(
            caption.text,
            "A dog is walking in a park.",
        )

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(
            VisionProviderError,
            "invalid fields",
        ):
            LlamaCppVisionProvider._parse_structured_output(
                {
                    "scenes": [],
                    "objects": [],
                    "activities": [],
                    "topics": [],
                    "caption": None,
                    "extra": "no",
                }
            )

    def test_invalid_confidence_is_rejected_as_malformed(self):
        with self.assertRaises(Exception):
            LlamaCppVisionProvider._parse_structured_output(
                {
                    "scenes": [
                        {"label": "park", "confidence": 2.0}
                    ],
                    "objects": [],
                    "activities": [],
                    "topics": [],
                    "caption": None,
                }
            )

    def test_duplicate_entries_are_rejected(self):
        with self.assertRaisesRegex(
            VisionProviderError,
            "duplicate",
        ):
            LlamaCppVisionProvider._parse_structured_output(
                {
                    "scenes": [],
                    "objects": [
                        {"label": "Dog", "confidence": 0.9},
                        {"label": "dog", "confidence": 0.8},
                    ],
                    "activities": [],
                    "topics": [],
                    "caption": None,
                }
            )

    def test_request_uses_llama_cpp_json_schema_wrapper(self):
        provider = LlamaCppVisionProvider(
            object(),
            _settings(),
        )
        captured = {}

        def capture(request):
            captured.update(
                json.loads(
                    request.data.decode("utf-8")
                )
            )
            return {"choices": []}

        with patch.object(
            provider,
            "_urlopen_json",
            side_effect=capture,
        ):
            provider._request_completion(b"png")
        response_format = captured["response_format"]
        self.assertEqual(
            response_format["type"],
            "json_schema",
        )
        self.assertEqual(
            response_format["json_schema"]["name"],
            "phase5_visual_evidence",
        )
        self.assertTrue(
            response_format["json_schema"]["strict"]
        )
        self.assertEqual(
            response_format["json_schema"]["schema"],
            __import__(
                "src.infrastructure.providers.vision.llama_cpp_vision_provider",
                fromlist=["_OUTPUT_SCHEMA"],
            )._OUTPUT_SCHEMA,
        )

    def test_completion_payload_requires_single_json_choice(self):
        with self.assertRaises(VisionProviderError):
            LlamaCppVisionProvider._parse_completion_payload(
                {"choices": []}
            )
        with self.assertRaises(VisionProviderError):
            LlamaCppVisionProvider._parse_completion_payload(
                {
                    "choices": [
                        {"message": {"content": "not json"}}
                    ]
                }
            )

    def test_runtime_timeout_is_sanitized(self):
        provider = LlamaCppVisionProvider(
            object(),
            _settings(),
        )
        request = urllib.request.Request(
            "http://127.0.0.1:8080/props"
        )
        with patch(
            "src.infrastructure.providers.vision.llama_cpp_vision_provider.urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                TimeoutError()
            ),
        ):
            with self.assertRaisesRegex(
                VisionProviderError,
                "timed out",
            ):
                provider._urlopen_json(request)

    def test_runtime_http_failure_is_sanitized(self):
        provider = LlamaCppVisionProvider(
            object(),
            _settings(),
        )
        request = urllib.request.Request(
            "http://127.0.0.1:8080/props"
        )
        error = urllib.error.HTTPError(
            request.full_url,
            500,
            "sensitive raw runtime message",
            hdrs=None,
            fp=None,
        )
        with patch(
            "src.infrastructure.providers.vision.llama_cpp_vision_provider.urllib.request.urlopen",
            side_effect=error,
        ):
            with self.assertRaisesRegex(
                VisionProviderError,
                "vision request failed",
            ) as context:
                provider._urlopen_json(request)
        self.assertNotIn(
            "sensitive raw runtime message",
            str(context.exception),
        )


class LlamaCppVisionProviderBehaviorTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_provider_returns_metadata_without_exposing_raw_response(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.png"
            path.write_bytes(
                b"png-bytes-for-mocked-request"
            )
            image = CanonicalImage(
                image_id="img-1",
                content_hash="a" * 64,
                storage_reference=str(path),
                mime_type="image/png",
                width=10,
                height=10,
                size_bytes=10,
                original_mime_type="image/png",
                original_width=10,
                original_height=10,
                original_size_bytes=10,
                was_resized=False,
                exif_orientation_applied=False,
            )
            provider = LlamaCppVisionProvider(
                object(),
                _settings(),
            )
            completion = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "scenes": [],
                                    "objects": [
                                        {
                                            "label": "dog",
                                            "confidence": 0.9,
                                        }
                                    ],
                                    "activities": [],
                                    "topics": [],
                                    "caption": None,
                                }
                            )
                        }
                    }
                ]
            }
            with (
                patch.object(
                    provider,
                    "_get_provider_version",
                    return_value="b123",
                ),
                patch.object(
                    provider,
                    "_request_completion",
                    return_value=completion,
                ),
            ):
                result = await provider.extract_async(image)
            self.assertEqual(result.provider, "llama_cpp")
            self.assertEqual(
                result.provider_version,
                "b123",
            )
            self.assertEqual(result.model_id, "candidate")
            self.assertEqual(result.model_version, "v1")
            self.assertEqual(len(result.observations), 1)

    async def test_missing_canonical_file_has_sanitized_error(
        self,
    ):
        image = CanonicalImage(
            image_id="img-1",
            content_hash="a" * 64,
            storage_reference=(
                "Z:/definitely-missing-phase5-image.png"
            ),
            mime_type="image/png",
            width=10,
            height=10,
            size_bytes=10,
            original_mime_type="image/png",
            original_width=10,
            original_height=10,
            original_size_bytes=10,
            was_resized=False,
            exif_orientation_applied=False,
        )
        provider = LlamaCppVisionProvider(
            object(),
            _settings(),
        )
        with self.assertRaisesRegex(
            VisionProviderError,
            "storage reference",
        ):
            await provider.extract_async(image)


if __name__ == "__main__":
    unittest.main()
