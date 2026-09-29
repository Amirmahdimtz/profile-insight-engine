import unittest

from src.core.profile_analysis.contracts import EvidenceType
from src.core.profile_analysis.image_contracts import CanonicalImage
from src.core.profile_analysis.ocr_contracts import (
    OcrProviderBlock,
    OcrProviderError,
    OcrProviderResult,
    OcrRegion,
    OcrValidationError,
)
from src.core.profile_analysis.ocr_provider import IOcrProvider
from src.core.services.profile_analysis.ocr_evidence_service import (
    OcrEvidenceService,
    infer_script_and_language_hint,
    normalize_ocr_text,
)


class FakeOcrProvider(IOcrProvider):
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def extract_async(self, image):
        if self._error is not None:
            raise self._error
        return self._result


def _canonical(image_id="img_1"):
    return CanonicalImage(
        image_id=image_id,
        content_hash="a" * 64,
        storage_reference="/tmp/canonical.png",
        mime_type="image/png",
        width=100,
        height=50,
        size_bytes=100,
        original_mime_type="image/png",
        original_width=100,
        original_height=50,
        original_size_bytes=100,
        was_resized=False,
        exif_orientation_applied=False,
    )


def _result(image_id="img_1", text="سلام OpenAI 2026", confidence=0.9):
    return OcrProviderResult(
        image_id=image_id,
        blocks=(
            OcrProviderBlock(
                order=0,
                raw_text=text,
                confidence=confidence,
                region=OcrRegion(1, 2, 30, 10),
            ),
        ),
        provider="test_ocr",
        provider_version="1.0",
        model_id="test-model",
        config_version="test-config",
    )


class OcrTextNormalizationTests(unittest.TestCase):
    def test_minimal_normalization_preserves_zwnj_and_normalizes_persian_forms(self):
        raw = "  مي\u200cروم\u00a0  كجا  \r\n OpenAI  "
        self.assertEqual(normalize_ocr_text(raw), "می\u200cروم کجا\nOpenAI")

    def test_script_and_language_hints_cover_persian_english_mixed_and_numeric(self):
        self.assertEqual(infer_script_and_language_hint("سلام"), ("arabic", "fa"))
        self.assertEqual(infer_script_and_language_hint("OpenAI"), ("latin", "en"))
        self.assertEqual(
            infer_script_and_language_hint("سلام OpenAI"),
            ("mixed_arabic_latin", "mixed"),
        )
        self.assertEqual(infer_script_and_language_hint("2026-09-29"), ("numeric", None))


class OcrEvidenceServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_persian_english_mixed_block_becomes_traceable_standard_evidence(self):
        service = OcrEvidenceService(FakeOcrProvider(_result()))
        extraction = await service.extract_async(_canonical())

        self.assertEqual(extraction.image_id, "img_1")
        self.assertEqual(extraction.raw_text, "سلام OpenAI 2026")
        self.assertEqual(extraction.normalized_text, "سلام OpenAI 2026")
        self.assertEqual(len(extraction.evidence), 1)
        evidence = extraction.evidence[0]
        self.assertEqual(evidence.type, EvidenceType.OCR_TEXT)
        self.assertEqual(evidence.image_id, "img_1")
        self.assertEqual(evidence.value, "سلام OpenAI 2026")
        self.assertEqual(evidence.confidence, 0.9)
        self.assertEqual(evidence.source, "test_ocr")
        self.assertEqual(evidence.metadata["raw_text"], "سلام OpenAI 2026")
        self.assertEqual(evidence.metadata["script_hint"], "mixed_arabic_latin")
        self.assertEqual(evidence.metadata["language_hint"], "mixed")
        self.assertEqual(evidence.metadata["region"]["width"], 30)

    async def test_empty_no_text_result_is_valid_and_deterministic(self):
        result = OcrProviderResult(
            image_id="img_1",
            blocks=(),
            provider="test_ocr",
            provider_version="1.0",
            model_id="test-model",
            config_version="test-config",
        )
        service = OcrEvidenceService(FakeOcrProvider(result))
        first = await service.extract_async(_canonical())
        second = await service.extract_async(_canonical())
        self.assertEqual(first, second)
        self.assertEqual(first.blocks, ())
        self.assertEqual(first.evidence, ())
        self.assertEqual(first.raw_text, "")

    async def test_evidence_id_and_serialized_content_are_deterministic_across_reruns(self):
        service = OcrEvidenceService(FakeOcrProvider(_result(text="CPU 3.20GHz")))
        first = await service.extract_async(_canonical())
        second = await service.extract_async(_canonical())
        self.assertEqual(first.evidence[0].id, second.evidence[0].id)
        self.assertEqual(first.evidence[0].to_dict(), second.evidence[0].to_dict())

    async def test_malformed_provider_result_type_is_rejected(self):
        service = OcrEvidenceService(FakeOcrProvider(result={"text": "unsafe raw schema"}))
        with self.assertRaisesRegex(OcrValidationError, "invalid result type"):
            await service.extract_async(_canonical())

    async def test_provider_result_must_reference_same_image(self):
        service = OcrEvidenceService(FakeOcrProvider(_result(image_id="other")))
        with self.assertRaisesRegex(OcrValidationError, "image_id"):
            await service.extract_async(_canonical())

    async def test_provider_failure_is_propagated_as_sanitized_provider_error(self):
        service = OcrEvidenceService(FakeOcrProvider(error=OcrProviderError("OCR provider failed")))
        with self.assertRaisesRegex(OcrProviderError, "OCR provider failed"):
            await service.extract_async(_canonical())

    def test_invalid_confidence_is_rejected_by_core_contract(self):
        for value in (-0.01, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(OcrValidationError):
                OcrProviderBlock(order=0, raw_text="text", confidence=value)


if __name__ == "__main__":
    unittest.main()
