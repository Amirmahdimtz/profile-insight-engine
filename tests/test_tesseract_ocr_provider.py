import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.profile_analysis.ocr_contracts import OcrProviderError
from src.infrastructure.providers.ocr.tesseract_ocr_provider import (
    TesseractOcrProvider,
    TesseractOcrSettings,
)


def _provider() -> TesseractOcrProvider:
    settings = TesseractOcrSettings(
        executable="tesseract",
        languages=("fas", "eng"),
        page_segmentation_mode=1,
        timeout_seconds=30,
        model_id="test-model",
        config_version="test-config",
    )
    return TesseractOcrProvider(config_reader=None, settings=settings)


class TesseractTsvParsingTests(unittest.TestCase):
    def test_valid_tsv_groups_words_into_deterministic_line_blocks(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t90\tسلام\n"
            "5\t1\t1\t1\t1\t2\t45\t20\t40\t10\t80\tOpenAI\n"
            "5\t1\t1\t1\t2\t1\t10\t40\t20\t10\t100\t2026\n"
        )
        blocks = TesseractOcrProvider._parse_tsv(tsv)
        self.assertEqual([item.raw_text for item in blocks], ["سلام OpenAI", "2026"])
        self.assertAlmostEqual(blocks[0].confidence, 0.85)
        self.assertEqual(
            blocks[0].region.to_dict(),
            {"x": 10, "y": 20, "width": 75, "height": 10},
        )
        self.assertEqual([item.order for item in blocks], [0, 1])

    def test_empty_tsv_is_valid_no_text_output(self):
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        self.assertEqual(TesseractOcrProvider._parse_tsv(tsv), ())

    def test_missing_columns_are_rejected(self):
        with self.assertRaisesRegex(OcrProviderError, "missing required columns"):
            TesseractOcrProvider._parse_tsv("level\ttext\n5\thello\n")

    def test_invalid_confidence_is_rejected(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t101\thello\n"
        )
        with self.assertRaisesRegex(OcrProviderError, "word row is invalid"):
            TesseractOcrProvider._parse_tsv(tsv)

    def test_negative_structural_confidence_row_without_text_is_ignored(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t\n"
        )
        self.assertEqual(TesseractOcrProvider._parse_tsv(tsv), ())


class TesseractFailureMappingTests(unittest.TestCase):
    def test_missing_executable_is_sanitized(self):
        provider = _provider()
        with patch(
            "src.infrastructure.providers.ocr.tesseract_ocr_provider.subprocess.run",
            side_effect=FileNotFoundError("private executable path"),
        ):
            with self.assertRaisesRegex(OcrProviderError, "executable was not found") as ctx:
                provider._run_tesseract(Path("canonical.png"))
        self.assertNotIn("private executable path", str(ctx.exception))

    def test_timeout_is_sanitized(self):
        provider = _provider()
        with patch(
            "src.infrastructure.providers.ocr.tesseract_ocr_provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["tesseract"], timeout=30),
        ):
            with self.assertRaisesRegex(OcrProviderError, "timed out"):
                provider._run_tesseract(Path("canonical.png"))

    def test_nonzero_exit_is_sanitized_without_stderr_leak(self):
        provider = _provider()
        completed = subprocess.CompletedProcess(
            args=["tesseract"],
            returncode=1,
            stdout=b"",
            stderr=b"SENSITIVE_OCR_CONTENT",
        )
        with patch(
            "src.infrastructure.providers.ocr.tesseract_ocr_provider.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(OcrProviderError, "^Tesseract OCR failed$") as ctx:
                provider._run_tesseract(Path("canonical.png"))
        self.assertNotIn("SENSITIVE_OCR_CONTENT", str(ctx.exception))

    def test_invalid_utf8_output_is_sanitized(self):
        provider = _provider()
        completed = subprocess.CompletedProcess(
            args=["tesseract"],
            returncode=0,
            stdout=b"\xff\xfe",
            stderr=b"",
        )
        with patch(
            "src.infrastructure.providers.ocr.tesseract_ocr_provider.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(OcrProviderError, "not valid UTF-8"):
                provider._run_tesseract(Path("canonical.png"))


    def test_custom_tessdata_dir_enables_tsv_without_external_config_file(self):
        settings = TesseractOcrSettings(
            executable="tesseract",
            languages=("fas", "eng"),
            page_segmentation_mode=1,
            timeout_seconds=30,
            model_id="test-model",
            config_version="test-config",
            tessdata_dir=r"E:\models\tessdata_fast",
        )
        provider = TesseractOcrProvider(config_reader=None, settings=settings)
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        )
        ocr_completed = subprocess.CompletedProcess(
            args=["tesseract"],
            returncode=0,
            stdout=tsv.encode("utf-8"),
            stderr=b"",
        )
        version_completed = subprocess.CompletedProcess(
            args=["tesseract", "--version"],
            returncode=0,
            stdout=b"tesseract 5.4.0\n",
            stderr=b"",
        )
        with patch(
            "src.infrastructure.providers.ocr.tesseract_ocr_provider.subprocess.run",
            side_effect=[ocr_completed, version_completed],
        ) as run:
            output, version = provider._run_tesseract(Path("canonical.png"))

        self.assertEqual(output, tsv)
        self.assertEqual(version, "tesseract 5.4.0")
        command = run.call_args_list[0].args[0]
        self.assertIn("--tessdata-dir", command)
        self.assertIn(r"E:\models\tessdata_fast", command)
        self.assertEqual(command[-2:], ["-c", "tessedit_create_tsv=1"])
        self.assertNotEqual(command[-1], "tsv")


if __name__ == "__main__":
    unittest.main()
