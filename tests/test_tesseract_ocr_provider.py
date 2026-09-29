import unittest

from src.core.profile_analysis.ocr_contracts import OcrProviderError
from src.infrastructure.providers.ocr.tesseract_ocr_provider import TesseractOcrProvider


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
        self.assertEqual(blocks[0].region.to_dict(), {"x": 10, "y": 20, "width": 75, "height": 10})
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


if __name__ == "__main__":
    unittest.main()
