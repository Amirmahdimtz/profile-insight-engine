import argparse
import unittest

from evaluation.ocr_benchmark import _parse_candidate


class OcrBenchmarkArgumentTests(unittest.TestCase):
    def test_candidate_parser_supports_default_and_explicit_tessdata_directory(self):
        default = _parse_candidate("fast=default")
        explicit = _parse_candidate(r"best=C:\\models\\tessdata_best")
        self.assertEqual(default.name, "fast")
        self.assertIsNone(default.tessdata_dir)
        self.assertEqual(explicit.name, "best")
        self.assertEqual(explicit.tessdata_dir, r"C:\\models\\tessdata_best")

    def test_candidate_parser_rejects_malformed_values(self):
        for value in ("", "name", "=path", "name="):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _parse_candidate(value)


if __name__ == "__main__":
    unittest.main()
