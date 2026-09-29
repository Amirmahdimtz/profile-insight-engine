from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE4_DOC = REPOSITORY_ROOT / "docs" / "phase_4_ocr_text_evidence.md"


class Phase4DocumentationTests(unittest.TestCase):
    def test_windows_verification_is_fail_fast_and_path_configurable(self):
        text = PHASE4_DOC.read_text(encoding="utf-8")
        section = text.split("## Windows local verification prerequisites", 1)[1]

        self.assertIn("Get-Command tesseract -ErrorAction SilentlyContinue", section)
        self.assertIn('$env:TESSDATA_FAST = Join-Path $env:MODEL_ROOT "tessdata_fast"', section)
        self.assertIn('$env:TESSDATA_BEST = Join-Path $env:MODEL_ROOT "tessdata_best"', section)
        self.assertIn(
            "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/$Language.traineddata",
            section,
        )
        self.assertIn(
            "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$Language.traineddata",
            section,
        )
        self.assertNotIn(r'E:\\tessdata_fast', section)
        self.assertNotIn(r'E:\\tessdata_best', section)

    def test_powershell_benchmark_command_uses_verified_variables_and_fingerprints(self):
        text = PHASE4_DOC.read_text(encoding="utf-8")
        benchmark = text.split(
            "PowerShell benchmark command, after the prerequisite checks above:", 1
        )[1].split("Inspect the report:", 1)[0]

        self.assertIn(
            'python -m evaluation.ocr_benchmark --manifest "$env:DATASET_ROOT\\manifest.json" '
            '--dataset-root "$env:DATASET_ROOT" --candidate "tessdata_fast=$env:TESSDATA_FAST" '
            '--candidate "tessdata_best=$env:TESSDATA_BEST" --iterations 2 '
            '--expected-manifest-fingerprint "$env:EXPECTED_MANIFEST_FINGERPRINT" '
            '--expected-dataset-content-fingerprint "$env:EXPECTED_DATASET_CONTENT_FINGERPRINT" '
            '--output "$env:PHASE4_REPORT"',
            benchmark,
        )
        self.assertNotIn("ocr_benchmark " + chr(96), benchmark)


if __name__ == "__main__":
    unittest.main()
