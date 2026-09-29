from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE3_DOC = REPOSITORY_ROOT / "docs" / "phase_3_image_ingestion_preprocessing.md"


class Phase3DocumentationTests(unittest.TestCase):
    def test_powershell_benchmark_command_is_copy_paste_safe_without_line_continuation(self):
        text = PHASE3_DOC.read_text(encoding="utf-8")
        powershell = text.split("PowerShell:", 1)[1].split("POSIX shell:", 1)[0]

        self.assertIn(
            'python -m evaluation.image_preprocessing_benchmark --manifest "$env:MANIFEST_PATH" '
            '--dataset-root "$env:DATASET_ROOT" --batch-sizes 1,5,20,50 --iterations 3 '
            '--output "$env:PHASE3_REPORT"',
            powershell,
        )
        self.assertNotIn("image_preprocessing_benchmark `", powershell)


if __name__ == "__main__":
    unittest.main()
