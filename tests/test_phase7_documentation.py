import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase_7_cross_image_aggregation.md"


class Phase7DocumentationTests(unittest.TestCase):
    def test_documentation_records_contract_and_keeps_phase_open_for_local_verification(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("unique canonical supporting image count / total batch image-id count", text)
        self.assertIn("min(per-image support count) / max(per-image support count)", text)
        self.assertIn("exact duplicate", text.lower())
        self.assertIn("near-duplicate", text.lower())
        self.assertIn("NFC", text)
        self.assertIn("casefold", text)
        self.assertIn("python -m evaluation.aggregation_benchmark", text)
        self.assertIn("python -m unittest discover", text)
        self.assertIn("git pull --ff-only origin main", text)
        self.assertNotIn("COMPLETE / READY_FOR_NEXT_PHASE", text)


if __name__ == "__main__":
    unittest.main()
