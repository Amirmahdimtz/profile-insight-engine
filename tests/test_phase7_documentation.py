import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase_7_cross_image_aggregation.md"


class Phase7DocumentationTests(unittest.TestCase):
    def test_documentation_records_verified_phase7_closure(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("COMPLETE / READY_FOR_NEXT_PHASE", text)
        self.assertNotIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("40d580b52b9ec98e51423dcf0226353f12e23537", text)
        self.assertIn("29 tests passed", text)
        self.assertIn("246 tests passed with no failures or errors", text)
        self.assertIn("phase7-aggregation-benchmark-v1", text)
        self.assertIn("deterministic_rerun = true", text)
        self.assertIn("reversed_input_deterministic = true", text)
        self.assertIn("unique canonical supporting image count / total batch image-id count", text)
        self.assertIn("min(per-image support count) / max(per-image support count)", text)
        self.assertIn("exact duplicate", text.lower())
        self.assertIn("near-duplicate", text.lower())
        self.assertIn("NFC", text)
        self.assertIn("casefold", text)
        self.assertIn("python -m evaluation.aggregation_benchmark", text)
        self.assertIn("python -m unittest discover", text)
        self.assertIn("git pull --ff-only origin main", text)


if __name__ == "__main__":
    unittest.main()
