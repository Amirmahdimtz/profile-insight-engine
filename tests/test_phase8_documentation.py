import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase_8_insight_engine.md"


class Phase8DocumentationTests(unittest.TestCase):
    def test_documentation_records_phase8_contract_and_keeps_phase_open(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertNotIn("COMPLETE / READY_FOR_NEXT_PHASE", text)
        self.assertIn("AggregatedTheme", text)
        self.assertIn("semantic-only", text)
        self.assertIn("minimum_evidence_count", text)
        self.assertIn("minimum_unique_image_count", text)
        self.assertIn("minimum_image_coverage", text)
        self.assertIn("cross_image_consistency", text)
        self.assertIn("source_diversity", text)
        self.assertIn("summary cannot add a new profile fact", text)
        self.assertIn("conflict ontology", text)
        self.assertIn("synthetic_contract_scenarios_not_product_quality", text)
        self.assertIn("confidence calibration", text.lower())
        self.assertIn("human review agreement", text.lower())
        self.assertIn("python -m evaluation.insight_benchmark", text)
        self.assertIn("python -m unittest discover", text)
        self.assertIn("git pull --ff-only origin main", text)


if __name__ == "__main__":
    unittest.main()
