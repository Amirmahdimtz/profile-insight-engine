import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase_8_insight_engine.md"


class Phase8DocumentationTests(unittest.TestCase):
    def test_documentation_records_verified_phase8_closure(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("COMPLETE / READY_FOR_NEXT_PHASE", text)
        self.assertNotIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("ef718cde5d3ad9f083563c1c1396e7436a9c83e1", text)
        self.assertIn("25 tests passed in 2.315 seconds", text)
        self.assertIn("271 tests passed with no failures or errors in 8.595 seconds", text)
        self.assertIn("phase8-insight-benchmark-v1", text)
        self.assertIn("synthetic_contract_scenarios_not_product_quality", text)
        self.assertIn("phase8-candidate-v1", text)
        self.assertIn("generated Insight count: `2`", text)
        self.assertIn("synthetic contract insight precision: `1.0`", text)
        self.assertIn("synthetic contract insight recall: `1.0`", text)
        self.assertIn("false unsupported Insight rate: `0.0`", text)
        self.assertIn("structural summary factuality: `1.0`", text)
        self.assertIn("deterministic rerun: `true`", text)
        self.assertIn("reversed-input deterministic: `true`", text)
        self.assertIn(
            "4350b2e4394bc3159570586e2a0e6b392138b529e89b8a64c04ae3069466fb88",
            text,
        )
        self.assertIn("confidence calibration", text.lower())
        self.assertIn("human review agreement", text.lower())
        self.assertIn("no tracked production, test, configuration, or documentation changes were pending", text)
        self.assertIn("AggregatedTheme", text)
        self.assertIn("semantic-only", text)
        self.assertIn("minimum_evidence_count", text)
        self.assertIn("minimum_unique_image_count", text)
        self.assertIn("minimum_image_coverage", text)
        self.assertIn("cross_image_consistency", text)
        self.assertIn("source_diversity", text)
        self.assertIn("summary cannot add a new profile fact", text)
        self.assertIn("conflict ontology", text)
        self.assertIn("python -m evaluation.insight_benchmark", text)
        self.assertIn("python -m unittest discover", text)
        self.assertIn("git pull --ff-only origin main", text)


if __name__ == "__main__":
    unittest.main()
