import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/phase_9_api_persistence_lifecycle.md"


class Phase9DocumentationTests(unittest.TestCase):
    def test_documentation_records_awaiting_local_verification_contract(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertNotIn("Phase 9 is locally verified", text)
        self.assertIn("POST /api/v1/profile_analysis/", text)
        self.assertIn("GET /api/v1/profile_analysis/{analysis_id}", text)
        self.assertIn("GET /api/v1/profile_analysis/{analysis_id}/result", text)
        self.assertIn("DELETE` is intentionally absent", text)
        self.assertIn("pending → in_progress → completed", text)
        self.assertIn("phase9-analysis-result-v1", text)
        self.assertIn("metadata.raw_text", text)
        self.assertIn("SemanticThemeService` is not invoked", text)
        self.assertIn("python scripts/verify_phase9.py", text)
        self.assertIn("POSTGRES_HOST_AUTH_METHOD=trust", text)
        self.assertIn("127.0.0.1", text)
        self.assertIn("stops at the first failing check", text)
        self.assertIn("entire repository regression suite in quiet mode", text)
        self.assertIn("phase9_api_persistence_benchmark.json", text)
        self.assertNotIn("PHASE9_DB_PASSWORD", text)
        self.assertNotIn("NewGuid().ToString", text)


if __name__ == "__main__":
    unittest.main()
