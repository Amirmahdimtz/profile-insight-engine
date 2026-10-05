import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/phase_9_api_persistence_lifecycle.md"


class Phase9DocumentationTests(unittest.TestCase):
    def test_documentation_records_verified_phase9_closure(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("COMPLETE / READY_FOR_NEXT_PHASE", text)
        self.assertNotIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("Phase 9 is locally verified", text)
        self.assertIn("60bb28e84fc22d43cbb6372ee5d6b7c62c6c5f7a", text)
        self.assertIn("33 tests passed in 7.394 seconds", text)
        self.assertIn("4 tests passed in 3.839 seconds", text)
        self.assertIn("308 tests passed with no failures or errors in 20.995 seconds", text)
        self.assertIn("20261005_0001 (head)", text)
        self.assertIn("no new upgrade operations detected", text)
        self.assertIn("SQLSTATE `23505`", text)
        self.assertIn("phase9-api-persistence-benchmark-v1", text)
        self.assertIn("phase9_contract_and_persistence_not_end_to_end_ml", text)
        self.assertIn("serialized analysis graph size: `1013` bytes", text)
        self.assertIn("concurrent PostgreSQL reads: `8`", text)
        self.assertIn("PHASE 9 LOCAL VERIFICATION PASSED", text)
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
        self.assertIn("phase9_api_persistence_benchmark.json", text)
        self.assertNotIn("PHASE9_DB_PASSWORD", text)
        self.assertNotIn("NewGuid().ToString", text)


if __name__ == "__main__":
    unittest.main()
