import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "verify_phase9.py"


class Phase9LocalVerificationTests(unittest.TestCase):
    def test_runner_is_parseable_and_uses_isolated_disposable_database(self):
        text = RUNNER.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn("profile-insight-phase9-verify-", text)
        self.assertIn("127.0.0.1:{port}:5432", text)
        self.assertIn("POSTGRES_HOST_AUTH_METHOD=trust", text)
        self.assertNotIn("POSTGRES_PASSWORD", text)
        self.assertIn("--rm", text)
        self.assertIn("PROFILE_INSIGHT_DATABASE_URL", text)
        self.assertIn("PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL", text)

    def test_runner_is_fail_fast_and_cleans_up(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("raise VerificationError", text)
        self.assertIn('["docker", "rm", "-f", CONTAINER_NAME]', text)
        self.assertIn("--untracked-files=no", text)
        self.assertIn("PHASE 9 LOCAL VERIFICATION FAILED", text)
        self.assertIn("PHASE 9 LOCAL VERIFICATION PASSED", text)

    def test_runner_covers_phase9_acceptance_checks(self):
        text = RUNNER.read_text(encoding="utf-8")
        required_fragments = (
            "compileall",
            "tests.test_phase9_contracts",
            "tests.test_phase9_di_discovery",
            "tests.test_profile_analysis_repository_integration",
            "alembic",
            "downgrade",
            "app.openapi()",
            "evaluation.phase9_api_persistence_benchmark",
            "phase9-api-persistence-benchmark-v1",
            "phase9_contract_and_persistence_not_end_to_end_ml",
            "unittest",
            "discover",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
