import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "verify_phase10.py"
HARDWARE = ROOT / "scripts" / "collect_phase10_hardware.py"


class Phase10LocalVerificationTests(unittest.TestCase):
    def test_runner_is_parseable_and_fail_fast(self):
        text = RUNNER.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn("raise VerificationError", text)
        self.assertIn("PHASE 10 IMPLEMENTATION VERIFICATION FAILED", text)
        self.assertIn("PHASE 10 IMPLEMENTATION VERIFICATION PASSED", text)
        self.assertIn("implementation/local verification is complete", text)
        self.assertIn("fail-closed scaffold only", text)
        self.assertIn("separately aggregated real evidence report", text)
        self.assertNotIn("acceptance remains blocked", text)
        self.assertNotIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn('docker", "rm", "-f", CONTAINER_NAME', text)
        self.assertIn("--untracked-files=no", text)
        self.assertIn("POSTGRES_READINESS_TIMEOUT_SECONDS = 120", text)
        self.assertIn("time.monotonic() + POSTGRES_READINESS_TIMEOUT_SECONDS", text)
        self.assertIn("secrets.token_hex", text)
        self.assertIn("POSTGRES_PASSWORD", text)
        self.assertNotIn("POSTGRES_HOST_AUTH_METHOD=trust", text)

    def test_runner_covers_phase10_implementation_checks(self):
        text = RUNNER.read_text(encoding="utf-8")
        for fragment in (
            "compileall",
            "tests.test_phase10_hardening",
            "tests.test_phase10_benchmark",
            "tests.test_phase10_documentation",
            "tests.test_transformers_vision_provider",
            "tests.test_transformers_vision_benchmark",
            "tests.test_siglip_zero_shot_vision_provider",
            "tests.test_siglip_zero_shot_vision_benchmark",
            "tests.test_phase10_holdout_dataset",
            "alembic",
            "collect_phase10_hardware.py",
            "evaluation.phase10_benchmark",
            "phase10-final-benchmark-v1",
            "unittest",
            "discover",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_runner_migrates_disposable_database_before_db_dependent_tests(self):
        text = RUNNER.read_text(encoding="utf-8")
        alembic_upgrade = text.index('"Alembic upgrade"')
        focused_tests = text.index('"Focused Phase 10 tests"')
        recovery_regression = text.index('"PostgreSQL recovery regression"')
        self.assertLess(alembic_upgrade, focused_tests)
        self.assertLess(alembic_upgrade, recovery_regression)

    def test_hardware_collector_does_not_read_secrets(self):
        text = HARDWARE.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertNotIn("PROFILE_INSIGHT_DATABASE_URL", text)
        self.assertNotIn("password", text.lower())
        self.assertIn("nvidia-smi", text)
        self.assertIn("tesseract", text)
        self.assertIn("llama-server", text)


if __name__ == "__main__":
    unittest.main()
