import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/phase_10_benchmark_hardening_local_deployment.md"


class Phase10DocumentationTests(unittest.TestCase):
    def test_documentation_records_phase10_gate_and_awaiting_status(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("594187e15b692944e126c485b7d9c1b4383cce1d", text)
        self.assertIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("Phase 10 is not `COMPLETE / PROJECT ACCEPTED`", text)
        self.assertIn("Production model/config selection status", text)
        self.assertIn("`UNRESOLVED`", text)
        self.assertIn("Full pipeline | `AVAILABLE`", text)
        self.assertIn("Evaluation dataset | `PARTIAL`", text)
        self.assertIn("Target hardware | `PARTIAL`", text)

    def test_runbook_is_copy_paste_oriented_and_documents_single_worker_recovery(self):
        text = DOC.read_text(encoding="utf-8")
        for fragment in (
            "python scripts/verify_phase10.py",
            "python scripts/collect_phase10_hardware.py",
            "python -m evaluation.phase10_end_to_end_benchmark",
            "python -m evaluation.phase10_benchmark",
            "python -m alembic -c src/infrastructure/alembic.ini upgrade head",
            "--workers 1",
            "--offline",
            "Restart/recovery",
            "Shutdown",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        self.assertIn("Do not use `--workers > 1`", text)

    def test_documentation_does_not_claim_missing_metrics_or_final_model_selection(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("`unavailable`", text)
        self.assertIn("No OCR traineddata choice, Vision model/config, Embedding model/config", text)
        self.assertIn("current Vision candidate has previously shown very high latency/timeouts", text)
        self.assertNotIn("Qwen2.5-VL-3B is the production model", text)
        self.assertNotIn("SigLIP2 is the production model", text)


if __name__ == "__main__":
    unittest.main()
