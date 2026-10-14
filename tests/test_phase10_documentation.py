import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/phase_10_benchmark_hardening_local_deployment.md"


class Phase10DocumentationTests(unittest.TestCase):
    def test_documentation_records_phase10_gate_and_awaiting_status(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("594187e15b692944e126c485b7d9c1b4383cce1d", text)
        self.assertIn("IMPLEMENTED / LOCAL VERIFICATION PASSED / ACCEPTANCE BLOCKED", text)
        self.assertIn("PHASE 10 IMPLEMENTATION VERIFICATION PASSED", text)
        self.assertIn("354 tests passed", text)
        self.assertIn("Phase 10 is not `COMPLETE / PROJECT ACCEPTED`", text)
        self.assertIn("Production model/config selection status", text)
        self.assertIn("`UNRESOLVED`", text)
        self.assertIn("Full pipeline | `AVAILABLE`", text)
        self.assertIn("Evaluation dataset | `PARTIAL`", text)
        self.assertIn("Target hardware | `AVAILABLE`", text)

    def test_runbook_is_copy_paste_oriented_and_documents_single_worker_recovery(self):
        text = DOC.read_text(encoding="utf-8")
        for fragment in (
            "python scripts/verify_phase10.py",
            "python scripts/collect_phase10_hardware.py",
            "python -m evaluation.transformers_vision_benchmark",
            "python -m evaluation.siglip_zero_shot_vision_benchmark",
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
        self.assertIn("single 512px image tile", text)
        self.assertIn("Outlines JSON-schema constrained decoding", text)
        self.assertIn("remains benchmark-only until holdout-safe evidence is reviewed", text)
        self.assertIn("deliberately remains unbound from DI", text)
        self.assertIn("benchmark_valid=true", text)
        self.assertIn("does **not** mean the candidate is production-ready", text)
        self.assertIn("failure rate was `0.5`", text)
        self.assertIn("model snapshot `067788b187b95ebe7b2e040b3e4299e342e5b8fd`", text)
        self.assertIn("current Transformers SmolVLM2-256M candidate is explicitly rejected", text)
        self.assertIn("No generative local Vision model/provider configuration is acceptable", text)
        self.assertIn("The finalization candidate is the non-generative SigLIP2 zero-shot adapter", text)
        self.assertIn("visual class universe from the TRAIN split only", text)
        self.assertIn("EVAL labels are never used to construct the candidate taxonomy", text)
        self.assertIn("current authorized dataset has no TRAIN visual taxonomy labels", text)
        self.assertIn("independent versioned product taxonomy", text)
        self.assertIn("sample-specific ground-truth mappings are never passed to the provider", text)
        self.assertIn("phase10_siglip_zero_shot_vision_benchmark.json", text)

    def test_documentation_does_not_claim_missing_metrics_or_final_model_selection(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("`unavailable`", text)
        self.assertIn("No OCR traineddata choice, Embedding model/config, semantic threshold or Insight threshold is production-final yet", text)
        self.assertIn("current Transformers SmolVLM2-256M candidate is explicitly rejected", text)
        self.assertIn("No generative local Vision model/provider configuration is acceptable", text)
        self.assertNotIn("Qwen2.5-VL-3B is the production model", text)
        self.assertNotIn("SigLIP2 is the production model", text)


if __name__ == "__main__":
    unittest.main()
