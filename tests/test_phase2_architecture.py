import ast
import importlib
import pathlib
import re
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVALUATION_ROOT = REPOSITORY_ROOT / "evaluation"
PHASE2_EVALUATION_FILES = (
    EVALUATION_ROOT / "benchmark.py",
    EVALUATION_ROOT / "common.py",
    EVALUATION_ROOT / "contracts.py",
    EVALUATION_ROOT / "dataset.py",
    EVALUATION_ROOT / "metrics.py",
    EVALUATION_ROOT / "predictions.py",
    EVALUATION_ROOT / "reporting.py",
    EVALUATION_ROOT / "validate_dataset.py",
)


class Phase2ArchitectureTests(unittest.TestCase):
    def test_evaluation_tooling_imports_without_third_party_dependencies(self):
        contracts = importlib.import_module("evaluation.contracts")
        metrics = importlib.import_module("evaluation.metrics")
        benchmark = importlib.import_module("evaluation.benchmark")
        self.assertTrue(hasattr(contracts, "EvaluationDatasetManifest"))
        self.assertTrue(hasattr(metrics, "character_error_rate"))
        self.assertTrue(hasattr(benchmark, "run_benchmark"))

    def test_phase2_evaluation_tooling_has_no_forbidden_runtime_or_ml_imports(self):
        forbidden_prefixes = (
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "torch",
            "transformers",
            "tensorflow",
            "cv2",
            "PIL",
            "numpy",
            "src.application",
            "src.infrastructure",
            "src.host",
        )
        violations = []
        for path in PHASE2_EVALUATION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                for name in imported:
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in forbidden_prefixes
                    ):
                        violations.append((path.name, name))
        self.assertEqual(violations, [])

    def test_phase2_files_do_not_gain_phase3_or_ml_provider_components(self):
        forbidden_terms = (
            "CanonicalImage",
            "ImageProcessingService",
            "OcrProvider",
            "VisionProvider",
            "EmbeddingProvider",
            "LocalVisionProvider",
        )
        production_text = "\n".join(
            path.read_text(encoding="utf-8") for path in PHASE2_EVALUATION_FILES
        )
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, production_text)

    def test_phase2_reuses_phase1_policy_instead_of_copying_policy_lists(self):
        contracts_text = (EVALUATION_ROOT / "contracts.py").read_text(encoding="utf-8")
        metrics_text = (EVALUATION_ROOT / "metrics.py").read_text(encoding="utf-8")
        self.assertIn("validate_observable_claim", contracts_text)
        self.assertIn("validate_observable_claim", metrics_text)
        self.assertNotIn("_SENSITIVE_TRAIT_PHRASES", contracts_text)
        self.assertNotIn("_SENSITIVE_TRAIT_KEYS", contracts_text)

    def test_production_python_has_no_todo_fixme_or_placeholder_markers(self):
        markers = re.compile(r"\b(TODO|FIXME|PLACEHOLDER)\b", re.IGNORECASE)
        violations = []
        production_files = [
            REPOSITORY_ROOT / "src" / "core" / "profile_analysis" / "contracts.py",
            *PHASE2_EVALUATION_FILES,
        ]
        for path in production_files:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if markers.search(line):
                    violations.append((str(path.relative_to(REPOSITORY_ROOT)), line_number, line.strip()))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
