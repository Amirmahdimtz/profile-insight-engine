import ast
import importlib
import inspect
import pathlib
import re
import unittest

from src.core.profile_analysis.ocr_provider import IOcrProvider
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE_FILE = REPOSITORY_ROOT / "src" / "core" / "services" / "profile_analysis" / "ocr_evidence_service.py"
PROVIDER_FILE = REPOSITORY_ROOT / "src" / "infrastructure" / "providers" / "ocr" / "tesseract_ocr_provider.py"
PRODUCTION_FILES = (
    REPOSITORY_ROOT / "src" / "core" / "profile_analysis" / "ocr_contracts.py",
    REPOSITORY_ROOT / "src" / "core" / "profile_analysis" / "ocr_provider.py",
    SERVICE_FILE,
    PROVIDER_FILE,
    REPOSITORY_ROOT / "evaluation" / "ocr_benchmark.py",
)


class Phase4ArchitectureTests(unittest.TestCase):
    def test_discovery_resolves_interface_provider_and_service(self):
        bootstrap_di()
        provider_module = importlib.import_module(
            "src.infrastructure.providers.ocr.tesseract_ocr_provider"
        )
        service_module = importlib.import_module(
            "src.core.services.profile_analysis.ocr_evidence_service"
        )
        self.assertIsInstance(resolve(IOcrProvider), provider_module.TesseractOcrProvider)
        self.assertIsInstance(resolve(service_module.OcrEvidenceService), service_module.OcrEvidenceService)

    def test_ocr_service_uses_constructor_injection_through_provider_interface(self):
        tree = ast.parse(SERVICE_FILE.read_text(encoding="utf-8"))
        service_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OcrEvidenceService")
        decorator_names = {node.id for node in service_class.decorator_list if isinstance(node, ast.Name)}
        self.assertIn("inject", decorator_names)
        init = next(node for node in service_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        self.assertEqual([arg.arg for arg in init.args.args], ["self", "ocr_provider"])
        self.assertEqual(ast.unparse(init.args.args[1].annotation), "IOcrProvider")

    def test_concrete_provider_is_in_infrastructure_and_implements_interface(self):
        module = importlib.import_module("src.infrastructure.providers.ocr.tesseract_ocr_provider")
        self.assertTrue(inspect.isclass(module.TesseractOcrProvider))
        self.assertTrue(issubclass(module.TesseractOcrProvider, IOcrProvider))

    def test_all_phase4_async_methods_use_async_suffix(self):
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                    violations.append((path.name, node.name))
        self.assertEqual(violations, [])

    def test_no_manual_ocr_provider_registration(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        self.assertNotIn("register(TesseractOcrProvider", production_text)
        self.assertNotIn("register(IOcrProvider", production_text)

    def test_core_contract_does_not_contain_tesseract_raw_schema(self):
        core_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PRODUCTION_FILES
            if "infrastructure" not in path.parts and path.name != "ocr_benchmark.py"
        )
        self.assertNotIn("page_num", core_text)
        self.assertNotIn("block_num", core_text)
        self.assertNotIn("word_num", core_text)
        self.assertNotIn("stdout", core_text)

    def test_phase5_plus_components_are_absent(self):
        forbidden_terms = (
            "VisionProvider",
            "EmbeddingProvider",
            "EvidenceAggregationService",
            "InsightGenerationService",
            "ProfileAnalysisController",
            "ProfileAnalysisRepository",
            "sqlalchemy",
            "fastapi",
        )
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, production_text)

    def test_production_has_no_todo_fixme_placeholder_fake_or_content_logging(self):
        markers = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|FAKE)\b", re.IGNORECASE)
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(markers.search(text), str(path))
            self.assertNotIn("logging", text)
            self.assertNotIn("logger", text)


if __name__ == "__main__":
    unittest.main()
