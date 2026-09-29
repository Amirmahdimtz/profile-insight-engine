import ast
import importlib
import inspect
import pathlib
import re
import unittest

from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE_FILE = REPOSITORY_ROOT / "src" / "core" / "services" / "profile_analysis" / "image_processing_service.py"
PRODUCTION_FILES = [
    REPOSITORY_ROOT / "evaluation" / "image_preprocessing_benchmark.py",
    REPOSITORY_ROOT / "src" / "core" / "profile_analysis" / "image_contracts.py",
    SERVICE_FILE,
    REPOSITORY_ROOT / "src" / "infrastructure" / "di" / "inject.py",
    REPOSITORY_ROOT / "src" / "infrastructure" / "di" / "bootstrap.py",
    REPOSITORY_ROOT / "src" / "infrastructure" / "utils" / "config_reader.py",
    REPOSITORY_ROOT / "src" / "infrastructure" / "providers" / "image" / "local_image_storage.py",
]


class Phase3ArchitectureTests(unittest.TestCase):
    def test_discovery_resolves_config_storage_and_image_processing_service(self):
        bootstrap_di()
        config_module = importlib.import_module("src.infrastructure.utils.config_reader")
        storage_module = importlib.import_module("src.infrastructure.providers.image.local_image_storage")
        service_module = importlib.import_module("src.core.services.profile_analysis.image_processing_service")
        self.assertIsInstance(resolve(config_module.ConfigReader), config_module.ConfigReader)
        self.assertIsInstance(resolve(storage_module.LocalImageStorage), storage_module.LocalImageStorage)
        self.assertIsInstance(resolve(service_module.ImageProcessingService), service_module.ImageProcessingService)

    def test_image_processing_service_uses_constructor_injection(self):
        tree = ast.parse(SERVICE_FILE.read_text(encoding="utf-8"))
        service_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageProcessingService")
        decorator_names = {node.id for node in service_class.decorator_list if isinstance(node, ast.Name)}
        self.assertIn("inject", decorator_names)
        init = next(node for node in service_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__")
        params = [arg.arg for arg in init.args.args]
        self.assertEqual(params, ["self", "config_reader", "image_storage"])

    def test_all_async_methods_use_async_suffix(self):
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                    violations.append((path.name, node.name))
        self.assertEqual(violations, [])

    def test_no_manual_feature_provider_registration(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        self.assertNotIn("register(ImageProcessingService", production_text)
        self.assertNotIn("register(LocalImageStorage", production_text)
        self.assertNotIn("register(ConfigReader", production_text)

    def test_phase4_plus_components_are_absent(self):
        forbidden_terms = (
            "OcrProvider",
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

    def test_production_has_no_todo_fixme_placeholder_or_logging_of_content(self):
        markers = re.compile(r"\b(TODO|FIXME|PLACEHOLDER)\b", re.IGNORECASE)
        violations = []
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            if markers.search(text):
                violations.append(path.name)
            self.assertNotIn("logging", text)
            self.assertNotIn("logger", text)
        self.assertEqual(violations, [])

    def test_storage_is_concrete_without_unnecessary_interface(self):
        module = importlib.import_module("src.infrastructure.providers.image.local_image_storage")
        self.assertTrue(inspect.isclass(module.LocalImageStorage))
        self.assertFalse((REPOSITORY_ROOT / "src" / "infrastructure" / "providers" / "image" / "i_image_storage.py").exists())


if __name__ == "__main__":
    unittest.main()
