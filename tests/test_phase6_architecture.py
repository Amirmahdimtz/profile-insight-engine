import ast
import importlib
import inspect
import pathlib
import re
import unittest

from src.core.profile_analysis.embedding_provider import IEmbeddingProvider
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE_FILE = ROOT / "src" / "core" / "services" / "profile_analysis" / "semantic_theme_service.py"
PROVIDER_FILE = ROOT / "src" / "infrastructure" / "providers" / "embedding" / "transformers_embedding_provider.py"
PRODUCTION_FILES = (
    ROOT / "src" / "core" / "profile_analysis" / "embedding_contracts.py",
    ROOT / "src" / "core" / "profile_analysis" / "embedding_provider.py",
    SERVICE_FILE,
    PROVIDER_FILE,
    ROOT / "evaluation" / "embedding_benchmark.py",
)


class Phase6ArchitectureTests(unittest.TestCase):
    def test_discovery_resolves_embedding_interface_provider_and_service(self):
        bootstrap_di()
        provider_module = importlib.import_module(
            "src.infrastructure.providers.embedding.transformers_embedding_provider"
        )
        service_module = importlib.import_module(
            "src.core.services.profile_analysis.semantic_theme_service"
        )
        self.assertIsInstance(resolve(IEmbeddingProvider), provider_module.TransformersEmbeddingProvider)
        self.assertIsInstance(resolve(service_module.SemanticThemeService), service_module.SemanticThemeService)

    def test_service_uses_constructor_injection_and_provider_interface(self):
        tree = ast.parse(SERVICE_FILE.read_text(encoding="utf-8"))
        service = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SemanticThemeService")
        decorators = {node.id for node in service.decorator_list if isinstance(node, ast.Name)}
        self.assertIn("inject", decorators)
        init = next(node for node in service.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        self.assertEqual([arg.arg for arg in init.args.args], ["self", "embedding_provider", "config_reader"])
        self.assertEqual(ast.unparse(init.args.args[1].annotation), "IEmbeddingProvider")

    def test_concrete_provider_is_in_infrastructure_and_implements_interface(self):
        module = importlib.import_module(
            "src.infrastructure.providers.embedding.transformers_embedding_provider"
        )
        self.assertTrue(inspect.isclass(module.TransformersEmbeddingProvider))
        self.assertTrue(issubclass(module.TransformersEmbeddingProvider, IEmbeddingProvider))

    def test_all_phase6_async_methods_use_async_suffix(self):
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                    violations.append((path.name, node.name))
        self.assertEqual(violations, [])

    def test_core_has_no_transformers_or_torch_runtime_imports(self):
        core_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PRODUCTION_FILES
            if "infrastructure" not in path.parts and path.name != "embedding_benchmark.py"
        ).lower()
        self.assertNotIn("import torch", core_text)
        self.assertNotIn("from transformers", core_text)
        self.assertNotIn("siglip", core_text)

    def test_phase7_plus_components_are_absent(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        for term in (
            "EvidenceAggregationService",
            "InsightGenerationService",
            "ProfileAnalysisController",
            "ProfileAnalysisRepository",
            "sqlalchemy",
            "fastapi",
        ):
            with self.subTest(term=term):
                self.assertNotIn(term, production_text)

    def test_no_manual_registration_todo_or_content_logging(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        self.assertNotIn("register(TransformersEmbeddingProvider", production_text)
        self.assertNotIn("register(IEmbeddingProvider", production_text)
        markers = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|FAKE)\b", re.IGNORECASE)
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(markers.search(text), str(path))
            self.assertNotIn("logging", text)
            self.assertNotIn("logger", text)


if __name__ == "__main__":
    unittest.main()
