import ast
import importlib
import inspect
import pathlib
import re
import unittest

from src.core.profile_analysis.vision_provider import IVisionProvider
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE_FILE = (
    REPOSITORY_ROOT
    / "src"
    / "core"
    / "services"
    / "profile_analysis"
    / "evidence_extraction_service.py"
)
PROVIDER_FILE = (
    REPOSITORY_ROOT
    / "src"
    / "infrastructure"
    / "providers"
    / "vision"
    / "llama_cpp_vision_provider.py"
)
DOC_FILE = (
    REPOSITORY_ROOT
    / "docs"
    / "phase_5_visual_understanding.md"
)
PRODUCTION_FILES = (
    REPOSITORY_ROOT
    / "src"
    / "core"
    / "profile_analysis"
    / "vision_contracts.py",
    REPOSITORY_ROOT
    / "src"
    / "core"
    / "profile_analysis"
    / "vision_provider.py",
    SERVICE_FILE,
    PROVIDER_FILE,
    REPOSITORY_ROOT
    / "evaluation"
    / "vision_benchmark.py",
)


class Phase5ArchitectureTests(unittest.TestCase):
    def test_discovery_resolves_interface_provider_and_service(
        self,
    ):
        bootstrap_di()
        provider_module = importlib.import_module(
            "src.infrastructure.providers.vision."
            "llama_cpp_vision_provider"
        )
        service_module = importlib.import_module(
            "src.core.services.profile_analysis."
            "evidence_extraction_service"
        )
        self.assertIsInstance(
            resolve(IVisionProvider),
            provider_module.LlamaCppVisionProvider,
        )
        self.assertIsInstance(
            resolve(service_module.EvidenceExtractionService),
            service_module.EvidenceExtractionService,
        )

    def test_service_uses_constructor_injection_through_provider_interface(
        self,
    ):
        tree = ast.parse(
            SERVICE_FILE.read_text(encoding="utf-8")
        )
        service_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "EvidenceExtractionService"
        )
        decorator_names = {
            node.id
            for node in service_class.decorator_list
            if isinstance(node, ast.Name)
        }
        self.assertIn("inject", decorator_names)
        init = next(
            node
            for node in service_class.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == "__init__"
        )
        self.assertEqual(
            [arg.arg for arg in init.args.args],
            ["self", "vision_provider"],
        )
        self.assertEqual(
            ast.unparse(init.args.args[1].annotation),
            "IVisionProvider",
        )

    def test_provider_is_in_infrastructure_and_implements_interface(
        self,
    ):
        module = importlib.import_module(
            "src.infrastructure.providers.vision."
            "llama_cpp_vision_provider"
        )
        self.assertTrue(
            inspect.isclass(
                module.LlamaCppVisionProvider
            )
        )
        self.assertTrue(
            issubclass(
                module.LlamaCppVisionProvider,
                IVisionProvider,
            )
        )

    def test_all_phase5_async_methods_use_async_suffix(self):
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(
                path.read_text(encoding="utf-8")
            )
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.AsyncFunctionDef)
                    and not node.name.endswith("_async")
                ):
                    violations.append(
                        (path.name, node.name)
                    )
        self.assertEqual(violations, [])

    def test_core_has_no_runtime_specific_imports(self):
        core_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PRODUCTION_FILES
            if "infrastructure" not in path.parts
            and path.name != "vision_benchmark.py"
        )
        for term in (
            "gguf",
            "llama_cpp",
            "urllib",
            "safetensors",
            "transformers",
            "onnx",
        ):
            with self.subTest(term=term):
                self.assertNotIn(
                    term,
                    core_text.lower(),
                )

    def test_phase6_plus_components_are_absent(self):
        forbidden_terms = (
            "EmbeddingProvider",
            "EvidenceAggregationService",
            "InsightGenerationService",
            "ProfileAnalysisController",
            "ProfileAnalysisRepository",
            "sqlalchemy",
            "fastapi",
            "near_duplicate",
            "similarity",
        )
        production_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PRODUCTION_FILES
        )
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, production_text)

    def test_windows_verification_contract_requires_synced_main_and_fresh_shell(
        self,
    ):
        documentation = DOC_FILE.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "git pull --ff-only origin main",
            documentation,
        )
        self.assertIn(
            "Microsoft\\WinGet\\Links\\llama-server.exe",
            documentation,
        )
        self.assertIn(
            "Microsoft\\WinGet\\Packages",
            documentation,
        )
        self.assertIn(
            "$env:LLAMA_SERVER_EXE",
            documentation,
        )
        self.assertIn(
            '--runtime-executable "$env:LLAMA_SERVER_EXE"',
            documentation,
        )
        self.assertNotIn(
            "Get-Command llama-server -ErrorAction Stop",
            documentation,
        )

    def test_no_manual_provider_registration_or_content_logging(
        self,
    ):
        production_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PRODUCTION_FILES
        )
        self.assertNotIn(
            "register(LlamaCppVisionProvider",
            production_text,
        )
        self.assertNotIn(
            "register(IVisionProvider",
            production_text,
        )
        markers = re.compile(
            r"\b(TODO|FIXME|PLACEHOLDER|FAKE)\b",
            re.IGNORECASE,
        )
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                markers.search(text),
                str(path),
            )
            self.assertNotIn("logging", text)
            self.assertNotIn("logger", text)


if __name__ == "__main__":
    unittest.main()
