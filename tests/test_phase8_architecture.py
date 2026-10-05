import ast
import importlib
import pathlib
import re
import unittest

from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "src" / "core" / "profile_analysis" / "insight_contracts.py"
SERVICE_FILE = ROOT / "src" / "core" / "services" / "profile_analysis" / "insight_generation_service.py"
BENCHMARK_FILE = ROOT / "evaluation" / "insight_benchmark.py"
CONFIG_FILES = (
    ROOT / "src" / "host" / "res" / "appsettings.yaml",
    ROOT / "src" / "host" / "res" / "appsettings.development.yaml",
)
PRODUCTION_FILES = (CONTRACT_FILE, SERVICE_FILE, BENCHMARK_FILE)


class Phase8ArchitectureTests(unittest.TestCase):
    def test_discovery_resolves_concrete_insight_service(self):
        bootstrap_di()
        module = importlib.import_module(
            "src.core.services.profile_analysis.insight_generation_service"
        )
        self.assertIsInstance(
            resolve(module.InsightGenerationService),
            module.InsightGenerationService,
        )

    def test_service_is_concrete_core_component_with_constructor_config_injection(self):
        tree = ast.parse(SERVICE_FILE.read_text(encoding="utf-8"))
        service = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "InsightGenerationService"
        )
        decorators = {node.id for node in service.decorator_list if isinstance(node, ast.Name)}
        self.assertIn("inject", decorators)
        init = next(
            node
            for node in service.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertEqual([arg.arg for arg in init.args.args], ["self", "config_reader"])
        self.assertEqual(ast.unparse(init.args.args[1].annotation), "ConfigReader")
        self.assertFalse(
            any(base.id.startswith("I") for base in service.bases if isinstance(base, ast.Name))
        )

    def test_core_contract_has_no_application_http_or_persistence_dependencies(self):
        tree = ast.parse(CONTRACT_FILE.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        forbidden = ("src.application", "fastapi", "sqlalchemy", "pydantic")
        violations = sorted(
            value
            for value in imports
            if any(value == prefix or value.startswith(prefix + ".") for prefix in forbidden)
        )
        self.assertEqual(violations, [])

    def test_phase9_components_and_persistence_are_absent(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        for term in (
            "ProfileAnalysisController",
            "ProfileAnalysisRepository",
            "PsqlDbContext",
            "sqlalchemy",
            "alembic",
            "fastapi",
            "APIRouter",
            "HTTPException",
        ):
            with self.subTest(term=term):
                self.assertNotIn(term, production_text)

    def test_no_manual_registration_logging_placeholders_or_generative_provider(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        self.assertNotIn("register(InsightGenerationService", production_text)
        self.assertNotIn("ExplanationProvider", production_text)
        self.assertNotIn("LLM", production_text)
        marker = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|FAKE)\b", re.IGNORECASE)
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(marker.search(text), str(path))
            self.assertNotIn("logging", text)
            self.assertNotIn("logger", text)

    def test_config_keys_exist_in_both_environment_files(self):
        required = (
            "policy_version:",
            "minimum_evidence_count:",
            "minimum_unique_image_count:",
            "minimum_image_coverage:",
            "minimum_source_diversity:",
            "minimum_cross_image_consistency:",
            "minimum_average_confidence:",
            "source_diversity_saturation:",
        )
        for path in CONFIG_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIn("insights:", text)
            for key in required:
                with self.subTest(path=path.name, key=key):
                    self.assertIn(key, text)

    def test_any_phase8_async_method_would_require_async_suffix(self):
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                    violations.append((path.name, node.name))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
