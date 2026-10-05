import ast
import importlib
import pathlib
import re
import unittest

from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "src" / "core" / "profile_analysis" / "aggregation_contracts.py"
SERVICE_FILE = (
    ROOT
    / "src"
    / "core"
    / "services"
    / "profile_analysis"
    / "evidence_aggregation_service.py"
)
BENCHMARK_FILE = ROOT / "evaluation" / "aggregation_benchmark.py"
PRODUCTION_FILES = (CONTRACT_FILE, SERVICE_FILE, BENCHMARK_FILE)


class Phase7ArchitectureTests(unittest.TestCase):
    def test_discovery_resolves_concrete_aggregation_service(self):
        bootstrap_di()
        module = importlib.import_module(
            "src.core.services.profile_analysis.evidence_aggregation_service"
        )
        self.assertIsInstance(
            resolve(module.EvidenceAggregationService),
            module.EvidenceAggregationService,
        )

    def test_service_is_concrete_in_core_and_has_no_unnecessary_dependency(self):
        tree = ast.parse(SERVICE_FILE.read_text(encoding="utf-8"))
        service = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "EvidenceAggregationService"
        )
        decorators = {node.id for node in service.decorator_list if isinstance(node, ast.Name)}
        self.assertIn("inject", decorators)
        init = next(
            node
            for node in service.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertEqual([arg.arg for arg in init.args.args], ["self"])
        self.assertFalse(any(base.id.startswith("I") for base in service.bases if isinstance(base, ast.Name)))

    def test_core_contract_has_no_application_infrastructure_or_persistence_imports(self):
        tree = ast.parse(CONTRACT_FILE.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        forbidden = ("src.application", "src.infrastructure", "fastapi", "sqlalchemy", "pydantic")
        violations = sorted(
            value
            for value in imports
            if any(value == prefix or value.startswith(prefix + ".") for prefix in forbidden)
        )
        self.assertEqual(violations, [])

    def test_phase8_plus_components_and_persistence_are_absent(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        for term in (
            "InsightGenerationService",
            "ProfileInsight",
            "evidence-backed summary",
            "ProfileAnalysisController",
            "ProfileAnalysisRepository",
            "sqlalchemy",
            "alembic",
            "fastapi",
            "minimum_evidence_count",
            "minimum_image_coverage",
        ):
            with self.subTest(term=term):
                self.assertNotIn(term, production_text)

    def test_no_config_dependency_manual_registration_logging_or_placeholders(self):
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        self.assertNotIn("ConfigReader", production_text)
        self.assertNotIn("register(EvidenceAggregationService", production_text)
        marker = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|FAKE)\b", re.IGNORECASE)
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(marker.search(text), str(path))
            self.assertNotIn("logging", text)
            self.assertNotIn("logger", text)

    def test_any_phase7_async_method_would_require_async_suffix(self):
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                    violations.append((path.name, node.name))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
