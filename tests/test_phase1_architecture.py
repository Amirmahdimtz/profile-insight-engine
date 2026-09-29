import ast
import importlib
import pathlib
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS_FILE = REPOSITORY_ROOT / "src" / "core" / "profile_analysis" / "contracts.py"


class Phase1ArchitectureTests(unittest.TestCase):
    def test_contract_module_imports_without_third_party_dependencies(self):
        module = importlib.import_module("src.core.profile_analysis.contracts")
        self.assertTrue(hasattr(module, "ProfileAnalysisResult"))

    def test_core_contract_does_not_import_forbidden_runtime_layers_or_ml_frameworks(self):
        tree = ast.parse(CONTRACTS_FILE.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

        forbidden_prefixes = (
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "torch",
            "transformers",
            "src.application",
            "src.infrastructure",
            "src.host",
        )
        violating = sorted(
            imported
            for imported in imports
            if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden_prefixes)
        )
        self.assertEqual(violating, [])

    def test_phase1_does_not_create_future_runtime_layers(self):
        src = REPOSITORY_ROOT / "src"
        self.assertFalse((src / "application").exists())
        self.assertFalse((src / "infrastructure").exists())
        self.assertFalse((src / "host").exists())
        self.assertFalse((src / "core" / "services").exists())


if __name__ == "__main__":
    unittest.main()
