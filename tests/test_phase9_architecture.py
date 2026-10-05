import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "src/application/profile_analysis/profile_analysis_controller.py"
DTO = ROOT / "src/application/profile_analysis/dtos/profile_analysis_dto.py"
SERVICE = ROOT / "src/core/services/profile_analysis/profile_analysis_service.py"
LIFECYCLE = ROOT / "src/core/profile_analysis/lifecycle_contracts.py"
REPOSITORY = ROOT / "src/infrastructure/repositories/profile_analysis/profile_analysis_repository.py"
BASE_REPOSITORY = ROOT / "src/infrastructure/repositories/base/base_repository.py"
MODEL = ROOT / "src/infrastructure/models/profile_analysis.py"
DB_CONTEXT = ROOT / "src/infrastructure/context/sql_db/psql_dbcontext.py"
WEB = ROOT / "src/application/web.py"
HOST = ROOT / "src/host/app.py"
ALEMBIC_ENV = ROOT / "src/infrastructure/alembic/env.py"
MIGRATION = ROOT / "src/infrastructure/alembic/versions/20261005_0001_create_profile_analysis.py"
CONFIG_FILES = (
    ROOT / "src/host/res/appsettings.yaml",
    ROOT / "src/host/res/appsettings.development.yaml",
)
PRODUCTION_FILES = (
    CONTROLLER,
    DTO,
    SERVICE,
    LIFECYCLE,
    REPOSITORY,
    BASE_REPOSITORY,
    MODEL,
    DB_CONTEXT,
    WEB,
    HOST,
    ALEMBIC_ENV,
    MIGRATION,
)


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


class Phase9ArchitectureTests(unittest.TestCase):
    def test_required_phase9_layers_exist_in_expected_locations(self):
        for path in PRODUCTION_FILES:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_controller_is_thin_and_has_no_repository_or_sqlalchemy_dependency(self):
        imports = _imports(CONTROLLER)
        self.assertFalse(any(name == "sqlalchemy" or name.startswith("sqlalchemy.") for name in imports))
        self.assertFalse(any("repositories" in name for name in imports))
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertNotIn("PsqlDbContext", text)
        self.assertNotIn("ProfileAnalysisRepository", text)

    def test_service_owns_workflow_and_has_no_fastapi_dependency(self):
        imports = _imports(SERVICE)
        self.assertFalse(any(name == "fastapi" or name.startswith("fastapi.") for name in imports))
        text = SERVICE.read_text(encoding="utf-8")
        for term in (
            "ImageProcessingService",
            "OcrEvidenceService",
            "EvidenceExtractionService",
            "EvidenceAggregationService",
            "InsightGenerationService",
            "ProfileAnalysisRepository",
        ):
            self.assertIn(term, text)
        self.assertNotIn("SemanticThemeService", text)
        self.assertNotIn("candidate_theme", text)

    def test_repository_is_the_only_feature_component_using_sqlalchemy_queries(self):
        self.assertTrue(any(name.startswith("sqlalchemy") for name in _imports(REPOSITORY)))
        self.assertFalse(any(name.startswith("sqlalchemy") for name in _imports(SERVICE)))
        self.assertFalse(any(name.startswith("sqlalchemy") for name in _imports(CONTROLLER)))

    def test_dependency_providers_use_inject_and_no_manual_registration_is_added(self):
        for path, class_name in (
            (CONTROLLER, "ProfileAnalysisController"),
            (SERVICE, "ProfileAnalysisService"),
            (REPOSITORY, "ProfileAnalysisRepository"),
            (DB_CONTEXT, "PsqlDbContext"),
            (WEB, "WebService"),
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
            decorators = {ast.unparse(item) for item in classes[class_name].decorator_list}
            self.assertIn("inject", decorators, f"{path}: {class_name}")
        production_text = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)
        self.assertNotRegex(production_text, re.compile(r"\bregister\s*\("))

    def test_async_service_and_repository_methods_use_async_suffix(self):
        violations = []
        for path in (SERVICE, REPOSITORY, BASE_REPOSITORY):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and not node.name.endswith("_async"):
                    violations.append((path.name, node.name))
        self.assertEqual(violations, [])

    def test_api_surface_has_post_status_result_and_no_delete(self):
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("@router.post", text)
        self.assertGreaterEqual(text.count("@router.get"), 2)
        self.assertNotIn("@router.delete", text)

    def test_persistence_preserves_traceability_and_redacts_ocr_content(self):
        text = SERVICE.read_text(encoding="utf-8")
        self.assertIn('metadata.pop("raw_text", None)', text)
        self.assertIn("value=None", text)
        dto_text = DTO.read_text(encoding="utf-8")
        for field in (
            "evidence_count",
            "image_coverage",
            "confidence",
            "supporting_evidence_ids",
            "supporting_image_ids",
            "explanation",
        ):
            self.assertIn(field, dto_text)

    def test_alembic_metadata_and_reversible_migration_are_present(self):
        env_text = ALEMBIC_ENV.read_text(encoding="utf-8")
        self.assertIn("Base.metadata", env_text)
        self.assertIn("ProfileAnalysisModel", env_text)
        migration_text = MIGRATION.read_text(encoding="utf-8")
        self.assertIn("def upgrade()", migration_text)
        self.assertIn("def downgrade()", migration_text)
        self.assertIn('op.create_table(', migration_text)
        self.assertIn('op.drop_table("profile_analysis")', migration_text)
        model_text = MODEL.read_text(encoding="utf-8")
        self.assertIn("JSONB(none_as_null=True)", model_text)
        self.assertIn("none_as_null=True", migration_text)

    def test_api_and_database_config_exist_in_both_environment_files(self):
        for path in CONFIG_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIn("api:\n  prefix: /api/v1", text)
            self.assertIn("database:\n  url_env: PROFILE_INSIGHT_DATABASE_URL", text)
            self.assertNotIn("postgresql+asyncpg://", text)

    def test_no_placeholders_or_raw_content_logging_are_added(self):
        marker = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|FAKE)\b", re.IGNORECASE)
        for path in PRODUCTION_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(marker.search(text), str(path))
            if path != ALEMBIC_ENV:
                self.assertNotIn("logging", text)
                self.assertNotIn("logger", text)


if __name__ == "__main__":
    unittest.main()
