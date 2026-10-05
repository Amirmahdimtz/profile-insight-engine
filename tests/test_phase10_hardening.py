import ast
import os
import pathlib
import tempfile
import unittest

from src.core.services.profile_analysis.profile_analysis_service import ProfileAnalysisService
from src.infrastructure.providers.image.local_image_storage import LocalImageStorage


ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "src/application/web.py"
SERVICE = ROOT / "src/core/services/profile_analysis/profile_analysis_service.py"
REPOSITORY = ROOT / "src/infrastructure/repositories/profile_analysis/profile_analysis_repository.py"
STORAGE = ROOT / "src/infrastructure/providers/image/local_image_storage.py"


class _RecoveryRepository:
    def __init__(self, count=0):
        self.count = count
        self.calls = 0

    async def fail_interrupted_async(self):
        self.calls += 1
        return self.count


class Phase10HardeningTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_recovery_delegates_to_repository(self):
        repository = _RecoveryRepository(count=3)
        service = ProfileAnalysisService(
            object(), object(), object(), object(), object(), repository
        )
        recovered = await service.recover_interrupted_async()
        self.assertEqual(recovered, 3)
        self.assertEqual(repository.calls, 1)

    async def test_local_image_storage_cleans_stale_scopes_without_content_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalImageStorage()
            storage._root = pathlib.Path(temp_dir) / "profile_insight_engine"
            stale = storage._root / "stale-scope"
            stale.mkdir(parents=True)
            (stale / "canonical.png").write_bytes(b"sensitive-image-bytes")
            orphan = storage._root / "orphan.tmp"
            orphan.write_bytes(b"sensitive-image-bytes")
            cleaned = await storage.cleanup_all_scopes_async()
            self.assertEqual(cleaned, 2)
            self.assertTrue(storage._root.is_dir())
            self.assertEqual(list(storage._root.iterdir()), [])

    def test_lifespan_runs_cleanup_and_database_recovery_before_serving(self):
        text = WEB.read_text(encoding="utf-8")
        ast.parse(text)
        cleanup_index = text.index("cleanup_all_scopes_async")
        recovery_index = text.index("recover_interrupted_async")
        yield_index = text.index("yield", recovery_index)
        self.assertLess(cleanup_index, yield_index)
        self.assertLess(recovery_index, yield_index)
        self.assertIn("lifespan=lifespan", text)

    def test_restart_recovery_is_repository_owned_and_does_not_add_manual_di_registration(self):
        repository_text = REPOSITORY.read_text(encoding="utf-8")
        service_text = SERVICE.read_text(encoding="utf-8")
        self.assertIn("fail_interrupted_async", repository_text)
        self.assertIn("ProfileAnalysisStatus.PENDING.value", repository_text)
        self.assertIn("ProfileAnalysisStatus.IN_PROGRESS.value", repository_text)
        self.assertIn("ProfileAnalysisStatus.FAILED.value", repository_text)
        self.assertIn("recover_interrupted_async", service_text)
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in (WEB, SERVICE, REPOSITORY, STORAGE)
        )
        self.assertNotIn("register(", combined)
        self.assertNotIn("raw_text", WEB.read_text(encoding="utf-8"))
        self.assertNotIn("logger", STORAGE.read_text(encoding="utf-8"))
        self.assertNotIn("logging", STORAGE.read_text(encoding="utf-8"))


@unittest.skipUnless(
    os.environ.get("PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"),
    "set PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL to a disposable PostgreSQL database",
)
class Phase10RecoveryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repository_marks_only_interrupted_rows_failed(self):
        from sqlalchemy import delete

        from src.core.profile_analysis.contracts import ProfileAnalysisStatus
        from src.infrastructure.context.sql_db.psql_dbcontext import PsqlDbContext
        from src.infrastructure.models.profile_analysis import ProfileAnalysisModel
        from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import (
            ProfileAnalysisRepository,
        )

        class _Config:
            def get_non_empty_string(self, key):
                if key != "database.url_env":
                    raise AssertionError(key)
                return "PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"

        db = PsqlDbContext(_Config())
        repository = ProfileAnalysisRepository(db)
        try:
            async with db.engine.begin() as connection:
                await connection.execute(delete(ProfileAnalysisModel))
            await repository.create_pending_async("pending", ("img-1",))
            await repository.create_pending_async("running", ("img-1",))
            await repository.transition_async(
                "running",
                expected_status=ProfileAnalysisStatus.PENDING,
                new_status=ProfileAnalysisStatus.IN_PROGRESS,
            )
            await repository.create_pending_async("completed", ("img-1",))
            await repository.transition_async(
                "completed",
                expected_status=ProfileAnalysisStatus.PENDING,
                new_status=ProfileAnalysisStatus.IN_PROGRESS,
            )
            await repository.transition_async(
                "completed",
                expected_status=ProfileAnalysisStatus.IN_PROGRESS,
                new_status=ProfileAnalysisStatus.COMPLETED,
                result_payload={"schema_version": "test"},
            )
            recovered = await repository.fail_interrupted_async()
            self.assertEqual(recovered, 2)
            self.assertEqual((await repository.get_by_id_async("pending")).status, "failed")
            self.assertEqual((await repository.get_by_id_async("running")).status, "failed")
            self.assertEqual((await repository.get_by_id_async("completed")).status, "completed")
        finally:
            async with db.engine.begin() as connection:
                await connection.execute(delete(ProfileAnalysisModel))
            await db.engine.dispose()


if __name__ == "__main__":
    unittest.main()
