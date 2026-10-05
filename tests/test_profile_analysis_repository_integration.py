import asyncio
import os
import unittest

from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError

from src.core.profile_analysis.contracts import ProfileAnalysisStatus
from src.infrastructure.context.sql_db.psql_dbcontext import PsqlDbContext
from src.infrastructure.models.profile_analysis import ProfileAnalysisModel
from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import (
    ProfileAnalysisAlreadyExistsError,
    ProfileAnalysisRepository,
    ProfileAnalysisTransitionConflictError,
)


_TEST_URL_ENV = "PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"


class _TestConfig:
    def get_non_empty_string(self, key):
        if key != "database.url_env":
            raise AssertionError(key)
        return _TEST_URL_ENV


@unittest.skipUnless(os.environ.get(_TEST_URL_ENV), f"set {_TEST_URL_ENV} to a disposable PostgreSQL database")
class ProfileAnalysisRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = PsqlDbContext(_TestConfig())
        async with self.db.engine.begin() as connection:
            await connection.execute(delete(ProfileAnalysisModel))
        self.repository = ProfileAnalysisRepository(self.db)

    async def asyncTearDown(self):
        async with self.db.engine.begin() as connection:
            await connection.execute(delete(ProfileAnalysisModel))
        await self.db.engine.dispose()

    async def test_read_after_write_atomic_completion_and_concurrent_reads(self):
        created = await self.repository.create_pending_async("analysis-1", ("img-1", "img-2"))
        self.assertEqual(created.status, ProfileAnalysisStatus.PENDING.value)
        read = await self.repository.get_by_id_async("analysis-1")
        self.assertEqual(read.image_ids, ["img-1", "img-2"])

        await self.repository.transition_async(
            "analysis-1",
            expected_status=ProfileAnalysisStatus.PENDING,
            new_status=ProfileAnalysisStatus.IN_PROGRESS,
        )
        payload = {
            "schema_version": "phase9-analysis-result-v1",
            "result": {"analysis_id": "analysis-1"},
            "insight_policy_version": "policy",
            "summary": "summary",
        }
        completed = await self.repository.transition_async(
            "analysis-1",
            expected_status=ProfileAnalysisStatus.IN_PROGRESS,
            new_status=ProfileAnalysisStatus.COMPLETED,
            result_payload=payload,
        )
        self.assertEqual(completed.status, ProfileAnalysisStatus.COMPLETED.value)
        self.assertEqual(completed.result_payload, payload)

        reads = await asyncio.gather(
            *(self.repository.get_by_id_async("analysis-1") for _ in range(8))
        )
        self.assertTrue(all(item.status == ProfileAnalysisStatus.COMPLETED.value for item in reads))

    async def test_duplicate_insert_rolls_back_and_session_remains_usable(self):
        await self.repository.create_pending_async("analysis-1", ("img-1",))
        with self.assertRaises(ProfileAnalysisAlreadyExistsError):
            await self.repository.create_pending_async("analysis-1", ("img-1",))
        created = await self.repository.create_pending_async("analysis-2", ("img-2",))
        self.assertEqual(created.id, "analysis-2")

    async def test_database_constraints_reject_invalid_completed_row(self):
        invalid = ProfileAnalysisModel(
            id="invalid",
            status=ProfileAnalysisStatus.COMPLETED.value,
            image_ids=["img-1"],
            result_payload=None,
        )
        with self.assertRaises(IntegrityError):
            await self.repository.insert_async(invalid)
        created = await self.repository.create_pending_async("valid", ("img-1",))
        self.assertEqual(created.status, ProfileAnalysisStatus.PENDING.value)

    async def test_stale_transition_is_rejected_deterministically(self):
        await self.repository.create_pending_async("analysis-1", ("img-1",))
        with self.assertRaises(ProfileAnalysisTransitionConflictError):
            await self.repository.transition_async(
                "analysis-1",
                expected_status=ProfileAnalysisStatus.IN_PROGRESS,
                new_status=ProfileAnalysisStatus.COMPLETED,
                result_payload={"result": "not-reached"},
            )
        current = await self.repository.get_by_id_async("analysis-1")
        self.assertEqual(current.status, ProfileAnalysisStatus.PENDING.value)
        self.assertIsNone(current.result_payload)


if __name__ == "__main__":
    unittest.main()
