import unittest

from sqlalchemy.exc import IntegrityError

from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import (
    ProfileAnalysisAlreadyExistsError,
    ProfileAnalysisRepository,
)


class _OrigIntegrityError(Exception):
    def __init__(self, sqlstate: str):
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


class ProfileAnalysisRepositoryErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_unique_violation_maps_to_already_exists(self):
        repository = ProfileAnalysisRepository(object())

        async def fail_unique(_entity):
            raise IntegrityError(
                "insert",
                {},
                _OrigIntegrityError("23505"),
            )

        repository.insert_async = fail_unique
        with self.assertRaises(ProfileAnalysisAlreadyExistsError):
            await repository.create_pending_async("analysis-1", ("img-1",))

    async def test_non_unique_integrity_error_is_not_masked_as_duplicate(self):
        repository = ProfileAnalysisRepository(object())

        async def fail_check(_entity):
            raise IntegrityError(
                "insert",
                {},
                _OrigIntegrityError("23514"),
            )

        repository.insert_async = fail_check
        with self.assertRaises(IntegrityError):
            await repository.create_pending_async("analysis-1", ("img-1",))


if __name__ == "__main__":
    unittest.main()
