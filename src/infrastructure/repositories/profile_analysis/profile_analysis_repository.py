from __future__ import annotations

from typing import Mapping, Sequence

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from src.core.profile_analysis.contracts import ProfileAnalysisStatus
from src.infrastructure.context.sql_db.psql_dbcontext import PsqlDbContext
from src.infrastructure.di.inject import inject
from src.infrastructure.models.profile_analysis import ProfileAnalysisModel
from src.infrastructure.repositories.base.base_repository import BaseRepository


class ProfileAnalysisRepositoryError(RuntimeError):
    """Base persistence error for profile analysis state."""


class ProfileAnalysisAlreadyExistsError(ProfileAnalysisRepositoryError):
    pass


class ProfileAnalysisTransitionConflictError(ProfileAnalysisRepositoryError):
    pass


_UNIQUE_VIOLATION_SQLSTATE = "23505"


def _is_unique_violation(exc: IntegrityError) -> bool:
    return getattr(exc.orig, "sqlstate", None) == _UNIQUE_VIOLATION_SQLSTATE


@inject
class ProfileAnalysisRepository(BaseRepository[ProfileAnalysisModel]):
    def __init__(self, db_context: PsqlDbContext):
        super().__init__(db_context=db_context, model=ProfileAnalysisModel)

    async def create_pending_async(
        self,
        analysis_id: str,
        image_ids: Sequence[str],
    ) -> ProfileAnalysisModel:
        entity = ProfileAnalysisModel(
            id=analysis_id,
            status=ProfileAnalysisStatus.PENDING.value,
            image_ids=list(image_ids),
            result_payload=None,
        )
        try:
            return await self.insert_async(entity)
        except IntegrityError as exc:
            if not _is_unique_violation(exc):
                raise
            raise ProfileAnalysisAlreadyExistsError(
                f"analysis '{analysis_id}' already exists"
            ) from exc

    async def transition_async(
        self,
        analysis_id: str,
        *,
        expected_status: ProfileAnalysisStatus,
        new_status: ProfileAnalysisStatus,
        result_payload: Mapping[str, object] | None = None,
    ) -> ProfileAnalysisModel:
        values: dict[str, object] = {
            "status": new_status.value,
            "result_payload": dict(result_payload) if result_payload is not None else None,
        }
        async with self._db_context.session() as session:
            try:
                statement = (
                    update(ProfileAnalysisModel)
                    .where(
                        ProfileAnalysisModel.id == analysis_id,
                        ProfileAnalysisModel.status == expected_status.value,
                    )
                    .values(**values)
                    .returning(ProfileAnalysisModel)
                )
                result = await session.execute(statement)
                entity = result.scalar_one_or_none()
                if entity is None:
                    await session.rollback()
                    raise ProfileAnalysisTransitionConflictError(
                        f"analysis '{analysis_id}' is not in expected status '{expected_status.value}'"
                    )
                await session.commit()
                return entity
            except ProfileAnalysisTransitionConflictError:
                raise
            except Exception:
                await session.rollback()
                raise

    async def fail_interrupted_async(self) -> int:
        """Fail unfinished rows left by a previous single-worker process lifetime."""

        async with self._db_context.session() as session:
            try:
                statement = (
                    update(ProfileAnalysisModel)
                    .where(
                        ProfileAnalysisModel.status.in_(
                            (
                                ProfileAnalysisStatus.PENDING.value,
                                ProfileAnalysisStatus.IN_PROGRESS.value,
                            )
                        ),
                        ProfileAnalysisModel.result_payload.is_(None),
                    )
                    .values(status=ProfileAnalysisStatus.FAILED.value)
                )
                result = await session.execute(statement)
                await session.commit()
                return int(result.rowcount or 0)
            except Exception:
                await session.rollback()
                raise
