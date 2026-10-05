from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from src.infrastructure.context.sql_db.psql_dbcontext import Base, PsqlDbContext
from src.infrastructure.di.inject import inject


TModel = TypeVar("TModel", bound=Base)


@inject
class BaseRepository(Generic[TModel]):
    def __init__(self, db_context: PsqlDbContext, model: type[TModel]):
        self._db_context = db_context
        self._model = model

    async def insert_async(self, entity: TModel) -> TModel:
        async with self._db_context.session() as session:
            try:
                session.add(entity)
                await session.commit()
                await session.refresh(entity)
                return entity
            except Exception:
                await session.rollback()
                raise

    async def get_by_id_async(self, entity_id: str) -> TModel | None:
        async with self._db_context.session() as session:
            return await session.get(self._model, entity_id)

    async def get_all_async(self) -> list[TModel]:
        async with self._db_context.session() as session:
            result = await session.execute(select(self._model))
            return list(result.scalars().all())

    async def update_async(self, entity: TModel) -> TModel:
        async with self._db_context.session() as session:
            try:
                merged = await session.merge(entity)
                await session.commit()
                await session.refresh(merged)
                return merged
            except Exception:
                await session.rollback()
                raise

    async def delete_async(self, entity_id: str) -> bool:
        async with self._db_context.session() as session:
            try:
                result = await session.execute(
                    delete(self._model).where(self._model.id == entity_id)
                )
                await session.commit()
                return bool(result.rowcount)
            except SQLAlchemyError:
                await session.rollback()
                raise
