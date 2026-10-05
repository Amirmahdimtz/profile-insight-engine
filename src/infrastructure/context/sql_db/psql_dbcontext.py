from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class Base(DeclarativeBase):
    pass


class DatabaseConfigurationError(RuntimeError):
    """Raised when the PostgreSQL async connection configuration is unavailable or invalid."""


@inject
class PsqlDbContext:
    __di_singleton__ = True

    def __init__(self, config_reader: ConfigReader):
        env_name = config_reader.get_non_empty_string("database.url_env")
        database_url = os.environ.get(env_name, "").strip()
        if not database_url:
            raise DatabaseConfigurationError(
                f"database URL environment variable '{env_name}' is not set"
            )
        if not database_url.startswith("postgresql+asyncpg://"):
            raise DatabaseConfigurationError(
                "database URL must use the postgresql+asyncpg SQLAlchemy dialect"
            )
        self._engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as session:
            yield session
