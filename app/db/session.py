"""
Database session setup.

Exposes:
  init_db(url)   — create tables, wire up the engine (called once at startup)
  get_session()  — async context manager yielding a transactional session
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base

logger = logging.getLogger(__name__)

_async_session: async_sessionmaker | None = None


async def init_db(database_url: str) -> None:
    """Create all tables and initialise the shared session factory."""
    global _async_session

    engine = create_async_engine(database_url, echo=False, future=True)
    _async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database ready: %s", database_url)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _async_session is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    async with _async_session() as session:
        async with session.begin():
            yield session
