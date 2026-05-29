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

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base

logger = logging.getLogger(__name__)

_async_session: async_sessionmaker | None = None


# ── lightweight idempotent migrations ────────────────────────────────────────
# We don't ship Alembic; this hook adds columns introduced by later versions
# to tables that were created by earlier ones. Idempotent — does nothing on
# DBs that are already up to date, or on fresh DBs (create_all already made
# the columns).
#
# Each entry: (column_name, DDL fragment to append after ADD COLUMN).
# Use defaults so existing rows get a sensible value without a separate UPDATE.

_ARTICLES_NEW_COLUMNS: list[tuple[str, str]] = [
    ("body_hash",            "VARCHAR"),
    ("sentences_per_chunk",  "INTEGER"),
    ("total_chunks",         "INTEGER"),
    ("chunks_processed",     "INTEGER NOT NULL DEFAULT 0"),
]


def _migrate_pending_columns(sync_conn: Connection) -> None:
    """ALTER TABLE for any expected column that's missing on the live DB."""
    inspector = inspect(sync_conn)
    if "articles" not in inspector.get_table_names():
        return  # fresh DB: create_all already made everything

    existing = {col["name"] for col in inspector.get_columns("articles")}
    added: list[str] = []
    for col_name, col_ddl in _ARTICLES_NEW_COLUMNS:
        if col_name in existing:
            continue
        sync_conn.execute(
            text(f"ALTER TABLE articles ADD COLUMN {col_name} {col_ddl}")
        )
        added.append(col_name)
    if added:
        logger.info("Migrated articles table: added column(s) %s", added)


async def init_db(database_url: str) -> None:
    """Create all tables, run pending column migrations, initialise the
    shared session factory."""
    global _async_session

    engine = create_async_engine(database_url, echo=False, future=True)
    _async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_pending_columns)

    logger.info("Database ready: %s", database_url)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _async_session is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    async with _async_session() as session:
        async with session.begin():
            yield session
