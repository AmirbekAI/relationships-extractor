"""
GraphRepository — all database reads and writes.

Every method works inside the session/transaction passed at construction time.
The caller (service layer) is responsible for session lifecycle.

Entity-resolution write path
─────────────────────────────
  get_or_create_person(canonical_name)
      → always returns a Person id; creates the row if missing.

  find_person_by_alias(surface_form)
      → looks up the normalised surface form in the aliases table.
        Returns (person_id, canonical_name) or None.

  add_alias(person_id, surface_form)
      → inserts; silently skips if the surface form already exists.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import (
    Alias,
    Article,
    Person,
    Provenance,
    Relationship,
    _uuid,
)

logger = logging.getLogger(__name__)


class GraphRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    # ──────────────────────────────────────────────────────────────── People

    async def get_or_create_person(self, canonical_name: str, bio: str | None = None) -> str:
        """Return existing person id or create a new one."""
        result = await self._s.execute(
            select(Person.id).where(Person.canonical_name == canonical_name)
        )
        existing_id = result.scalar_one_or_none()
        if existing_id:
            return existing_id

        person_id = _uuid()
        try:
            # SAVEPOINT: a concurrent insert of the same canonical_name rolls
            # back only this insert, not the caller's outer transaction.
            async with self._s.begin_nested():
                self._s.add(Person(id=person_id, canonical_name=canonical_name, bio=bio))
        except IntegrityError:
            result = await self._s.execute(
                select(Person.id).where(Person.canonical_name == canonical_name)
            )
            return result.scalar_one()
        return person_id

    async def get_person(self, person_id: str) -> Person | None:
        """Fetch a person with aliases eager-loaded (safe to use post-session)."""
        result = await self._s.execute(
            select(Person)
            .options(selectinload(Person.aliases))
            .where(Person.id == person_id)
        )
        return result.scalar_one_or_none()

    async def list_people(self, page: int, page_size: int) -> tuple[list[Person], int]:
        """Paginated list of Person rows (aliases eager-loaded) + total count."""
        total = await self._s.scalar(select(func.count()).select_from(Person)) or 0

        result = await self._s.execute(
            select(Person)
            .options(selectinload(Person.aliases))
            .order_by(Person.canonical_name)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def count_people(self) -> int:
        return await self._s.scalar(select(func.count()).select_from(Person)) or 0

    async def get_person_relationships(self, person_id: str) -> list[Relationship]:
        """
        All relationships touching *person_id*, with both endpoint Persons and
        each Provenance.article eager-loaded so the route can build full DTOs
        after the session closes.
        """
        result = await self._s.execute(
            select(Relationship)
            .options(
                selectinload(Relationship.source_person),
                selectinload(Relationship.target_person),
                selectinload(Relationship.provenance).selectinload(Provenance.article),
            )
            .where(
                (Relationship.source_person_id == person_id)
                | (Relationship.target_person_id == person_id)
            )
        )
        return list(result.scalars().all())

    async def count_relationships(self) -> int:
        return await self._s.scalar(select(func.count()).select_from(Relationship)) or 0

    # ──────────────────────────────────────────────────────────────── Aliases

    async def find_person_by_alias(self, surface_form: str) -> tuple[str, str] | None:
        """
        Look up a normalised surface form in the aliases table.
        Returns (person_id, canonical_name) or None.
        """
        result = await self._s.execute(
            select(Alias.person_id)
            .where(Alias.surface_form == surface_form)
        )
        person_id = result.scalar_one_or_none()
        if not person_id:
            return None
        person = await self.get_person(person_id)
        if not person:
            return None
        return person_id, person.canonical_name

    async def add_alias(self, person_id: str, surface_form: str) -> None:
        """Insert alias; silently ignore if the surface form already exists."""
        existing = await self._s.execute(
            select(Alias.id).where(Alias.surface_form == surface_form)
        )
        if existing.scalar_one_or_none():
            return
        try:
            # SAVEPOINT: if the surface form was inserted concurrently, undo
            # just this insert and leave the outer transaction usable.
            async with self._s.begin_nested():
                self._s.add(Alias(id=_uuid(), surface_form=surface_form, person_id=person_id))
        except IntegrityError:
            pass  # inserted concurrently — fine

    async def get_all_aliases(self) -> list[tuple[str, str, str]]:
        """
        Return all (alias.surface_form, person.id, person.canonical_name) rows.
        Used by the entity resolver to build its lookup table.
        """
        result = await self._s.execute(
            select(Alias.surface_form, Person.id, Person.canonical_name)
            .join(Person, Person.id == Alias.person_id)
        )
        return result.all()

    # ──────────────────────────────────────────────────────────────── Articles

    async def get_article_by_url(self, url: str) -> Article | None:
        """Return the Article row for *url*, or None."""
        result = await self._s.execute(select(Article).where(Article.url == url))
        return result.scalar_one_or_none()

    async def upsert_article(
        self,
        url: str,
        title: str | None,
        published_at: datetime | None,
        author: str | None,
        source: str | None,
        *,
        body_hash: str | None = None,
        sentences_per_chunk: int | None = None,
        total_chunks: int | None = None,
    ) -> str:
        """
        Insert a fresh Article row, or return the existing id without touching
        its checkpoint fields. The first insert seeds body_hash / sentences_per_chunk
        / total_chunks (frozen for the article's lifetime); subsequent calls leave
        those alone so a partially-processed row keeps its pointer.
        """
        result = await self._s.execute(select(Article.id).where(Article.url == url))
        existing_id = result.scalar_one_or_none()
        if existing_id:
            return existing_id

        article_id = _uuid()
        self._s.add(Article(
            id=article_id,
            url=url,
            title=title,
            published_at=published_at,
            author=author,
            source=source,
            body_hash=body_hash,
            sentences_per_chunk=sentences_per_chunk,
            total_chunks=total_chunks,
            chunks_processed=0,
        ))
        await self._s.flush()
        return article_id

    async def update_chunk_progress(self, article_id: str, chunks_processed: int) -> None:
        """Bump the checkpoint pointer after a chunk has been durably stored."""
        article = await self._s.get(Article, article_id)
        if article is not None:
            article.chunks_processed = chunks_processed
            await self._s.flush()

    async def reset_article_for_rerun(
        self,
        article_id: str,
        *,
        body_hash: str,
        sentences_per_chunk: int,
        total_chunks: int,
    ) -> None:
        """
        The article's body changed (or chunking config did) since the last
        run. Wipe its provenance + relationships derived purely from this
        article, and reset the checkpoint so processing starts fresh.

        Person + Alias rows are intentionally left intact — they may be
        referenced by other articles and the resolver will re-resolve names
        the same way on the next run.
        """
        # Provenance referencing this article (CASCADE wipes them via FK if we
        # delete the article, but here we're not deleting — we're rewinding).
        from sqlalchemy import delete
        await self._s.execute(
            delete(Provenance).where(Provenance.article_id == article_id)
        )
        # Now delete relationships that have no remaining provenance.
        await self._s.flush()
        orphan_rels = await self._s.execute(
            select(Relationship.id).where(
                ~select(Provenance.id)
                .where(Provenance.relationship_id == Relationship.id)
                .exists()
            )
        )
        orphan_ids = [r for (r,) in orphan_rels.all()]
        if orphan_ids:
            await self._s.execute(
                delete(Relationship).where(Relationship.id.in_(orphan_ids))
            )

        # Reset the article's checkpoint fields.
        article = await self._s.get(Article, article_id)
        if article is not None:
            article.body_hash = body_hash
            article.sentences_per_chunk = sentences_per_chunk
            article.total_chunks = total_chunks
            article.chunks_processed = 0
        await self._s.flush()

    # ────────────────────────────────────────────────────────── Relationships

    async def upsert_relationship(
        self,
        source_person_id: str,
        target_person_id: str,
        relation_type: str,
        explanation: str,
    ) -> str:
        """Return existing relationship id or create a new one."""
        result = await self._s.execute(
            select(Relationship.id).where(
                Relationship.source_person_id == source_person_id,
                Relationship.target_person_id == target_person_id,
                Relationship.relation_type == relation_type,
            )
        )
        existing_id = result.scalar_one_or_none()
        if existing_id:
            return existing_id

        rel_id = _uuid()
        self._s.add(Relationship(
            id=rel_id,
            source_person_id=source_person_id,
            target_person_id=target_person_id,
            relation_type=relation_type,
            explanation=explanation,
        ))
        await self._s.flush()
        return rel_id

    async def add_provenance(
        self,
        relationship_id: str,
        article_id: str,
        quote: str | None,
    ) -> None:
        """Add provenance record; silently skip if (relationship, article) pair already exists."""
        existing = await self._s.execute(
            select(Provenance.id).where(
                Provenance.relationship_id == relationship_id,
                Provenance.article_id == article_id,
            )
        )
        if existing.scalar_one_or_none():
            return
        self._s.add(Provenance(
            id=_uuid(),
            relationship_id=relationship_id,
            article_id=article_id,
            quote=quote,
        ))
        await self._s.flush()

    async def get_provenance_for_relationship(self, relationship_id: str) -> list[Provenance]:
        result = await self._s.execute(
            select(Provenance).where(Provenance.relationship_id == relationship_id)
        )
        return result.scalars().all()
