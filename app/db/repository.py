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

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
        self._s.add(Person(id=person_id, canonical_name=canonical_name, bio=bio))
        try:
            await self._s.flush()
        except IntegrityError:
            await self._s.rollback()
            result = await self._s.execute(
                select(Person.id).where(Person.canonical_name == canonical_name)
            )
            return result.scalar_one()
        return person_id

    async def get_person(self, person_id: str) -> Person | None:
        result = await self._s.execute(select(Person).where(Person.id == person_id))
        return result.scalar_one_or_none()

    async def list_people(self, page: int, page_size: int) -> tuple[list[Person], int]:
        count_result = await self._s.execute(select(Person.id))
        total = len(count_result.all())

        result = await self._s.execute(
            select(Person)
            .order_by(Person.canonical_name)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return result.scalars().all(), total

    async def get_person_relationships(self, person_id: str) -> list[Relationship]:
        result = await self._s.execute(
            select(Relationship).where(
                (Relationship.source_person_id == person_id)
                | (Relationship.target_person_id == person_id)
            )
        )
        return result.scalars().all()

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
        self._s.add(Alias(id=_uuid(), surface_form=surface_form, person_id=person_id))
        try:
            await self._s.flush()
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

    async def upsert_article(
        self,
        url: str,
        title: str | None,
        published_at: datetime | None,
        author: str | None,
        source: str | None,
    ) -> str:
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
        ))
        await self._s.flush()
        return article_id

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
