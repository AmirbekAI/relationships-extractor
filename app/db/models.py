"""
ORM models — SQLAlchemy table definitions.

Schema design
─────────────
Person      canonical node; one row per real-world individual
Alias       many-to-one → Person; every surface form seen in articles
Article     provenance unit; one row per processed URL
Relationship  directed typed edge between two Person nodes
Provenance  links a Relationship to the Article + quote that justifies it
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp; replaces deprecated datetime.utcnow."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "people"

    id = Column(String, primary_key=True, default=_uuid)
    canonical_name = Column(String, nullable=False, unique=True, index=True)
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    aliases = relationship(
        "Alias", back_populates="person", cascade="all, delete-orphan"
    )
    outgoing = relationship(
        "Relationship",
        foreign_keys="Relationship.source_person_id",
        back_populates="source_person",
        cascade="all, delete-orphan",
    )
    incoming = relationship(
        "Relationship",
        foreign_keys="Relationship.target_person_id",
        back_populates="target_person",
        cascade="all, delete-orphan",
    )


class Alias(Base):
    __tablename__ = "aliases"
    __table_args__ = (UniqueConstraint("surface_form"),)

    id = Column(String, primary_key=True, default=_uuid)
    surface_form = Column(String, nullable=False, index=True)  # normalised form
    person_id = Column(
        String, ForeignKey("people.id", ondelete="CASCADE"), nullable=False, index=True
    )

    person = relationship("Person", back_populates="aliases")


class Article(Base):
    __tablename__ = "articles"

    id = Column(String, primary_key=True, default=_uuid)
    url = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    author = Column(String, nullable=True)
    source = Column(String, nullable=True)  # e.g. "techcrunch"
    processed_at = Column(DateTime(timezone=True), default=_utcnow)

    # ── crash-recovery checkpoint ────────────────────────────────────────
    # body_hash:           SHA-256 of body_text at chunking time. Used to
    #                      detect that the article changed since the
    #                      previous run; on mismatch we restart from zero.
    # sentences_per_chunk: the value used at first chunking. Frozen for
    #                      the life of the article so changing the global
    #                      setting mid-flight doesn't shift chunk boundaries.
    # total_chunks:        how many chunks the body splits into.
    # chunks_processed:    pointer — the next chunk to extract on resume.
    #                      When chunks_processed == total_chunks > 0, the
    #                      article is complete (API surfaces "already_exists").
    body_hash = Column(String, nullable=True)
    sentences_per_chunk = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)
    chunks_processed = Column(Integer, nullable=False, default=0)

    provenance = relationship("Provenance", back_populates="article")


class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        # One logical edge per (source, target, type)
        UniqueConstraint("source_person_id", "target_person_id", "relation_type"),
    )

    id = Column(String, primary_key=True, default=_uuid)
    source_person_id = Column(
        String, ForeignKey("people.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_person_id = Column(
        String, ForeignKey("people.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type = Column(String, nullable=False)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    source_person = relationship(
        "Person", foreign_keys=[source_person_id], back_populates="outgoing"
    )
    target_person = relationship(
        "Person", foreign_keys=[target_person_id], back_populates="incoming"
    )
    provenance = relationship(
        "Provenance", back_populates="rel", cascade="all, delete-orphan"
    )


class Provenance(Base):
    __tablename__ = "provenance"
    __table_args__ = (UniqueConstraint("relationship_id", "article_id"),)

    id = Column(String, primary_key=True, default=_uuid)
    relationship_id = Column(
        String,
        ForeignKey("relationships.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    article_id = Column(
        String, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False
    )
    quote = Column(Text, nullable=True)

    rel = relationship("Relationship", back_populates="provenance")
    article = relationship("Article", back_populates="provenance")
