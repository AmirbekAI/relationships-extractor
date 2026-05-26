"""
Shared domain dataclasses.

These are plain Python — no ORM, no Pydantic. Every other layer imports from
here so the types stay consistent across extractor, resolver, and service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedPerson:
    name: str
    role: Optional[str] = None          # e.g. "CEO of OpenAI", "journalist"


@dataclass
class ExtractedRelationship:
    source_person: str
    target_person: str
    relation_type: str                  # e.g. "criticizes", "partners with"
    explanation: str
    supporting_quote: str


@dataclass
class ExtractionResult:
    """Output of the extractor for one article (or one chunk)."""
    article_url: str
    people: list[ExtractedPerson] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    error: Optional[str] = None
