"""
API contract — request and response DTOs.

All request/response shapes for the four endpoints:
  POST /articles
  POST /rescan
  GET  /people
  GET  /people/{id}
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, AnyHttpUrl, Field


# ─────────────────────────────────────────────
# POST /articles
# ─────────────────────────────────────────────

class ArticleSubmitRequest(BaseModel):
    url: AnyHttpUrl = Field(
        description="URL of a TechCrunch (or other supported) article to fetch and process."
    )
    sentences_per_chunk: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "How many sentences to send to the LLM in a single extraction call. "
            "Overrides the server default when provided. "
            "Tune this to find the sweet spot between context richness and token cost."
        ),
    )


class ArticleSubmitResponse(BaseModel):
    article_id: str = Field(description="Internal ID assigned to this article.")
    url: str
    title: Optional[str]
    people_found: int = Field(description="Number of distinct people extracted.")
    relationships_found: int = Field(description="Number of relationships extracted.")
    status: str = Field(description="'processed' or 'already_exists'.")


# ─────────────────────────────────────────────
# POST /rescan
# ─────────────────────────────────────────────

class RescanRequest(BaseModel):
    pages: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of topic-listing pages to crawl (page 1 = most recent).",
    )
    sentences_per_chunk: Optional[int] = Field(
        default=None,
        ge=1,
        description="Sentences-per-chunk override applied to every article in this scan.",
    )
    max_parallel: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Override the server's max_parallel_articles for this scan only. "
            "Higher values process articles concurrently — the crawler still "
            "enforces per-host politeness internally."
        ),
    )


class RescanResponse(BaseModel):
    pages_crawled: int
    articles_processed: int
    articles_skipped: int
    new_people: int = Field(description="People nodes created during this scan.")
    new_relationships: int = Field(description="Relationship edges created during this scan.")
    status: str = Field(description="'complete' or 'partial' (if some articles failed).")


# ─────────────────────────────────────────────
# GET /people  (paginated list)
# ─────────────────────────────────────────────

class PersonSummary(BaseModel):
    id: str
    canonical_name: str
    aliases: list[str] = Field(description="All known surface forms for this person.")


class PaginatedPeopleResponse(BaseModel):
    items: list[PersonSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


# ─────────────────────────────────────────────
# GET /people/{id}
# ─────────────────────────────────────────────

class ProvenanceDTO(BaseModel):
    article_id: str
    article_url: str
    article_title: Optional[str]
    quote: Optional[str] = Field(
        description="The sentence / passage from the article that justifies this relationship."
    )


class RelationshipDTO(BaseModel):
    id: str
    source_person_id: str
    source_person_name: str
    target_person_id: str
    target_person_name: str
    relation_type: str = Field(description="Short verb phrase, e.g. 'criticizes', 'partners with'.")
    explanation: str = Field(description="1-2 sentence human-readable description.")
    provenance: list[ProvenanceDTO]


class PersonDetailResponse(BaseModel):
    id: str
    canonical_name: str
    aliases: list[str]
    bio: Optional[str]
    relationships: list[RelationshipDTO]
