"""
GraphService — top-level pipeline orchestrator.

Crawl → Extract → Resolve names → Persist → Read

Public API
──────────
  process_article(url, sentences_per_chunk)
      Full pipeline for one URL. Returns a summary dict.

  rescan(pages, sentences_per_chunk, source_ids)
      Iterate listing pages across all (or a subset of) registered crawlers
      and call process_article() for each discovered URL.

  get_people(page, page_size)
      Paginated list of Person ORM rows + total count.

  get_person_detail(person_id)
      Person row + all Relationships with their Provenance records.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import get_settings
from app.core.entity_resolver import (
    build_token_owners,
    normalize,
    resolve_person,
    update_token_owners_and_recency,
)
from app.crawlers.base import CrawlerRegistry
from app.db.repository import GraphRepository
from app.db.session import get_session
from app.extractors.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 5


class GraphService:
    """
    Stateless service — every public method opens and closes its own DB
    session so callers do not need to manage transactions.

    Inject an *LLMExtractor* at construction time; you can swap the
    underlying client (OpenAI vs local model) without touching this class.
    """

    def __init__(self, extractor: LLMExtractor) -> None:
        self._extractor = extractor

    # ──────────────────────────────────────────────── write: single article

    async def process_article(
        self,
        url: str,
        sentences_per_chunk: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline for one article URL.

        Steps
        -----
        1. Pick the right crawler from the registry
        2. Crawl the article (title, author, published_at, body)
        3. Run LLM extraction in sentence chunks
        4. For every extracted person:
              alias lookup → Levenshtein → LLM fallback → create new Person
        5. Upsert Relationship + Provenance rows for each extracted relationship
        6. Return a summary dict with counts

        Raises
        ------
        ValueError   if no crawler is registered for the URL
        RuntimeError if the crawler returns nothing
        """
        chunk_size = sentences_per_chunk or _DEFAULT_CHUNK_SIZE

        # ── 1. select crawler ─────────────────────────────────────────────────
        crawler = CrawlerRegistry.for_url(url)
        if crawler is None:
            raise ValueError(f"No crawler registered for URL: {url}")

        # ── 1b. fast-path: skip already-processed URLs ────────────────────────
        # Avoids re-running the LLM on every re-scan; the API surfaces this
        # as status="already_exists".
        async with get_session() as session:
            existing = await GraphRepository(session).get_article_by_url(url)
        if existing is not None:
            return {
                "article_url": url,
                "article_id": existing.id,
                "title": existing.title,
                "people_resolved": 0,
                "relationships_stored": 0,
                "extraction_error": None,
                "status": "already_exists",
            }

        # ── 2. crawl ──────────────────────────────────────────────────────────
        article = await crawler.fetch_article(url)
        if article is None:
            raise RuntimeError(f"Crawler returned nothing for: {url}")

        # ── 3. extract ────────────────────────────────────────────────────────
        result = await self._extractor.extract(article, sentences_per_chunk=chunk_size)
        if result.error:
            logger.warning("Extraction error for %s: %s", url, result.error)

        async with get_session() as session:
            repo = GraphRepository(session)

            # ── 4. upsert the Article row ─────────────────────────────────────
            article_id = await repo.upsert_article(
                url=article.url,
                title=article.title,
                published_at=article.published_at,
                author=article.author,
                source=article.source,
            )

            # ── 4b. optional: prepare per-article recency state ───────────────
            # When the setting is on: seed the long-token → owners map once
            # from the current DB snapshot, and start an empty recency map.
            # Both are maintained dynamically by _resolve_or_create — every
            # resolution or creation can add a new owner to a token, and the
            # moment a token has 2+ owners (contested) the recency map starts
            # tracking it. So a collision that emerges mid-article gets
            # picked up immediately, not at the next article boundary.
            use_recency = get_settings().resolver_recency_enabled
            recency: dict[str, str] = {}
            token_owners: dict[str, set[str]] = {}
            if use_recency:
                token_owners = build_token_owners(await repo.get_all_aliases())

            # ── 5. resolve / create every extracted Person ────────────────────
            # raw_name → person_id; built up incrementally so we never resolve
            # the same name twice within the same article.
            name_to_id: dict[str, str] = {}

            for ep in result.people:
                await self._resolve_or_create(
                    ep.name, repo, name_to_id,
                    recency=recency, token_owners=token_owners,
                )

            # ── 6. upsert Relationships + Provenance ──────────────────────────
            rel_count = 0
            for er in result.relationships:
                src_id = name_to_id.get(er.source_person) or await self._resolve_or_create(
                    er.source_person, repo, name_to_id,
                    recency=recency, token_owners=token_owners,
                )
                tgt_id = name_to_id.get(er.target_person) or await self._resolve_or_create(
                    er.target_person, repo, name_to_id,
                    recency=recency, token_owners=token_owners,
                )

                rel_id = await repo.upsert_relationship(
                    source_person_id=src_id,
                    target_person_id=tgt_id,
                    relation_type=er.relation_type,
                    explanation=er.explanation,
                )
                await repo.add_provenance(rel_id, article_id, er.supporting_quote)
                rel_count += 1

        return {
            "article_url": url,
            "article_id": article_id,
            "title": article.title,
            "people_resolved": len(name_to_id),
            "relationships_stored": rel_count,
            "extraction_error": result.error,
            "status": "processed",
        }

    async def _resolve_or_create(
        self,
        raw_name: str,
        repo: GraphRepository,
        cache: Optional[dict[str, str]] = None,
        *,
        recency: Optional[dict[str, str]] = None,
        token_owners: Optional[dict[str, set[str]]] = None,
    ) -> str:
        """
        Resolve *raw_name* to a person_id, creating a new Person if needed.
        Optionally update *cache* with the result.

        When *recency* and *token_owners* are both supplied (recency mode on),
        every successful resolution OR creation updates the maps in-place:
        the person's long tokens get added to ``token_owners``, and any token
        whose owner set is now >= 2 also gets ``recency[token] = person_id``.
        That's how a collision that emerges *mid-article* is detected the
        moment it happens, rather than at the next article boundary.
        """
        resolved = await resolve_person(
            raw_name, repo, self._extractor,
            recency=recency,
        )
        if resolved:
            person_id, canonical_name = resolved
        else:
            person_id = await repo.get_or_create_person(raw_name)
            await repo.add_alias(person_id, normalize(raw_name))
            canonical_name = raw_name

        if recency is not None and token_owners is not None:
            update_token_owners_and_recency(
                person_id, canonical_name, token_owners, recency,
            )

        if cache is not None:
            cache[raw_name] = person_id
        return person_id

    # ──────────────────────────────────────────────────────── write: rescan

    async def rescan(
        self,
        pages: int = 1,
        sentences_per_chunk: Optional[int] = None,
        source_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Iterate up to *pages* listing pages for each registered crawler
        (or just those in *source_ids*) and process every discovered article.

        Returns aggregate counts and a list of per-URL error strings.
        """
        crawlers = list(CrawlerRegistry.all().values())
        if source_ids:
            crawlers = [c for c in crawlers if c.source_id in source_ids]

        total_processed = 0
        total_skipped = 0
        total_relationships = 0
        errors: list[str] = []

        for crawler in crawlers:
            for page in range(1, pages + 1):
                try:
                    urls = await crawler.get_article_urls(page=page)
                except Exception as exc:
                    msg = f"{crawler.source_id} page {page}: {exc}"
                    logger.error(msg)
                    errors.append(msg)
                    continue

                for url in urls:
                    try:
                        summary = await self.process_article(url, sentences_per_chunk)
                        if summary.get("status") == "already_exists":
                            total_skipped += 1
                            continue
                        total_processed += 1
                        total_relationships += summary["relationships_stored"]
                        logger.info(
                            "Processed %s: %d people, %d relationships",
                            url,
                            summary["people_resolved"],
                            summary["relationships_stored"],
                        )
                    except Exception as exc:
                        msg = f"{url}: {exc}"
                        logger.error(msg)
                        errors.append(msg)

        return {
            "pages_crawled": pages,
            "articles_processed": total_processed,
            "articles_skipped": total_skipped,
            "relationships_stored": total_relationships,
            "errors": errors,
        }

    # ──────────────────────────────────────────────────────────────── helpers

    async def get_counts(self) -> tuple[int, int]:
        """Return (people_count, relationship_count) for the whole graph."""
        async with get_session() as session:
            repo = GraphRepository(session)
            return await repo.count_people(), await repo.count_relationships()

    async def get_article_by_url(self, url: str):
        """Return the Article row for *url*, or None."""
        async with get_session() as session:
            return await GraphRepository(session).get_article_by_url(url)

    # ──────────────────────────────────────────────────────────── read

    async def get_people(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list, int]:
        """
        Return (list[Person], total_count) — paginated, ordered by canonical_name.
        """
        async with get_session() as session:
            repo = GraphRepository(session)
            return await repo.list_people(page=page, page_size=page_size)

    async def get_person_detail(self, person_id: str) -> Optional[dict[str, Any]]:
        """
        Return a rich detail dict for one person, or ``None`` if not found.

        Shape
        -----
        {
          "person": Person,
          "relationships": [
            {
              "relationship": Relationship,
              "provenance":   [Provenance, ...]
            },
            ...
          ]
        }
        """
        async with get_session() as session:
            repo = GraphRepository(session)
            person = await repo.get_person(person_id)
            if person is None:
                return None

            # Relationships come with both endpoints + provenance.article
            # eager-loaded by the repo, so the route can build DTOs after
            # the session closes without triggering lazy-load errors.
            relationships = await repo.get_person_relationships(person_id)

            return {
                "person": person,
                "relationships": relationships,
            }
