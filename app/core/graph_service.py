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

from app.core.entity_resolver import normalize, resolve_person
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

            # ── 5. resolve / create every extracted Person ────────────────────
            # raw_name → person_id; built up incrementally so we never resolve
            # the same name twice within the same article.
            name_to_id: dict[str, str] = {}

            for ep in result.people:
                await self._resolve_or_create(ep.name, repo, name_to_id)

            # ── 6. upsert Relationships + Provenance ──────────────────────────
            rel_count = 0
            for er in result.relationships:
                src_id = name_to_id.get(er.source_person) or await self._resolve_or_create(
                    er.source_person, repo, name_to_id
                )
                tgt_id = name_to_id.get(er.target_person) or await self._resolve_or_create(
                    er.target_person, repo, name_to_id
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
            "people_resolved": len(name_to_id),
            "relationships_stored": rel_count,
            "extraction_error": result.error,
        }

    async def _resolve_or_create(
        self,
        raw_name: str,
        repo: GraphRepository,
        cache: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Resolve *raw_name* to a person_id, creating a new Person if needed.
        Optionally update *cache* with the result.
        """
        resolved = await resolve_person(raw_name, repo, self._extractor)
        if resolved:
            person_id = resolved[0]
        else:
            person_id = await repo.get_or_create_person(raw_name)
            await repo.add_alias(person_id, normalize(raw_name))

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

        total_articles = 0
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
                        total_articles += 1
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
            "articles_processed": total_articles,
            "relationships_stored": total_relationships,
            "errors": errors,
        }

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

            relationships = await repo.get_person_relationships(person_id)
            rels_with_prov = []
            for rel in relationships:
                prov = await repo.get_provenance_for_relationship(rel.id)
                rels_with_prov.append({
                    "relationship": rel,
                    "provenance": prov,
                })

            return {
                "person": person,
                "relationships": rels_with_prov,
            }
