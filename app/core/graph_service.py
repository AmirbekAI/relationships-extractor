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

import hashlib
import logging
from typing import Any, Optional

from app.config import get_settings
from app.core.entity_resolver import (
    build_token_owners,
    normalize,
    resolve_person,
    update_token_owners_and_recency,
)
from app.crawlers.base import ArticleContent, CrawlerRegistry
from app.db.repository import GraphRepository
from app.db.session import get_session


def _body_hash(text: str) -> str:
    """Stable fingerprint of the article body; used to detect mid-flight edits."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
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
        Run the chunk-checkpointed pipeline for one article URL.

        Steps
        -----
        1. Pick a crawler, crawl the article.
        2. Open ONE setup session/transaction:
             - upsert the Article row (first time: seeds body_hash,
               sentences_per_chunk, total_chunks, chunks_processed=0).
             - if the body_hash changed since the last run → wipe this
               article's provenance + orphan relationships and reset
               chunks_processed to 0 (clean restart).
             - if already complete (chunks_processed == total_chunks > 0)
               return status="already_exists" without extracting.
        3. For each chunk from chunks_processed to total_chunks-1, in its
           OWN session/transaction:
             a. Call extractor.extract_one_chunk(...) — one LLM call.
             b. Resolve people + upsert relationships + provenance from
                this chunk.
             c. Bump article.chunks_processed = i+1.
           A crash anywhere costs at most this one chunk's LLM call on
           resume.

        Raises
        ------
        ValueError   if no crawler is registered for the URL
        RuntimeError if the crawler returns nothing
        """
        chunk_size = sentences_per_chunk or _DEFAULT_CHUNK_SIZE

        # ── 1. select crawler + crawl ─────────────────────────────────────────
        crawler = CrawlerRegistry.for_url(url)
        if crawler is None:
            raise ValueError(f"No crawler registered for URL: {url}")

        article = await crawler.fetch_article(url)
        if article is None:
            raise RuntimeError(f"Crawler returned nothing for: {url}")

        # ── 2. compute chunks + open setup transaction ───────────────────────
        body_hash = _body_hash(article.body_text)
        chunks = self._extractor.split_chunks(article, chunk_size)
        total_chunks = len(chunks)

        settings = get_settings()
        use_recency = settings.resolver_recency_enabled
        use_llm_fallback = settings.resolver_llm_fallback_enabled

        async with get_session() as session:
            repo = GraphRepository(session)
            existing = await repo.get_article_by_url(url)

            if existing is None:
                # Fresh article: insert with frozen chunking params.
                article_id = await repo.upsert_article(
                    url=article.url,
                    title=article.title,
                    published_at=article.published_at,
                    author=article.author,
                    source=article.source,
                    body_hash=body_hash,
                    sentences_per_chunk=chunk_size,
                    total_chunks=total_chunks,
                )
                start_idx = 0
                stored_total = total_chunks
                stored_chunk_size = chunk_size
                logger.info(
                    "Article %s: starting fresh, %d chunk(s) to process.",
                    url, stored_total,
                )
            else:
                article_id = existing.id

                # Legacy row from before checkpointing existed (migration
                # added the columns with NULL/0 defaults). Trust that it
                # was fully processed under the old code path and skip;
                # the user can delete it manually to force a re-run.
                if existing.total_chunks is None:
                    logger.info(
                        "Article %s: pre-checkpoint row, treating as complete.",
                        url,
                    )
                    return {
                        "article_url": url,
                        "article_id": article_id,
                        "title": existing.title,
                        "people_resolved": 0,
                        "relationships_stored": 0,
                        "extraction_error": None,
                        "status": "already_exists",
                    }

                # Body changed since last run? rewind and replay.
                if existing.body_hash != body_hash:
                    logger.info(
                        "Article %s: body changed since last run "
                        "(stored hash %.8s, new hash %.8s); resetting "
                        "checkpoint and replaying from chunk 1/%d.",
                        url, existing.body_hash or "", body_hash, total_chunks,
                    )
                    await repo.reset_article_for_rerun(
                        article_id,
                        body_hash=body_hash,
                        sentences_per_chunk=chunk_size,
                        total_chunks=total_chunks,
                    )
                    start_idx = 0
                    stored_total = total_chunks
                    stored_chunk_size = chunk_size
                else:
                    start_idx = existing.chunks_processed or 0
                    stored_total = existing.total_chunks or total_chunks
                    stored_chunk_size = existing.sentences_per_chunk or chunk_size

                    if start_idx >= stored_total and stored_total > 0:
                        logger.info(
                            "Article %s: already complete (%d/%d chunks), skipping.",
                            url, stored_total, stored_total,
                        )
                        return {
                            "article_url": url,
                            "article_id": article_id,
                            "title": existing.title,
                            "people_resolved": 0,
                            "relationships_stored": 0,
                            "extraction_error": None,
                            "status": "already_exists",
                        }
                    elif start_idx > 0:
                        logger.info(
                            "Article %s: found progress at chunk %d/%d, "
                            "resuming (%d chunk(s) left).",
                            url, start_idx, stored_total, stored_total - start_idx,
                        )
                    else:
                        # Existed but never advanced past chunk 0 (e.g.
                        # crashed on the very first chunk last time).
                        logger.info(
                            "Article %s: row exists but no chunks done yet, "
                            "starting from chunk 1/%d.",
                            url, stored_total,
                        )

        # ── 3. re-chunk under the FROZEN sentences_per_chunk ─────────────────
        # If the global setting differs from what the row was first chunked
        # with, honour the stored value so chunk boundaries don't shift.
        if stored_chunk_size != chunk_size:
            chunks = self._extractor.split_chunks(article, stored_chunk_size)

        if len(chunks) != stored_total:
            # Should be impossible given the body_hash check above, but
            # guard anyway — better to refuse than to scramble.
            raise RuntimeError(
                f"Chunk count drift for {url}: expected {stored_total}, "
                f"computed {len(chunks)}. Delete the article row and re-run."
            )

        # ── 4. per-article recency state, used across all chunks ─────────────
        # Initialised ONCE so a person resolved in chunk 1 disambiguates a
        # bare reference in chunk 4. token_owners is seeded from the current
        # alias snapshot; _resolve_or_create then maintains both maps in
        # place via update_token_owners_and_recency as new people get added.
        # On crash + resume the maps start empty (we lose cross-chunk recency
        # for the resumed chunks); acceptable cost — alias-table hits still
        # land for anything we already wrote.
        recency: dict[str, str] = {}
        token_owners: dict[str, set[str]] = {}
        if use_recency:
            async with get_session() as session:
                token_owners = build_token_owners(
                    await GraphRepository(session).get_all_aliases()
                )

        # ── 5. per-chunk loop, each in its own transaction ───────────────────
        # On a chunk failure we STOP and leave chunks_processed where it is,
        # so the next call to process_article retries that exact chunk. Any
        # other behaviour would either skip the failure forever or scramble
        # the checkpoint semantics.
        people_seen: set[str] = set()
        rels_stored = 0
        extraction_error: Optional[str] = None
        final_chunks_processed = start_idx

        for i in range(start_idx, stored_total):
            chunk_text = chunks[i]
            chunk_result = await self._extractor.extract_one_chunk(
                article, chunk_text, i, stored_total,
            )
            if chunk_result.error:
                logger.warning(
                    "Extraction error on chunk %d/%d of %s: %s — stopping; "
                    "next run will retry from here.",
                    i + 1, stored_total, url, chunk_result.error,
                )
                extraction_error = f"chunk {i + 1}: {chunk_result.error}"
                break

            async with get_session() as session:
                repo = GraphRepository(session)

                name_to_id: dict[str, str] = {}
                for ep in chunk_result.people:
                    pid = await self._resolve_or_create(
                        ep.name, repo, name_to_id,
                        recency=recency, token_owners=token_owners,
                        use_llm_fallback=use_llm_fallback,
                    )
                    people_seen.add(pid)

                for er in chunk_result.relationships:
                    src_id = name_to_id.get(er.source_person) or await self._resolve_or_create(
                        er.source_person, repo, name_to_id,
                        recency=recency, token_owners=token_owners,
                        use_llm_fallback=use_llm_fallback,
                    )
                    tgt_id = name_to_id.get(er.target_person) or await self._resolve_or_create(
                        er.target_person, repo, name_to_id,
                        recency=recency, token_owners=token_owners,
                        use_llm_fallback=use_llm_fallback,
                    )
                    rel_id = await repo.upsert_relationship(
                        source_person_id=src_id,
                        target_person_id=tgt_id,
                        relation_type=er.relation_type,
                        explanation=er.explanation,
                    )
                    await repo.add_provenance(rel_id, article_id, er.supporting_quote)
                    rels_stored += 1

                # Advance the checkpoint *inside the same transaction* as
                # the chunk's writes. Crash before commit → nothing
                # persists for this chunk and chunks_processed stays at i.
                await repo.update_chunk_progress(article_id, i + 1)

            final_chunks_processed = i + 1  # set only after successful commit

        return {
            "article_url": url,
            "article_id": article_id,
            "title": article.title,
            "people_resolved": len(people_seen),
            "relationships_stored": rels_stored,
            "extraction_error": extraction_error,
            "status": "processed",
            "chunks_processed": final_chunks_processed,
            "total_chunks": stored_total,
        }

    async def _resolve_or_create(
        self,
        raw_name: str,
        repo: GraphRepository,
        cache: Optional[dict[str, str]] = None,
        *,
        recency: Optional[dict[str, str]] = None,
        token_owners: Optional[dict[str, set[str]]] = None,
        use_llm_fallback: bool = True,
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

        *use_llm_fallback* is forwarded to ``resolve_person`` — set to False
        for cost-sensitive deployments where missed merges are acceptable.
        """
        resolved = await resolve_person(
            raw_name, repo, self._extractor,
            recency=recency,
            use_llm_fallback=use_llm_fallback,
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
