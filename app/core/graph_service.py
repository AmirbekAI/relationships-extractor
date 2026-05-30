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

import asyncio
import hashlib
import logging
from dataclasses import dataclass
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
from app.extractors.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)


def _body_hash(text: str) -> str:
    """Stable fingerprint of the article body; used to detect mid-flight edits."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_DEFAULT_CHUNK_SIZE = 5


@dataclass(frozen=True)
class _Checkpoint:
    """
    Outcome of the article setup transaction in :meth:`_prepare_checkpoint`.

    If *done* is True the article is already fully processed; the caller
    returns a status="already_exists" summary without running the extractor.
    Otherwise *start_idx* / *total_chunks* / *chunk_size* describe where the
    per-chunk loop should resume.
    """

    article_id: str
    title: Optional[str]
    start_idx: int
    total_chunks: int
    chunk_size: int
    done: bool


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
        1. Pick a crawler, crawl the article (:meth:`_crawl`).
        2. Open ONE setup transaction to decide where to start
           (:meth:`_prepare_checkpoint`): fresh insert, resume from a stored
           pointer, restart on body-hash mismatch, or skip if already done.
        3. Per chunk from start_idx onward, in its OWN transaction
           (:meth:`_process_chunk`): extract, resolve, persist, advance the
           pointer. A crash mid-chunk costs at most that one LLM call.

        Raises
        ------
        ValueError   if no crawler is registered for the URL
        RuntimeError if the crawler returns nothing
        """
        chunk_size = sentences_per_chunk or _DEFAULT_CHUNK_SIZE

        article = await self._crawl(url)
        body_hash = _body_hash(article.body_text)
        chunks = self._extractor.split_chunks(article, chunk_size)
        total_chunks = len(chunks)

        checkpoint = await self._prepare_checkpoint(
            url=url,
            article=article,
            body_hash=body_hash,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
        )

        if checkpoint.done:
            return {
                "article_url": url,
                "article_id": checkpoint.article_id,
                "title": checkpoint.title,
                "people_resolved": 0,
                "relationships_stored": 0,
                "extraction_error": None,
                "status": "already_exists",
            }

        # Honour the stored chunking — if it differs from the caller's request
        # we re-split so chunk boundaries don't shift mid-article.
        if checkpoint.chunk_size != chunk_size:
            chunks = self._extractor.split_chunks(article, checkpoint.chunk_size)

        if len(chunks) != checkpoint.total_chunks:
            # Should be impossible given the body_hash check, but guard
            # anyway — better to refuse than to scramble the checkpoint.
            raise RuntimeError(
                f"Chunk count drift for {url}: expected {checkpoint.total_chunks}, "
                f"computed {len(chunks)}. Delete the article row and re-run."
            )

        settings = get_settings()
        use_recency = settings.resolver_recency_enabled
        use_llm_fallback = settings.resolver_llm_fallback_enabled

        # Per-article recency state: initialised ONCE so a person resolved in
        # chunk 1 still disambiguates a bare reference in chunk 4. Seeded from
        # the current alias snapshot; _resolve_or_create maintains both maps in
        # place as new people are resolved or created mid-article. On crash +
        # resume these start empty (cross-chunk recency lost for resumed
        # chunks); acceptable — alias-table hits still land for anything we
        # already wrote.
        recency: dict[str, str] = {}
        token_owners: dict[str, set[str]] = {}
        if use_recency:
            async with get_session() as session:
                token_owners = build_token_owners(
                    await GraphRepository(session).get_all_aliases()
                )

        people_seen: set[str] = set()
        rels_stored = 0
        extraction_error: Optional[str] = None
        final_chunks_processed = checkpoint.start_idx

        for i in range(checkpoint.start_idx, checkpoint.total_chunks):
            chunk_outcome = await self._process_chunk(
                article=article,
                article_id=checkpoint.article_id,
                chunk_text=chunks[i],
                chunk_idx=i,
                total_chunks=checkpoint.total_chunks,
                recency=recency,
                token_owners=token_owners,
                use_llm_fallback=use_llm_fallback,
            )
            if chunk_outcome is None:
                # Extractor reported an error for this chunk — stop here so
                # the next invocation retries from the same index.
                extraction_error = f"chunk {i + 1}: extractor error"
                break

            people_seen.update(chunk_outcome[0])
            rels_stored += chunk_outcome[1]
            final_chunks_processed = i + 1

        return {
            "article_url": url,
            "article_id": checkpoint.article_id,
            "title": article.title,
            "people_resolved": len(people_seen),
            "relationships_stored": rels_stored,
            "extraction_error": extraction_error,
            "status": "processed",
            "chunks_processed": final_chunks_processed,
            "total_chunks": checkpoint.total_chunks,
        }

    # ─────────────────────────────────────────── process_article: sub-steps

    async def _crawl(self, url: str) -> ArticleContent:
        """Pick the right crawler for *url* and fetch the article body."""
        crawler = CrawlerRegistry.for_url(url)
        if crawler is None:
            raise ValueError(f"No crawler registered for URL: {url}")
        article = await crawler.fetch_article(url)
        if article is None:
            raise RuntimeError(f"Crawler returned nothing for: {url}")
        return article

    async def _prepare_checkpoint(
        self,
        *,
        url: str,
        article: ArticleContent,
        body_hash: str,
        chunk_size: int,
        total_chunks: int,
    ) -> _Checkpoint:
        """
        Decide where this run should start, inside one setup transaction.

        Four outcomes:
          * fresh article → insert and start at 0
          * pre-checkpoint legacy row → treat as done (caller short-circuits)
          * body_hash mismatch → reset and start at 0
          * existing row, partial / complete progress → resume or signal done
        """
        async with get_session() as session:
            repo = GraphRepository(session)
            existing = await repo.get_article_by_url(url)

            if existing is None:
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
                logger.info(
                    "Article %s: starting fresh, %d chunk(s) to process.",
                    url,
                    total_chunks,
                )
                return _Checkpoint(
                    article_id=article_id,
                    title=article.title,
                    start_idx=0,
                    total_chunks=total_chunks,
                    chunk_size=chunk_size,
                    done=False,
                )

            article_id = existing.id

            # Legacy row from before checkpointing existed (migration added
            # the columns with NULL/0 defaults). Trust that it was fully
            # processed under the old code path; the user can delete it
            # manually to force a re-run.
            if existing.total_chunks is None:
                logger.info(
                    "Article %s: pre-checkpoint row, treating as complete.",
                    url,
                )
                return _Checkpoint(
                    article_id=article_id,
                    title=existing.title,
                    start_idx=0,
                    total_chunks=0,
                    chunk_size=chunk_size,
                    done=True,
                )

            # Body changed since last run? rewind and replay.
            if existing.body_hash != body_hash:
                logger.info(
                    "Article %s: body changed since last run "
                    "(stored hash %.8s, new hash %.8s); resetting "
                    "checkpoint and replaying from chunk 1/%d.",
                    url,
                    existing.body_hash or "",
                    body_hash,
                    total_chunks,
                )
                await repo.reset_article_for_rerun(
                    article_id,
                    body_hash=body_hash,
                    sentences_per_chunk=chunk_size,
                    total_chunks=total_chunks,
                )
                return _Checkpoint(
                    article_id=article_id,
                    title=existing.title,
                    start_idx=0,
                    total_chunks=total_chunks,
                    chunk_size=chunk_size,
                    done=False,
                )

            start_idx = existing.chunks_processed or 0
            stored_total = existing.total_chunks or total_chunks
            stored_chunk_size = existing.sentences_per_chunk or chunk_size

            if start_idx >= stored_total and stored_total > 0:
                logger.info(
                    "Article %s: already complete (%d/%d chunks), skipping.",
                    url,
                    stored_total,
                    stored_total,
                )
                return _Checkpoint(
                    article_id=article_id,
                    title=existing.title,
                    start_idx=start_idx,
                    total_chunks=stored_total,
                    chunk_size=stored_chunk_size,
                    done=True,
                )

            if start_idx > 0:
                logger.info(
                    "Article %s: found progress at chunk %d/%d, "
                    "resuming (%d chunk(s) left).",
                    url,
                    start_idx,
                    stored_total,
                    stored_total - start_idx,
                )
            else:
                logger.info(
                    "Article %s: row exists but no chunks done yet, "
                    "starting from chunk 1/%d.",
                    url,
                    stored_total,
                )

            return _Checkpoint(
                article_id=article_id,
                title=existing.title,
                start_idx=start_idx,
                total_chunks=stored_total,
                chunk_size=stored_chunk_size,
                done=False,
            )

    async def _process_chunk(
        self,
        *,
        article: ArticleContent,
        article_id: str,
        chunk_text: str,
        chunk_idx: int,
        total_chunks: int,
        recency: dict[str, str],
        token_owners: dict[str, set[str]],
        use_llm_fallback: bool,
    ) -> Optional[tuple[set[str], int]]:
        """
        Extract + persist one chunk inside a single transaction.

        Returns
        -------
        (people_ids_seen, relationships_stored)
            on success — checkpoint pointer is already bumped.
        None
            if the extractor reported an error for this chunk. Pointer is
            NOT bumped, so the next invocation retries from the same index.
        """
        url = article.url
        chunk_result = await self._extractor.extract_one_chunk(
            article,
            chunk_text,
            chunk_idx,
            total_chunks,
        )
        if chunk_result.error:
            logger.warning(
                "Extraction error on chunk %d/%d of %s: %s — stopping; "
                "next run will retry from here.",
                chunk_idx + 1,
                total_chunks,
                url,
                chunk_result.error,
            )
            return None

        people_seen: set[str] = set()
        rels_stored = 0

        async with get_session() as session:
            repo = GraphRepository(session)

            name_to_id: dict[str, str] = {}
            for ep in chunk_result.people:
                pid = await self._resolve_or_create(
                    ep.name,
                    repo,
                    name_to_id,
                    recency=recency,
                    token_owners=token_owners,
                    use_llm_fallback=use_llm_fallback,
                )
                people_seen.add(pid)

            for er in chunk_result.relationships:
                src_id = name_to_id.get(
                    er.source_person
                ) or await self._resolve_or_create(
                    er.source_person,
                    repo,
                    name_to_id,
                    recency=recency,
                    token_owners=token_owners,
                    use_llm_fallback=use_llm_fallback,
                )
                tgt_id = name_to_id.get(
                    er.target_person
                ) or await self._resolve_or_create(
                    er.target_person,
                    repo,
                    name_to_id,
                    recency=recency,
                    token_owners=token_owners,
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

            # Advance the checkpoint *inside the same transaction* as the
            # chunk's writes. Crash before commit → nothing persists for
            # this chunk and chunks_processed stays at chunk_idx.
            await repo.update_chunk_progress(article_id, chunk_idx + 1)

        return people_seen, rels_stored

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
            raw_name,
            repo,
            self._extractor,
            recency=recency,
            use_llm_fallback=use_llm_fallback,
        )
        if resolved:
            person_id = resolved.person_id
            canonical_name = resolved.canonical_name
        else:
            person_id = await repo.get_or_create_person(raw_name)
            await repo.add_alias(person_id, normalize(raw_name))
            canonical_name = raw_name

        if recency is not None and token_owners is not None:
            update_token_owners_and_recency(
                person_id,
                canonical_name,
                token_owners,
                recency,
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
        max_parallel: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Iterate up to *pages* listing pages for each registered crawler
        (or just those in *source_ids*) and process every discovered article.

        Concurrency
        -----------
        Articles are processed under an ``asyncio.Semaphore`` bounded by
        *max_parallel* (defaulting to ``settings.max_parallel_articles``).
        Politeness to a single host is the crawler's responsibility — it
        holds an internal per-host async lock + delay across delay+GET so
        the burst-rate floor is preserved even when this method fans out.

        Listing pages are fetched sequentially per crawler (cheap, one
        request per page), then every discovered URL is dispatched in
        parallel up to the semaphore cap.

        Returns aggregate counts and a list of per-URL error strings.
        """
        if max_parallel is None:
            max_parallel = get_settings().max_parallel_articles

        crawlers = list(CrawlerRegistry.all().values())
        if source_ids:
            crawlers = [c for c in crawlers if c.source_id in source_ids]

        # ── 1. listing pass — collect URLs (sequential per crawler) ─────────
        work: list[str] = []
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
                work.extend(urls)

        # ── 2. article pass — bounded parallelism ───────────────────────────
        sem = asyncio.Semaphore(max_parallel)

        async def _one(url: str) -> Optional[dict[str, Any]]:
            async with sem:
                try:
                    return await self.process_article(url, sentences_per_chunk)
                except Exception as exc:
                    msg = f"{url}: {exc}"
                    logger.error(msg)
                    errors.append(msg)
                    return None

        summaries = await asyncio.gather(*(_one(u) for u in work))

        # ── 3. roll up ──────────────────────────────────────────────────────
        total_processed = 0
        total_skipped = 0
        total_relationships = 0
        for summary in summaries:
            if summary is None:
                continue
            if summary.get("status") == "already_exists":
                total_skipped += 1
                continue
            total_processed += 1
            total_relationships += summary["relationships_stored"]
            logger.info(
                "Processed %s: %d people, %d relationships",
                summary["article_url"],
                summary["people_resolved"],
                summary["relationships_stored"],
            )

        return {
            "pages_crawled": pages,
            "articles_processed": total_processed,
            "articles_skipped": total_skipped,
            "relationships_stored": total_relationships,
            "errors": errors,
            "max_parallel": max_parallel,
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
