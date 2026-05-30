"""
End-to-end tests for the chunk checkpoint / crash-resume logic in
GraphService.process_article.

Three scenarios pinned here:
  1. Crash mid-article  → chunks_processed records exactly how far we got;
                          relationships from completed chunks are durably
                          stored; resume picks up at the failed chunk.
  2. Body-hash mismatch → on the next call with a different body for the
                          same URL, the row is reset and processed afresh.
  3. Already-complete   → re-invoking process_article on a finished URL
                          returns status="already_exists" without calling
                          the extractor.

A fake crawler + counting/raising extractor stub drive the pipeline against
an ephemeral SQLite DB so no real LLM or network is involved.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.core.graph_service import GraphService, _body_hash
from app.core.models import (
    ExtractedPerson,
    ExtractedRelationship,
    ExtractionResult,
)
from app.crawlers.base import ArticleContent, BaseCrawler, CrawlerRegistry
from app.db.repository import GraphRepository
from app.db.session import get_session, init_db

# ── stubs ────────────────────────────────────────────────────────────────────

# 4 chunks worth of sentences. With sentences_per_chunk=2 the splitter will
# produce 4 chunks of 2 sentences each.
_BODY = (
    "Sam Altman returned to OpenAI as CEO. Elon Musk criticized the board's decision. "
    "Satya Nadella publicly backed Altman. Greg Brockman resigned in solidarity. "
    "Reid Hoffman expressed support online. Marc Andreessen weighed in. "
    "Vinod Khosla called the firing reckless. Peter Thiel declined to comment."
)


class _FakeCrawler(BaseCrawler):
    source_id = "test.local"

    def __init__(self, body: str = _BODY) -> None:
        self._body = body

    async def get_article_urls(self, page: int = 1) -> list[str]:
        return []

    async def fetch_article(self, url: str) -> ArticleContent:
        return ArticleContent(
            url=url,
            title="Test article",
            author="Jane Doe",
            published_at=datetime.now(timezone.utc),
            body_text=self._body,
            source=self.source_id,
        )


class _CountingExtractor:
    """
    Stand-in for LLMExtractor that's deterministic, never calls a real LLM,
    and can be configured to RAISE on a specific chunk index — emulating
    crash recovery exactly.

    Mirrors LLMExtractor's two-method contract used by process_article:
        split_chunks(article, n)   → list[str]
        extract_one_chunk(article, text, idx, total)  → ExtractionResult
    """

    def __init__(self, *, fail_on_chunk: int | None = None) -> None:
        self.fail_on_chunk = fail_on_chunk
        self.calls: list[int] = []  # chunk indices we extracted (success path)

    def split_chunks(self, article, sentences_per_chunk):
        # Same splitter LLMExtractor uses, so chunk count matches what the
        # service computes.
        from app.extractors.llm_extractor import _chunk, _split_sentences

        return _chunk(_split_sentences(article.body_text), sentences_per_chunk)

    async def extract_one_chunk(self, article, chunk_text, idx, total):
        if self.fail_on_chunk is not None and idx == self.fail_on_chunk:
            return ExtractionResult(
                article_url=article.url,
                error=f"simulated crash on chunk {idx}",
            )
        self.calls.append(idx)
        # Emit one person + one self-loop relationship tagged by chunk index
        # so each chunk's writes are distinguishable in the DB.
        person_name = f"Person {idx}"
        return ExtractionResult(
            article_url=article.url,
            people=[ExtractedPerson(name=person_name)],
            relationships=[
                ExtractedRelationship(
                    source_person=person_name,
                    target_person=person_name,
                    relation_type=f"acted on chunk {idx}",
                    explanation="x",
                    supporting_quote=chunk_text[:40],
                ),
            ],
        )

    # The resolver-eval path inside resolve_person needs this on the
    # extractor object even when use_llm_fallback=True; we never expect it
    # to actually fire for these self-edge chunks (each name is unique).
    async def resolve_alias_with_llm(self, name, candidates):  # pragma: no cover
        raise AssertionError(
            f"LLM fallback unexpectedly invoked: {name!r} in {candidates!r}"
        )


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def temp_db():
    """A throwaway SQLite DB for the duration of one test."""
    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tf.close()
    try:
        await init_db(f"sqlite+aiosqlite:///{tf.name}")
        yield
    finally:
        if os.path.exists(tf.name):
            os.unlink(tf.name)


@pytest.fixture(autouse=True)
def _reset_crawler_registry():
    """Each test gets a clean CrawlerRegistry so we can register a fake."""
    saved = dict(CrawlerRegistry._registry)
    CrawlerRegistry._registry.clear()
    yield
    CrawlerRegistry._registry.clear()
    CrawlerRegistry._registry.update(saved)


# ── tests ────────────────────────────────────────────────────────────────────

URL = "https://test.local/checkpoint-article"


@pytest.mark.asyncio
async def test_crash_mid_article_then_resume(temp_db):
    """
    Two-phase scenario:
      Phase A: extractor crashes on chunk 2 (0-indexed). The article row
               must exist with chunks_processed == 2 and chunks 0–1 worth
               of data must be in the DB.
      Phase B: re-invoke process_article on the same URL with a now-working
               extractor. It must:
                 - call extract_one_chunk only for chunks 2, 3 (the
                   remaining ones — never for 0, 1).
                 - leave chunks_processed == total_chunks at the end.
    """
    CrawlerRegistry.register(_FakeCrawler())

    # Phase A — crash on chunk index 2.
    failing = _CountingExtractor(fail_on_chunk=2)
    svc = GraphService(extractor=failing)
    summary_a = await svc.process_article(URL, sentences_per_chunk=2)

    assert summary_a["status"] == "processed"
    assert summary_a["chunks_processed"] == 2  # 0 and 1 succeeded
    assert summary_a["total_chunks"] == 4  # 8 sentences / 2 = 4 chunks
    assert summary_a["extraction_error"] is not None
    assert failing.calls == [0, 1]  # only the successful ones

    # DB-side: article row exists, chunks_processed == 2, two relationships.
    async with get_session() as session:
        repo = GraphRepository(session)
        art = await repo.get_article_by_url(URL)
        assert art is not None
        assert art.chunks_processed == 2
        assert art.total_chunks == 4
        assert art.sentences_per_chunk == 2
        n_rels = await repo.count_relationships()
        assert n_rels == 2  # chunks 0 and 1 each stored one self-edge

    # Phase B — replace the extractor with a working one and re-run.
    working = _CountingExtractor(fail_on_chunk=None)
    svc.b_extractor = working  # keep a handle for inspection
    svc._extractor = working
    summary_b = await svc.process_article(URL, sentences_per_chunk=2)

    assert summary_b["status"] == "processed"
    assert summary_b["chunks_processed"] == 4  # done
    assert summary_b["total_chunks"] == 4
    assert summary_b["extraction_error"] is None
    assert working.calls == [2, 3]  # ONLY the remaining chunks

    async with get_session() as session:
        repo = GraphRepository(session)
        art = await repo.get_article_by_url(URL)
        assert art.chunks_processed == 4
        n_rels = await repo.count_relationships()
        assert n_rels == 4  # 0+1+2+3 stored


@pytest.mark.asyncio
async def test_body_hash_change_triggers_clean_restart(temp_db):
    """
    Process the article fully, then re-process the same URL with a *different
    body*. The body_hash mismatch must wipe the article's provenance +
    orphaned relationships and re-run from chunk 0.
    """
    # Phase 1: fully process with the original body.
    CrawlerRegistry.register(_FakeCrawler())
    extractor1 = _CountingExtractor()
    svc = GraphService(extractor=extractor1)
    summary1 = await svc.process_article(URL, sentences_per_chunk=2)
    assert summary1["chunks_processed"] == summary1["total_chunks"] == 4
    assert extractor1.calls == [0, 1, 2, 3]

    # Sanity: store the original body_hash.
    original_hash = _body_hash(_BODY)
    async with get_session() as session:
        repo = GraphRepository(session)
        art = await repo.get_article_by_url(URL)
        assert art.body_hash == original_hash
        n_rels_before = await repo.count_relationships()
        assert n_rels_before == 4

    # Phase 2: swap the crawler for one that returns a SHORTER body
    # (2 chunks, not 4) so the chunk-count mismatch is also exercised.
    new_body = "Sam Altman gave a keynote. " "Microsoft announced a new partnership."
    CrawlerRegistry._registry.clear()
    CrawlerRegistry.register(_FakeCrawler(body=new_body))

    extractor2 = _CountingExtractor()
    svc._extractor = extractor2
    summary2 = await svc.process_article(URL, sentences_per_chunk=2)

    assert summary2["status"] == "processed"
    # New body → 2 sentences → 1 chunk of size 2.
    assert summary2["total_chunks"] == 1
    assert summary2["chunks_processed"] == 1
    assert extractor2.calls == [0]  # re-ran from scratch

    async with get_session() as session:
        repo = GraphRepository(session)
        art = await repo.get_article_by_url(URL)
        # body_hash updated; previous provenance + relationships wiped.
        assert art.body_hash == _body_hash(new_body)
        assert art.total_chunks == 1
        n_rels_after = await repo.count_relationships()
        # only the one self-edge from the new (single) chunk
        assert n_rels_after == 1


@pytest.mark.asyncio
async def test_already_complete_skips_extractor(temp_db):
    """A second call on a finished URL returns already_exists, doesn't extract."""
    CrawlerRegistry.register(_FakeCrawler())
    extractor1 = _CountingExtractor()
    svc = GraphService(extractor=extractor1)
    await svc.process_article(URL, sentences_per_chunk=2)
    assert extractor1.calls == [0, 1, 2, 3]

    # Second invocation: a fresh extractor that would fail if called.
    boom = _CountingExtractor(fail_on_chunk=0)
    svc._extractor = boom
    summary = await svc.process_article(URL, sentences_per_chunk=2)

    assert summary["status"] == "already_exists"
    assert boom.calls == []
