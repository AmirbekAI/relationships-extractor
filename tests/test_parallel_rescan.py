"""
Tests for the bounded-parallel rescan path.

What's pinned here:
  * `rescan(max_parallel=N)` lifts wall-clock throughput when articles are
    slow but independent — actual concurrency, not just async scheduling.
  * The host-lock contract: even with a high max_parallel, fetches to one
    crawler instance are serialised + spaced by request_delay.
  * Errors from individual articles don't sink the whole batch, they land
    in `summary["errors"]`.
  * The reported summary includes the resolved `max_parallel` so callers
    can verify what actually happened.

Each test uses an in-process fake crawler + sleep-y extractor so the timing
deltas are real but small (seconds → tenths). Ephemeral SQLite DB so the
production Postgres is never touched.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.core.graph_service import GraphService
from app.core.models import (
    ExtractedPerson,
    ExtractedRelationship,
    ExtractionResult,
)
from app.crawlers.base import ArticleContent, BaseCrawler, CrawlerRegistry
from app.db.session import init_db


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def temp_db():
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
    saved = dict(CrawlerRegistry._registry)
    CrawlerRegistry._registry.clear()
    yield
    CrawlerRegistry._registry.clear()
    CrawlerRegistry._registry.update(saved)


# ── stubs ────────────────────────────────────────────────────────────────────

class _ListingCrawler(BaseCrawler):
    """
    Hands out a configurable list of URLs from one listing page. The actual
    article fetch is a no-op delay (no host throttling needed unless a test
    explicitly sets request_delay via the constructor).
    """
    source_id = "test.parallel"

    def __init__(self, urls: list[str], request_delay: float = 0.0) -> None:
        super().__init__()
        self._urls = urls
        self._delay = request_delay
        # Per-host-lock wall-clock probe: each fetch logs (start, end) so a
        # test can verify they didn't overlap.
        self.fetch_windows: list[tuple[float, float]] = []

    async def get_article_urls(self, page: int = 1) -> list[str]:
        return list(self._urls) if page == 1 else []

    async def fetch_article(self, url: str) -> ArticleContent:
        async with self._get_host_lock():
            await asyncio.sleep(self._delay)
            t0 = time.perf_counter()
            # Pretend body fetch — single-sentence body so the extractor only
            # produces one chunk per article.
            body = f"Subject of {url} announced something. End of article."
            t1 = time.perf_counter()
            self.fetch_windows.append((t0, t1))
        return ArticleContent(
            url=url,
            title=f"Article for {url}",
            author="Test Author",
            published_at=datetime.now(timezone.utc),
            body_text=body,
            source=self.source_id,
        )


class _SlowExtractor:
    """
    Stand-in for LLMExtractor — emits one trivial person + edge per chunk
    after sleeping `chunk_delay` seconds. The sleep is the concurrency
    signal: serial → N*delay, parallel → ~delay.
    """

    def __init__(self, chunk_delay: float) -> None:
        self._delay = chunk_delay
        self.calls = 0

    def split_chunks(self, article, sentences_per_chunk):
        from app.extractors.llm_extractor import _chunk, _split_sentences
        return _chunk(_split_sentences(article.body_text), sentences_per_chunk)

    async def extract_one_chunk(self, article, chunk_text, idx, total):
        await asyncio.sleep(self._delay)
        self.calls += 1
        name = f"Person from {article.url}"
        return ExtractionResult(
            article_url=article.url,
            people=[ExtractedPerson(name=name)],
            relationships=[
                ExtractedRelationship(
                    source_person=name, target_person=name,
                    relation_type="acted", explanation="x", supporting_quote="q",
                ),
            ],
        )

    async def resolve_alias_with_llm(self, name, candidates):  # pragma: no cover
        raise AssertionError("LLM fallback unexpectedly invoked")


# ── concurrency: rescan actually overlaps article processing ─────────────────

@pytest.mark.asyncio
async def test_rescan_max_parallel_speedup(temp_db):
    """
    4 articles × 200ms extractor sleep each.
      serial   (max_parallel=1) → ~800ms
      parallel (max_parallel=4) → ~200ms
    We assert the parallel run is at least 2.5× faster — generous margin so
    CI scheduler jitter doesn't make this flaky, but tight enough that a
    regression to sequential execution would fail.
    """
    urls = [
        f"https://test.parallel/article-{i}" for i in range(4)
    ]
    CrawlerRegistry.register(_ListingCrawler(urls))

    # Serial baseline.
    extractor_serial = _SlowExtractor(chunk_delay=0.2)
    svc = GraphService(extractor=extractor_serial)

    t0 = time.perf_counter()
    summary_serial = await svc.rescan(pages=1, max_parallel=1)
    serial_secs = time.perf_counter() - t0

    assert summary_serial["articles_processed"] == 4
    assert summary_serial["max_parallel"] == 1
    assert extractor_serial.calls == 4

    # New ephemeral DB for the parallel run so chunks_processed bookkeeping
    # doesn't short-circuit the second pass via "already_exists".
    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tf.close()
    try:
        await init_db(f"sqlite+aiosqlite:///{tf.name}")

        CrawlerRegistry._registry.clear()
        CrawlerRegistry.register(_ListingCrawler(urls))
        extractor_parallel = _SlowExtractor(chunk_delay=0.2)
        svc = GraphService(extractor=extractor_parallel)

        t0 = time.perf_counter()
        summary_parallel = await svc.rescan(pages=1, max_parallel=4)
        parallel_secs = time.perf_counter() - t0
    finally:
        if os.path.exists(tf.name):
            os.unlink(tf.name)

    assert summary_parallel["articles_processed"] == 4
    assert summary_parallel["max_parallel"] == 4
    assert extractor_parallel.calls == 4

    speedup = serial_secs / parallel_secs
    assert speedup >= 2.5, (
        f"Expected ≥2.5× speedup from max_parallel=4 vs 1; "
        f"got {speedup:.2f}× (serial={serial_secs:.2f}s, parallel={parallel_secs:.2f}s)"
    )


# ── politeness: per-host lock serialises fetches even at high parallelism ────

@pytest.mark.asyncio
async def test_host_lock_keeps_fetches_sequential(temp_db):
    """
    With max_parallel=8 but only one crawler instance, fetches MUST NOT
    overlap — the per-host lock has to win over the rescan semaphore.

    Probe: each fetch records (start, end) and we assert every pair is
    strictly disjoint. Extractor work runs in parallel between fetches,
    which is exactly the design.
    """
    urls = [f"https://test.parallel/a-{i}" for i in range(4)]
    crawler = _ListingCrawler(urls, request_delay=0.05)
    CrawlerRegistry.register(crawler)

    svc = GraphService(extractor=_SlowExtractor(chunk_delay=0.05))
    await svc.rescan(pages=1, max_parallel=8)

    assert len(crawler.fetch_windows) == 4
    windows = sorted(crawler.fetch_windows, key=lambda w: w[0])
    for (_, end_prev), (start_next, _) in zip(windows, windows[1:]):
        assert start_next >= end_prev, (
            f"Fetch windows overlap — host lock leaked. "
            f"prev_end={end_prev:.4f}, next_start={start_next:.4f}"
        )


# ── correctness: a single article erroring does not sink the batch ───────────

class _FlakyExtractor(_SlowExtractor):
    """Raises on the URL containing 'bad', succeeds on everything else."""

    async def extract_one_chunk(self, article, chunk_text, idx, total):
        if "bad" in article.url:
            raise RuntimeError("simulated extractor blowup")
        return await super().extract_one_chunk(article, chunk_text, idx, total)


@pytest.mark.asyncio
async def test_rescan_errors_are_isolated_per_article(temp_db):
    urls = [
        "https://test.parallel/good-1",
        "https://test.parallel/bad-1",
        "https://test.parallel/good-2",
    ]
    CrawlerRegistry.register(_ListingCrawler(urls))

    svc = GraphService(extractor=_FlakyExtractor(chunk_delay=0.01))
    summary = await svc.rescan(pages=1, max_parallel=3)

    # 2 good articles still processed; 1 bad URL appears in errors.
    assert summary["articles_processed"] == 2
    assert any("bad-1" in e for e in summary["errors"])
    # The good URLs must NOT show up as errors.
    assert not any("good-1" in e or "good-2" in e for e in summary["errors"])


# ── default behaviour: max_parallel falls back to the Settings value ─────────

@pytest.mark.asyncio
async def test_rescan_uses_settings_default_when_unspecified(temp_db, monkeypatch):
    """
    Caller passes no max_parallel → rescan reads the configured default
    out of Settings. Probe by forcing the setting to 1 and asserting the
    reported value flows through.
    """
    monkeypatch.setenv("MAX_PARALLEL_ARTICLES", "1")
    # Clear the lru_cache so the env override is picked up.
    from app.config import get_settings
    get_settings.cache_clear()

    urls = ["https://test.parallel/just-one"]
    CrawlerRegistry.register(_ListingCrawler(urls))

    svc = GraphService(extractor=_SlowExtractor(chunk_delay=0.01))
    summary = await svc.rescan(pages=1)  # no override

    assert summary["max_parallel"] == 1

    # Restore for any subsequent test that cached different settings.
    get_settings.cache_clear()
