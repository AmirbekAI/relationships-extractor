"""
Tests for TechCrunchCrawler.

All HTTP calls are intercepted with httpx's MockTransport — no real network
requests are made.  Each test provides the exact HTML the crawler would see,
so we verify parsing logic in isolation.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.crawlers.base import ArticleContent
from app.crawlers.techcrunch import (
    TechCrunchCrawler,
    _authors,
    _body,
    _first_text,
    _published_at,
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal HTML fixtures
# ─────────────────────────────────────────────────────────────────────────────

LISTING_HTML = """
<html><body>
  <a class="loop-card__title-link" href="https://techcrunch.com/2024/01/02/article-two/">Article Two</a>
  <a class="loop-card__title-link" href="https://techcrunch.com/2024/01/01/article-one/">Article One</a>
  <a class="loop-card__title-link" href="https://techcrunch.com/2024/01/01/article-one/">Article One</a>
</body></html>
"""

LISTING_HTML_FALLBACK = """
<html><body>
  <h2><a href="https://techcrunch.com/2024/01/03/article-three/">Article Three</a></h2>
  <h3><a href="/2024/01/04/article-four/">Article Four</a></h3>
</body></html>
"""

ARTICLE_HTML = """
<html><body>
  <h1 class="article__title">Sam Altman Returns to OpenAI</h1>
  <a class="article__author-name">Jane Doe</a>
  <time class="article__date" datetime="2024-01-15T10:30:00">January 15, 2024</time>
  <div class="article-content">
    <p>Sam Altman has returned to OpenAI as CEO.</p>
    <p>Elon Musk criticized the board's decision.</p>
    <script>bad script</script>
    <nav>nav noise</nav>
  </div>
</body></html>
"""

ARTICLE_HTML_NO_BODY = """
<html><body>
  <h1 class="article__title">No Content Article</h1>
</body></html>
"""

ARTICLE_HTML_FALLBACK_BODY = """
<html><body>
  <main>
    <p>Fallback paragraph one.</p>
    <p>Fallback paragraph two.</p>
  </main>
</body></html>
"""

# Podcast layout: different title/author/date markup, multiple (and in the
# real DOM, repeated) author cards, and a free-text date with no <time> tag.
PODCAST_ARTICLE_HTML = """
<html><head>
  <meta property="article:published_time" content="2026-05-22T09:00:00+00:00">
</head><body>
  <div class="wp-block-techcrunch-podcast-single-hero__inner">
    <h1 class="wp-block-techcrunch-podcast-single-hero__title">Elon Musk can’t hear you</h1>
    <div class="wp-block-techcrunch-podcast-single-hero__meta">
      <span class="wp-block-techcrunch-podcast-single-hero__author-list">
        <a class="wp-block-tc23-author-card-name__link" href="/author/theresa-loconsolo/">Theresa Loconsolo</a>
        <a class="wp-block-tc23-author-card-name__link" href="/author/kirsten-korosec/">Kirsten Korosec</a>
        <a class="wp-block-tc23-author-card-name__link" href="/author/sean-okane/">Sean O'Kane</a>
        <a class="wp-block-tc23-author-card-name__link" href="/author/anthony-ha/">Anthony Ha</a>
        <a class="wp-block-tc23-author-card-name__link" href="/author/anthony-ha/">Anthony Ha</a>
      </span>
      <span>May 22, 2026</span>
    </div>
  </div>
  <div class="article-content">
    <p>SpaceX filed its S-1.</p>
    <p>Elon Musk commented on the IPO.</p>
  </div>
</body></html>
"""

# Same podcast layout but WITHOUT the published-time meta tag, so the date has
# to come from the free-text "May 22, 2026" span.
PODCAST_ARTICLE_HTML_NO_META = PODCAST_ARTICLE_HTML.replace(
    '<meta property="article:published_time" content="2026-05-22T09:00:00+00:00">',
    "",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_crawler(html_map: dict[str, str]) -> TechCrunchCrawler:
    """
    Return a TechCrunchCrawler whose HTTP client is replaced with a mock that
    serves HTML from *html_map* keyed by URL.
    """
    crawler = TechCrunchCrawler(request_delay=0)  # no delay in tests

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        html = html_map.get(url)
        if html is None:
            return httpx.Response(404, text="Not found")
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    crawler._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )
    return crawler


# ─────────────────────────────────────────────────────────────────────────────
# _listing_url
# ─────────────────────────────────────────────────────────────────────────────


def test_listing_url_page_1():
    crawler = TechCrunchCrawler()
    assert crawler._listing_url(1) == "https://techcrunch.com/tag/openai/"


def test_listing_url_page_2():
    crawler = TechCrunchCrawler()
    assert crawler._listing_url(2) == "https://techcrunch.com/tag/openai/page/2/"


def test_listing_url_page_5():
    crawler = TechCrunchCrawler()
    assert crawler._listing_url(5) == "https://techcrunch.com/tag/openai/page/5/"


# ─────────────────────────────────────────────────────────────────────────────
# get_article_urls
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_article_urls_returns_deduplicated_links():
    crawler = _make_crawler(
        {
            "https://techcrunch.com/tag/openai/": LISTING_HTML,
        }
    )
    urls = await crawler.get_article_urls(page=1)

    # Duplicate href appears once in HTML — must appear once in output
    assert urls.count("https://techcrunch.com/2024/01/01/article-one/") == 1
    assert "https://techcrunch.com/2024/01/02/article-two/" in urls
    assert len(urls) == 2


@pytest.mark.asyncio
async def test_get_article_urls_page_2():
    crawler = _make_crawler(
        {
            "https://techcrunch.com/tag/openai/page/2/": LISTING_HTML,
        }
    )
    urls = await crawler.get_article_urls(page=2)
    assert len(urls) == 2


@pytest.mark.asyncio
async def test_get_article_urls_fallback_selector():
    """When the primary selector finds nothing, fall back to h2/h3 links."""
    crawler = _make_crawler(
        {
            "https://techcrunch.com/tag/openai/": LISTING_HTML_FALLBACK,
        }
    )
    urls = await crawler.get_article_urls(page=1)

    assert "https://techcrunch.com/2024/01/03/article-three/" in urls
    # Relative href should be resolved to absolute
    assert "https://techcrunch.com/2024/01/04/article-four/" in urls


@pytest.mark.asyncio
async def test_get_article_urls_returns_empty_on_http_error():
    crawler = _make_crawler({})  # every URL → 404
    urls = await crawler.get_article_urls(page=1)
    assert urls == []


# ─────────────────────────────────────────────────────────────────────────────
# fetch_article
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_article_parses_all_fields():
    url = "https://techcrunch.com/2024/01/15/sample/"
    crawler = _make_crawler({url: ARTICLE_HTML})

    result = await crawler.fetch_article(url)

    assert result is not None
    assert result.title == "Sam Altman Returns to OpenAI"
    assert result.author == "Jane Doe"
    assert result.published_at == datetime(2024, 1, 15, 10, 30, 0)
    assert result.source == "techcrunch"
    assert result.url == url


@pytest.mark.asyncio
async def test_fetch_article_body_strips_noise():
    url = "https://techcrunch.com/2024/01/15/sample/"
    crawler = _make_crawler({url: ARTICLE_HTML})

    result = await crawler.fetch_article(url)

    assert result is not None
    assert "Sam Altman has returned" in result.body_text
    assert "Elon Musk criticized" in result.body_text
    # Script and nav content must be stripped
    assert "bad script" not in result.body_text
    assert "nav noise" not in result.body_text


@pytest.mark.asyncio
async def test_fetch_article_fallback_body():
    """If article-content div is absent, fall back to <main> paragraphs."""
    url = "https://techcrunch.com/2024/01/15/fallback/"
    crawler = _make_crawler({url: ARTICLE_HTML_FALLBACK_BODY})

    result = await crawler.fetch_article(url)

    assert result is not None
    assert "Fallback paragraph one" in result.body_text
    assert "Fallback paragraph two" in result.body_text


@pytest.mark.asyncio
async def test_fetch_article_returns_none_when_no_body():
    url = "https://techcrunch.com/2024/01/15/nobody/"
    crawler = _make_crawler({url: ARTICLE_HTML_NO_BODY})

    result = await crawler.fetch_article(url)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_article_returns_none_on_http_error():
    crawler = _make_crawler({})  # every URL → 404
    result = await crawler.fetch_article("https://techcrunch.com/2024/01/15/missing/")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_article_parses_podcast_layout():
    """Podcast pages use different markup — title/authors/date must still fill."""
    url = "https://techcrunch.com/podcast/elon-musk-ipo/"
    crawler = _make_crawler({url: PODCAST_ARTICLE_HTML})

    result = await crawler.fetch_article(url)

    assert result is not None
    # &nbsp; (\xa0) collapsed to a regular space.
    assert result.title == "Elon Musk can’t hear you"
    # All authors joined, de-duplicated, order preserved.
    assert result.author == (
        "Theresa Loconsolo, Kirsten Korosec, Sean O'Kane, Anthony Ha"
    )
    # Date from the published-time meta tag.
    assert result.published_at == datetime(
        2026, 5, 22, 9, 0, 0, tzinfo=result.published_at.tzinfo
    )
    assert result.published_at.tzinfo is not None
    assert "SpaceX filed its S-1" in result.body_text


@pytest.mark.asyncio
async def test_fetch_article_podcast_date_from_free_text():
    """With no meta tag, the date falls back to the 'May 22, 2026' span."""
    url = "https://techcrunch.com/podcast/no-meta/"
    crawler = _make_crawler({url: PODCAST_ARTICLE_HTML_NO_META})

    result = await crawler.fetch_article(url)

    assert result is not None
    assert result.published_at == datetime(2026, 5, 22)


def test_authors_dedupes_and_joins():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(PODCAST_ARTICLE_HTML, "lxml")
    assert _authors(soup) == (
        "Theresa Loconsolo, Kirsten Korosec, Sean O'Kane, Anthony Ha"
    )


def test_first_text_falls_back_through_selectors():
    from bs4 import BeautifulSoup

    # Only the podcast title selector matches → fallback must find it.
    soup = BeautifulSoup(PODCAST_ARTICLE_HTML, "lxml")
    from app.crawlers.techcrunch import _SEL_TITLE

    assert _first_text(soup, _SEL_TITLE) == "Elon Musk can’t hear you"


def test_published_at_returns_none_when_absent():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<html><body><p>no date here</p></body></html>", "lxml")
    assert _published_at(soup) is None


# ─────────────────────────────────────────────────────────────────────────────
# close
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_is_idempotent():
    crawler = TechCrunchCrawler(request_delay=0)
    await crawler.close()  # client never opened — should not raise
    await crawler.close()  # second call also fine
