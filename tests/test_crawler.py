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
from app.crawlers.techcrunch import TechCrunchCrawler, _body, _datetime, _text

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


# ─────────────────────────────────────────────────────────────────────────────
# close
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_is_idempotent():
    crawler = TechCrunchCrawler(request_delay=0)
    await crawler.close()  # client never opened — should not raise
    await crawler.close()  # second call also fine
