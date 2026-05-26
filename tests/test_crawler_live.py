"""
Integration test — hits the real TechCrunch website.

Run manually to verify the crawler still parses live pages correctly:

    .venv/bin/python -m pytest tests/test_crawler_live.py -v -s

Not run in CI (no network access assumed).
"""

from __future__ import annotations

import pytest

from app.crawlers.techcrunch import TechCrunchCrawler

TOPIC_URL = "https://techcrunch.com/tag/openai/"


@pytest.mark.asyncio
async def test_live_parse_listing_and_first_article():
    crawler = TechCrunchCrawler(topic_url=TOPIC_URL, request_delay=1.0)

    try:
        # ── Step 1: get article URLs from page 1 ──────────────────────────
        urls = await crawler.get_article_urls(page=1)

        print(f"\n\n{'═' * 60}")
        print(f"LISTING PAGE — found {len(urls)} URLs")
        print('═' * 60)
        for u in urls:
            print(f"  {u}")

        assert len(urls) > 0, "No article URLs found — selector may be broken"

        # ── Step 2: fetch and parse the first article ─────────────────────
        first_url = urls[0]
        article = await crawler.fetch_article(first_url)

        print(f"\n{'═' * 60}")
        print("ARTICLE CONTENT")
        print('═' * 60)
        print(f"  URL          : {first_url}")
        print(f"  Title        : {article.title if article else 'N/A'}")
        print(f"  Author       : {article.author if article else 'N/A'}")
        print(f"  Published at : {article.published_at if article else 'N/A'}")
        if article:
            preview = article.body_text[:800].replace("\n", " ")
            print(f"  Body preview : {preview} …")
        print('═' * 60)

        assert article is not None, f"fetch_article returned None for {first_url}"
        assert article.body_text.strip(), "Body text is empty"

    finally:
        await crawler.close()
