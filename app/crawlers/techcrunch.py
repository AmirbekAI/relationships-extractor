"""
TechCrunch crawler.

Topic listing:  https://techcrunch.com/tag/openai/
                https://techcrunch.com/tag/openai/page/2/  etc.

Selectors are isolated in constants at the top of the file — update them here
if TechCrunch's markup changes, without touching anything else.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.crawlers.base import ArticleContent, BaseCrawler

logger = logging.getLogger(__name__)

# ── CSS selectors ────────────────────────────────────────────────────────────
_SEL_LISTING_LINK   = "a.loop-card__title-link"
_SEL_TITLE          = "h1.article__title"
_SEL_AUTHOR         = "a.article__author-name"
_SEL_DATE           = "time.article__date"
_SEL_BODY           = "div.article-content"

_USER_AGENT = (
    "Mozilla/5.0 (compatible; RelationshipFinderBot/1.0)"
)
_TIMEOUT        = 30    # seconds per request
_POLITE_DELAY   = 1.5   # seconds between requests to the same host


class TechCrunchCrawler(BaseCrawler):
    source_id = "techcrunch"

    def __init__(
        self,
        topic_url: str = "https://techcrunch.com/tag/openai/",
        request_delay: float = _POLITE_DELAY,
    ) -> None:
        self._topic_url = topic_url.rstrip("/") + "/"
        self._delay = request_delay
        self._client: Optional[httpx.AsyncClient] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _client_(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── listing page ─────────────────────────────────────────────────────────

    def _listing_url(self, page: int) -> str:
        return self._topic_url if page <= 1 else f"{self._topic_url}page/{page}/"

    async def get_article_urls(self, page: int = 1) -> list[str]:
        url = self._listing_url(page)
        logger.info("Fetching listing page %d: %s", page, url)

        try:
            resp = await self._client_().get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Listing fetch failed (%s): %s", url, exc)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        anchors = soup.select(_SEL_LISTING_LINK)

        # Fallback if selector misses (layout change)
        if not anchors:
            anchors = soup.select("h2 a, h3 a")

        urls: list[str] = []
        for a in anchors:
            href = a.get("href", "")
            if not href:
                continue
            if href.startswith("http"):
                urls.append(href)
            else:
                urls.append(urljoin("https://techcrunch.com", href))

        # Deduplicate, preserve order
        seen: set[str] = set()
        unique: list[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)

        logger.info("Found %d article URLs on listing page %d", len(unique), page)
        return unique

    # ── article page ─────────────────────────────────────────────────────────

    async def fetch_article(self, url: str) -> Optional[ArticleContent]:
        await asyncio.sleep(self._delay)

        try:
            resp = await self._client_().get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Article fetch failed (%s): %s", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        title       = _text(soup, _SEL_TITLE)
        author      = _text(soup, _SEL_AUTHOR)
        published_at = _datetime(soup, _SEL_DATE)
        body_text   = _body(soup)

        if not body_text:
            logger.warning("No body text extracted from %s", url)
            return None

        return ArticleContent(
            url=url,
            title=title,
            author=author,
            published_at=published_at,
            body_text=body_text,
            source=self.source_id,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _text(soup: BeautifulSoup, selector: str) -> Optional[str]:
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else None


def _datetime(soup: BeautifulSoup, selector: str) -> Optional[datetime]:
    el = soup.select_one(selector)
    if not el:
        return None
    raw = el.get("datetime") or el.get_text(strip=True)
    try:
        return datetime.fromisoformat(raw.rstrip("Z"))
    except (ValueError, AttributeError):
        return None


def _body(soup: BeautifulSoup) -> str:
    container = soup.select_one(_SEL_BODY)
    if not container:
        container = soup.find("main") or soup

    for tag in container.find_all(["script", "style", "nav", "aside", "figure"]):
        tag.decompose()

    paragraphs = container.find_all("p")
    return "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs).strip()
