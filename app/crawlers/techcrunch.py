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
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.crawlers.base import ArticleContent, BaseCrawler

logger = logging.getLogger(__name__)

# ── CSS selectors ────────────────────────────────────────────────────────────
# TechCrunch ships more than one article layout (the standard post, the
# podcast "hero", …). Each field lists selectors in priority order; the first
# one that matches wins, so a new layout is supported by appending a selector
# here rather than touching the parsing code.
_SEL_LISTING_LINK = "a.loop-card__title-link"
_SEL_TITLE = (
    "h1.article__title",
    "h1.wp-block-techcrunch-podcast-single-hero__title",
    "h1.wp-block-post-title",
)
_SEL_AUTHOR = (
    "a.article__author-name",
    "a.wp-block-tc23-author-card-name__link",
)
_SEL_TIME = (
    "time.article__date",
    "time[datetime]",
)
# Containers whose text is scanned for a free-text date ("May 22, 2026") when
# no <time> element or published-time meta tag is present (podcast layout).
_SEL_DATE_CONTAINER = (
    '[class*="hero__meta"], [class*="byline"], [class*="post-meta"], '
    '[class*="post-date"]'
)
_SEL_BODY = "div.article-content"

_USER_AGENT = "Mozilla/5.0 (compatible; RelationshipFinderBot/1.0)"
_TIMEOUT = 30  # seconds per request
_POLITE_DELAY = 1.5  # seconds between requests to the same host


class TechCrunchCrawler(BaseCrawler):
    source_id = "techcrunch"

    def __init__(
        self,
        topic_url: str = "https://techcrunch.com/tag/openai/",
        request_delay: float = _POLITE_DELAY,
    ) -> None:
        super().__init__()
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

        # Serialise via the per-host lock so concurrent callers (e.g. parallel
        # rescan) don't burst requests at TechCrunch. No sleep here — listing
        # pages aren't on the polite-delay budget; only article fetches are.
        try:
            async with self._get_host_lock():
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
        # Hold the per-host lock across delay + GET so:
        #   (a) only one fetch is outstanding to this host at a time, and
        #   (b) consecutive fetches are spaced by at least `self._delay`.
        # Concurrent rescan callers therefore can't violate politeness even
        # with max_parallel_articles > 1 — they queue here.
        try:
            async with self._get_host_lock():
                await asyncio.sleep(self._delay)
                resp = await self._client_().get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Article fetch failed (%s): %s", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        title = _first_text(soup, _SEL_TITLE)
        author = _authors(soup)
        published_at = _published_at(soup)
        body_text = _body(soup)

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

# "May 22, 2026" — the human-readable date the podcast layout renders instead
# of a machine-readable <time datetime="…"> element.
_LONG_DATE_RE = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}")


def _clean(text: str) -> str:
    """Collapse the non-breaking spaces TechCrunch sprinkles through titles."""
    return text.replace("\xa0", " ").strip()


def _first_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> Optional[str]:
    """First non-empty text matching *selectors* in priority order."""
    for selector in selectors:
        el = soup.select_one(selector)
        if el:
            text = _clean(el.get_text(strip=True))
            if text:
                return text
    return None


def _authors(soup: BeautifulSoup) -> Optional[str]:
    """
    Join every byline author into one string, de-duplicated and order-preserving.

    The standard layout has a single ``a.article__author-name``; the podcast
    layout lists several ``a.wp-block-tc23-author-card-name__link`` cards (and
    repeats them in the DOM), so we dedupe.
    """
    for selector in _SEL_AUTHOR:
        anchors = soup.select(selector)
        if not anchors:
            continue
        names: list[str] = []
        seen: set[str] = set()
        for a in anchors:
            name = _clean(a.get_text(strip=True))
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
        if names:
            return ", ".join(names)
    return None


def _parse_iso(raw: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; tolerate a trailing 'Z' on Python <3.11."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_long_date(raw: Optional[str]) -> Optional[datetime]:
    """Parse a 'May 22, 2026' style human date."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%B %d, %Y")
    except (ValueError, AttributeError):
        return None


def _published_at(soup: BeautifulSoup) -> Optional[datetime]:
    """
    Resolve the publish date across layouts, most-reliable source first:
      1. ``<meta property="article:published_time">`` (Yoast/WordPress — ISO).
      2. A ``<time datetime="…">`` element (standard article layout).
      3. A free-text date inside a byline/meta container (podcast layout).
    """
    meta = soup.find("meta", attrs={"property": "article:published_time"})
    if meta is not None:
        dt = _parse_iso(meta.get("content"))
        if dt:
            return dt

    for selector in _SEL_TIME:
        el = soup.select_one(selector)
        if el is not None:
            raw = el.get("datetime") or el.get_text(strip=True)
            dt = _parse_iso(raw) or _parse_long_date(raw)
            if dt:
                return dt

    for container in soup.select(_SEL_DATE_CONTAINER):
        match = _LONG_DATE_RE.search(container.get_text(" ", strip=True))
        if match:
            dt = _parse_long_date(match.group())
            if dt:
                return dt

    return None


def _body(soup: BeautifulSoup) -> str:
    container = soup.select_one(_SEL_BODY)
    if not container:
        container = soup.find("main") or soup

    for tag in container.find_all(["script", "style", "nav", "aside", "figure"]):
        tag.decompose()

    paragraphs = container.find_all("p")
    return "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs).strip()
