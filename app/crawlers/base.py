"""
Base crawler interface.

Any new source (The Verge, Wired, …) is added by:
  1. Subclassing BaseCrawler in a new file
  2. Implementing get_article_urls() and fetch_article()
  3. Registering the instance in CrawlerRegistry

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse


@dataclass
class ArticleContent:
    """Structured content returned by a crawler for a single article."""

    url: str
    title: Optional[str]
    author: Optional[str]
    published_at: Optional[datetime]
    body_text: str  # clean plain-text body, ready for chunking
    source: str  # matches BaseCrawler.source_id


class BaseCrawler(ABC):
    source_id: str = ""  # must be overridden

    def __init__(self) -> None:
        # Per-instance lock used by subclasses to serialise outbound HTTP
        # requests to this crawler's host. Lazy-init so the Lock binds to
        # whatever event loop the first await runs on.
        self._host_lock: Optional[asyncio.Lock] = None

    def _get_host_lock(self) -> asyncio.Lock:
        if self._host_lock is None:
            self._host_lock = asyncio.Lock()
        return self._host_lock

    @abstractmethod
    async def get_article_urls(self, page: int = 1) -> list[str]:
        """Return article URLs from listing page *page* (1 = most recent)."""
        ...

    @abstractmethod
    async def fetch_article(self, url: str) -> Optional[ArticleContent]:
        """Fetch and parse a single article. Returns None on unrecoverable failure."""
        ...

    async def close(self) -> None:
        """Optional teardown — close HTTP client, etc."""


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class CrawlerRegistry:
    _registry: dict[str, BaseCrawler] = {}

    @classmethod
    def register(cls, crawler: BaseCrawler) -> None:
        cls._registry[crawler.source_id] = crawler

    @classmethod
    def get(cls, source_id: str) -> Optional[BaseCrawler]:
        return cls._registry.get(source_id)

    @classmethod
    def for_url(cls, url: str) -> Optional[BaseCrawler]:
        """Pick the right crawler based on the article URL hostname."""
        host = (urlparse(url).hostname or "").lower()
        for crawler in cls._registry.values():
            if crawler.source_id in host:
                return crawler
        return None

    @classmethod
    def all(cls) -> dict[str, "BaseCrawler"]:
        """Return all registered crawlers keyed by source_id."""
        return dict(cls._registry)
