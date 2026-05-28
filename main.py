"""
FastAPI application entry point.

Run with:
    uvicorn main:app --reload

The lifespan handler wires up the singletons the rest of the app depends on:
  - the DB schema (init_db)
  - all crawler instances (CrawlerRegistry)
  - the GraphService (stored on app.state for the dependency provider)
and closes crawler HTTP clients on shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings
from app.core.graph_service import GraphService
from app.crawlers.base import CrawlerRegistry
from app.crawlers.techcrunch import TechCrunchCrawler
from app.db.session import init_db
from app.extractors.llm_extractor import LLMExtractor
from app.extractors.openai_client import OpenAIClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    # ── DB ────────────────────────────────────────────────────────────────
    await init_db(settings.database_url)

    # ── Crawlers ──────────────────────────────────────────────────────────
    CrawlerRegistry.register(
        TechCrunchCrawler(request_delay=settings.request_delay)
    )

    # ── LLM extractor + GraphService singleton ────────────────────────────
    client = OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_model)
    extractor = LLMExtractor(
        client=client,
        default_sentences_per_chunk=settings.sentences_per_chunk,
    )
    app.state.graph_service = GraphService(extractor=extractor)

    try:
        yield
    finally:
        for crawler in CrawlerRegistry.all().values():
            await crawler.close()


app = FastAPI(
    title="RelationshipFinder",
    description=(
        "Crawl news articles, extract the people mentioned and the relationships "
        "between them with an LLM, store the result as a provenance-tracked graph, "
        "and expose it over a small read API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
