"""
Article ingestion endpoints.

POST /articles   submit one URL for crawl + extract + persist
POST /rescan     walk listing pages across all crawlers, process each URL

Both surface 'already_exists' / 'skipped' bookkeeping so callers can tell
fresh work apart from no-ops.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_graph_service
from app.core.graph_service import GraphService
from app.schemas.api import (
    ArticleSubmitRequest,
    ArticleSubmitResponse,
    RescanRequest,
    RescanResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["articles"])


@router.post(
    "/articles",
    response_model=ArticleSubmitResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a single article URL for processing",
)
async def submit_article(
    body: ArticleSubmitRequest,
    service: GraphService = Depends(get_graph_service),
) -> ArticleSubmitResponse:
    url = str(body.url)
    try:
        summary = await service.process_article(
            url=url,
            sentences_per_chunk=body.sentences_per_chunk,
        )
    except ValueError as exc:
        # No crawler registered for this hostname
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        # Crawler reached the page but couldn't parse it
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    return ArticleSubmitResponse(
        article_id=summary["article_id"],
        url=url,
        title=summary.get("title"),
        people_found=summary["people_resolved"],
        relationships_found=summary["relationships_stored"],
        status=summary["status"],
    )


@router.post(
    "/rescan",
    response_model=RescanResponse,
    status_code=status.HTTP_200_OK,
    summary="Crawl listing pages across all registered sources",
)
async def rescan(
    body: RescanRequest,
    service: GraphService = Depends(get_graph_service),
) -> RescanResponse:
    # Snapshot graph size so we can report deltas.
    people_before, rels_before = await service.get_counts()

    summary = await service.rescan(
        pages=body.pages,
        sentences_per_chunk=body.sentences_per_chunk,
        max_parallel=body.max_parallel,
    )

    people_after, rels_after = await service.get_counts()

    return RescanResponse(
        pages_crawled=summary["pages_crawled"],
        articles_processed=summary["articles_processed"],
        articles_skipped=summary["articles_skipped"],
        new_people=max(0, people_after - people_before),
        new_relationships=max(0, rels_after - rels_before),
        status="partial" if summary["errors"] else "complete",
    )
