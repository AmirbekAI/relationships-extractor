"""Top-level API router — composes every sub-router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.articles import router as articles_router
from app.api.routes.people import router as people_router

router = APIRouter()
router.include_router(articles_router)
router.include_router(people_router)
