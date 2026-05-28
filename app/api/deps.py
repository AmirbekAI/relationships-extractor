"""
FastAPI dependency providers.

The GraphService is built once in the app's lifespan handler (see main.py)
and stored on app.state. Routes pull it via Depends(get_graph_service) so
they never instantiate or own service lifecycle themselves.
"""

from __future__ import annotations

from fastapi import Request

from app.core.graph_service import GraphService


def get_graph_service(request: Request) -> GraphService:
    return request.app.state.graph_service
