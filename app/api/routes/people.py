"""
Read-only endpoints over the people graph.

GET /people          paginated list of all known people
GET /people/{id}     one person + every relationship they touch, with provenance
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_graph_service
from app.core.graph_service import GraphService
from app.schemas.api import (
    PaginatedPeopleResponse,
    PersonDetailResponse,
    PersonSummary,
    ProvenanceDTO,
    RelationshipDTO,
)

router = APIRouter(prefix="/people", tags=["people"])


@router.get(
    "",
    response_model=PaginatedPeopleResponse,
    summary="Paginated list of all people in the graph",
)
async def list_people(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    service: GraphService = Depends(get_graph_service),
) -> PaginatedPeopleResponse:
    people, total = await service.get_people(page=page, page_size=page_size)
    total_pages = math.ceil(total / page_size) if total else 0

    return PaginatedPeopleResponse(
        items=[
            PersonSummary(
                id=p.id,
                canonical_name=p.canonical_name,
                aliases=[a.surface_form for a in p.aliases],
            )
            for p in people
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{person_id}",
    response_model=PersonDetailResponse,
    summary="Full detail for one person: aliases + relationships + provenance",
)
async def get_person(
    person_id: str,
    service: GraphService = Depends(get_graph_service),
) -> PersonDetailResponse:
    detail = await service.get_person_detail(person_id)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Person not found")

    person = detail["person"]
    relationships = detail["relationships"]

    return PersonDetailResponse(
        id=person.id,
        canonical_name=person.canonical_name,
        aliases=[a.surface_form for a in person.aliases],
        bio=person.bio,
        relationships=[
            RelationshipDTO(
                id=rel.id,
                source_person_id=rel.source_person_id,
                source_person_name=rel.source_person.canonical_name,
                target_person_id=rel.target_person_id,
                target_person_name=rel.target_person.canonical_name,
                relation_type=rel.relation_type,
                explanation=rel.explanation or "",
                provenance=[
                    ProvenanceDTO(
                        article_id=p.article_id,
                        article_url=p.article.url,
                        article_title=p.article.title,
                        quote=p.quote,
                    )
                    for p in rel.provenance
                ],
            )
            for rel in relationships
        ],
    )
