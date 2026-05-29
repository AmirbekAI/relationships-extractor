"""
Black-box tests for the FastAPI surface.

The HTTP layer is exercised end-to-end with FastAPI's TestClient against a
GraphService whose methods are stubbed in-process. We're NOT re-testing the
service logic here (that lives in test_chunk_checkpoint.py etc.); we're
testing that the routes:

  * pull the service via the Depends(get_graph_service) wiring,
  * marshal request bodies through the Pydantic DTOs,
  * map service exceptions to the right HTTP status codes,
  * shape responses to match the schema.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router as api_router
from app.core.graph_service import GraphService


# ── stub service ─────────────────────────────────────────────────────────────

class _StubService:
    """
    Implements just the GraphService methods the routes touch. Each test
    swaps in its own instance via dependency override.
    """

    def __init__(self) -> None:
        self.process_article_calls: list[tuple[str, Optional[int]]] = []
        self.rescan_calls: list[tuple[int, Optional[int]]] = []
        # Configurable per test
        self.process_article_result: dict[str, Any] = {}
        self.process_article_raises: Optional[Exception] = None
        self.rescan_result: dict[str, Any] = {}
        self.counts_seq: list[tuple[int, int]] = [(0, 0), (0, 0)]
        self.people_result: tuple[list, int] = ([], 0)
        self.detail_result: Optional[dict[str, Any]] = None

    async def process_article(self, url: str, sentences_per_chunk=None):
        self.process_article_calls.append((url, sentences_per_chunk))
        if self.process_article_raises is not None:
            raise self.process_article_raises
        return self.process_article_result

    async def rescan(self, pages, sentences_per_chunk=None, source_ids=None):
        self.rescan_calls.append((pages, sentences_per_chunk))
        return self.rescan_result

    async def get_counts(self) -> tuple[int, int]:
        return self.counts_seq.pop(0)

    async def get_people(self, page, page_size):
        return self.people_result

    async def get_person_detail(self, person_id: str):
        return self.detail_result


# ── helpers to build TestClient with stub injected ────────────────────────────

def _make_app(svc: _StubService) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    # Override the dependency provider — routes get the stub, not the real svc.
    from app.api.deps import get_graph_service
    app.dependency_overrides[get_graph_service] = lambda: svc
    return app


# ─────────────────────────────────────────────────────────────────────────────
# POST /articles
# ─────────────────────────────────────────────────────────────────────────────

def test_post_articles_returns_200_with_summary():
    svc = _StubService()
    svc.process_article_result = {
        "article_id": "a-1",
        "title": "Sam returns",
        "people_resolved": 3,
        "relationships_stored": 4,
        "status": "processed",
    }
    client = TestClient(_make_app(svc))

    resp = client.post(
        "/articles",
        json={"url": "https://techcrunch.com/2024/01/15/example/"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["article_id"] == "a-1"
    assert body["people_found"] == 3
    assert body["relationships_found"] == 4
    assert body["status"] == "processed"
    assert svc.process_article_calls == [
        ("https://techcrunch.com/2024/01/15/example/", None),
    ]


def test_post_articles_passes_sentences_per_chunk_through():
    svc = _StubService()
    svc.process_article_result = {
        "article_id": "a-2", "title": None,
        "people_resolved": 0, "relationships_stored": 0,
        "status": "already_exists",
    }
    client = TestClient(_make_app(svc))

    resp = client.post(
        "/articles",
        json={
            "url": "https://techcrunch.com/2024/01/15/example/",
            "sentences_per_chunk": 7,
        },
    )

    assert resp.status_code == 200
    assert svc.process_article_calls[0][1] == 7


def test_post_articles_maps_valueerror_to_400():
    svc = _StubService()
    svc.process_article_raises = ValueError("No crawler registered for URL: …")
    client = TestClient(_make_app(svc))

    resp = client.post(
        "/articles",
        json={"url": "https://unsupported.example.com/some-article/"},
    )

    assert resp.status_code == 400
    assert "No crawler registered" in resp.json()["detail"]


def test_post_articles_maps_runtimeerror_to_502():
    svc = _StubService()
    svc.process_article_raises = RuntimeError("Crawler returned nothing for: …")
    client = TestClient(_make_app(svc))

    resp = client.post(
        "/articles",
        json={"url": "https://techcrunch.com/2024/01/15/dead-link/"},
    )

    assert resp.status_code == 502


def test_post_articles_rejects_invalid_url():
    svc = _StubService()
    client = TestClient(_make_app(svc))

    # Pydantic AnyHttpUrl rejects "not-a-url" → 422 before service ever runs.
    resp = client.post("/articles", json={"url": "not-a-url"})

    assert resp.status_code == 422
    assert svc.process_article_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# POST /rescan
# ─────────────────────────────────────────────────────────────────────────────

def test_post_rescan_reports_complete_when_no_errors():
    svc = _StubService()
    svc.counts_seq = [(2, 5), (4, 9)]  # before, after
    svc.rescan_result = {
        "pages_crawled": 2,
        "articles_processed": 3,
        "articles_skipped": 1,
        "relationships_stored": 4,
        "errors": [],
    }
    client = TestClient(_make_app(svc))

    resp = client.post("/rescan", json={"pages": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert body["pages_crawled"] == 2
    assert body["new_people"] == 2          # 4 - 2
    assert body["new_relationships"] == 4   # 9 - 5
    assert body["status"] == "complete"


def test_post_rescan_reports_partial_on_errors():
    svc = _StubService()
    svc.counts_seq = [(0, 0), (1, 1)]
    svc.rescan_result = {
        "pages_crawled": 1, "articles_processed": 1, "articles_skipped": 0,
        "relationships_stored": 1, "errors": ["http://x: boom"],
    }
    client = TestClient(_make_app(svc))

    resp = client.post("/rescan", json={"pages": 1})

    assert resp.status_code == 200
    assert resp.json()["status"] == "partial"


# ─────────────────────────────────────────────────────────────────────────────
# GET /people
# ─────────────────────────────────────────────────────────────────────────────

class _FakePerson:
    """Stand-in for the Person ORM row — only fields the route reads."""
    def __init__(self, pid: str, name: str, aliases: list[str]) -> None:
        self.id = pid
        self.canonical_name = name
        self.aliases = [_FakeAlias(a) for a in aliases]


class _FakeAlias:
    def __init__(self, surface: str) -> None:
        self.surface_form = surface


def test_get_people_paginates():
    svc = _StubService()
    svc.people_result = (
        [
            _FakePerson("p1", "Elon Musk", ["elon musk"]),
            _FakePerson("p2", "Sam Altman", ["sam altman", "altman"]),
        ],
        7,  # total
    )
    client = TestClient(_make_app(svc))

    resp = client.get("/people?page=2&page_size=2")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 7
    assert body["page"] == 2
    assert body["page_size"] == 2
    assert body["total_pages"] == 4  # ceil(7/2)
    assert [item["canonical_name"] for item in body["items"]] == [
        "Elon Musk", "Sam Altman",
    ]


def test_get_people_rejects_bad_pagination():
    client = TestClient(_make_app(_StubService()))

    # page < 1 → 422 (FastAPI Query(ge=1))
    assert client.get("/people?page=0").status_code == 422
    # page_size > 200 → 422
    assert client.get("/people?page_size=999").status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /people/{id}
# ─────────────────────────────────────────────────────────────────────────────

class _FakeRel:
    def __init__(self, rid, src, tgt, rel_type, explanation, provenance) -> None:
        self.id = rid
        self.source_person_id = src.id
        self.source_person = src
        self.target_person_id = tgt.id
        self.target_person = tgt
        self.relation_type = rel_type
        self.explanation = explanation
        self.provenance = provenance


class _FakeArticle:
    def __init__(self, aid, url, title) -> None:
        self.id = aid
        self.url = url
        self.title = title


class _FakeProv:
    def __init__(self, prov_id, article, quote) -> None:
        self.id = prov_id
        self.article_id = article.id
        self.article = article
        self.quote = quote


def test_get_person_returns_404_when_missing():
    svc = _StubService()
    svc.detail_result = None
    client = TestClient(_make_app(svc))

    resp = client.get("/people/unknown-id")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Person not found"


def test_get_person_returns_full_detail_with_provenance():
    svc = _StubService()
    p1 = _FakePerson("p1", "Sam Altman", ["sam altman"])
    p1.bio = "CEO of OpenAI"
    p2 = _FakePerson("p2", "Elon Musk", ["elon musk"])
    article = _FakeArticle("art-1", "https://example.com/a", "Title")
    rel = _FakeRel(
        "r-1", p2, p1, "criticizes", "explanation here",
        [_FakeProv("pv-1", article, "Musk slammed Altman's decision.")],
    )
    svc.detail_result = {"person": p1, "relationships": [rel]}
    client = TestClient(_make_app(svc))

    resp = client.get("/people/p1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "p1"
    assert body["canonical_name"] == "Sam Altman"
    assert body["aliases"] == ["sam altman"]
    assert body["bio"] == "CEO of OpenAI"
    assert len(body["relationships"]) == 1
    edge = body["relationships"][0]
    assert edge["source_person_name"] == "Elon Musk"
    assert edge["target_person_name"] == "Sam Altman"
    assert edge["relation_type"] == "criticizes"
    assert edge["provenance"][0]["article_url"] == "https://example.com/a"
    assert edge["provenance"][0]["quote"] == "Musk slammed Altman's decision."


# ─────────────────────────────────────────────────────────────────────────────
# /health (smoke)
# ─────────────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    # /health lives on the root app (main.py), not on api_router, so build
    # a minimal app that mirrors what main.py wires up.
    app = FastAPI()
    app.include_router(api_router)

    @app.get("/health")
    async def _health():
        return {"status": "ok"}

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
