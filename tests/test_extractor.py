"""
Tests for the LLM extractor.

Two test groups:

1. Unit tests (no API calls) — verify sentence splitting, chunking, and
   cross-chunk merging logic using internal helpers directly.

2. Live integration tests (real OpenAI API) — use a fixed, deterministic
   article text with a known set of people and relationships, and assert
   that the model finds them. Run with:
       .venv/bin/python -m pytest tests/test_extractor.py -v -s

   Reads OPENAI_API_KEY from .env automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.crawlers.base import ArticleContent
from app.extractors.llm_extractor import (
    LLMExtractor,
    _ChunkResult,
    _Person,
    _Relationship,
    _chunk,
    _merge,
    _split_sentences,
)
from app.extractors.openai_client import OpenAIClient

# Load .env so OPENAI_API_KEY is available
load_dotenv(Path(__file__).parent.parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _article(body: str, title: str = "Test Article", author: str = "Jane Doe") -> ArticleContent:
    return ArticleContent(
        url="https://techcrunch.com/2024/01/01/test/",
        title=title,
        author=author,
        published_at=None,
        body_text=body,
        source="techcrunch",
    )


def _make_extractor(chunk_size: int = 10) -> LLMExtractor:
    api_key = os.environ["OPENAI_API_KEY"]
    return LLMExtractor(
        client=OpenAIClient(api_key=api_key, model="gpt-4o-mini"),
        default_sentences_per_chunk=chunk_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: helpers (no API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "Sam Altman returned. Elon Musk criticized the board. Satya Nadella expressed support."
    result = _split_sentences(text)
    assert result == [
        "Sam Altman returned.",
        "Elon Musk criticized the board.",
        "Satya Nadella expressed support.",
    ]


def test_split_sentences_question_and_exclamation():
    text = "Did Altman resign? No! He was fired."
    result = _split_sentences(text)
    assert result == ["Did Altman resign?", "No!", "He was fired."]


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_chunk_even_split():
    sentences = ["S1.", "S2.", "S3.", "S4."]
    assert _chunk(sentences, size=2) == ["S1. S2.", "S3. S4."]


def test_chunk_uneven_split():
    sentences = ["S1.", "S2.", "S3."]
    assert _chunk(sentences, size=2) == ["S1. S2.", "S3."]


def test_chunk_size_larger_than_input():
    assert _chunk(["S1.", "S2."], size=10) == ["S1. S2."]


def test_merge_deduplicates_people():
    chunk1 = _ChunkResult(people=[_Person(name="Sam Altman", role="CEO")], relationships=[])
    chunk2 = _ChunkResult(
        people=[_Person(name="Sam Altman", role="CEO"), _Person(name="Elon Musk")],
        relationships=[],
    )
    people, _ = _merge([chunk1, chunk2])
    names = [p.name for p in people]
    assert names.count("Sam Altman") == 1
    assert "Elon Musk" in names


def test_merge_deduplicates_relationships():
    rel = _Relationship(
        source_person="Elon Musk", target_person="Sam Altman",
        relation_type="criticizes", explanation="x", supporting_quote="q",
    )
    _, rels = _merge([
        _ChunkResult(people=[], relationships=[rel]),
        _ChunkResult(people=[], relationships=[rel]),
    ])
    assert len(rels) == 1


def test_merge_keeps_distinct_relationships():
    rel1 = _Relationship(
        source_person="Elon Musk", target_person="Sam Altman",
        relation_type="criticizes", explanation="A", supporting_quote="Q1",
    )
    rel2 = _Relationship(
        source_person="Satya Nadella", target_person="Sam Altman",
        relation_type="supports", explanation="B", supporting_quote="Q2",
    )
    _, rels = _merge([
        _ChunkResult(people=[], relationships=[rel1]),
        _ChunkResult(people=[], relationships=[rel2]),
    ])
    assert len(rels) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Live integration tests (real OpenAI API)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed article text with a known, unambiguous set of people and relationships.
# Expected findings are defined below and compared against the model output.
FIXED_ARTICLE = ArticleContent(
    url="https://techcrunch.com/2024/01/15/altman-returns/",
    title="Sam Altman Returns to OpenAI as CEO",
    author="Jane Doe",
    published_at=None,
    body_text=(
        "Sam Altman has returned to OpenAI as CEO after a brief but dramatic firing by the board. "
        "Elon Musk, an early investor and co-founder of OpenAI, publicly criticized the board's decision to remove Altman. "
        "Satya Nadella, the CEO of Microsoft, expressed strong support for Altman and offered him a senior role at Microsoft. "
        "Greg Brockman, OpenAI's president, resigned in solidarity with Altman shortly after the firing. "
        "Altman and Brockman were later reinstated following significant pressure from OpenAI employees and investors."
    ),
    source="techcrunch",
)

# What we expect the model to find — used for assertions.
EXPECTED_PEOPLE = {"Sam Altman", "Elon Musk", "Satya Nadella", "Greg Brockman", "Jane Doe"}

EXPECTED_RELATIONSHIPS = [
    # (source, target, relation_type_substring)
    ("Elon Musk",    "Sam Altman",    "criticiz"),
    ("Satya Nadella","Sam Altman",    "support"),
    ("Greg Brockman","Sam Altman",    "resign"),
    ("Jane Doe",     "Sam Altman",    "reports"),
]


@pytest.mark.asyncio
async def test_live_extract_finds_expected_people():
    extractor = _make_extractor(chunk_size=10)
    result = await extractor.extract(FIXED_ARTICLE)

    print(f"\n── People found ({len(result.people)}) ──")
    for p in result.people:
        print(f"  {p.name!r:30s}  role={p.role}")

    assert result.error is None, f"Extraction error: {result.error}"

    found_names = {p.name for p in result.people}
    for expected in EXPECTED_PEOPLE:
        assert any(expected.lower() in n.lower() for n in found_names), (
            f"Expected person '{expected}' not found. Got: {found_names}"
        )


@pytest.mark.asyncio
async def test_live_extract_finds_expected_relationships():
    extractor = _make_extractor(chunk_size=10)
    result = await extractor.extract(FIXED_ARTICLE)

    print(f"\n── Relationships found ({len(result.relationships)}) ──")
    for r in result.relationships:
        print(f"  {r.source_person!r:20s} —{r.relation_type}→ {r.target_person!r}")
        print(f"    quote: {r.supporting_quote[:80]}")

    assert result.error is None

    for src, tgt, rel_substr in EXPECTED_RELATIONSHIPS:
        match = any(
            src.lower() in r.source_person.lower()
            and tgt.lower() in r.target_person.lower()
            and rel_substr.lower() in r.relation_type.lower()
            for r in result.relationships
        )
        assert match, (
            f"Expected relationship '{src} —[{rel_substr}]→ {tgt}' not found.\n"
            f"Got: {[(r.source_person, r.relation_type, r.target_person) for r in result.relationships]}"
        )


@pytest.mark.asyncio
async def test_live_extract_chunked_same_result():
    """
    Sending the article in chunks of 2 sentences should produce the same
    essential people and relationships as a single chunk.
    """
    extractor_single = _make_extractor(chunk_size=20)
    extractor_chunked = _make_extractor(chunk_size=2)

    result_single  = await extractor_single.extract(FIXED_ARTICLE)
    result_chunked = await extractor_chunked.extract(FIXED_ARTICLE)

    names_single  = {p.name for p in result_single.people}
    names_chunked = {p.name for p in result_chunked.people}

    print(f"\n── Single chunk  : {sorted(names_single)}")
    print(f"── Chunked (2 s) : {sorted(names_chunked)}")

    # Every person found in the single-chunk run should also appear in the
    # chunked run (chunking must not lose information).
    for name in names_single:
        assert any(name.lower() in n.lower() for n in names_chunked), (
            f"'{name}' found in single run but missing from chunked run"
        )


@pytest.mark.asyncio
async def test_live_resolve_alias_altman():
    """Model should resolve 'Altman' → 'Sam Altman'."""
    extractor = _make_extractor()
    result = await extractor.resolve_alias_with_llm(
        unknown_name="Altman",
        candidates=["Sam Altman", "Elon Musk", "Satya Nadella", "Greg Brockman"],
    )
    print(f"\n── resolve 'Altman' → {result!r}")
    assert result == "Sam Altman"


@pytest.mark.asyncio
async def test_live_resolve_alias_openai_ceo():
    """Model should resolve 'OpenAI CEO' → 'Sam Altman'."""
    extractor = _make_extractor()
    result = await extractor.resolve_alias_with_llm(
        unknown_name="OpenAI CEO",
        candidates=["Sam Altman", "Elon Musk", "Satya Nadella"],
    )
    print(f"\n── resolve 'OpenAI CEO' → {result!r}")
    assert result == "Sam Altman"


@pytest.mark.asyncio
async def test_live_resolve_alias_unknown_returns_none():
    """Completely unrelated name should return None."""
    extractor = _make_extractor()
    result = await extractor.resolve_alias_with_llm(
        unknown_name="Vladimir Putin",
        candidates=["Sam Altman", "Elon Musk", "Satya Nadella"],
    )
    print(f"\n── resolve 'Vladimir Putin' → {result!r}")
    assert result is None
