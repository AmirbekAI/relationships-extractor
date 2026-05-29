"""
Unit tests for the resolver's ambiguous-sub-name path and the dynamic
ambiguity-detection mechanism.

What's being tested:
  * resolve_person treats `recency` as read-only and never writes it.
  * update_token_owners_and_recency does the right bookkeeping (adds
    owners, records recency only when a token has 2+ owners).
  * The four behavioural cases of ambiguous sub-name resolution:
      1. Unique sub-name match               → auto-accept (no recency consulted)
      2. Ambiguous + no recency arg          → refuse (None)
      3. Ambiguous + recency points at one
         of the candidates                   → accept (recency hit)
      4. Ambiguous + recency exists but
         doesn't match any candidate         → refuse (None)
  * Dynamic detection: a token that's NOT contested at the start of the
    article becomes contested the moment a second owner is added — and
    the very next ambiguous reference uses recency to resolve.

The repo and extractor are minimal stand-ins; LLM is asserted-never-called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from app.core.entity_resolver import (
    ResolveResult,
    build_token_owners,
    resolve_person,
    update_token_owners_and_recency,
)


# ── minimal async test doubles ───────────────────────────────────────────────

@dataclass
class _StubRepo:
    """Just enough of GraphRepository to drive resolve_person."""
    aliases: list[tuple[str, str, str]]          # (surface_form, person_id, canonical_name)
    added_aliases: list[tuple[str, str]] = field(default_factory=list)

    async def find_person_by_alias(self, surface_form: str) -> Optional[tuple[str, str]]:
        for sf, pid, canonical in self.aliases:
            if sf == surface_form:
                return pid, canonical
        return None

    async def get_all_aliases(self) -> list[tuple[str, str, str]]:
        return list(self.aliases)

    async def add_alias(self, person_id: str, surface_form: str) -> None:
        self.added_aliases.append((person_id, surface_form))


class _StubExtractor:
    """resolve_person should never reach the LLM for any of these tests."""
    async def resolve_alias_with_llm(self, name, candidates):  # pragma: no cover
        raise AssertionError(
            f"LLM was called unexpectedly: name={name!r}, candidates={candidates!r}"
        )


def _seed(*pairs: tuple[str, str]) -> _StubRepo:
    """Build a repo from (person_id, canonical_name) pairs. Each canonical
    is seeded as an alias of itself (the normalised form)."""
    aliases = [(canonical.lower(), pid, canonical) for pid, canonical in pairs]
    return _StubRepo(aliases=aliases)


# ── update_token_owners_and_recency: the bookkeeping primitive ───────────────

def test_bookkeeping_only_records_recency_when_token_is_contested():
    """A token's first owner just gets added; recency stays empty until 2+ owners exist."""
    token_owners: dict[str, set[str]] = {}
    recency: dict[str, str] = {}

    # First Anthony: token_owners["anthony"] = {p1}; recency untouched (1 owner).
    update_token_owners_and_recency("p1", "Anthony Ha", token_owners, recency)
    assert token_owners == {"anthony": {"p1"}}
    assert recency == {}

    # Second Anthony: 2 owners → recency now tracks "anthony" → p2.
    update_token_owners_and_recency("p2", "Anthony Garcia", token_owners, recency)
    assert token_owners == {"anthony": {"p1", "p2"}, "garcia": {"p2"}}
    assert recency == {"anthony": "p2"}

    # Third Anthony (already-collided token): recency updates to the new one.
    update_token_owners_and_recency("p3", "Anthony Smith", token_owners, recency)
    assert recency["anthony"] == "p3"


def test_bookkeeping_ignores_short_tokens():
    """Anything below the min-len floor (default 4) is irrelevant."""
    token_owners: dict[str, set[str]] = {}
    recency: dict[str, str] = {}
    update_token_owners_and_recency("p1", "Sam Lo", token_owners, recency)  # "sam" + "lo"
    # "lo" is 2 chars, "sam" is 3 chars → neither qualifies.
    assert token_owners == {}
    assert recency == {}


# ── resolve_person: ambiguous sub-name behavioural cases ─────────────────────

@pytest.mark.asyncio
async def test_unique_subname_auto_accepts():
    """'Altman' → 'Sam Altman' when Sam Altman is the only Altman in the DB."""
    repo = _seed(("p1", "Sam Altman"))
    recency: dict[str, str] = {}

    result = await resolve_person(
        "Altman", repo, _StubExtractor(),
        recency=recency,
    )

    assert result == ResolveResult(person_id="p1", canonical_name="Sam Altman", stage="subname")
    # resolve_person doesn't write recency itself — that's _resolve_or_create's job.
    assert recency == {}


@pytest.mark.asyncio
async def test_ambiguous_subname_no_recency_refuses():
    """No recency arg → ambiguous resolves to None instead of guessing."""
    repo = _seed(("p1", "Anthony Ha"), ("p2", "Anthony Garcia"))

    result = await resolve_person(
        "Anthony", repo, _StubExtractor(),
        # recency intentionally NOT passed
    )

    assert result is None


@pytest.mark.asyncio
async def test_ambiguous_subname_recency_disambiguates():
    """Recency points at one candidate → that one wins, no LLM call."""
    repo = _seed(("p1", "Anthony Ha"), ("p2", "Anthony Garcia"))
    recency = {"anthony": "p1"}  # we just resolved Anthony Ha earlier in this article

    result = await resolve_person(
        "Anthony", repo, _StubExtractor(),
        recency=recency,
    )

    assert result == ResolveResult(person_id="p1", canonical_name="Anthony Ha", stage="subname")


@pytest.mark.asyncio
async def test_ambiguous_subname_recency_off_target_still_refuses():
    """Recency is non-empty but doesn't share any token with the surface → refuse."""
    repo = _seed(("p1", "Anthony Ha"), ("p2", "Anthony Garcia"))
    recency = {"smith": "px"}  # different family entirely

    result = await resolve_person(
        "Anthony", repo, _StubExtractor(),
        recency=recency,
    )

    assert result is None


# ── dynamic detection: collision emerges *mid-article* ───────────────────────

def test_dynamic_detection_mid_article_collision_bookkeeping():
    """
    Bookkeeping unit-test: the DB starts with only Anthony Ha (no
    collision). A second Anthony is added mid-article. The token
    "anthony" must transition from uncontested to contested at that
    moment — recency only starts tracking once the 2nd owner shows up.
    """
    # Start-of-article DB snapshot: only one Anthony.
    initial_aliases = [("anthony ha", "p1", "Anthony Ha")]
    token_owners = build_token_owners(initial_aliases)
    recency: dict[str, str] = {}

    # First resolution: Anthony Ha (already in DB). No collision yet.
    update_token_owners_and_recency("p1", "Anthony Ha", token_owners, recency)
    assert recency == {}, "single-owner token must stay out of recency"

    # Mid-article: a NEW person Anthony Garcia gets created.
    update_token_owners_and_recency("p2", "Anthony Garcia", token_owners, recency)
    assert recency == {"anthony": "p2"}, (
        "the moment the 2nd Anthony is added, the token becomes contested "
        "and recency starts tracking it"
    )


@pytest.mark.asyncio
async def test_llm_fallback_disabled_skips_llm_returns_none():
    """
    Cost mode: when use_llm_fallback=False, the resolver must not call the
    LLM even if it has plausible candidates. It returns None so the caller
    creates a fresh Person.

    Setup: a surface "Sam Altmen" (typo, sim ~0.91 — actually triggers the
    Levenshtein auto-accept). To exercise the LLM fallback path specifically,
    use a longer typo that lands in the 0.50–0.80 LLM-candidate window.
    "Samuel Altmen" vs "Sam Altman": substantial similarity but well below
    0.80, with no exact alias, no subname subset.
    """
    repo = _seed(("p1", "Sam Altman"))
    extractor = _StubExtractor()  # raises AssertionError if .resolve_alias_with_llm is hit

    # With LLM disabled, we expect None even though "samuel altmen" would
    # otherwise reach the LLM (similarity in the 0.5-0.8 window via the
    # shared 'altmen' Levenshtein noise — and not a subname subset).
    result = await resolve_person(
        "Samuel Altmen", repo, extractor,
        use_llm_fallback=False,
    )

    assert result is None  # treated as a new person — caller will create it


@pytest.mark.asyncio
async def test_end_to_end_mid_article_collision_then_bare_reference():
    """
    The full scenario the snapshot design got wrong, end-to-end.

    DB at article start: only Anthony Ha.
      1) Article mentions "Anthony Ha"    → alias hit, no collision yet.
      2) Article introduces "Anthony Garcia" (fresh person, mid-article).
         → bookkeeping flips "anthony" to contested and records recency=p2.
      3) Article then mentions just "Anthony".
         → MUST resolve to Anthony Garcia via recency.
         A snapshot-based design computes contested_tokens={} at step 0
         and would refuse here, treating "Anthony" as yet another new
         person. This implementation does not.
    """
    # Step 0: DB seed.
    repo = _seed(("p1", "Anthony Ha"))
    token_owners = build_token_owners(repo.aliases)
    recency: dict[str, str] = {}

    # Step 1: "Anthony Ha" — alias hit, no LLM, no collision yet.
    res1 = await resolve_person("Anthony Ha", repo, _StubExtractor(), recency=recency)
    assert res1 == ResolveResult(person_id="p1", canonical_name="Anthony Ha", stage="alias")
    update_token_owners_and_recency("p1", "Anthony Ha", token_owners, recency)
    assert recency == {}, "still only one owner of 'anthony'"

    # Step 2: "Anthony Garcia" — fresh person. Simulate the create that
    # _resolve_or_create would do (insert into the repo's aliases) and run
    # the bookkeeping the same way the service would.
    repo.aliases.append(("anthony garcia", "p2", "Anthony Garcia"))
    update_token_owners_and_recency("p2", "Anthony Garcia", token_owners, recency)
    assert recency == {"anthony": "p2"}, "collision detected mid-article"

    # Step 3: bare "Anthony" — must use recency to land on Anthony Garcia.
    res3 = await resolve_person("Anthony", repo, _StubExtractor(), recency=recency)
    assert res3 == ResolveResult(
        person_id="p2", canonical_name="Anthony Garcia", stage="subname",
    ), (
        "bare 'Anthony' should resolve to the most-recently-saved Anthony "
        "via the recency mechanism populated mid-article"
    )
