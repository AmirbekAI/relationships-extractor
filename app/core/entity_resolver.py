"""
Entity resolver — maps a raw extracted name to a Person in the database.

Resolution pipeline (in order):
  1. Exact alias lookup   — O(1) DB lookup on the normalised surface form
  2. Levenshtein distance — catches typos / nickname variations across all
                            known aliases; accepts if similarity ≥ 0.80
  3. LLM fallback         — last resort; LLM picks from candidates with
                            similarity ≥ 0.50

If all three fail the caller should treat the name as a new Person.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from app.db.repository import GraphRepository
    from app.extractors.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)


# Which tier of the pipeline actually produced the answer. Returned from
# resolve_person so callers (the eval, future telemetry) don't have to sniff
# log strings. "none" covers every refuse-or-give-up path.
Stage = Literal["alias", "levenshtein", "subname", "llm", "none"]


@dataclass(frozen=True)
class ResolveResult:
    """What resolve_person decided, and which tier made the call."""

    person_id: str
    canonical_name: str
    stage: Stage


# ── normalisation ─────────────────────────────────────────────────────────────
_TITLES = re.compile(
    r"\b(mr|mrs|ms|dr|prof|sir|dame|lord|lady|rev|gen|col|capt|sgt|cpl|pvt|jr|sr)\.?\b",
    re.IGNORECASE,
)
_NON_ALPHA = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")


def normalize(name: str) -> str:
    """
    Produce a normalised surface form for alias-table storage and lookup.

    Examples
    --------
    "Dr. Sam Altman Jr."  →  "sam altman"
    "Elon  MUSK"          →  "elon musk"
    "Naïve Çağrı"         →  "naive cagri"
    """
    # unicode → ASCII approximation
    n = unicodedata.normalize("NFKD", name)
    n = n.encode("ascii", "ignore").decode()
    # strip honorifics / suffixes
    n = _TITLES.sub(" ", n)
    # lowercase + remove punctuation
    n = _NON_ALPHA.sub(" ", n.lower())
    return _MULTI_SPACE.sub(" ", n).strip()


# ── Levenshtein ───────────────────────────────────────────────────────────────


def _levenshtein(a: str, b: str) -> int:
    """Classic O(m·n) edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,  # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + (ca != cb),  # substitution
                )
            )
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """1.0 = identical strings, 0.0 = nothing in common."""
    dist = _levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


# ── thresholds ────────────────────────────────────────────────────────────────
# Accept Levenshtein match automatically above this similarity
_LEVENSHTEIN_ACCEPT: float = 0.80
# Pass to LLM as a candidate above this (but below accept threshold)
_LLM_CANDIDATE_MIN: float = 0.50
# Also pass to LLM if surface and alias share a token of at least this length.
# Catches "Satya" → "Satya Nadella" and "Altman" → "Sam Altman" — short
# surface forms that fail the similarity floor for length reasons alone.
# 4 chars is the floor so we don't flood the LLM with first-name collisions.
_TOKEN_OVERLAP_MIN_LEN: int = 4


def _shares_long_token(a: str, b: str, min_len: int = _TOKEN_OVERLAP_MIN_LEN) -> bool:
    """True iff *a* and *b* share at least one whitespace token of length >= min_len."""
    a_tokens = {t for t in a.split() if len(t) >= min_len}
    b_tokens = {t for t in b.split() if len(t) >= min_len}
    return bool(a_tokens & b_tokens)


def build_token_owners(
    all_aliases: list[tuple[str, str, str]],
    min_len: int = _TOKEN_OVERLAP_MIN_LEN,
) -> dict[str, set[str]]:
    """
    Build the initial ``long_token -> {person_id, ...}`` map from the alias
    table snapshot. Used to seed the recency mechanism at the start of an
    article; the service then maintains this map as new people are resolved
    or created mid-article, so newly-introduced collisions get caught the
    moment they happen instead of at the next article boundary.
    """
    owners: dict[str, set[str]] = {}
    for surface_form, person_id, _ in all_aliases:
        for token in surface_form.split():
            if len(token) >= min_len:
                owners.setdefault(token, set()).add(person_id)
    return owners


def update_token_owners_and_recency(
    person_id: str,
    canonical_name: str,
    token_owners: dict[str, set[str]],
    recency: dict[str, str],
    min_len: int = _TOKEN_OVERLAP_MIN_LEN,
) -> None:
    """
    Bookkeeping after a person was just resolved or created.

    For every long token of *canonical_name*:
      1. Add *person_id* to that token's owner set in *token_owners*.
      2. If the owner set now has 2+ entries (i.e. the token is currently
         contested in the DB), set ``recency[token] = person_id``.

    Net effect: ``recency`` always reflects the most-recently-handled owner
    of every token that has at least one collision. Tokens with a single
    owner stay out of ``recency`` entirely (no point tracking them).
    """
    for token in normalize(canonical_name).split():
        if len(token) < min_len:
            continue
        owners = token_owners.setdefault(token, set())
        owners.add(person_id)
        if len(owners) >= 2:
            recency[token] = person_id


def _is_subname_of(
    surface_norm: str, alias_norm: str, min_len: int = _TOKEN_OVERLAP_MIN_LEN
) -> bool:
    """
    True iff every long (>= min_len) token of *surface_norm* also appears as
    a whitespace token of *alias_norm*. Captures the "first name only" /
    "last name only" / "title + last name" abbreviations of a canonical
    name *without* spending an LLM call.

    Examples that return True:
        ("anthony",     "anthony ha")     — first name only
        ("altman",      "sam altman")     — last name only
        ("satya",       "satya nadella")  — first name only
        ("ceo altman",  "sam altman")     — title dropped by short-token filter

    Examples that return False:
        ("anthony garcia", "anthony ha")  — surface has token not in alias
        ("sam",            "sam altman")  — surface has no token >= min_len
        ("",               "x")           — no usable tokens
    """
    surface_long = {t for t in surface_norm.split() if len(t) >= min_len}
    if not surface_long:
        return False
    return surface_long.issubset(set(alias_norm.split()))


# ── main entry point ──────────────────────────────────────────────────────────


async def resolve_person(
    raw_name: str,
    repo: "GraphRepository",
    extractor: "LLMExtractor",
    *,
    recency: Optional[dict[str, str]] = None,
    use_llm_fallback: bool = True,
) -> Optional[ResolveResult]:
    """
    Resolve *raw_name* → :class:`ResolveResult` or ``None``.

    Side-effect: on a Levenshtein / sub-name / LLM match the normalised surface
    form is written to the aliases table so future lookups are O(1).

    Parameters
    ----------
    raw_name:           the name string as it appeared in the article
    repo:               open GraphRepository (caller owns the session/transaction)
    extractor:          LLMExtractor instance used as the last-resort resolver
    recency:            optional READ-ONLY per-article ``token -> last_person_id``
                        map. When supplied, an ambiguous sub-name match is broken
                        by preferring the person whose id is currently mapped from
                        a token of the surface form. The resolver never *writes*
                        this map — that bookkeeping happens in ``_resolve_or_create``
                        so it covers both resolutions and freshly-created people.
    use_llm_fallback:   when False, skip the final LLM disambiguator and return
                        None instead. Trades accuracy for cost: every fuzzy match
                        that would have gone to the LLM now becomes a fresh
                        Person row, recoverable via later dedupe.
    """
    norm = normalize(raw_name)
    if not norm:
        return None

    # ── 1. exact alias lookup ─────────────────────────────────────────────────
    hit = await repo.find_person_by_alias(norm)
    if hit:
        logger.debug("resolve '%s' → alias hit '%s'", raw_name, hit[1])
        return ResolveResult(person_id=hit[0], canonical_name=hit[1], stage="alias")

    # ── 2. Levenshtein distance over all known aliases ────────────────────────
    # rows: (surface_form, person_id, canonical_name)
    all_aliases: list[tuple[str, str, str]] = await repo.get_all_aliases()

    best_sim = 0.0
    best_match: Optional[tuple[str, str]] = None  # (person_id, canonical_name)
    llm_candidates: list[str] = []
    # person_id → canonical_name for every alias where the surface's long
    # tokens are a subset of the alias's tokens. Deduped by person_id so
    # multiple alias rows for the same person count as one match.
    subname_matches: dict[str, str] = {}

    for surface_form, person_id, canonical_name in all_aliases:
        sim = _similarity(norm, surface_form)
        if sim >= _LEVENSHTEIN_ACCEPT:
            if sim > best_sim:
                best_sim = sim
                best_match = (person_id, canonical_name)
        else:
            if _is_subname_of(norm, surface_form):
                subname_matches[person_id] = canonical_name
            if sim >= _LLM_CANDIDATE_MIN or _shares_long_token(norm, surface_form):
                llm_candidates.append(canonical_name)

    if best_match:
        logger.debug(
            "resolve '%s' → levenshtein hit '%s' (sim=%.2f)",
            raw_name,
            best_match[1],
            best_sim,
        )
        await repo.add_alias(best_match[0], norm)
        return ResolveResult(
            person_id=best_match[0],
            canonical_name=best_match[1],
            stage="levenshtein",
        )

    # ── 2.5 unique sub-name (e.g. first-name-only) → no LLM needed ────────────
    # If the surface form is a strict abbreviation of exactly one canonical
    # name in the DB, accept it deterministically — no LLM round-trip.
    if len(subname_matches) == 1:
        person_id, canonical_name = next(iter(subname_matches.items()))
        logger.debug(
            "resolve '%s' → subname hit '%s' (unique long-token subset)",
            raw_name,
            canonical_name,
        )
        await repo.add_alias(person_id, norm)
        return ResolveResult(
            person_id=person_id,
            canonical_name=canonical_name,
            stage="subname",
        )

    # ── 2.6 ambiguous sub-name (multiple matches) ─────────────────────────────
    # Try article-recency first (opt-in via the `recency` kwarg). For each
    # long token of the surface, see whether we've already resolved someone
    # with that contested token earlier in this article — if exactly one such
    # recent person is also a subname candidate, accept it.
    if len(subname_matches) > 1:
        if recency:
            surface_long = {t for t in norm.split() if len(t) >= _TOKEN_OVERLAP_MIN_LEN}
            recent_hits = {recency[t] for t in surface_long if t in recency} & set(
                subname_matches.keys()
            )
            if len(recent_hits) == 1:
                person_id = next(iter(recent_hits))
                canonical_name = subname_matches[person_id]
                logger.debug(
                    "resolve '%s' → subname recency hit '%s' "
                    "(last seen earlier in this article)",
                    raw_name,
                    canonical_name,
                )
                await repo.add_alias(person_id, norm)
                return ResolveResult(
                    person_id=person_id,
                    canonical_name=canonical_name,
                    stage="subname",
                )

        # Either recency wasn't enabled, or it didn't disambiguate.
        # Refuse rather than letting the LLM guess without article context.
        logger.debug(
            "resolve '%s' → subname ambiguous (%d candidates: %s), "
            "treating as new person rather than guessing",
            raw_name,
            len(subname_matches),
            list(subname_matches.values()),
        )
        return None

    # ── 3. LLM fallback ───────────────────────────────────────────────────────
    if not llm_candidates:
        logger.debug("resolve '%s' → no candidates, treating as new person", raw_name)
        return None

    if not use_llm_fallback:
        # Cost-sensitive mode: skip the LLM round-trip even though we have
        # plausible candidates. Caller will create a fresh Person; manual
        # / batch dedupe can merge later if it turns out to be wrong.
        logger.debug(
            "resolve '%s' → LLM fallback disabled (had %d candidate(s)), "
            "treating as new person",
            raw_name,
            len(set(llm_candidates)),
        )
        return None

    # de-duplicate while preserving first-seen order
    llm_candidates = list(dict.fromkeys(llm_candidates))
    logger.debug(
        "resolve '%s' → asking LLM among %d candidates", raw_name, len(llm_candidates)
    )

    matched_name = await extractor.resolve_alias_with_llm(raw_name, llm_candidates)
    if not matched_name:
        logger.debug("resolve '%s' → LLM returned no match", raw_name)
        return None

    # map the LLM-chosen canonical name back to a person_id
    for _, person_id, canonical_name in all_aliases:
        if canonical_name == matched_name:
            logger.debug("resolve '%s' → LLM hit '%s'", raw_name, canonical_name)
            await repo.add_alias(person_id, norm)
            return ResolveResult(
                person_id=person_id,
                canonical_name=canonical_name,
                stage="llm",
            )

    return None
