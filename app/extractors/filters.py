"""
Post-extraction filtering.

Belt-and-suspenders defense against the LLM emitting non-human entities
(companies, products, government bodies) as people. The prompt is the
primary defense — this module is the safety net for cases that slip through.

What it does:
  • is_likely_organization(name) — predicate that flags org-shaped names by
        either (a) exact match against a blocklist of well-known companies /
        media outlets, or (b) a corporate-suffix tail token (Inc, Corp, Ltd…).
  • filter_extraction(people, relationships) — drops any flagged person AND
        any relationship whose source or target is a flagged name. Returns
        the cleaned lists plus a small report dict for logging.

Kept tiny on purpose: the blocklist is editable in one place, the suffix
list in another, and adding a new term is a one-line change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── blocklist ────────────────────────────────────────────────────────────────
# Stored as lowercase, whitespace-collapsed. Add new entries here.
# Focused on tech orgs + major media since the pipeline ingests tech news.

KNOWN_ORGANIZATIONS: frozenset[str] = frozenset(
    {
        # AI / tech companies
        "openai",
        "anthropic",
        "google",
        "alphabet",
        "google deepmind",
        "deepmind",
        "microsoft",
        "meta",
        "facebook",
        "instagram",
        "whatsapp",
        "amazon",
        "aws",
        "apple",
        "nvidia",
        "tesla",
        "spacex",
        "x",
        "twitter",
        "xai",
        "x.ai",
        "ibm",
        "oracle",
        "salesforce",
        "adobe",
        "intel",
        "amd",
        "samsung",
        "sony",
        "cerebras",
        "groq",
        "perplexity",
        "stability ai",
        "stability",
        "hugging face",
        "huggingface",
        "mistral",
        "mistral ai",
        "scale ai",
        "scale",
        "y combinator",
        "yc",
        # Major media — pipeline source itself, but body text might mention others
        "techcrunch",
        "the verge",
        "wired",
        "bloomberg",
        "reuters",
        "the new york times",
        "nyt",
        "the washington post",
        "wsj",
        "the wall street journal",
        "bbc",
        "cnn",
        "cnbc",
        "the information",
        # Government / regulatory bodies that occasionally appear as subjects
        "ftc",
        "sec",
        "doj",
        "fbi",
        "eu",
        "european union",
        "white house",
        "congress",
        "senate",
        "house",
        "supreme court",
        # Vague collective subjects the LLM sometimes anthropomorphises
        "the board",
        "the company",
        "the team",
        "the firm",
    }
)


# ── corporate suffixes ───────────────────────────────────────────────────────
# Matched against the LAST whitespace token of the name, after stripping
# trailing punctuation. Conservative — only unambiguous corporate tails.

ORG_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "llc",
        "llp",
        "gmbh",
        "ag",
        "sa",
        "plc",
        "pty",
        "bv",
        "nv",
        "holdings",
    }
)


# ── placeholder / unknown markers ────────────────────────────────────────────
# The LLM sometimes emits these as people because the prompt feeds it
# 'Author: Unknown' when the crawler couldn't find a byline. Caught as
# "non-person" by the same predicate the org check uses.

KNOWN_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "unknown",
        "anonymous",
        "anon",
        "n/a",
        "na",
        "unspecified",
        "unattributed",
        "not specified",
        "no author",
        "no byline",
        "staff writer",
        "staff",
        "editor",
        "editorial staff",
        "the editors",
        "tba",
        "tbd",
    }
)

_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")


# ── core predicate ───────────────────────────────────────────────────────────


def _normalise_for_match(name: str) -> str:
    """Lowercase + collapse internal whitespace — does NOT strip honorifics
    (we want 'Dr. OpenAI' to still flag as OpenAI). Keep simple."""
    return " ".join(name.lower().split())


def is_likely_organization(name: str) -> bool:
    """
    True if *name* shouldn't be treated as a real person — covers companies
    / orgs / media / government bodies AND placeholder markers like
    'Unknown' / 'Anonymous' / 'staff writer' that the LLM sometimes emits
    because the prompt feeds it 'Author: Unknown' when there's no byline.

    Kept under this name (rather than ``is_likely_non_person``) so existing
    call sites and tests don't break — the predicate is the same shape, the
    blocklists just grew.
    """
    if not name or not name.strip():
        return True  # nothing usable — treat as garbage

    norm = _normalise_for_match(name)
    if norm in KNOWN_ORGANIZATIONS:
        return True
    if norm in KNOWN_PLACEHOLDERS:
        return True

    # Suffix check on the last token.
    tokens = norm.split()
    if not tokens:
        return True
    last = _TRAILING_PUNCT.sub("", tokens[-1])
    if last in ORG_SUFFIXES:
        return True

    return False


# ── filter ───────────────────────────────────────────────────────────────────


@dataclass
class FilterReport:
    dropped_people: list[str]
    dropped_relationships: list[tuple[str, str, str]]  # (src, tgt, type)

    @property
    def n_dropped_people(self) -> int:
        return len(self.dropped_people)

    @property
    def n_dropped_relationships(self) -> int:
        return len(self.dropped_relationships)


def filter_extraction(people, relationships):
    """
    Drop org-shaped people + any relationship that touches one.

    *people* is anything iterable whose elements have a ``name`` attribute.
    *relationships* is anything iterable whose elements have
    ``source_person``, ``target_person`` and ``relation_type`` attributes.

    Returns (kept_people, kept_relationships, FilterReport).
    """
    kept_people = []
    dropped_names: set[str] = set()

    for p in people:
        if is_likely_organization(p.name):
            dropped_names.add(_normalise_for_match(p.name))
            continue
        kept_people.append(p)

    kept_rels = []
    dropped_rels: list[tuple[str, str, str]] = []
    for r in relationships:
        src_org = is_likely_organization(r.source_person)
        tgt_org = is_likely_organization(r.target_person)
        if src_org or tgt_org:
            dropped_rels.append((r.source_person, r.target_person, r.relation_type))
            # also remember the org names in case they slipped past the people pass
            if src_org:
                dropped_names.add(_normalise_for_match(r.source_person))
            if tgt_org:
                dropped_names.add(_normalise_for_match(r.target_person))
            continue
        kept_rels.append(r)

    report = FilterReport(
        dropped_people=sorted(dropped_names),
        dropped_relationships=dropped_rels,
    )

    if report.n_dropped_people or report.n_dropped_relationships:
        logger.info(
            "Filter dropped %d org-people, %d org-touching relationships: %s",
            report.n_dropped_people,
            report.n_dropped_relationships,
            report.dropped_people,
        )

    return kept_people, kept_rels, report
