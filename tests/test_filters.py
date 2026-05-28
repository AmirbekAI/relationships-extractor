"""
Tests for the post-extraction org filter.

Pure unit tests — no LLM, no DB.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.extractors.filters import (
    filter_extraction,
    is_likely_organization,
)


# ── is_likely_organization ───────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "OpenAI", "openai", "OPENAI",                    # blocklist exact + case
    "Microsoft", "Anthropic", "Google", "Amazon",
    "Google DeepMind",                               # multi-word blocklist entry
    "TechCrunch", "Reuters",                         # media
    "the board", "the company",                      # collective subjects
    "Acme Inc", "Foo Corp.", "Bar Ltd",              # suffix tails
    "Foo LLC", "Baz GmbH", "Quux Holdings",
    "",                                              # empty
    "   ",                                           # whitespace only
])
def test_is_organization_positives(name):
    assert is_likely_organization(name) is True, f"{name!r} should flag as org"


@pytest.mark.parametrize("name", [
    "Sam Altman", "Elon Musk", "Satya Nadella",
    "Sundar Pichai", "Dario Amodei", "Greg Brockman",
    "Mary Smith", "Pat Lee",
    "Jean-Luc Picard",
    "Andrew Ng",
])
def test_is_organization_negatives(name):
    assert is_likely_organization(name) is False, f"{name!r} should NOT flag as org"


# ── filter_extraction ────────────────────────────────────────────────────────

@dataclass
class _P:
    name: str
    role: str | None = None


@dataclass
class _R:
    source_person: str
    target_person: str
    relation_type: str
    explanation: str = ""
    supporting_quote: str = ""


def test_filter_drops_org_person_and_touching_edge():
    people = [
        _P("Sam Altman"),
        _P("OpenAI"),                  # org — must drop
        _P("Satya Nadella"),
    ]
    rels = [
        _R("Satya Nadella", "Sam Altman", "partners with"),
        _R("Sam Altman", "OpenAI", "leads"),                       # edge touches org
        _R("OpenAI", "Microsoft", "partners with"),                 # both org
    ]

    kept_p, kept_r, report = filter_extraction(people, rels)

    assert [p.name for p in kept_p] == ["Sam Altman", "Satya Nadella"]
    assert len(kept_r) == 1
    assert kept_r[0].source_person == "Satya Nadella"
    assert report.n_dropped_people == 2     # OpenAI + Microsoft (seen via edge)
    assert report.n_dropped_relationships == 2


def test_filter_no_op_when_clean():
    people = [_P("Sam Altman"), _P("Elon Musk")]
    rels = [_R("Elon Musk", "Sam Altman", "criticizes")]

    kept_p, kept_r, report = filter_extraction(people, rels)

    assert kept_p == people
    assert kept_r == rels
    assert report.n_dropped_people == 0
    assert report.n_dropped_relationships == 0


def test_filter_drops_corporate_suffix():
    people = [_P("Sam Altman"), _P("Acme Inc")]
    rels = [_R("Sam Altman", "Acme Inc", "leads")]

    kept_p, kept_r, _ = filter_extraction(people, rels)

    assert [p.name for p in kept_p] == ["Sam Altman"]
    assert kept_r == []
