"""
Scoring primitives.

Two kinds of comparisons:
  • people  — set-based match on normalised names (uses the same normalise()
              the production resolver uses, so the eval can't drift from it).
  • edges   — match on (normalised_src, normalised_tgt, relation_match)
              where relation_match has two flavours:
                strict  → predicted relation_type == one of gold.type_keywords
                fuzzy   → ANY gold.type_keywords appears as a substring
                          of the predicted relation_type (case-insensitive)
              We report both so the reader can see the pessimistic floor and
              the synonym-tolerant ceiling.

P / R / F1 are plain set arithmetic — no learning, no thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.entity_resolver import normalize

# ── core P/R/F1 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "Score") -> "Score":
        return Score(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def score_sets(predicted: set, gold: set) -> Score:
    return Score(
        tp=len(predicted & gold),
        fp=len(predicted - gold),
        fn=len(gold - predicted),
    )


# ── people ──────────────────────────────────────────────────────────────────


def normalise_people(names: list[str]) -> set[str]:
    """De-duplicate a list of names down to their normalised surface form."""
    return {n for n in (normalize(x) for x in names) if n}


# ── edges ───────────────────────────────────────────────────────────────────


def _relation_matches_strict(predicted_type: str, gold_keywords: list[str]) -> bool:
    p = predicted_type.strip().lower()
    return any(p == k.strip().lower() for k in gold_keywords)


def _relation_matches_fuzzy(predicted_type: str, gold_keywords: list[str]) -> bool:
    p = predicted_type.strip().lower()
    return any(k.strip().lower() in p for k in gold_keywords)


def score_edges(
    predicted: list[tuple[str, str, str]],  # (src, tgt, relation_type)
    gold: list[tuple[str, str, list[str]]],  # (src, tgt, type_keywords)
    *,
    fuzzy: bool,
) -> Score:
    """
    Match predicted edges to gold edges.

    A predicted edge (src, tgt, relation_type) matches a gold edge iff
      normalize(src) == normalize(gold.src)
      AND normalize(tgt) == normalize(gold.tgt)
      AND relation_type matches gold.type_keywords (strict or fuzzy).

    Each gold edge can only be matched once (greedy, first-come-first-served).
    """
    matcher = _relation_matches_fuzzy if fuzzy else _relation_matches_strict

    # Normalise predicted edges once.
    norm_pred = [(normalize(s), normalize(t), rt) for (s, t, rt) in predicted]
    norm_pred = [(s, t, rt) for (s, t, rt) in norm_pred if s and t]

    # Index gold edges so we can mark them consumed.
    gold_remaining = [(normalize(s), normalize(t), kws) for (s, t, kws) in gold]

    tp = 0
    for s, t, rt in norm_pred:
        for i, (gs, gt, kws) in enumerate(gold_remaining):
            if s == gs and t == gt and matcher(rt, kws):
                tp += 1
                gold_remaining.pop(i)
                break

    return Score(tp=tp, fp=len(norm_pred) - tp, fn=len(gold_remaining))
