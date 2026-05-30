"""
Extractor-quality evaluation.

For each gold article we:
  1. Build an ArticleContent from the fixture body (no crawl, no HTTP).
  2. Run the real LLMExtractor on it.
  3. Score predicted people + predicted edges against the hand-labelled gold,
     reporting both strict and fuzzy relation-type matches.

Per-article and aggregate P/R/F1 are printed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.entity_resolver import normalize
from app.crawlers.base import ArticleContent
from app.extractors.llm_extractor import LLMExtractor
from evaluation.metrics import (
    Score,
    normalise_people,
    score_edges,
    score_sets,
)

logger = logging.getLogger(__name__)

GOLD_PATH = Path(__file__).parent / "gold" / "articles.json"


@dataclass
class ArticleResult:
    article_id: str
    people: Score
    edges_strict: Score
    edges_fuzzy: Score
    # Populated only when --show-diff is requested.
    diff: "EdgeDiff | None" = None


@dataclass
class EdgeDiff:
    """Per-article record of which gold edges matched vs missed and which
    predicted edges were extras (FPs). Matching uses the FUZZY rule so the
    diff reflects the most lenient pass."""

    matched: list[tuple[str, str, str, str]]  # (src, tgt, gold_kw_repr, predicted_type)
    missed: list[tuple[str, str, list[str]]]  # (src, tgt, type_keywords)
    extra: list[tuple[str, str, str]]  # (src, tgt, relation_type)


def _load_gold() -> list[dict[str, Any]]:
    with GOLD_PATH.open() as f:
        return json.load(f)


def _article_from_gold(g: dict[str, Any]) -> ArticleContent:
    return ArticleContent(
        url=g["url"],
        title=g["title"],
        author=g.get("author"),
        published_at=datetime.now(timezone.utc),
        body_text=g["body_text"],
        source="techcrunch",
    )


def _compute_edge_diff(
    pred_edges: list[tuple[str, str, str]],
    gold_edges: list[tuple[str, str, list[str]]],
) -> EdgeDiff:
    """
    Run the same fuzzy-match walk score_edges does, but record which gold
    edge each predicted edge matched (or that it didn't). Returns the lists
    needed for the human-readable diff.
    """
    norm_pred = [
        (normalize(s), normalize(t), rt, (s, t, rt)) for (s, t, rt) in pred_edges
    ]
    gold_remaining = [
        (normalize(s), normalize(t), kws, (s, t, kws)) for (s, t, kws) in gold_edges
    ]

    matched: list[tuple[str, str, str, str]] = []
    extra: list[tuple[str, str, str]] = []

    for ps, pt, prt, raw_p in norm_pred:
        hit_idx = None
        for i, (gs, gt, kws, _) in enumerate(gold_remaining):
            if ps == gs and pt == gt and any(k.lower() in prt.lower() for k in kws):
                hit_idx = i
                break
        if hit_idx is not None:
            _, _, kws, raw_g = gold_remaining.pop(hit_idx)
            matched.append((raw_g[0], raw_g[1], "|".join(kws), prt))
        else:
            extra.append(raw_p)

    missed = [(raw_g[0], raw_g[1], raw_g[2]) for (_, _, _, raw_g) in gold_remaining]
    return EdgeDiff(matched=matched, missed=missed, extra=extra)


async def evaluate_extractor(
    extractor: LLMExtractor,
    sentences_per_chunk: int = 10,
    *,
    capture_diff: bool = False,
) -> tuple[list[ArticleResult], Score, Score, Score]:
    """
    Returns (per_article_results, total_people, total_edges_strict, total_edges_fuzzy).

    When *capture_diff* is True, each ArticleResult.diff is populated with the
    edge-level diff (matched / missed / extra) for human-readable inspection.
    """
    gold_articles = _load_gold()
    per_article: list[ArticleResult] = []
    total_people = Score(0, 0, 0)
    total_strict = Score(0, 0, 0)
    total_fuzzy = Score(0, 0, 0)

    for g in gold_articles:
        result = await extractor.extract(
            _article_from_gold(g),
            sentences_per_chunk=sentences_per_chunk,
        )
        if result.error:
            logger.warning("Extraction error on %s: %s", g["id"], result.error)

        pred_people = normalise_people([p.name for p in result.people])
        gold_people = normalise_people(g["people"])
        people_score = score_sets(pred_people, gold_people)

        pred_edges = [
            (r.source_person, r.target_person, r.relation_type)
            for r in result.relationships
        ]
        gold_edges = [
            (e["source"], e["target"], e["type_keywords"]) for e in g["relationships"]
        ]
        edges_strict = score_edges(pred_edges, gold_edges, fuzzy=False)
        edges_fuzzy = score_edges(pred_edges, gold_edges, fuzzy=True)

        diff = _compute_edge_diff(pred_edges, gold_edges) if capture_diff else None

        per_article.append(
            ArticleResult(g["id"], people_score, edges_strict, edges_fuzzy, diff)
        )
        total_people += people_score
        total_strict += edges_strict
        total_fuzzy += edges_fuzzy

    return per_article, total_people, total_strict, total_fuzzy


# ── pretty printing ──────────────────────────────────────────────────────────


def _row(label: str, s: Score) -> str:
    return (
        f"  {label:<18s}  P={s.precision:>5.2f}  R={s.recall:>5.2f}  "
        f"F1={s.f1:>5.2f}   (tp={s.tp} fp={s.fp} fn={s.fn})"
    )


def render_report(
    per_article: list[ArticleResult],
    total_people: Score,
    total_strict: Score,
    total_fuzzy: Score,
) -> str:
    lines = ["═" * 72, "EXTRACTOR EVAL", "═" * 72]
    for r in per_article:
        lines.append(f"\n── {r.article_id} ──")
        lines.append(_row("people", r.people))
        lines.append(_row("edges (strict)", r.edges_strict))
        lines.append(_row("edges (fuzzy)", r.edges_fuzzy))
        if r.diff is not None:
            lines.append(_render_diff(r.diff))

    lines.append("\n── AGGREGATE ──")
    lines.append(_row("people", total_people))
    lines.append(_row("edges (strict)", total_strict))
    lines.append(_row("edges (fuzzy)", total_fuzzy))
    lines.append("═" * 72)
    return "\n".join(lines)


def _render_diff(d: EdgeDiff) -> str:
    """One block per article: matched / missed / extra edges, fuzzy-rule."""
    out = ["    EDGE DIFF (fuzzy-match basis):"]

    if d.matched:
        out.append(f"      ✓ matched ({len(d.matched)}):")
        for src, tgt, kws, predicted_type in d.matched:
            out.append(
                f"          {src} → {tgt}  gold=[{kws}]  pred='{predicted_type}'"
            )

    if d.missed:
        out.append(f"      ✗ missed gold ({len(d.missed)}):")
        for src, tgt, kws in d.missed:
            out.append(f"          {src} → {tgt}  expected one of {kws}")

    if d.extra:
        out.append(f"      + extra predicted ({len(d.extra)}):")
        for src, tgt, rt in d.extra:
            out.append(f"          {src} → {tgt}  '{rt}'")

    if not (d.matched or d.missed or d.extra):
        out.append("      (no edges)")

    return "\n".join(out)
