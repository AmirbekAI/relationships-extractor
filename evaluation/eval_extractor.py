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
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _load_gold() -> list[dict[str, Any]]:
    with GOLD_PATH.open() as f:
        return json.load(f)


def _article_from_gold(g: dict[str, Any]) -> ArticleContent:
    return ArticleContent(
        url=g["url"],
        title=g["title"],
        author=g.get("author"),
        published_at=datetime.utcnow(),
        body_text=g["body_text"],
        source="techcrunch",
    )


async def evaluate_extractor(
    extractor: LLMExtractor,
    sentences_per_chunk: int = 10,
) -> tuple[list[ArticleResult], Score, Score, Score]:
    """
    Returns (per_article_results, total_people, total_edges_strict, total_edges_fuzzy).
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
            (e["source"], e["target"], e["type_keywords"])
            for e in g["relationships"]
        ]
        edges_strict = score_edges(pred_edges, gold_edges, fuzzy=False)
        edges_fuzzy = score_edges(pred_edges, gold_edges, fuzzy=True)

        per_article.append(
            ArticleResult(g["id"], people_score, edges_strict, edges_fuzzy)
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

    lines.append("\n── AGGREGATE ──")
    lines.append(_row("people", total_people))
    lines.append(_row("edges (strict)", total_strict))
    lines.append(_row("edges (fuzzy)", total_fuzzy))
    lines.append("═" * 72)
    return "\n".join(lines)
