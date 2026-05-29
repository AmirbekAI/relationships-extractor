"""
Resolver-quality evaluation.

Spins up a temp SQLite DB, seeds it with the canonical_seed people from
gold/alias_pairs.json (one Person row + one Alias row each), then for every
(surface, expected_canonical, expected_stage) pair:

  • calls resolve_person(surface, repo, extractor)
  • captures which pipeline stage actually fired
    (alias / levenshtein / llm / none) by sniffing the resolver's own
    debug logs — keeps this eval in sync with the resolver without
    modifying it.
  • compares resolved canonical_name + stage to the gold expectation.

Reports overall accuracy and a per-stage confusion summary.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.entity_resolver import normalize, resolve_person
from app.db.repository import GraphRepository
from app.db.session import get_session, init_db
from app.extractors.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)

GOLD_PATH = Path(__file__).parent / "gold" / "alias_pairs.json"


# ── stage detection via log sniffing ─────────────────────────────────────────

class _StageSniffer(logging.Handler):
    """
    Listens on the app.core.entity_resolver logger and infers which stage
    fired from the log line text. resolver.py emits one of:
        "→ alias hit '...'"
        "→ levenshtein hit '...' (sim=...)"
        "→ LLM hit '...'"
        "→ LLM returned no match"
        "→ no candidates, treating as new person"
    """

    STAGE_MARKERS = [
        ("alias hit", "alias"),
        ("levenshtein hit", "levenshtein"),
        ("subname hit", "subname"),
        ("subname recency hit", "subname"),  # ambiguous, disambiguated by recency
        ("subname ambiguous", "none"),       # ambiguous + no recency → refuse
        ("LLM hit", "llm"),
        ("LLM returned no match", "llm"),
        ("LLM fallback disabled", "none"),   # cost mode: skip LLM, treat as new
        ("no candidates", "none"),
    ]

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.last_stage: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for marker, stage in self.STAGE_MARKERS:
            if marker in msg:
                self.last_stage = stage
                return

    def reset(self) -> None:
        self.last_stage = None


# ── eval ─────────────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    surface: str
    expected: Optional[str]
    expected_stage: str
    got: Optional[str]
    got_stage: Optional[str]

    @property
    def name_correct(self) -> bool:
        return self.got == self.expected

    @property
    def stage_correct(self) -> bool:
        return self.got_stage == self.expected_stage


def _load_gold() -> dict[str, Any]:
    with GOLD_PATH.open() as f:
        return json.load(f)


async def _seed_db(gold: dict[str, Any]) -> None:
    async with get_session() as session:
        repo = GraphRepository(session)
        for entry in gold["canonical_seed"]:
            pid = await repo.get_or_create_person(entry["canonical_name"])
            for alias in entry["seed_aliases"]:
                await repo.add_alias(pid, normalize(alias))


async def evaluate_resolver(extractor: LLMExtractor) -> list[PairResult]:
    """
    Returns a list of PairResult, one per gold pair. The caller is
    responsible for setting DATABASE_URL to a *throwaway* DB before
    invoking this — see __main__.py for the temp-sqlite wiring.
    """
    gold = _load_gold()
    await _seed_db(gold)

    # Wire up the stage sniffer on the resolver's logger.
    resolver_logger = logging.getLogger("app.core.entity_resolver")
    old_level = resolver_logger.level
    resolver_logger.setLevel(logging.DEBUG)
    sniffer = _StageSniffer()
    resolver_logger.addHandler(sniffer)

    results: list[PairResult] = []
    try:
        for pair in gold["pairs"]:
            sniffer.reset()
            async with get_session() as session:
                repo = GraphRepository(session)
                resolved = await resolve_person(pair["surface"], repo, extractor)

            got_name = resolved[1] if resolved else None
            results.append(PairResult(
                surface=pair["surface"],
                expected=pair["expected"],
                expected_stage=pair["expected_stage"],
                got=got_name,
                got_stage=sniffer.last_stage,
            ))
    finally:
        resolver_logger.removeHandler(sniffer)
        resolver_logger.setLevel(old_level)

    return results


# ── temp-DB wiring ───────────────────────────────────────────────────────────

class TempEvalDB:
    """
    Async context manager that initialises a throwaway SQLite DB for the
    duration of an eval run and cleans up the file afterwards.

    Used by __main__.py so the production Postgres is never touched.
    """

    def __init__(self) -> None:
        self._path: Optional[str] = None

    async def __aenter__(self) -> str:
        tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tf.close()
        self._path = tf.name
        url = f"sqlite+aiosqlite:///{self._path}"
        await init_db(url)
        return url

    async def __aexit__(self, *_exc: Any) -> None:
        if self._path and os.path.exists(self._path):
            os.unlink(self._path)


# ── pretty printing ──────────────────────────────────────────────────────────

def render_report(results: list[PairResult]) -> str:
    total = len(results)
    name_correct = sum(1 for r in results if r.name_correct)
    stage_correct = sum(1 for r in results if r.stage_correct)

    lines = ["═" * 72, "RESOLVER EVAL", "═" * 72]
    lines.append(
        f"  {'Surface':<22s} → {'Expected':<15s} | {'Got':<15s} "
        f"| {'stage exp':<11s} / {'stage got':<11s} | ok?"
    )
    lines.append("  " + "─" * 82)
    for r in results:
        exp = r.expected if r.expected is not None else "<none>"
        got = r.got if r.got is not None else "<none>"
        ok = "✓" if r.name_correct else "✗"
        lines.append(
            f"  {r.surface:<22s} → {exp:<15s} | {got:<15s} "
            f"| {r.expected_stage:<11s} / {(r.got_stage or '?'):<11s} | {ok}"
        )
    lines.append("")
    lines.append(f"  Name accuracy:  {name_correct}/{total}  "
                 f"({100 * name_correct / total:.1f}%)")
    lines.append(f"  Stage accuracy: {stage_correct}/{total}  "
                 f"({100 * stage_correct / total:.1f}%)")
    lines.append("═" * 72)
    return "\n".join(lines)
