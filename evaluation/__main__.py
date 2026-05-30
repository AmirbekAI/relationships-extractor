"""
Evaluation CLI.

Run with:
    .venv/bin/python -m evaluation

Or to run just one suite:
    .venv/bin/python -m evaluation --only extractor
    .venv/bin/python -m evaluation --only resolver

Reads OPENAI_API_KEY from .env (or the env). Uses an ephemeral SQLite DB
for the resolver eval — your real DATABASE_URL is never touched.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.config import get_settings
from app.extractors.llm_extractor import LLMExtractor
from app.extractors.openai_client import OpenAIClient
from evaluation.eval_extractor import (
    evaluate_extractor,
)
from evaluation.eval_extractor import render_report as render_extractor_report
from evaluation.eval_resolver import (
    TempEvalDB,
    evaluate_resolver,
)
from evaluation.eval_resolver import render_report as render_resolver_report

load_dotenv(Path(__file__).parent.parent / ".env")


def _build_extractor() -> LLMExtractor:
    s = get_settings()
    client = OpenAIClient(api_key=s.openai_api_key, model=s.openai_model)
    return LLMExtractor(
        client=client, default_sentences_per_chunk=s.sentences_per_chunk
    )


async def _run(only: str | None, show_diff: bool) -> int:
    extractor = _build_extractor()

    if only in (None, "extractor"):
        per_article, p, es, ef = await evaluate_extractor(
            extractor,
            capture_diff=show_diff,
        )
        print(render_extractor_report(per_article, p, es, ef))

    if only in (None, "resolver"):
        # Force the resolver eval onto a throwaway SQLite DB so we never
        # write seed people into production Postgres.
        async with TempEvalDB():
            results = await evaluate_resolver(extractor)
        print(render_resolver_report(results))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("extractor", "resolver"),
        default=None,
        help="Run just one of the two eval suites (default: both).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print extractor + resolver debug logs as the eval runs.",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Per-article: print matched / missed / extra edges so you can "
        "see exactly what the model got right and wrong.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    return asyncio.run(_run(args.only, args.show_diff))


if __name__ == "__main__":
    sys.exit(main())
