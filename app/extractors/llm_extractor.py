"""
LLM extractor.

Two responsibilities:
  1. extract(article, sentences_per_chunk)
       Splits article body into sentence chunks, calls the LLM for each,
       merges the results into a single ExtractionResult.

  2. resolve_alias_with_llm(name, candidates)
       Backup alias resolver: used when Levenshtein distance is inconclusive.
       Sends the unknown name + a filtered list of candidate canonical names
       to the LLM and asks which one (if any) it refers to.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from app.core.models import ExtractionResult, ExtractedPerson, ExtractedRelationship
from app.crawlers.base import ArticleContent
from app.extractors.base import BaseLLMClient
from app.extractors.filters import filter_extraction

logger = logging.getLogger(__name__)

# ── sentence splitter ─────────────────────────────────────────────────────────
# Splits on . ! ? followed by whitespace or end-of-string.
# Good enough for news prose; no NLTK dependency needed.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# ── Pydantic schemas for LLM structured output ───────────────────────────────

class _Person(BaseModel):
    name: str = Field(description="Full canonical name, e.g. 'Sam Altman'")
    role: Optional[str] = Field(default=None, description="Title or role, e.g. 'CEO of OpenAI'")


class _Relationship(BaseModel):
    source_person: str = Field(description="Full name of the person doing the action")
    target_person: str = Field(description="Full name of the person receiving the action")
    relation_type: str = Field(description="Short verb phrase: 'criticizes', 'partners with', 'funds', 'leads', …")
    explanation: str = Field(description="1-2 sentence description of the relationship")
    supporting_quote: str = Field(description="Verbatim sentence from the text that justifies this relationship")


class _ChunkResult(BaseModel):
    people: list[_Person] = Field(default_factory=list)
    relationships: list[_Relationship] = Field(default_factory=list)


class _AliasResolution(BaseModel):
    canonical_name: Optional[str] = Field(
        default=None,
        description="The canonical name from the candidates list this name refers to, or null if none match.",
    )


# ── prompts ───────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are an expert at extracting structured information from news article excerpts.
Identify all PEOPLE mentioned and the relationships between them.

PEOPLE rules:
- Only real human individuals. NEVER list a company, organization, product,
  government body, agency, or any non-human entity as a person.
  Examples of what NOT to extract as a person:
    "OpenAI", "Microsoft", "Anthropic", "Google", "Amazon",
    "the board", "the company", "TechCrunch", "the Federal Reserve",
    "ChatGPT", "GPT-4"
- Use full canonical names (e.g. "Sam Altman", not "Altman" or "OpenAI's CEO").
- Include the article author as a person with role "journalist".
- If the same person appears under different surface forms use one canonical entry.

RELATIONSHIP rules:
- For the author add a relationship: author —reports on→ each main subject
  with relation_type "reports on".
- Only include relationships clearly stated or strongly implied — do not invent.
- When the text attributes an action to an organization (e.g. "OpenAI sued Musk",
  "Microsoft invested in OpenAI", "Anthropic partnered with Amazon"):
    * If the article names a human responsible for the org's action ANYWHERE
      in the text (CEO, founder, named actor), you MUST emit the relationship
      between the corresponding humans. Do NOT skip the edge.
      Examples (assume the article also names Altman as OpenAI's CEO and
      Nadella as Microsoft's CEO):
        "OpenAI and Microsoft announced a partnership"
            → Sam Altman —partners with→ Satya Nadella
        "Microsoft invested in OpenAI"
            → Satya Nadella —invests in→ Sam Altman
        "Elon Musk sued OpenAI and CEO Sam Altman"
            → Elon Musk —sues→ Sam Altman
    * Only if NO responsible human is named anywhere in the article should
      you omit the relationship.
- Relation types must be short verb phrases: "criticizes", "partners with",
  "invests in", "sues", "leads", "co-founded", "reports on", etc.
- Each relationship must have a verbatim supporting quote from the text.
- If no people or relationships are present in this excerpt return empty lists.
"""

_ALIAS_SYSTEM = """\
You are a named-entity resolution expert.
You will be given an unknown name and a list of known canonical person names.
Decide which canonical name (if any) the unknown name refers to.
Return null if you are not confident.
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _chunk(sentences: list[str], size: int) -> list[str]:
    """Group sentences into chunks of *size* and join each as a paragraph."""
    return [
        " ".join(sentences[i : i + size])
        for i in range(0, len(sentences), size)
    ]


def _merge(results: list[_ChunkResult]) -> tuple[list[_Person], list[_Relationship]]:
    """
    Deduplicate people and relationships across chunks.
    People are keyed by normalised name; relationships by (src, tgt, type).
    """
    people: dict[str, _Person] = {}
    rels: dict[tuple[str, str, str], _Relationship] = {}

    for r in results:
        for p in r.people:
            key = p.name.strip().lower()
            if key not in people:
                people[key] = p

        for rel in r.relationships:
            key = (
                rel.source_person.strip().lower(),
                rel.target_person.strip().lower(),
                rel.relation_type.strip().lower(),
            )
            if key not in rels:
                rels[key] = rel

    return list(people.values()), list(rels.values())


# ── extractor ─────────────────────────────────────────────────────────────────

class LLMExtractor:
    def __init__(self, client: BaseLLMClient, default_sentences_per_chunk: int = 10) -> None:
        self._client = client
        self._default_chunk_size = default_sentences_per_chunk

    async def extract(
        self,
        article: ArticleContent,
        sentences_per_chunk: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Split the article body into sentence chunks, run extraction on each,
        and merge into one ExtractionResult.

        Args:
            article:              parsed article content from a crawler.
            sentences_per_chunk:  overrides the instance default when provided.
        """
        chunk_size = sentences_per_chunk or self._default_chunk_size

        if not article.body_text.strip():
            return ExtractionResult(article_url=article.url, error="Empty article body")

        sentences = _split_sentences(article.body_text)
        chunks = _chunk(sentences, chunk_size)

        logger.info(
            "Extracting from '%s' — %d sentence(s) in %d chunk(s) of %d",
            article.url, len(sentences), len(chunks), chunk_size,
        )

        chunk_results: list[_ChunkResult] = []

        for idx, chunk_text in enumerate(chunks):
            user_msg = (
                f"Article title: {article.title or 'Unknown'}\n"
                f"Author: {article.author or 'Unknown'}\n"
                f"URL: {article.url}\n\n"
                f"Excerpt ({idx + 1}/{len(chunks)}):\n{chunk_text}"
            )
            try:
                result = await self._client.structured_complete(
                    system_prompt=_EXTRACTION_SYSTEM,
                    user_message=user_msg,
                    response_schema=_ChunkResult,
                )
                chunk_results.append(result)
                logger.debug(
                    "Chunk %d/%d — %d people, %d rels",
                    idx + 1, len(chunks), len(result.people), len(result.relationships),
                )
            except Exception as exc:
                logger.error("Extraction failed on chunk %d of %s: %s", idx + 1, article.url, exc)
                # Continue — partial results are better than nothing

        if not chunk_results:
            return ExtractionResult(article_url=article.url, error="All chunks failed extraction")

        people, rels = _merge(chunk_results)

        # Belt-and-suspenders: even with the tightened prompt, the LLM can
        # still slip an org into the people list. Drop them and any edge
        # they touch before they reach the graph.
        people, rels, _ = filter_extraction(people, rels)

        logger.info(
            "Merged: %d people, %d relationships from %s",
            len(people), len(rels), article.url,
        )

        return ExtractionResult(
            article_url=article.url,
            people=[ExtractedPerson(name=p.name, role=p.role) for p in people],
            relationships=[
                ExtractedRelationship(
                    source_person=r.source_person,
                    target_person=r.target_person,
                    relation_type=r.relation_type,
                    explanation=r.explanation,
                    supporting_quote=r.supporting_quote,
                )
                for r in rels
            ],
        )

    async def resolve_alias_with_llm(
        self,
        unknown_name: str,
        candidates: list[str],
    ) -> Optional[str]:
        """
        Ask the LLM which canonical name (if any) *unknown_name* refers to.

        This is the last-resort fallback in the entity resolution pipeline,
        called only when both the alias table lookup and Levenshtein distance
        are inconclusive.

        Args:
            unknown_name:  the raw surface form seen in an article.
            candidates:    pre-filtered list of canonical names that are
                           somewhat similar (passed in by the resolver so we
                           don't burn tokens on obviously unrelated names).

        Returns:
            A canonical name from *candidates*, or None if no confident match.
        """
        if not candidates:
            return None

        candidate_list = "\n".join(f"- {c}" for c in candidates)
        user_msg = (
            f"Unknown name: \"{unknown_name}\"\n\n"
            f"Candidate canonical names:\n{candidate_list}\n\n"
            f"Which canonical name does \"{unknown_name}\" refer to? "
            f"Reply null if none are a confident match."
        )

        try:
            result = await self._client.structured_complete(
                system_prompt=_ALIAS_SYSTEM,
                user_message=user_msg,
                response_schema=_AliasResolution,
            )
            matched = result.canonical_name
            if matched and matched in candidates:
                logger.info("LLM resolved '%s' → '%s'", unknown_name, matched)
                return matched
            logger.info("LLM could not resolve '%s'", unknown_name)
            return None
        except Exception as exc:
            logger.error("LLM alias resolution failed for '%s': %s", unknown_name, exc)
            return None
