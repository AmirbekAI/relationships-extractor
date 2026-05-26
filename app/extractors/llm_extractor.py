"""
LLM-based extractor — sends article text to an OpenAI model and parses the
structured extraction result.

Design decisions
────────────────
- Uses the *Responses API* (`client.responses.parse`) with a Pydantic response
  schema so the model is constrained to return valid JSON in one inference
  call — no fragile string parsing.
- The model is configurable (default gpt-4o-mini — cheap, sufficient for this
  task).  Swapping to Claude or Gemini requires only a compatible client.
- The article author is explicitly included as a person who "reports on" the
  main subjects; this is a deliberate call documented in the README.
- Body text is truncated to ~8 000 tokens (~32 kB) to stay well within context
  limits for all supported models.
- Relationship types use an open-ended vocabulary: the LLM picks the best verb
  phrase.  A closed taxonomy would improve consistency at the cost of recall;
  for a 2-page prototype the open approach captures more signal.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.core.models import ExtractionResult, ExtractedPerson, ExtractedRelationship
from app.crawlers.base import ArticleContent

logger = logging.getLogger(__name__)

_MAX_BODY_CHARS = 32_000   # ~8 k tokens — safe for gpt-4o / gpt-4o-mini

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class _LLMPerson(BaseModel):
    name: str = Field(description="Full canonical name, e.g. 'Sam Altman'")
    role: Optional[str] = Field(default=None, description="Title or role, e.g. 'CEO of OpenAI'")


class _LLMRelationship(BaseModel):
    source_person: str = Field(description="Full name of the person initiating / doing the action")
    target_person: str = Field(description="Full name of the person affected / receiving the action")
    relation_type: str = Field(description="Short verb phrase, e.g. 'criticizes', 'partners with', 'funds'")
    explanation: str = Field(description="1-2 sentence description of the relationship in this article")
    supporting_quote: str = Field(description="Verbatim sentence or phrase from the article that justifies this relationship")


class _LLMExtractionResult(BaseModel):
    people: list[_LLMPerson] = Field(default_factory=list)
    relationships: list[_LLMRelationship] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert at extracting structured information from news articles.
Your task is to identify all people mentioned and the relationships between them.

Rules:
- Use full canonical names (e.g. "Sam Altman" not "Altman" or "OpenAI's CEO").
- Include the article's author(s) as people with role "journalist".
- For the author, create a relationship: author —reports on→ each main subject
  with relation_type "reports on".
- Only include relationships that are clearly stated or strongly implied by the
  article text — do not invent relationships.
- Relationship types should be concise verb phrases: "criticizes", "partners with",
  "invests in", "sues", "leads", "co-founded", "reports on", etc.
- Each relationship must have a supporting quote from the article.
- De-duplicate: if someone is mentioned by different surface forms, use one
  canonical entry.
"""


class LLMExtractor:
    """Extracts people and relationships from article text using an LLM."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def extract(self, article: ArticleContent) -> ExtractionResult:
        body = article.body_text[:_MAX_BODY_CHARS]
        if not body.strip():
            return ExtractionResult(
                article_url=article.url,
                error="Empty article body",
            )

        user_message = (
            f"Article title: {article.title or 'Unknown'}\n"
            f"Author: {article.author or 'Unknown'}\n"
            f"URL: {article.url}\n\n"
            f"Article body:\n{body}"
        )

        try:
            response = await self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                text_format=_LLMExtractionResult,
            )
            parsed: _LLMExtractionResult = response.output_parsed
        except Exception as exc:
            logger.error("LLM extraction failed for %s: %s", article.url, exc)
            return ExtractionResult(article_url=article.url, error=str(exc))

        people = [
            ExtractedPerson(name=p.name, role=p.role)
            for p in parsed.people
        ]
        relationships = [
            ExtractedRelationship(
                source_person=r.source_person,
                target_person=r.target_person,
                relation_type=r.relation_type,
                explanation=r.explanation,
                supporting_quote=r.supporting_quote,
            )
            for r in parsed.relationships
        ]

        logger.info(
            "Extracted %d people, %d relationships from %s",
            len(people), len(relationships), article.url,
        )
        return ExtractionResult(
            article_url=article.url,
            people=people,
            relationships=relationships,
        )
