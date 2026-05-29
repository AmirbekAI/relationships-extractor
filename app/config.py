"""
Application settings — loaded from environment variables (or .env).

Centralises every tunable so the rest of the codebase never touches os.environ.
Fields map 1-1 to the keys in .env.example.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key (required).")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model identifier.")

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/relationship_finder",
        description="SQLAlchemy async URL. Defaults to local Postgres via asyncpg.",
    )

    # ── Crawling defaults ──────────────────────────────────────────────────
    default_pages: int = Field(2, ge=1, le=50)
    request_delay: float = Field(1.5, ge=0)
    sentences_per_chunk: int = Field(5, ge=1)

    # ── Resolver ───────────────────────────────────────────────────────────
    resolver_recency_enabled: bool = Field(
        True,
        description=(
            "If True, the resolver disambiguates ambiguous first/last-name "
            "mentions within an article by preferring the most-recently-"
            "resolved person who shares a contested token. On by default — "
            "trade-off: better coverage of the introduce-then-shorten "
            "journalism pattern, accepts the risk of a wrong merge when the "
            "article text is genuinely ambiguous. Set to False to revert to "
            "strict refuse-on-ambiguity behaviour."
        ),
    )
    resolver_llm_fallback_enabled: bool = Field(
        True,
        description=(
            "If True, the resolver asks the LLM to pick the canonical name "
            "when the alias / Levenshtein / sub-name rules can't decide. "
            "On by default — accurate but each call costs a token round-trip. "
            "Set to False for cost-sensitive deployments where some missed "
            "merges (duplicate Person rows) are acceptable: the resolver "
            "skips the LLM and treats the surface form as a new person."
        ),
    )

    # ── API ────────────────────────────────────────────────────────────────
    default_page_size: int = Field(20, ge=1, le=200)
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build once, reuse — Settings reads the env on first construction."""
    return Settings()  # type: ignore[call-arg]
