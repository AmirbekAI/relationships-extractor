"""
Application settings — loaded from environment variables (or .env).

Centralises every tunable so the rest of the codebase never touches os.environ.
Fields map 1-1 to the keys in .env.example.
"""

from __future__ import annotations

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
        False,
        description=(
            "If True, the resolver disambiguates ambiguous first/last-name "
            "mentions within an article by preferring the most-recently-"
            "resolved person who shares a contested token. Off by default — "
            "trade-off: better coverage, accepts risk of wrong-merge when the "
            "article text is genuinely ambiguous."
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


def get_settings() -> Settings:
    """Build once, reuse — Settings reads the env on first construction."""
    return Settings()  # type: ignore[call-arg]
