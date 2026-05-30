"""
Local model client — talks to any OpenAI-compatible HTTP endpoint.

Works with Ollama, LM Studio, llama.cpp server, vLLM, etc.

Structured output strategy
───────────────────────────
Local models don't all support the Responses API `text_format` parameter.
Instead we:
  1. Append the JSON schema to the system prompt.
  2. Ask the model to reply with ONLY a JSON object.
  3. Parse and validate the reply with Pydantic.

If the model returns text outside the JSON we strip it with a lenient regex
before parsing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.extractors.base import BaseLLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LocalModelClient(BaseLLMClient):
    """
    Client for a locally-running OpenAI-compatible server.

    Args:
        base_url:  e.g. "http://localhost:11434/v1" for Ollama
        model:     model tag as the server knows it, e.g. "llama3:8b"
        api_key:   most local servers accept any non-empty string
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def structured_complete(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Type[T],
    ) -> T:
        schema_str = json.dumps(response_schema.model_json_schema(), indent=2)

        augmented_system = (
            f"{system_prompt}\n\n"
            f"Reply with ONLY a valid JSON object that matches this schema "
            f"(no markdown, no explanation):\n{schema_str}"
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content or ""

        # Strip any text outside the JSON object
        match = _JSON_BLOCK_RE.search(raw)
        if not match:
            raise ValueError(f"Model did not return a JSON object. Raw output:\n{raw}")

        return response_schema.model_validate_json(match.group())
