"""
OpenAI API client — uses the Responses API with structured output
(response_format / text_format) so the model is constrained to return
valid JSON matching the Pydantic schema in one call.
"""

from __future__ import annotations

from typing import Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.extractors.base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def structured_complete(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Type[T],
    ) -> T:
        response = await self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            text_format=response_schema,
        )
        return response.output_parsed
