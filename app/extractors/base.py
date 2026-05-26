"""
Abstract LLM client interface.

Decouples the extractor logic from any specific provider. To add a new
backend (Anthropic, Gemini, …) implement BaseLLMClient and pass it in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """
    Minimal interface the extractor depends on.

    Implementations must support structured output — i.e. constrain the model
    to return valid JSON that deserialises into a given Pydantic schema.
    """

    @abstractmethod
    async def structured_complete(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Type[T],
    ) -> T:
        """
        Call the model and return a parsed instance of *response_schema*.
        Raises on failure — callers are responsible for catching.
        """
        ...
