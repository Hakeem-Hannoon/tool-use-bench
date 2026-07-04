"""OpenRouter provider adapter (OpenAI-compatible API surface)."""

from __future__ import annotations

from .openai_provider import OpenAIChatFamilyProvider


class OpenRouterProvider(OpenAIChatFamilyProvider):
    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
