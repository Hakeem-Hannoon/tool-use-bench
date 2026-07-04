"""Generic OpenAI-compatible provider adapter.

Covers DeepSeek, Groq, Together, Fireworks, Mistral, Cerebras, Perplexity,
xAI, local gateways, and anything else exposing OpenAI-style chat/tool APIs.
Configure via ``endpoint``, ``api_key_env``, the model ``name``, and any
extra ``api_params``.
"""

from __future__ import annotations

from .openai_provider import OpenAIChatFamilyProvider


class OpenAICompatibleProvider(OpenAIChatFamilyProvider):
    name = "openai_compatible"
    requires_endpoint = True
