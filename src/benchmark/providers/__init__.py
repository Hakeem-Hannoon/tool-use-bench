"""Provider adapter registry.

SDK-backed adapters are imported lazily so that importing the package (e.g.
for scoring tests) never requires every provider SDK to be importable.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ..config import ModelConfig
from .base import BaseProvider, ProviderSettings

if TYPE_CHECKING:  # pragma: no cover
    pass

_PROVIDER_MODULES: dict[str, tuple[str, str]] = {
    "openai": ("benchmark.providers.openai_provider", "OpenAIChatFamilyProvider"),
    "azure_openai": ("benchmark.providers.azure_openai_provider", "AzureOpenAIProvider"),
    "anthropic": ("benchmark.providers.anthropic_provider", "AnthropicProvider"),
    "gemini": ("benchmark.providers.gemini_provider", "GeminiProvider"),
    "vertexai": ("benchmark.providers.vertexai_provider", "VertexAIProvider"),
    "openrouter": ("benchmark.providers.openrouter_provider", "OpenRouterProvider"),
    "openai_compatible": (
        "benchmark.providers.openai_compatible_provider",
        "OpenAICompatibleProvider",
    ),
}


def create_provider(cfg: ModelConfig, settings: ProviderSettings) -> BaseProvider:
    try:
        module_name, class_name = _PROVIDER_MODULES[cfg.provider]
    except KeyError:  # config validation should prevent this
        raise ValueError(f"unsupported provider: {cfg.provider!r}") from None
    module = import_module(module_name)
    provider_cls: type[BaseProvider] = getattr(module, class_name)
    return provider_cls(cfg, settings)


__all__ = ["BaseProvider", "ProviderSettings", "create_provider"]
