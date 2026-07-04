"""Azure OpenAI provider adapter."""

from __future__ import annotations

from openai import AsyncAzureOpenAI

from ..utils.env import require_env
from .openai_provider import OpenAIChatFamilyProvider


class AzureOpenAIProvider(OpenAIChatFamilyProvider):
    name = "azure_openai"

    def check_credentials(self) -> None:
        super().check_credentials()
        if not self.cfg.endpoint:
            raise ValueError(
                "azure_openai models require 'endpoint' "
                "(https://YOUR_RESOURCE.openai.azure.com)"
            )
        if not (self.cfg.deployment or self.cfg.name):
            raise ValueError("azure_openai models require a 'deployment'")

    def _make_client(self) -> AsyncAzureOpenAI:
        api_key = require_env(self.cfg.api_key_env) if self.cfg.api_key_env else None
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self.cfg.endpoint,
            api_version=self.cfg.api_version or "2025-01-01-preview",
            max_retries=0,
        )

    def _model_identifier(self) -> str:
        # Azure routes by deployment name, not model name.
        return self.cfg.deployment or self.cfg.name
