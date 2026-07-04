"""OpenAI provider adapter (Chat Completions), plus the shared extraction
logic reused by Azure OpenAI, OpenRouter, and OpenAI-compatible endpoints."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from ..schema import ToolCall
from ..utils.env import require_env
from .base import BaseProvider, normalize_arguments, safe_dump


def extract_openai_tool_calls(response: dict[str, Any]) -> tuple[list[ToolCall], str]:
    """Extract native tool calls + text from an OpenAI-style response dict.

    Handles both Chat Completions (``choices[].message.tool_calls``) and
    Responses-API-style payloads (``output[].type == "function_call"``).
    Text-only JSON in ``content`` is deliberately NOT treated as a tool call.
    """
    calls: list[ToolCall] = []
    text_parts: list[str] = []

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        for tc in message.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            if tc.get("type") not in (None, "function"):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                calls.append(ToolCall(
                    name=name,
                    arguments=normalize_arguments(fn.get("arguments")),
                ))
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):  # multimodal-style content parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

    output = response.get("output")
    if isinstance(output, list):  # Responses API shape
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                name = item.get("name")
                if isinstance(name, str) and name:
                    calls.append(ToolCall(
                        name=name,
                        arguments=normalize_arguments(item.get("arguments")),
                    ))
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])

    return calls, "\n".join(p for p in text_parts if p)


class OpenAIChatFamilyProvider(BaseProvider):
    """Shared Chat Completions implementation for OpenAI-compatible APIs."""

    name = "openai"
    default_base_url: str | None = None
    requires_endpoint = False

    def __init__(self, cfg, settings):
        super().__init__(cfg, settings)
        self._client: AsyncOpenAI | None = None

    def check_credentials(self) -> None:
        super().check_credentials()
        if self.requires_endpoint and not (self.cfg.endpoint or self.default_base_url):
            raise ValueError(
                f"provider {self.cfg.provider!r} requires an 'endpoint' in "
                f"the model config"
            )

    def _make_client(self) -> AsyncOpenAI:
        api_key = require_env(self.cfg.api_key_env) if self.cfg.api_key_env else None
        base_url = self.cfg.endpoint or self.default_base_url
        return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _model_identifier(self) -> str:
        return self.cfg.name

    async def _call(self, prompt: str, system_prompt: str,
                    tools: list[dict[str, Any]]):
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        params = self.request_params(
            model=self._model_identifier(),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.settings.temperature,
        )
        if params.get("temperature") is None:
            params.pop("temperature", None)
        response = await self.client.chat.completions.create(**params)
        raw = safe_dump(response)
        calls, text = extract_openai_tool_calls(
            raw if isinstance(raw, dict) else {})
        return raw, calls, text
