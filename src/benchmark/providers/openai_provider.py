"""OpenAI provider adapter (Chat Completions), plus the shared extraction
logic reused by Azure OpenAI, OpenRouter, and OpenAI-compatible endpoints."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from ..schema import ToolCall, Usage
from ..utils.env import require_env
from .base import (
    BaseProvider, Conversation, NativeCall, StepOutput, normalize_arguments,
    safe_dump,
)


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


def extract_openai_native_calls(response: dict[str, Any]) -> list[NativeCall]:
    """Chat Completions tool calls with their ids (synthesized when absent)."""
    out: list[NativeCall] = []
    choices = response.get("choices")
    if not (isinstance(choices, list) and choices):
        return out
    message = choices[0].get("message") or {}
    for i, tc in enumerate(message.get("tool_calls") or []):
        if not isinstance(tc, dict) or tc.get("type") not in (None, "function"):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        if isinstance(name, str) and name:
            out.append(NativeCall(
                id=tc.get("id") or f"call_{i}",
                name=name,
                arguments=normalize_arguments(fn.get("arguments")),
                raw_arguments=fn.get("arguments"),
            ))
    return out


def extract_openai_usage(response: dict[str, Any]) -> Usage:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cached_input_tokens=int(details.get("cached_tokens") or 0),
    )


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
        if self.requires_endpoint and not (self.endpoint() or self.default_base_url):
            raise ValueError(
                f"provider {self.cfg.provider!r} requires an 'endpoint' (or "
                f"'endpoint_env') in the model config"
            )

    def _make_client(self) -> AsyncOpenAI:
        api_key = require_env(self.cfg.api_key_env) if self.cfg.api_key_env else None
        base_url = self.endpoint() or self.default_base_url
        return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _model_identifier(self) -> str:
        return self.cfg.name

    # -- wire format ------------------------------------------------------

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, step: StepOutput) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": step.text or None}
        if step.calls:
            message["tool_calls"] = [{
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.raw_arguments if isinstance(c.raw_arguments, str)
                    else json.dumps(c.arguments, ensure_ascii=False),
                },
            } for c in step.calls]
        elif message["content"] is None:
            message["content"] = ""
        return message

    def tool_result_messages(self, step: StepOutput,
                             results: list[tuple[NativeCall, str]]) -> list[Any]:
        return [{"role": "tool", "tool_call_id": call.id, "content": payload}
                for call, payload in results]

    async def step(self, conv: Conversation, tools: list[dict[str, Any]] | None,
                   *, json_mode: bool = False) -> StepOutput:
        messages: list[dict[str, Any]] = []
        if conv.system:
            messages.append({"role": "system", "content": conv.system})
        messages.extend(conv.messages)
        base: dict[str, Any] = {
            "model": self._model_identifier(),
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            base["tools"] = tools
            base["tool_choice"] = "auto"
        if json_mode:
            base["response_format"] = {"type": "json_object"}
        params = self.request_params(**base)
        if params.get("temperature") is None:
            params.pop("temperature", None)
        response = await self.client.chat.completions.create(**params)
        raw = safe_dump(response)
        raw_dict = raw if isinstance(raw, dict) else {}
        _, text = extract_openai_tool_calls(raw_dict)
        return StepOutput(
            raw=raw,
            calls=extract_openai_native_calls(raw_dict),
            text=text,
            usage=extract_openai_usage(raw_dict),
        )
