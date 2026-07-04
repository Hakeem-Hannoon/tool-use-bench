"""Anthropic provider adapter (Messages API, ``tool_use`` blocks)."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from ..schema import ToolCall
from ..utils.env import require_env
from .base import BaseProvider, safe_dump


def convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style tool defs -> Anthropic ``tools`` entries."""
    converted = []
    for tool in tools:
        fn = tool.get("function", {})
        converted.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object"}),
        })
    return converted


def extract_anthropic_tool_calls(response: dict[str, Any]) -> tuple[list[ToolCall], str]:
    """Extract ``tool_use`` blocks + text from an Anthropic Messages response
    dict. JSON printed inside text blocks is NOT treated as a tool call."""
    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in response.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                arguments = block.get("input")
                calls.append(ToolCall(
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                ))
        elif block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return calls, "\n".join(p for p in text_parts if p)


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, cfg, settings):
        super().__init__(cfg, settings)
        self._client: AsyncAnthropic | None = None

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            api_key = require_env(self.cfg.api_key_env) \
                if self.cfg.api_key_env else None
            self._client = AsyncAnthropic(
                api_key=api_key,
                base_url=self.cfg.endpoint or None,
                max_retries=0,
            )
        return self._client

    async def _call(self, prompt: str, system_prompt: str,
                    tools: list[dict[str, Any]]):
        params = self.request_params(
            model=self.cfg.name,
            max_tokens=self.settings.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            tools=convert_tools_to_anthropic(tools),
            temperature=self.settings.temperature,
        )
        if system_prompt:
            params.setdefault("system", system_prompt)
        # Some newer Anthropic models reject sampling params entirely; a
        # config can drop them via "api_params": {"temperature": null}.
        if params.get("temperature") is None:
            params.pop("temperature", None)
        response = await self.client.messages.create(**params)
        raw = safe_dump(response)
        calls, text = extract_anthropic_tool_calls(
            raw if isinstance(raw, dict) else {})
        return raw, calls, text
