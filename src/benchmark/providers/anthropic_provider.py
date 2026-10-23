"""Anthropic provider adapter (Messages API, ``tool_use`` blocks)."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from ..schema import ToolCall, Usage
from ..utils.env import require_env
from .base import BaseProvider, Conversation, NativeCall, StepOutput, safe_dump


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


def extract_anthropic_native_calls(response: dict[str, Any]) -> list[NativeCall]:
    out: list[NativeCall] = []
    for block in response.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                arguments = block.get("input")
                out.append(NativeCall(
                    id=block.get("id"), name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    raw_arguments=arguments,
                ))
    return out


def extract_anthropic_usage(response: dict[str, Any]) -> Usage:
    """``input_tokens`` excludes cache reads/writes on Anthropic; fold them in
    so ``Usage.input_tokens`` is total prompt size everywhere."""
    usage = response.get("usage") or {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0) + cache_read + cache_write,
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=cache_read,
    )


def replayable_content(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant content blocks trimmed to what the Messages API accepts back
    (SDK dumps carry response-only fields such as ``citations: null``)."""
    blocks: list[dict[str, Any]] = []
    for block in response.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif kind == "tool_use":
            blocks.append({"type": "tool_use", "id": block.get("id"),
                           "name": block.get("name"),
                           "input": block.get("input") or {}})
        elif kind == "thinking":
            blocks.append({"type": "thinking", "thinking": block.get("thinking", ""),
                           "signature": block.get("signature", "")})
        elif kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": block.get("data", "")})
    return blocks


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

    # -- wire format ------------------------------------------------------

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, step: StepOutput) -> dict[str, Any]:
        content = step.native if step.native else [{"type": "text", "text": step.text}]
        return {"role": "assistant", "content": content}

    def tool_result_messages(self, step: StepOutput,
                             results: list[tuple[NativeCall, str]]) -> list[Any]:
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": payload}
            for call, payload in results
        ]}]

    async def step(self, conv: Conversation, tools: list[dict[str, Any]] | None,
                   *, json_mode: bool = False) -> StepOutput:
        # The Messages API has no JSON mode; the router prompt itself demands
        # a single JSON object, exactly as in production.
        base: dict[str, Any] = {
            "model": self.cfg.name,
            "max_tokens": self.settings.max_output_tokens,
            "messages": conv.messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            base["tools"] = convert_tools_to_anthropic(tools)
        params = self.request_params(**base)
        if conv.system:
            params.setdefault("system", conv.system)
        # Some newer Anthropic models reject sampling params entirely; a
        # config can drop them via "api_params": {"temperature": null}.
        if params.get("temperature") is None:
            params.pop("temperature", None)
        response = await self.client.messages.create(**params)
        raw = safe_dump(response)
        raw_dict = raw if isinstance(raw, dict) else {}
        _, text = extract_anthropic_tool_calls(raw_dict)
        return StepOutput(
            raw=raw,
            calls=extract_anthropic_native_calls(raw_dict),
            text=text,
            usage=extract_anthropic_usage(raw_dict),
            native=replayable_content(raw_dict),
        )
