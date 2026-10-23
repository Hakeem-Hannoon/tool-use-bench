"""Gemini API provider adapter (``google-genai`` SDK, ``function_call`` parts)."""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types as genai_types

from ..schema import ToolCall, Usage
from ..utils.env import require_env
from .base import BaseProvider, Conversation, NativeCall, StepOutput, safe_dump


def _strip_unsupported_schema_keys(schema: Any) -> Any:
    """Gemini function declarations reject some JSON Schema keywords
    (notably ``additionalProperties``); strip them recursively."""
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported_schema_keys(v)
            for k, v in schema.items()
            if k not in ("additionalProperties", "$schema")
        }
    if isinstance(schema, list):
        return [_strip_unsupported_schema_keys(v) for v in schema]
    return schema


def convert_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style tool defs -> Gemini ``function_declarations``."""
    declarations = []
    for tool in tools:
        fn = tool.get("function", {})
        declarations.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": _strip_unsupported_schema_keys(
                fn.get("parameters", {"type": "object"})),
        })
    return declarations


def extract_gemini_tool_calls(response: dict[str, Any]) -> tuple[list[ToolCall], str]:
    """Extract ``function_call`` parts + text from a Gemini/Vertex response
    dict (``generate_content`` shape)."""
    calls: list[ToolCall] = []
    text_parts: list[str] = []
    candidates = response.get("candidates") or []
    if candidates:
        content = candidates[0].get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            fc = part.get("function_call") or part.get("functionCall")
            if isinstance(fc, dict):
                name = fc.get("name")
                if isinstance(name, str) and name:
                    args = fc.get("args", fc.get("arguments"))
                    calls.append(ToolCall(
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                    ))
            elif isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    return calls, "\n".join(p for p in text_parts if p)


def extract_gemini_usage(response: dict[str, Any]) -> Usage:
    """Thinking tokens are billed as output on Gemini, so they count there."""
    meta = response.get("usage_metadata") or response.get("usageMetadata") or {}

    def num(*keys: str) -> int:
        for key in keys:
            if meta.get(key) is not None:
                return int(meta[key])
        return 0

    return Usage(
        input_tokens=num("prompt_token_count", "promptTokenCount"),
        output_tokens=num("candidates_token_count", "candidatesTokenCount")
        + num("thoughts_token_count", "thoughtsTokenCount"),
        cached_input_tokens=num("cached_content_token_count",
                                "cachedContentTokenCount"),
    )


def _function_response_payload(payload: str) -> dict[str, Any]:
    """FunctionResponse.response must be an object; wrap anything else."""
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        value = payload
    return value if isinstance(value, dict) else {"result": value}


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, cfg, settings):
        super().__init__(cfg, settings)
        self._client: genai.Client | None = None

    def _make_client(self) -> genai.Client:
        api_key = require_env(self.cfg.api_key_env) \
            if self.cfg.api_key_env else None
        return genai.Client(api_key=api_key)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    # -- wire format ------------------------------------------------------

    def user_message(self, text: str) -> Any:
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def assistant_message(self, step: StepOutput) -> Any:
        # Replay the SDK's own Content so thought signatures survive.
        if step.native is not None:
            return step.native
        return genai_types.Content(role="model",
                                   parts=[genai_types.Part(text=step.text)])

    def tool_result_messages(self, step: StepOutput,
                             results: list[tuple[NativeCall, str]]) -> list[Any]:
        return [genai_types.Content(role="user", parts=[
            genai_types.Part.from_function_response(
                name=call.name, response=_function_response_payload(payload))
            for call, payload in results
        ])]

    async def step(self, conv: Conversation, tools: list[dict[str, Any]] | None,
                   *, json_mode: bool = False) -> StepOutput:
        base: dict[str, Any] = {"temperature": self.settings.temperature}
        if tools:
            base["tools"] = [genai_types.Tool(
                function_declarations=convert_tools_to_gemini(tools))]
        if json_mode:
            base["response_mime_type"] = "application/json"
        config_kwargs = self.request_params(**base)
        if conv.system:
            config_kwargs.setdefault("system_instruction", conv.system)
        if config_kwargs.get("temperature") is None:
            config_kwargs.pop("temperature", None)
        response = await self.client.aio.models.generate_content(
            model=self.cfg.name,
            contents=conv.messages,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        raw = safe_dump(response)
        raw_dict = raw if isinstance(raw, dict) else {}
        calls, text = extract_gemini_tool_calls(raw_dict)
        native_calls = [NativeCall(id=None, name=c.name, arguments=c.arguments)
                        for c in calls]
        native = None
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None) is not None:
            native = candidates[0].content
        return StepOutput(raw=raw, calls=native_calls, text=text,
                          usage=extract_gemini_usage(raw_dict), native=native)
