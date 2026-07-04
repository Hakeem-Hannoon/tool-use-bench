"""Gemini API provider adapter (``google-genai`` SDK, ``function_call`` parts)."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types as genai_types

from ..schema import ToolCall
from ..utils.env import require_env
from .base import BaseProvider, safe_dump


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

    async def _call(self, prompt: str, system_prompt: str,
                    tools: list[dict[str, Any]]):
        config_kwargs: dict[str, Any] = self.request_params(
            temperature=self.settings.temperature,
            tools=[genai_types.Tool(
                function_declarations=convert_tools_to_gemini(tools))],
        )
        if system_prompt:
            config_kwargs.setdefault("system_instruction", system_prompt)
        if config_kwargs.get("temperature") is None:
            config_kwargs.pop("temperature", None)
        response = await self.client.aio.models.generate_content(
            model=self.cfg.name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        raw = safe_dump(response)
        calls, text = extract_gemini_tool_calls(
            raw if isinstance(raw, dict) else {})
        return raw, calls, text
