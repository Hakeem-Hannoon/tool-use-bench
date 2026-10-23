"""Provider response normalization — tested on recorded response shapes,
never against live SDKs."""

import pytest

from fakes import ScriptedProvider, model_cfg
from fakes import native as fake_native, text as fake_text

from benchmark.providers.anthropic_provider import (
    convert_tools_to_anthropic, extract_anthropic_tool_calls,
)
from benchmark.providers.base import (
    ProviderSettings, parse_tool_calls_from_text,
)
from benchmark.providers.gemini_provider import (
    convert_tools_to_gemini, extract_gemini_tool_calls,
)
from benchmark.providers.openai_provider import extract_openai_tool_calls
from benchmark.samples import build_all_samples
from benchmark.schema import Sample

WEATHER_ARGS = {"location": "Toronto, Canada", "unit": "celsius"}


# ---------------------------------------------------------------------------
# OpenAI-style (also covers Azure OpenAI, OpenRouter, OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_openai_chat_completions_tool_calls():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Toronto, Canada", '
                                     '"unit": "celsius"}',
                    },
                }],
            },
        }],
    }
    calls, text = extract_openai_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == WEATHER_ARGS
    assert text == ""


def test_openai_parallel_tool_calls_preserve_order():
    response = {
        "choices": [{
            "message": {
                "tool_calls": [
                    {"type": "function",
                     "function": {"name": "a", "arguments": '{"i": 1}'}},
                    {"type": "function",
                     "function": {"name": "b", "arguments": '{"i": 2}'}},
                ],
            },
        }],
    }
    calls, _ = extract_openai_tool_calls(response)
    assert [c.name for c in calls] == ["a", "b"]


def test_openai_responses_api_function_call_items():
    response = {
        "output": [
            {"type": "function_call", "name": "get_weather",
             "arguments": '{"location": "Toronto, Canada", "unit": "celsius"}'},
            {"type": "message",
             "content": [{"type": "output_text", "text": "done"}]},
        ],
    }
    calls, text = extract_openai_tool_calls(response)
    assert calls[0].arguments == WEATHER_ARGS
    assert text == "done"


def test_openai_json_in_text_is_not_a_tool_call():
    response = {
        "choices": [{
            "message": {
                "content": '{"name": "get_weather", "arguments": '
                           '{"location": "Toronto, Canada"}}',
                "tool_calls": None,
            },
        }],
    }
    calls, text = extract_openai_tool_calls(response)
    assert calls == []
    assert "get_weather" in text


def test_openai_unparseable_arguments_become_sentinel():
    response = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": "{not json"},
                }],
            },
        }],
    }
    calls, _ = extract_openai_tool_calls(response)
    assert calls[0].arguments == {"_unparseable_arguments": "{not json"}


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_tool_use_blocks():
    response = {
        "content": [
            {"type": "text", "text": "Checking the weather."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": WEATHER_ARGS},
        ],
        "stop_reason": "tool_use",
    }
    calls, text = extract_anthropic_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == WEATHER_ARGS
    assert text == "Checking the weather."


def test_anthropic_text_only_yields_no_calls():
    response = {"content": [{"type": "text",
                             "text": '{"name": "get_weather"}'}]}
    calls, text = extract_anthropic_tool_calls(response)
    assert calls == []
    assert text


def test_anthropic_tool_conversion_uses_input_schema():
    sample_tools = [{
        "type": "function",
        "function": {"name": "get_weather", "description": "d",
                     "parameters": {"type": "object", "properties": {}}},
    }]
    converted = convert_tools_to_anthropic(sample_tools)
    assert converted == [{
        "name": "get_weather", "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }]


# ---------------------------------------------------------------------------
# Gemini / Vertex AI
# ---------------------------------------------------------------------------


def test_gemini_function_call_parts():
    response = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Sure."},
                    {"function_call": {"name": "get_weather",
                                       "args": WEATHER_ARGS}},
                ],
                "role": "model",
            },
        }],
    }
    calls, text = extract_gemini_tool_calls(response)
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == WEATHER_ARGS
    assert text == "Sure."


def test_gemini_camelcase_function_call_variant():
    response = {
        "candidates": [{
            "content": {"parts": [
                {"functionCall": {"name": "get_weather",
                                  "args": WEATHER_ARGS}},
            ]},
        }],
    }
    calls, _ = extract_gemini_tool_calls(response)
    assert calls[0].name == "get_weather"


def test_gemini_tool_conversion_strips_additional_properties():
    sample_tools = [{
        "type": "function",
        "function": {
            "name": "t", "description": "d",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        },
    }]
    (decl,) = convert_tools_to_gemini(sample_tools)
    assert "additionalProperties" not in decl["parameters"]
    assert decl["parameters"]["required"] == ["x"]


# ---------------------------------------------------------------------------
# Strict-by-default text fallback
# ---------------------------------------------------------------------------


def test_parse_tool_calls_from_text_fenced_json():
    text = ('Here you go:\n```json\n{"name": "get_weather", "arguments": '
            '{"location": "Toronto, Canada", "unit": "celsius"}}\n```')
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].arguments == WEATHER_ARGS


def test_parse_tool_calls_from_text_handles_lists_and_wrappers():
    text = '{"tool_calls": [{"name": "a", "arguments": {}}, ' \
           '{"name": "b", "arguments": {"x": 1}}]}'
    calls = parse_tool_calls_from_text(text)
    assert [c.name for c in calls] == ["a", "b"]


def test_parse_tool_calls_from_plain_prose_is_empty():
    assert parse_tool_calls_from_text("I would call get_weather here.") == []


TEXT_CALL = ('```json\n{"name": "get_weather", "arguments": '
             '{"location": "Toronto, Canada", "unit": "celsius"}}\n```')


def _weather_sample() -> Sample:
    raw = next(s for s in build_all_samples()
               if s["id"] == "0001_weather_current")
    return Sample.model_validate(raw)


async def test_text_tool_call_fails_by_default():
    provider = ScriptedProvider(model_cfg(), ProviderSettings(
        allow_text_tool_call_fallback=False), [fake_text(TEXT_CALL)])
    result = await provider.complete(_weather_sample(), 0, "")
    assert result.tool_calls == []  # strict: text JSON is not a tool call


async def test_text_fallback_only_when_enabled():
    provider = ScriptedProvider(model_cfg(), ProviderSettings(
        allow_text_tool_call_fallback=True), [fake_text(TEXT_CALL)])
    result = await provider.complete(_weather_sample(), 0, "")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == WEATHER_ARGS


async def test_native_calls_win_over_fallback_parsing():
    settings = ProviderSettings(allow_text_tool_call_fallback=True)
    provider = ScriptedProvider(model_cfg(), settings, [
        fake_native(("native", {}), text='{"name": "text_call", "arguments": {}}'),
    ])
    result = await provider.complete(_weather_sample(), 0, "")
    assert [c.name for c in result.tool_calls] == ["native"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


def test_gemini_strips_zod_keywords_but_keeps_property_names():
    from benchmark.providers.gemini_provider import _strip_unsupported_schema_keys
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "id": {"type": "string", "format": "uuid"},
        "when": {"type": "string", "format": "date-time"},
        "n": {"type": "integer", "exclusiveMinimum": 0, "default": 1},
        "default": {"type": "string"}}}
    out = _strip_unsupported_schema_keys(schema)
    assert out == {"type": "object", "properties": {
        "id": {"type": "string"},
        "when": {"type": "string", "format": "date-time"},
        "n": {"type": "integer"},
        "default": {"type": "string"}}}
