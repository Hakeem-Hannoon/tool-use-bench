"""Multi-step episodes, the Spotter router protocol, wire formats, usage."""

import json

import pytest

from fakes import ScriptedProvider, envelope, model_cfg, native, text

from benchmark import router
from benchmark.episode import MOCK_NOT_FOUND, mock_result, system_prompt_for
from benchmark.providers.anthropic_provider import (
    extract_anthropic_usage, replayable_content,
)
from benchmark.providers.base import NativeCall, ProviderSettings, StepOutput
from benchmark.providers.gemini_provider import extract_gemini_usage
from benchmark.providers.openai_provider import (
    OpenAIChatFamilyProvider, extract_openai_native_calls, extract_openai_usage,
)
from benchmark.runner import _record_from_result
from benchmark.schema import Sample, ToolCall, Usage

WORKOUT_ID = "0b6f6c55-3f51-4c1e-9a52-0d1f8a8e7c11"
SET_ID = "8d3a1e2f-6b7c-4d5e-8f90-a1b2c3d4e5f6"


def _tool(name, **props):
    return {"type": "function", "function": {
        "name": name, "description": f"{name} tool",
        "parameters": {"type": "object", "properties": props}}}


def edit_sample(**overrides) -> Sample:
    data = {
        "id": "0001_edit_flow", "category": "id_grounded_edits",
        "difficulty": "hard",
        "prompt": "make yesterday's bench 195",
        "tools": [_tool("get_recent_workouts", limit={"type": "integer"}),
                  _tool("create_pending_workout_edit",
                        workoutId={"type": "string"}),
                  _tool("create_pending_macro_log")],
        "expected_tool_calls": [{"name": "create_pending_workout_edit",
                                 "arguments": {"workoutId": WORKOUT_ID}}],
        "scoring": {"require_exact_arguments": False,
                    "allowed_extra_tools": ["get_recent_workouts"]},
        "max_steps": 4,
        "terminal_tools": ["create_pending_workout_edit",
                           "create_pending_macro_log"],
        "tool_results": [
            {"tool": "get_recent_workouts", "when": {"limit": {"$range": [1, 3]}},
             "result": {"workouts": [{"workoutId": WORKOUT_ID}], "few": True}},
            {"tool": "get_recent_workouts",
             "result": {"workouts": [{"workoutId": WORKOUT_ID}]}},
        ],
    }
    data.update(overrides)
    return Sample.model_validate(data)


# ------------------------------------------------------------- mock results


def test_mock_result_uses_first_matching_rule():
    sample = edit_sample()
    few = mock_result(sample, ToolCall(name="get_recent_workouts",
                                       arguments={"limit": 1}))
    assert few["few"] is True
    many = mock_result(sample, ToolCall(name="get_recent_workouts",
                                        arguments={"limit": 20}))
    assert "few" not in many
    assert mock_result(sample, ToolCall(name="create_pending_macro_log")) \
        == MOCK_NOT_FOUND


# ------------------------------------------------------------ native episodes


async def test_native_episode_feeds_results_and_stops_at_terminal():
    provider = ScriptedProvider(model_cfg(), ProviderSettings(), [
        native(("get_recent_workouts", {"limit": 1})),
        native(("create_pending_workout_edit",
                {"workoutId": WORKOUT_ID, "summary": "bench 195"})),
        text("should never be requested"),
    ])
    result = await provider.complete(edit_sample(), 0, "sys")
    assert [c.name for c in result.tool_calls] == [
        "get_recent_workouts", "create_pending_workout_edit"]
    assert len(provider.requests) == 2  # terminal call ended the episode
    second = provider.requests[1]
    assert second["tools"] == ["get_recent_workouts",
                               "create_pending_workout_edit",
                               "create_pending_macro_log"]
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert json.loads(tool_msg["content"])["few"] is True
    assert result.usage.input_tokens == 200
    record = _record_from_result(edit_sample(), result, save_raw=False,
                                 model_cfg=model_cfg())
    assert record.success, record.scoring_result.reason
    assert record.step_count == 2


async def test_native_text_reply_ends_episode():
    provider = ScriptedProvider(model_cfg(), ProviderSettings(),
                                [text("which workout?")])
    result = await provider.complete(edit_sample(), 0, "")
    assert result.tool_calls == []
    assert len(provider.requests) == 1


async def test_native_max_steps_caps_the_episode():
    loop = [native(("get_recent_workouts", {"limit": 5}))] * 10
    provider = ScriptedProvider(model_cfg(), ProviderSettings(), loop)
    result = await provider.complete(edit_sample(max_steps=3), 0, "")
    assert len(provider.requests) == 3
    assert len(result.tool_calls) == 3


async def test_continue_after_terminal_reports_proposal_created():
    sample = edit_sample(
        continue_after_terminal=True,
        expected_tool_calls=[
            {"name": "create_pending_workout_edit", "arguments": {}},
            {"name": "create_pending_macro_log", "arguments": {}}],
    )
    provider = ScriptedProvider(model_cfg(), ProviderSettings(), [
        native(("create_pending_workout_edit", {"workoutId": WORKOUT_ID})),
        native(("create_pending_macro_log", {})),
        text("done"),
    ])
    result = await provider.complete(sample, 0, "")
    assert [c.name for c in result.tool_calls] == [
        "create_pending_workout_edit", "create_pending_macro_log", ]
    tool_msg = provider.requests[1]["messages"][-1]
    assert "proposal_created" in tool_msg["content"]


# ------------------------------------------------------------ router episodes


def router_cfg():
    return model_cfg(protocol="spotter_router")


async def test_router_sends_no_tools_in_json_mode_and_parses_envelopes():
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [
        envelope("get_recent_workouts", limit=1),
        envelope("create_pending_workout_edit", workoutId=WORKOUT_ID,
                 summary="bench 195"),
    ])
    sample = edit_sample()
    result = await provider.complete(sample, 0, "base prompt")
    first = provider.requests[0]
    assert first["tools"] is None and first["json_mode"] is True
    assert "Available tools:" in first["system"]
    assert "create_pending_workout_edit" in first["system"]
    follow_up = provider.requests[1]["messages"][-1]["content"]
    assert follow_up.startswith("TOOL_RESULT get_recent_workouts:\n")
    assert follow_up.endswith(router.CONTINUE_SUFFIX)
    assert [s.envelope for s in result.steps] == ["tool", "tool"]
    record = _record_from_result(sample, result, save_raw=True,
                                 model_cfg=router_cfg())
    assert record.success, record.scoring_result.reason
    assert record.protocol == "spotter_router"


async def test_router_uses_sample_prompt_verbatim_when_provided():
    sample = edit_sample(system_prompt_by_protocol={
        "spotter_router": "SPOTTER PROMPT", "native": "NATIVE PROMPT"})
    assert system_prompt_for(sample, "spotter_router", "x") == "SPOTTER PROMPT"
    assert system_prompt_for(sample, "native", "x") == "NATIVE PROMPT"
    provider = ScriptedProvider(router_cfg(), ProviderSettings(),
                                [envelope(reply="hey")])
    await provider.complete(sample, 0, "ignored")
    assert provider.requests[0]["system"] == "SPOTTER PROMPT"


async def test_router_canonicalizes_plural_tool_names():
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [
        envelope("Create_Pending_Workout_Edits", workoutId=WORKOUT_ID),
    ])
    result = await provider.complete(edit_sample(max_steps=1), 0, "")
    assert result.tool_calls[0].name == "create_pending_workout_edit"


async def test_router_invalid_json_gets_one_nudge_then_recovers():
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [
        text('{"kind":"tool","tool":"create_pending_workout_edit","argu'),
        envelope("create_pending_workout_edit", workoutId=WORKOUT_ID),
    ])
    result = await provider.complete(edit_sample(), 0, "")
    assert result.recoveries == 1
    assert result.protocol_error is None
    assert provider.requests[1]["messages"][-1]["content"] == router.INVALID_JSON_NUDGE
    assert [s.envelope for s in result.steps] == ["invalid_json", "tool"]


async def test_router_unparseable_on_last_step_is_protocol_error():
    provider = ScriptedProvider(router_cfg(), ProviderSettings(),
                                [text('{"kind": "tool", "tool": ')])
    result = await provider.complete(edit_sample(max_steps=1), 0, "")
    assert result.protocol_error
    record = _record_from_result(edit_sample(max_steps=1), result,
                                 save_raw=False, model_cfg=router_cfg())
    assert record.failure_type == "protocol_error"


async def test_router_unknown_tool_nudges_with_close_names():
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [
        envelope("pending_workout_edit_tool"),
        envelope("create_pending_workout_edit", workoutId=WORKOUT_ID),
    ])
    result = await provider.complete(edit_sample(), 0, "")
    nudge = provider.requests[1]["messages"][-1]["content"]
    assert nudge.startswith('UNKNOWN TOOL: "pending_workout_edit_tool"')
    assert result.recoveries == 1
    assert [c.name for c in result.tool_calls] == ["create_pending_workout_edit"]


async def test_router_prose_and_reply_end_with_no_calls():
    for step in (text("just answer in prose"), envelope(reply="hi")):
        provider = ScriptedProvider(router_cfg(), ProviderSettings(), [step])
        result = await provider.complete(edit_sample(), 0, "")
        assert result.tool_calls == [] and result.protocol_error is None


async def test_router_empty_response_is_protocol_error():
    provider = ScriptedProvider(router_cfg(), ProviderSettings(), [text("  ")])
    result = await provider.complete(edit_sample(), 0, "")
    assert result.protocol_error == "empty response"


# ------------------------------------------------------- router parser port


@pytest.mark.parametrize("raw, kind, tool", [
    ('{"kind":"tool","tool":"x","arguments":{}}', "tool", "x"),
    ('Sure!\n```json\n{"kind":"reply","message":"hi"}\n```', "reply", None),
    ('prefix {"kind":"reply","message":"a {b} c"} suffix', "reply", None),
])
def test_parse_router_output_valid(raw, kind, tool):
    env = router.parse_router_output(raw)
    assert env is not None and env.kind == kind and env.tool == tool


@pytest.mark.parametrize("raw", [
    "", "no json here", '{"kind":"maybe"}',
    '{"kind":"reply","options":["1","2","3","4","5","6","7"]}',
    '{"kind":"tool","tool":5}',
])
def test_parse_router_output_invalid(raw):
    assert router.parse_router_output(raw) is None


def test_looks_like_router_json():
    assert router.looks_like_router_json('```json\n{"kind": "to')
    assert router.looks_like_router_json('blah "tool": "x"')
    assert router.looks_like_router_json("<svg viewBox='0 0 1 1'>")
    assert not router.looks_like_router_json("Bench 3x5 sounds great!")


def test_canonicalize_and_close_names():
    names = ["search_workouts", "get_recent_workouts", "show_table"]
    assert router.canonicalize_tool_name("search_workout", names) == "search_workouts"
    assert router.canonicalize_tool_name("SHOW_TABLE", names) == "show_table"
    assert router.canonicalize_tool_name("show_tables", names) == "show_table"
    assert router.canonicalize_tool_name("tables", names) is None
    assert router.close_tool_names("recent_workouts", names) == ["get_recent_workouts"]
    assert router.close_tool_names("tab", names) == []


# ------------------------------------------------------------- wire formats


def test_openai_assistant_message_replays_raw_arguments():
    provider = OpenAIChatFamilyProvider(model_cfg(), ProviderSettings())
    step = StepOutput(raw={}, text="", calls=[
        NativeCall(id="c1", name="f", arguments={"a": 1}, raw_arguments='{"a":1}')])
    msg = provider.assistant_message(step)
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert msg["content"] is None
    results = provider.tool_result_messages(step, [(step.calls[0], '{"ok":true}')])
    assert results == [{"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'}]


def test_openai_native_calls_get_synthesized_ids():
    raw = {"choices": [{"message": {"tool_calls": [
        {"type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]}
    assert extract_openai_native_calls(raw)[0].id == "call_0"


def test_anthropic_replayable_content_drops_response_only_fields():
    raw = {"content": [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "ok", "citations": None},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1},
         "caller": None},
    ]}
    assert replayable_content(raw) == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
    ]


def test_usage_extraction_per_provider():
    assert extract_openai_usage({"usage": {
        "prompt_tokens": 1000, "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 800}}}) == Usage(
            input_tokens=1000, output_tokens=50, cached_input_tokens=800)
    assert extract_openai_usage({"usage": {"prompt_tokens": 5,
                                           "prompt_tokens_details": None}}) \
        == Usage(input_tokens=5)
    assert extract_anthropic_usage({"usage": {
        "input_tokens": 100, "output_tokens": 20,
        "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}}) \
        == Usage(input_tokens=1000, output_tokens=20, cached_input_tokens=900)
    assert extract_gemini_usage({"usage_metadata": {
        "prompt_token_count": 400, "candidates_token_count": 30,
        "thoughts_token_count": 70, "cached_content_token_count": 100}}) \
        == Usage(input_tokens=400, output_tokens=100, cached_input_tokens=100)


def test_pricing_cost_uses_cached_rate():
    cfg = model_cfg(pricing={"input_per_mtok": 0.2, "output_per_mtok": 0.5,
                             "cached_input_per_mtok": 0.05})
    usage = Usage(input_tokens=10_000, output_tokens=400, cached_input_tokens=7_000)
    # 3k fresh * 0.2 + 7k cached * 0.05 + 400 out * 0.5, per million
    assert cfg.pricing.cost(usage) == pytest.approx((600 + 350 + 200) / 1e6)


def test_model_key_distinguishes_protocol_and_label():
    a = model_cfg()
    b = model_cfg(protocol="spotter_router")
    c = model_cfg(protocol="spotter_router", label="hot")
    assert a.key == "openai/fake-model"  # unchanged for existing configs
    assert len({a.key, b.key, c.key}) == 3
    assert b.display == "fake-model (spotter_router)" and c.display == "hot"
