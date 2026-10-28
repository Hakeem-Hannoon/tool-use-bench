"""Episode runner: one sample, one model, one run — possibly several steps.

A classic sample (``max_steps: 1``) is a single model response, exactly as
before. With ``max_steps > 1`` the model works in a loop the way an agent
does in production: each non-terminal tool call gets the sample's scripted
result (:class:`benchmark.schema.MockToolResult`) and the model continues
until it replies, calls a terminal tool, or runs out of steps. Every call it
made across the episode is scored together.

Two protocols:

* ``native`` — the provider's tool-calling API; results go back as native
  tool-result messages.
* ``spotter_router`` — Spotter's JSON-router contract (:mod:`benchmark.router`):
  no ``tools`` parameter, JSON mode, one envelope per step, results returned
  as ``TOOL_RESULT`` user messages, with production's nudges for invalid JSON
  and unknown tool names (each counted as a recovery).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from . import router
from .schema import (
    EpisodeStep, ProviderResult, Sample, ScoringConfig, ToolCall, Usage,
)
from .scoring import compare_arguments, schema_errors_for
from .world import ApplyResult, World

if TYPE_CHECKING:  # pragma: no cover
    from .providers.base import BaseProvider, NativeCall, StepOutput

#: Returned when no scripted result matches a call.
MOCK_NOT_FOUND = {"error": "not_found",
                  "message": "No data matched this request."}

_SUBSET = ScoringConfig(require_exact_arguments=False)


def system_prompt_for(sample: Sample, protocol: str, default: str) -> str:
    """The system prompt a model sees for this sample under ``protocol``."""
    if protocol in sample.system_prompt_by_protocol:
        return sample.system_prompt_by_protocol[protocol]
    base = sample.system_prompt or default
    if protocol == "spotter_router":
        return router.build_router_system_prompt(
            base, [t.model_dump(mode="json") for t in sample.tools],
            terminal=set(sample.terminal_tools))
    return base


def mock_result(sample: Sample, call: ToolCall) -> Any:
    """First scripted result whose tool and ``when`` subset match the call."""
    for mock in sample.tool_results:
        if mock.tool != call.name:
            continue
        if not compare_arguments(call.name, mock.when, call.arguments, _SUBSET):
            return mock.result
    return MOCK_NOT_FOUND


#: Spotter caps every tool result it feeds back (MAX_TOOL_RESULT_CHARS in
#: orchestrator.ts); the benchmark applies the same cap so a model never sees
#: more than production would show it.
MAX_TOOL_RESULT_CHARS = 6000
_TRUNCATION_NOTE = ("Result truncated. Narrow the query (smaller limit, tighter "
                    "date range, one exercise) and try again for full detail.")


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def serialize_result(value: Any) -> str:
    """Compact JSON, capped like Spotter's serializeToolResult: an oversized
    result becomes {truncated, note, preview} with the preview shrunk until
    the whole message fits."""
    full = _compact(value if value is not None else {})
    if len(full) <= MAX_TOOL_RESULT_CHARS:
        return full
    keep = MAX_TOOL_RESULT_CHARS - 200
    out = _compact({"truncated": True, "note": _TRUNCATION_NOTE,
                    "preview": full[:keep]})
    while len(out) > MAX_TOOL_RESULT_CHARS and keep > 0:
        keep //= 2
        out = _compact({"truncated": True, "note": _TRUNCATION_NOTE,
                        "preview": full[:keep]})
    return out


def _arguments_of(envelope_arguments: Any) -> dict[str, Any]:
    if envelope_arguments is None:
        return {}
    if isinstance(envelope_arguments, dict):
        return envelope_arguments
    # Production passes non-object arguments straight to Zod, which rejects
    # them; keep the value visible so scoring fails deterministically.
    return {"_non_object_arguments": envelope_arguments}


def _proposal_payload() -> dict[str, Any]:
    return json.loads(router.PROPOSAL_CREATED_RESULT)


def resolve_effect(sample: Sample, world: World, call: ToolCall) -> ApplyResult:
    """Schema-check, then apply. Invalid calls and unknown ids do not commit
    and do not end the turn, so the model can correct them."""
    if sample.scoring.validate_arguments_schema:
        schema = sample.tool_schemas().get(call.name)
        if schema is not None:
            errs = schema_errors_for(call, schema)
            if errs:
                return ApplyResult(
                    {"error": "invalid_arguments",
                     "issues": [f"{e.argument_path or '<root>'}: {e.message}"
                                for e in errs]},
                    ends_turn=False)
    simulated = world.apply(call)
    if simulated is not None:
        return simulated
    if call.name in sample.terminal_tools and not sample.continue_after_terminal:
        return ApplyResult(_proposal_payload(), ends_turn=True)
    return ApplyResult(mock_result(sample, call), ends_turn=False)


def _router_feedback(name: str, payload: Any, *, ends_turn: bool) -> str:
    body = serialize_result(payload)
    if ends_turn:
        return f"TOOL_RESULT {name}:\n{body}"
    return router.tool_result_message(name, body)


async def run_dialogue(provider: "BaseProvider", sample: Sample, run_index: int,
                       default_system_prompt: str) -> ProviderResult:
    """One user turn per message. ``max_steps`` is the cap inside each turn.

    A terminal write commits to the fixture world and ends that turn. The
    next user message is appended, and reads from then on see the write.
    """
    protocol = getattr(provider.cfg, "protocol", "native")
    router_mode = protocol == "spotter_router"
    tools = [t.model_dump(mode="json") for t in sample.tools]
    offered = [t.function.name for t in sample.tools]
    world = World.from_tool_results(sample.tool_results)
    initial = world.project()

    system = system_prompt_for(sample, protocol, default_system_prompt)
    conv = provider.new_conversation(system, sample.turns[0])

    steps: list[EpisodeStep] = []
    calls: list[ToolCall] = []
    usage = Usage()
    recoveries = 0
    protocol_error: str | None = None
    started = time.monotonic()

    for turn_number, turn in enumerate(sample.turns, start=1):
        if turn_number > 1:
            conv.messages.append(provider.user_message(turn))
        json_nudge_used = False
        unknown_nudge_used = False

        for index in range(sample.max_steps):
            last = index == sample.max_steps - 1
            t0 = time.monotonic()
            out: StepOutput = await provider.step(
                conv, None if router_mode else tools, json_mode=router_mode)
            step = EpisodeStep(text=out.text, usage=out.usage, turn=turn_number,
                               latency_ms=int((time.monotonic() - t0) * 1000),
                               raw_response=out.raw)
            steps.append(step)
            usage = usage + out.usage

            if not router_mode:
                step_calls = [ToolCall(name=c.name, arguments=c.arguments)
                              for c in out.calls]
                from_text = False
                if not step_calls and provider.settings.allow_text_tool_call_fallback:
                    from .providers.base import parse_tool_calls_from_text
                    step_calls = parse_tool_calls_from_text(out.text)
                    from_text = bool(step_calls)
                step.tool_calls = step_calls
                calls.extend(step_calls)
                conv.messages.append(provider.assistant_message(out))
                if not step_calls:
                    break
                effects = [resolve_effect(sample, world, call) for call in step_calls]
                if out.calls and not from_text:
                    results = [
                        (native, serialize_result(effect.payload))
                        for native, effect in zip(out.calls, effects)
                    ]
                    conv.messages.extend(provider.tool_result_messages(out, results))
                if from_text or any(effect.ends_turn for effect in effects) or last:
                    break
                continue

            envelope = router.parse_router_output(out.text)
            if envelope is None:
                if router.looks_like_router_json(out.text):
                    step.envelope = "invalid_json"
                    if not last:
                        recoveries += 1
                        json_nudge_used = True
                        conv.messages.append(provider.assistant_message(out))
                        conv.messages.append(provider.user_message(
                            router.INVALID_JSON_NUDGE))
                        continue
                    protocol_error = ("no parseable JSON envelope"
                                      + (" after a retry" if json_nudge_used else ""))
                    break
                if not out.text.strip():
                    step.envelope = "empty"
                    protocol_error = "empty response"
                    break
                step.envelope = "prose"
                conv.messages.append(provider.assistant_message(out))
                break
            if envelope.kind == "reply":
                step.envelope = "reply"
                conv.messages.append(provider.assistant_message(out))
                break

            name = router.canonicalize_tool_name(envelope.tool or "", offered)
            if name is None:
                step.envelope = "unknown_tool"
                if not unknown_nudge_used and not last:
                    unknown_nudge_used = True
                    recoveries += 1
                    conv.messages.append(provider.assistant_message(out))
                    conv.messages.append(provider.user_message(
                        router.unknown_tool_nudge(envelope.tool or "", offered)))
                    continue
                calls.append(ToolCall(name=envelope.tool or "",
                                      arguments=_arguments_of(envelope.arguments)))
                protocol_error = f"unknown tool {envelope.tool!r}"
                break

            step.envelope = "tool"
            call = ToolCall(name=name, arguments=_arguments_of(envelope.arguments))
            step.tool_calls = [call]
            calls.append(call)
            effect = resolve_effect(sample, world, call)
            conv.messages.append(provider.assistant_message(out))
            conv.messages.append(provider.user_message(
                _router_feedback(name, effect.payload, ends_turn=effect.ends_turn)))
            if effect.ends_turn or last:
                break
        if protocol_error:
            break

    return ProviderResult(
        model=provider.cfg.name,
        provider=provider.cfg.provider,
        sample_id=sample.id,
        run_index=run_index,
        raw_response=steps[0].raw_response if len(steps) == 1 else None,
        tool_calls=calls,
        text="\n".join(s.text for s in steps if s.text),
        error=None,
        latency_ms=int((time.monotonic() - started) * 1000),
        usage=usage,
        steps=steps,
        protocol_error=protocol_error,
        recoveries=recoveries,
        world_state=world.project(),
        initial_world=initial,
    )


async def run_episode(provider: "BaseProvider", sample: Sample, run_index: int,
                      default_system_prompt: str) -> ProviderResult:
    if sample.expected_state is not None:
        return await run_dialogue(provider, sample, run_index,
                                  default_system_prompt)
    protocol = getattr(provider.cfg, "protocol", "native")
    router_mode = protocol == "spotter_router"
    tools = [t.model_dump(mode="json") for t in sample.tools]
    offered = [t.function.name for t in sample.tools]
    terminal = set(sample.terminal_tools)

    system = system_prompt_for(sample, protocol, default_system_prompt)
    conv = provider.new_conversation(system, sample.prompt)

    steps: list[EpisodeStep] = []
    calls: list[ToolCall] = []
    usage = Usage()
    recoveries = 0
    protocol_error: str | None = None
    json_nudge_used = False
    unknown_nudge_used = False
    started = time.monotonic()

    for index in range(sample.max_steps):
        last = index == sample.max_steps - 1
        t0 = time.monotonic()
        out: StepOutput = await provider.step(
            conv, None if router_mode else tools, json_mode=router_mode)
        step = EpisodeStep(text=out.text, usage=out.usage,
                           latency_ms=int((time.monotonic() - t0) * 1000),
                           raw_response=out.raw)
        steps.append(step)
        usage = usage + out.usage

        if not router_mode:
            step_calls = [ToolCall(name=c.name, arguments=c.arguments)
                          for c in out.calls]
            from_text = False
            if not step_calls and provider.settings.allow_text_tool_call_fallback:
                from .providers.base import parse_tool_calls_from_text
                step_calls = parse_tool_calls_from_text(out.text)
                from_text = bool(step_calls)
            step.tool_calls = step_calls
            calls.extend(step_calls)
            # A text reply ends the episode; so does a text-parsed call, which
            # has no native id to answer.
            if not out.calls or from_text:
                break
            if (any(c.name in terminal for c in step_calls)
                    and not sample.continue_after_terminal) or last:
                break
            results: list[tuple[NativeCall, str]] = []
            for native_call, call in zip(out.calls, step_calls):
                payload = (router.PROPOSAL_CREATED_RESULT if call.name in terminal
                           else serialize_result(mock_result(sample, call)))
                results.append((native_call, payload))
            conv.messages.append(provider.assistant_message(out))
            conv.messages.extend(provider.tool_result_messages(out, results))
            continue

        # ---- spotter_router -------------------------------------------
        envelope = router.parse_router_output(out.text)
        if envelope is None:
            if router.looks_like_router_json(out.text):
                step.envelope = "invalid_json"
                if not last:
                    recoveries += 1
                    json_nudge_used = True
                    conv.messages.append(provider.assistant_message(out))
                    conv.messages.append(provider.user_message(
                        router.INVALID_JSON_NUDGE))
                    continue
                protocol_error = ("no parseable JSON envelope"
                                  + (" after a retry" if json_nudge_used else ""))
                break
            if not out.text.strip():
                step.envelope = "empty"
                protocol_error = "empty response"
                break
            step.envelope = "prose"  # production shows prose as the reply
            break
        if envelope.kind == "reply":
            step.envelope = "reply"
            break

        name = router.canonicalize_tool_name(envelope.tool or "", offered)
        if name is None:
            step.envelope = "unknown_tool"
            if not unknown_nudge_used and not last:
                unknown_nudge_used = True
                recoveries += 1
                conv.messages.append(provider.assistant_message(out))
                conv.messages.append(provider.user_message(
                    router.unknown_tool_nudge(envelope.tool or "", offered)))
                continue
            # Production answers with an "unknown tool" error card.
            calls.append(ToolCall(name=envelope.tool or "",
                                  arguments=_arguments_of(envelope.arguments)))
            protocol_error = f"unknown tool {envelope.tool!r}"
            break

        step.envelope = "tool"
        call = ToolCall(name=name, arguments=_arguments_of(envelope.arguments))
        step.tool_calls = [call]
        calls.append(call)
        if (name in terminal and not sample.continue_after_terminal) or last:
            break
        conv.messages.append(provider.assistant_message(out))
        if name in terminal:
            conv.messages.append(provider.user_message(
                router.proposal_result_message(name)))
        else:
            conv.messages.append(provider.user_message(router.tool_result_message(
                name, serialize_result(mock_result(sample, call)))))

    return ProviderResult(
        model=provider.cfg.name,
        provider=provider.cfg.provider,
        sample_id=sample.id,
        run_index=run_index,
        raw_response=steps[0].raw_response if len(steps) == 1 else None,
        tool_calls=calls,
        text="\n".join(s.text for s in steps if s.text),
        error=None,
        latency_ms=int((time.monotonic() - started) * 1000),
        usage=usage,
        steps=steps,
        protocol_error=protocol_error,
        recoveries=recoveries,
    )
