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
from .scoring import compare_arguments

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


def serialize_result(value: Any) -> str:
    """Compact JSON, like Spotter's JSON.stringify of a tool result."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _arguments_of(envelope_arguments: Any) -> dict[str, Any]:
    if envelope_arguments is None:
        return {}
    if isinstance(envelope_arguments, dict):
        return envelope_arguments
    # Production passes non-object arguments straight to Zod, which rejects
    # them; keep the value visible so scoring fails deterministically.
    return {"_non_object_arguments": envelope_arguments}


async def run_episode(provider: "BaseProvider", sample: Sample, run_index: int,
                      default_system_prompt: str) -> ProviderResult:
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
